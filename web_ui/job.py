"""Single current-job runner.

Deliberately minimal for a local single-user tool: at most one job runs at a
time, nothing is persisted across restarts. The job is launched under a
**pseudo-terminal (PTY)** so the child sees a real TTY — this keeps progress
bars (the manual ``\\r`` writer in ``precision_track/outputs/display.py`` and
``tqdm`` in the pipelined tracker) behaving as they would in a terminal. The
raw byte stream is parsed with carriage-return semantics into:
  - committed lines  -> kept in a rolling ``tail`` (SSE ``event: line``)
  - the in-progress line -> ``live`` (SSE ``event: live``), updated in place
so a progress bar renders as a single updating line in the embedded terminal.
"""

import asyncio
import codecs
import datetime as _dt
import fcntl
import os
import pty
import re
import signal
import struct
import sys
import termios
from collections import deque
from typing import AsyncIterator, Dict, List, Optional

from .paths import TOOLS_DIR

_MAX_TAIL = 2000  # committed lines kept for refresh-replay
_COLS, _ROWS = 120, 40

# CSI / OSC / other escape sequences — stripped for a clean text display.
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI ... final byte
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC ... BEL/ST
    r"|\x1b[@-Z\\-_]"  # two-char escapes
)


class Job:
    def __init__(self, tool: str, argv: list):
        self.tool = tool
        self.argv = argv
        self.status = "running"  # running | done | failed | stopped
        self.started_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.ended_at: Optional[str] = None
        self.returncode: Optional[int] = None
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.master_fd: Optional[int] = None
        self.live: str = ""  # current in-progress (unterminated) line
        self._tail: deque = deque(maxlen=_MAX_TAIL)
        self._subscribers: List[asyncio.Queue] = []
        self._reader_done = asyncio.Event()
        # assembler state
        self._cur = ""
        self._pending_cr = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    # ----------------------------------------------------------- emit/stream
    def _publish(self, event: Dict) -> None:
        for q in list(self._subscribers):
            q.put_nowait(event)

    def emit_line(self, text: str) -> None:
        """Commit a finished line."""
        self._tail.append(text)
        self.live = ""
        self._publish({"kind": "line", "text": text})

    def emit_live(self, text: str) -> None:
        """Update the current in-progress line in place."""
        self.live = text
        self._publish({"kind": "live", "text": text})

    def feed(self, data: bytes) -> None:
        """Parse a chunk of raw PTY output with carriage-return semantics.

        ``\\r\\n`` is treated as a single newline; a lone ``\\r`` moves the cursor
        to column 0 so progress bars overwrite the current line in place.
        """
        text = self._ansi_strip(self._decoder.decode(data))
        for ch in text:
            if self._pending_cr:
                self._pending_cr = False
                if ch == "\n":
                    self.emit_line(self._cur)
                    self._cur = ""
                    continue
                self._cur = ""  # lone CR: overwrite the current line
            if ch == "\r":
                self._pending_cr = True
            elif ch == "\n":
                self.emit_line(self._cur)
                self._cur = ""
            elif ch == "\b":
                self._cur = self._cur[:-1]
            elif ch == "\t":
                self._cur += "    "
            elif ord(ch) >= 32:
                self._cur += ch
            # other control chars dropped
        if self._cur:
            self.emit_live(self._cur)

    @staticmethod
    def _ansi_strip(text: str) -> str:
        return _ANSI_RE.sub("", text)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def tail(self) -> list:
        return list(self._tail)

    def info(self) -> Dict:
        return {
            "tool": self.tool,
            "argv": self.argv,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "returncode": self.returncode,
        }


_current: Optional[Job] = None


def current() -> Optional[Job]:
    return _current


async def start(tool: str, argv: list) -> Job:
    global _current
    if _current is not None and _current.status == "running":
        raise RuntimeError("A job is already running. Stop it before launching another.")

    job = Job(tool, argv)
    _current = job

    env = dict(os.environ)
    env.setdefault("ATEN_CPU_CAPABILITY", "default")  # per tools/README CPU note
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = str(_COLS)
    env["LINES"] = str(_ROWS)

    master_fd, slave_fd = pty.openpty()
    try:
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", _ROWS, _COLS, 0, 0))
    except OSError:
        pass
    try:
        # Disable output post-processing so the PTY does not translate the
        # child's "\n" into "\r\n" (which would otherwise corrupt our CR-based
        # progress-bar handling).
        attrs = termios.tcgetattr(slave_fd)
        attrs[1] &= ~termios.OPOST  # oflag
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    except (OSError, termios.error):
        pass

    job.emit_line(f"$ python {tool} {' '.join(argv)}")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        tool,
        *argv,
        cwd=str(TOOLS_DIR),
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        start_new_session=True,  # own session/process group -> stop() kills children
    )
    os.close(slave_fd)  # parent keeps only the master end
    job.proc = proc
    job.master_fd = master_fd

    _attach_reader(job)
    asyncio.create_task(_finalize(job))
    return job


def _attach_reader(job: Job) -> None:
    loop = asyncio.get_event_loop()
    fd = job.master_fd
    os.set_blocking(fd, False)

    def on_readable():
        try:
            data = os.read(fd, 65536)
        except BlockingIOError:
            return
        except OSError:
            data = b""  # EIO when the slave side closes -> EOF
        if data:
            job.feed(data)
        else:
            loop.remove_reader(fd)
            job._reader_done.set()

    loop.add_reader(fd, on_readable)


async def _finalize(job: Job) -> None:
    try:
        await job._reader_done.wait()
        if job._cur:  # commit a trailing line that had no newline
            job.emit_line(job._cur)
            job._cur = ""
        rc = await job.proc.wait()
        job.returncode = rc
        if job.status != "stopped":
            job.status = "done" if rc == 0 else "failed"
    except Exception as exc:  # pragma: no cover - defensive
        job.status = "failed"
        job.emit_line(f"[web_ui] reader error: {exc}")
    finally:
        if job.master_fd is not None:
            try:
                os.close(job.master_fd)
            except OSError:
                pass
            job.master_fd = None
        job.ended_at = _dt.datetime.now().isoformat(timespec="seconds")
        job.emit_line(f"[web_ui] process {job.status} (exit code {job.returncode}).")
        job._publish({"kind": "end", "text": job.status})


async def stop() -> Dict:
    job = _current
    if job is None or job.status != "running" or job.proc is None:
        return {"stopped": False, "message": "No job is running."}
    job.status = "stopped"
    try:
        os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(job.proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"stopped": True}


async def stream(job: Job) -> AsyncIterator[str]:
    """SSE frames: replay committed tail + current live line, then live events."""
    for line in job.tail():
        yield _sse_event("line", line)
    if job.live:
        yield _sse_event("live", job.live)
    if job.status != "running":
        yield _sse_event("end", job.status)
        return
    q = job.subscribe()
    try:
        while True:
            event = await q.get()
            yield _sse_event(event["kind"], event["text"])
            if event["kind"] == "end":
                break
    finally:
        job.unsubscribe(q)


def _sse_event(event: str, data: str) -> str:
    # data never contains a newline (committed/live lines are single-line).
    return f"event: {event}\ndata: {data}\n\n"
