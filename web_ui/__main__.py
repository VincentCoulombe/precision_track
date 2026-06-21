"""Entry point: ``python -m web_ui`` launches the local UI on 127.0.0.1:8000."""

import argparse
import threading
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="PrecisionTrack local configuration & launch UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"PrecisionTrack UI running at {url}")
    uvicorn.run("web_ui.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
