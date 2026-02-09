from collections.abc import Iterable
from typing import Any, List, Optional
from time import perf_counter
import numpy as np
import os

from precision_track.registry import OUTPUTS


class Result:

    def __init__(self, outputs: Optional[List[dict]] = None) -> None:
        """Manage the multiple outputs and synchronize their saving, reading
        and iteration.

        Args:
            outputs (Optional[List[dict]]): A list of output's config.
        """
        self._outputs = []
        if outputs is not None:
            if not isinstance(outputs, Iterable):
                raise TypeError("`outputs` must be a list of dicts")
            self._outputs = [OUTPUTS.build(output) for output in outputs]
        self._length = 0

    @property
    def outputs(self):
        return self._outputs

    def __call__(self, data: Any, profile: Optional[list] = None) -> None:
        """Load new data into the outputs.

        Args:
            data (Any): The data to load into the outputs
        """
        if isinstance(profile, list):
            saving_result_start = perf_counter()
        for output in self._outputs:
            output(data)
        if isinstance(profile, list):
            profile.append(perf_counter() - saving_result_start)

    def __iter__(self):
        self._current = 0
        return self

    def __len__(self):
        return self._length

    def __next__(self) -> List[dict]:
        if self._current >= self._length:
            raise StopIteration

        results = []
        for output in self._outputs:
            key = output.__class__.__name__
            if hasattr(output, "subtype"):
                key += f"-{output.subtype}"
            try:
                results.append({key: np.array(output[self._current])})
            except IndexError:
                results.append({key: np.array([])})

        self._current += 1
        return results

    def reset(self) -> None:
        for output in self._outputs:
            output.reset()

    def save(self) -> None:
        for output in self._outputs:
            output.save()

    def read(self, not_exists_ok: Optional[bool] = False) -> None:
        output_to_remove = []
        for i, output in enumerate(self._outputs):
            if not os.path.exists(output.path):
                if not_exists_ok:
                    output_to_remove.append(i)
                    continue
                else:
                    raise RuntimeError(f"The '{output.__class_.__name__}' output as an invalid path: {output.path}")
            output.read()
            if len(output) > self._length:
                self._length = len(output)
        for o in reversed(output_to_remove):
            self._outputs.pop(o)
