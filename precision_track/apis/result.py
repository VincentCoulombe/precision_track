from collections.abc import Iterable
from typing import Any, List, Optional

import numpy as np

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

    def __call__(self, data: Any) -> None:
        """Load new data into the outputs.

        Args:
            data (Any): The data to load into the outputs
        """
        for output in self._outputs:
            output(data)

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
            try:
                results.append({output.__class__.__name__: np.array(output[self._current])})
            except IndexError:
                results.append({output.__class__.__name__: np.array([])})

        self._current += 1
        return results

    def reset(self) -> None:
        for output in self._outputs:
            output.reset()

    def save(self) -> None:
        for output in self._outputs:
            output.save()

    def read(self) -> None:
        for output in self._outputs:
            output.read()
            if len(output) > self._length:
                self._length = len(output)
