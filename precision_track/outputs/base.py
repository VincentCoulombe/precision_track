import abc


class BaseOutput(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, data: dict) -> None:
        """Load data into the output.

        Args:
            data (dict): The data to load into the output
        """

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the output, but keep its original initialization."""

    @abc.abstractmethod
    def save(self) -> None:
        """Save an output to disk at sink_path."""

    @abc.abstractmethod
    def read(self, source_path: str) -> None:
        """Read an output from source_path and load it into memory.

        Args:
            source_path (str): The path to the output, the extension is inferred.
        """
