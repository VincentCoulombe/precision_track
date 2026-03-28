from typing import Any, List, Optional

import numpy as np
import seaborn as sns
import supervision as sv


class ColorPalette(sv.ColorPalette):

    def __init__(
        self,
        size: Optional[int] = 20,
        names: Optional[List[Any]] = None,
        colors: Optional[List[List[int]]] = None,
        nan_color: Optional[List[int]] = None,
        normalized_colors: Optional[bool] = False,
    ):
        """A color palette based on seaborn's. For more palette options, you can visit:
        https://seaborn.pydata.org/tutorial/color_palettes.html

        Args:
            size (Optional[int], optional): The size of the color palette (how many ids do you expect). Defaults to 20.
            names (Optional[List[Any]], optional): A list of seaborn's color palette, they will split the size evenly. Defaults to ["deep", "Spectral"].
            colors (Optional[List[List[int]]], optional): A list of RGB colors to use. If provided with names=None, only these colors are used. Otherwise,
            they are combined with seaborn palettes.
        """
        np.random.seed(42)

        if colors is not None and names is None:
            self.colors = [sv.Color(*color) for color in colors]
        else:
            if names is None:
                names = ["deep", "Spectral"]

            # Calculate how many colors to get from seaborn palettes
            num_custom_colors = len(colors) if colors is not None else 0
            sns_size = size - num_custom_colors

            palette = np.zeros((sns_size, 3), dtype=float)
            splits = self._split_evenly(sns_size, len(names))
            sns_palettes = [sns.color_palette(name, split) for name, split in zip(names, splits)]
            i = 0
            for sns_palette in sns_palettes:
                for color in sns_palette:
                    palette[i, ...] = np.array(color)
                    i += 1
            if not normalized_colors:
                palette *= 255
                palette = palette.astype(int)
            np.random.shuffle(palette)
            self.colors = []
            for color in palette.tolist():
                self.colors.append(sv.Color(*color))

            # Append custom colors at the end
            if colors is not None:
                for color in colors:
                    self.colors.append(sv.Color(*color))

        if nan_color is None:
            nan_color = [0, 0, 0]
        self.nan_color = nan_color

    def by_idx(self, idx: int) -> sv.Color:
        if idx < 0:
            return sv.Color(*self.nan_color)
        else:
            return super().by_idx(int(idx))

    @staticmethod
    def _split_evenly(size: int, nb_names: int):
        q = size // nb_names
        r = size % nb_names
        splits = [q] * nb_names
        for i in range(r):
            splits[i] += 1
        return splits
