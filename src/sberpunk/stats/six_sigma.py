__all__ = ["QualityClass", "SberProcess", "SigmaClassifier"]


# imports: standard library
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Union

# imports: 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm

# imports: user modules
from sberpunk import RUNTIME_CONFIG


# settings
plt.rcParams.update(RUNTIME_CONFIG)
plt.style.use("seaborn-v0_8-ticks")


scalar = int | float
array_like = Union[list, tuple, np.ndarray, pd.Series]


class QualityClass:
    """
    Quality class defined by a color label and supremum.

    Attributes:
        color (str):
            Quality class color label. Must be a valid `matplotlib` color name.

        supremum (int | float):
            Unreachable upper bound of the sigma interval that corresponds to
            the quality class.

            E.g., "red" class never reaches sigma=2.1 supremum which is the
            exact lower bound of the next, "yellow", class and "yellow" never
            reaches 4.1 which is the lower bound of the "green" class.
    """

    def __init__(self, color: str, supremum: scalar) -> None:
        self.color = color
        self.supremum = supremum

    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.supremum < other.supremum
        return NotImplemented


class SigmaClassifier:
    """
    A classifier responsible for getting the quality class color label for a
    sigma value.

    Attributes:
        quality_classes (list[QualityClass]):
            List of `QualityClass` instances defining the classifier. They are
            not supposed to be sorted by a supremum. A quality class with
            `float("inf")` supremum is mandatory.
    """

    def __init__(self, quality_classes: list[QualityClass]) -> None:
        if not quality_classes:
            raise ValueError("No quality classes defined")

        self.quality_classes = sorted(quality_classes)

        if self.quality_classes[-1].supremum != float("inf"):
            msg = (
                'Top quality class is not defined: `(-inf, +inf)` is not '
                'covered. Set `float("inf")` supremum for a top quality class'
            )
            raise ValueError(msg)

        self.get_color_label = np.vectorize(
            self._get_color_label, excluded="self"
        )

    def _get_color_label(self, sigma: np.ndarray) -> str:
        """Get the quality class and return its color label. """
        for quality_class in self:
            if sigma < quality_class.supremum:
                return quality_class.color
        raise ValueError("Line of code unreachable under any circumstances")

    def __iter__(self) -> Iterator:
        return iter(self.quality_classes)


@dataclass
class SberProcess:
    """
    Process to evaluate with the "6 sigma" approach.

    Attributes:
        classifier (SigmaClassifier):
            A classifier responsible for getting the quality class color label
            for a given sigma value.

        actions (int | float | list | tuple | numpy.ndarray | pandas.Series):
            Total number of actions.

        fails (int | float | list | tuple | numpy.ndarray | pandas.Series):
            Number of actions qualified as failed.

        name (str | list | tuple | numpy.ndarray | pandas.Series | None):
            Name of the process. Defaults to `None`.
    """

    classifier: SigmaClassifier

    # action considered as a single run/try of the process
    actions: scalar | array_like
    fails: scalar | array_like
    name: str | array_like | None = None

    def __post_init__(self) -> None:
        actions, fails = map(self._to_iterable, (self.actions, self.fails))

        if actions.size != fails.size:
            msg = "`actions` and `fails` must be of the same length"
            raise ValueError(msg)

        if (actions < 0).any() or (fails < 0).any():
            msg = (
                "Neither total number of actions nor number of fails can be "
                "negative"
            )
            raise ValueError(msg)

        if (fails > actions).any():
            msg = (
                "Number of fails can't be greater than the total number of "
                "actions"
            )
            raise ValueError(msg)

        if self.name is not None:
            name = self._to_iterable(self.name)
            if len(name) != actions.size:
                msg = (
                    "Keep name as `None` (default) or explicitly provide every "
                    "process with a meaningful name"
                )
                raise ValueError(msg)
        else:
            name = [None] * actions.size

        self.defect_ratio = fails / actions

        # normal continuous random variable with loc=1.5 and scale=1
        self._norm = norm(1.5)

        # ppf stands for "percent point function"
        sigma = self._norm.ppf(1 - self.defect_ratio)
        self.sigma = np.minimum(sigma, sys.maxsize)
        self.label = self.classifier.get_color_label(self.sigma)

        self._attrs = [
            actions, fails, name,
            self.defect_ratio, self.sigma, self.label
        ]

    @staticmethod
    def _to_iterable(value) -> list[str] | np.ndarray | pd.Series:
        """Cast value to an iterable. """
        # numpy array for vectorized calculation
        if isinstance(value, (int, float)):
            return np.array([value])
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return np.array(value)
        return value

    def plot(self) -> None:
        """Visualization of the sigma value and defect ratio. """
        nrows = self._attrs[0].size
        ax = plt.subplots(nrows=nrows, figsize=(8, 2*nrows), squeeze=False)[1]
        ax_iter = ax.flat

        xmin, xmax = -3, 6
        x = np.linspace(xmin, xmax, 100*(xmax - xmin) + 1)
        y = self._norm.pdf(x)  # pdf stands for "probability density function"

        for actions, fails, name, ratio, sigma, label in zip(*self._attrs):
            extra_xticks = [1.5, sigma]
            xticks = list(range(xmin, xmax + 1)) + extra_xticks
            xfill = np.linspace(sigma, xmax)
            color = label.item().lower()
            label = f"Defect ratio = {ratio * 100:.2f}%"
            aes = {"label": label, "color": color, "alpha": 0.5}
            sigma_annotation = f"$\\sigma$ = {sigma:.4g}"

            label = f"$N(\\mu = 1.5, \\sigma = 1)$"
            name = f", {name=}" if name else ""
            title = f"{self.__class__.__name__}({actions=}, {fails=}{name})"

            ax = next(ax_iter)
            ax.plot(x, y, lw=2.2, color="k", label=label)
            ax.fill_between(xfill, self._norm.pdf(xfill), 0, **aes)
            ax.annotate(sigma_annotation, size=15, xy=(0.9, 0.22))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, y.max() + 0.02)
            ax.set_xticks(xticks)
            ax.tick_params(axis="both", labelsize=8)
            ax.tick_params(axis="x", labelrotation=45)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.4g"))
            ax.grid(lw=0.5)
            ax.legend(frameon=True, loc="upper left")
            ax.set_title(title)

        plt.tight_layout()
