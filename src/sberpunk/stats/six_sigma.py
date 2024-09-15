__all__ = ['Process', 'Threshold']


# imports: standard library
from dataclasses import dataclass
from enum import Enum
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
plt.style.use('seaborn-v0_8-ticks')


array_like = Union[list, np.ndarray, pd.Series]


class Threshold(Enum):
    """
    Supremums of the quality classes.
    """

    # 'red' never reaches 2.1 which is the lower bound of the 'yellow' class
    # 'yellow' never reaches 4.1 which is the lower bound of the 'green' class

    RED = 2.1
    YELLOW = 4.1


@dataclass
class Process:
    """
    Process to evaluate with '6 sigma' approach.

    Attributes:
        actions (int | list | numpy.ndarray | pandas.Series):
            Total number of actions.

        fails (int | list | numpy.ndarray | pandas.Series):
            Number of actions qualified as failed.

        name (str | list | numpy.ndarray | pandas.Series | None):
            Name of the process. Defaults to None.
    """

    # action considered as a single run/try of the process
    actions: int | array_like
    fails: int | array_like
    name: str | array_like | None = None

    def __post_init__(self) -> None:
        actions, fails = map(self._to_array_like, (self.actions, self.fails))

        if actions.size != fails.size:
            msg = '`actions` and `fails` must be of the same length'
            raise ValueError(msg)

        if (fails > actions).any():
            msg = (
                'Number of fails can\'t be greater than the total '
                'number of actions'
            )
            raise ValueError(msg)

        if self.name is not None:
            name = self._to_array_like(self.name)
            if len(name) != actions.size:
                msg = (
                    'Keep name as `None` or explicitly provide '
                    'every process with a meaningful name'
                )
                raise ValueError(msg)
        else:
            name = [None] * actions.size

        self.defect_ratio = fails / actions
        # normal continuous random variable with loc=1.5 and scale=1
        self._norm = norm(1.5)
        # ppf stands for 'percent point function'
        self.sigma = self._norm.ppf(1 - self.defect_ratio)
        self.label = self._get_label(self.sigma)

        self._attrs = [
            actions, fails, name,
            self.defect_ratio, self.sigma, self.label
        ]

    @staticmethod
    def _to_array_like(attr) -> array_like:
        """Cast `attr` to an iterable interface. """
        # numpy array for vectorized calculation
        if isinstance(attr, int):
            return np.array([attr])
        if isinstance(attr, str):
            return [attr]
        if isinstance(attr, list):
            return np.array(attr)
        return attr

    @staticmethod
    @np.vectorize
    def _get_label(sigma: float) -> str:
        """
        Get the label of the quality class.
        """
        for thres in Threshold:
            if sigma < thres.value:
                return thres.name
        return 'GREEN'

    def plot(self) -> None:
        """
        Process sigma and defect ratio visualization.
        """
        nrows = self._attrs[0].size
        ax = plt.subplots(nrows=nrows, figsize=(8, 2*nrows), squeeze=False)[1]
        ax_iter = ax.flat

        xmin, xmax = -3, 6
        x = np.linspace(xmin, xmax, 100*(xmax - xmin) + 1)
        y = self._norm.pdf(x)  # pdf stands for 'probability density function'

        for actions, fails, name, ratio, sigma, label in zip(*self._attrs):
            extra_xticks = [1.5, sigma]
            xticks = list(range(xmin, xmax + 1)) + extra_xticks
            xfill = np.linspace(sigma, xmax)
            color = label.item().lower()
            label = f'Defect ratio = {ratio * 100:.2f}%'
            aes = {'label': label, 'color': color, 'alpha': 0.67}
            sigma_annotation = f'$\\sigma$ = {sigma:.4g}'

            label = f'$N(\\mu = 1.5, \\sigma = 1)$'
            name = f', {name=}' if name else ''
            title = f'{self.__class__.__name__}({actions=}, {fails=}{name})'

            ax = next(ax_iter)
            ax.plot(x, y, lw=2.2, color='k', label=label)
            ax.fill_between(xfill, self._norm.pdf(xfill), 0, **aes)
            ax.annotate(sigma_annotation, size=15, xy=(0.9, 0.22))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, y.max() + 0.02)
            ax.set_xticks(xticks)
            ax.tick_params(axis='both', labelsize=8)
            ax.tick_params(axis='x', labelrotation=45)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
            ax.grid(lw=0.5)
            ax.legend(frameon=True, loc='upper left')
            ax.set_title(title)

        plt.tight_layout()
