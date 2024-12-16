__all__ = ['QueteletIndex']


# imports: standard library
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, ClassVar, Optional, Union

# imports: 3rd party libraries
import numpy as np
import pandas as pd
from scipy.stats.contingency import margins


array_like = Union[list, tuple, np.ndarray, pd.Series]


@dataclass
class Feature:
    """
    Feature with a name.
    """

    _id: ClassVar[int] = cycle([1, 2])
    data: array_like
    name: Optional[str]

    def __post_init__(self) -> None:
        # update instance id even if name defined explicitly
        generic_name = f'feature #{next(self.__class__._id)}'
        if self.name is None:
            # assign generic if name not defined explicitly
            self.name = generic_name
            # if `data` is a pandas series try to use its name
            if isinstance(self.data, pd.Series):
                self.name = self.data.name or self.name

    @property
    def values(self) -> list[Any]:
        """Sorted list of unique values. """
        return sorted(set(self.data))

    def __len__(self) -> int:
        return len(self.values)

    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return len(self) < len(other)
        return NotImplemented


@dataclass
class QueteletIndex:
    """
    Quetelet indices for a pair of categorical features.
    https://urait.ru/book/vvedenie-v-analiz-dannyh-511121

    Attributes:
        feature1 (list | tuple | numpy.ndarray | pandas.Series):
            First feature.

        feature2 (list | tuple | numpy.ndarray | pandas.Series):
            Second feature.

        names (list[str]):
            Names of the features. Defaults to an empty list.
    """

    feature1: array_like
    feature2: array_like
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.feature1) != len(self.feature2):
            raise ValueError('Features must be of the same size')

        # feature with greater number of factors located horizontally
        self.names += [None] * (2 - len(self.names))
        pairs = zip([self.feature1, self.feature2], self.names)
        yfeat, xfeat = sorted(Feature(*pair) for pair in pairs)
        self._xrot = any(len(str(v)) > 3 for v in xfeat.values)
        self._ratio = len(yfeat) / len(xfeat)

        # cross tabulation
        self.crosstab = pd.crosstab(yfeat.data, xfeat.data)
        self.crosstab_marg_norm = pd.crosstab(
            yfeat.data, xfeat.data, margins=True, normalize=True
        )

        # Quetelet indices evaluation
        values = self.crosstab.values
        values = values / values.sum()
        rows_margin, columns_margin = margins(values)
        data = (values / rows_margin - columns_margin) / columns_margin

        self.qtable = pd.DataFrame(data, yfeat.values, xfeat.values)
        self.qtable.index.name = self.crosstab.index.name = yfeat.name
        self.qtable.columns.name = self.crosstab.columns.name = xfeat.name

    def plot(self, *, annot_kws: Optional[dict[str, Any]] = None) -> None:
        """Plot Quetelet indices as a heatmap. """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sberpunk import RUNTIME_CONFIG

        sns.set_theme(palette='tab10', style='ticks', rc=RUNTIME_CONFIG)

        title = (
            f'Quetelet indices for features {self.qtable.index.name} /'
            f' {self.qtable.columns.name}'
        )

        ha, labelrotation = [('center', 0), ('left', 30)][self._xrot]

        plt.figure(figsize=(10, 10*self._ratio))
        sns.heatmap(
            self.qtable,
            annot=True,
            fmt='.4f',
            annot_kws=annot_kws or {'size': 15},
            linewidths=0.5,
            square=True,
            cmap='coolwarm'
        )
        plt.gca().xaxis.set_label_position('top')
        plt.tick_params(
            axis='x',
            labelrotation=labelrotation,
            labelbottom=False,
            bottom=False,
            labeltop=True,
            top=True
        )
        plt.xticks(ha=ha)
        plt.yticks(rotation=0)
        plt.title(title, size=15)
