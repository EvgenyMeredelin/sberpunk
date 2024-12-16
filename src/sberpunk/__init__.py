# imports: standard library
from functools import partialmethod
from pathlib import Path
from typing import Any


DATETIME_DEFAULT_FORMAT = '%Y-%m-%d %H-%M-%S'

RUNTIME_CONFIG = {
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.titlepad': 15,
    'figure.dpi': 400
}

NLTK_RUS_STOPWORDS = {
    'а', 'без', 'более', 'больше', 'будет', 'будто', 'бы',
    'был', 'была', 'были', 'было', 'быть', 'в', 'вам', 'вас',
    'вдруг', 'ведь', 'во', 'вот', 'впрочем', 'все', 'всегда',
    'всего', 'всех', 'всю', 'вы', 'где', 'да', 'даже', 'два',
    'для', 'до', 'другой', 'его', 'ее', 'ей', 'ему', 'если',
    'есть', 'еще', 'ж', 'же', 'за', 'зачем', 'здесь', 'и',
    'из', 'или', 'им', 'иногда', 'их', 'к', 'как', 'какая',
    'какой', 'когда', 'конечно', 'кто', 'куда', 'ли', 'лучше',
    'между', 'меня', 'мне', 'много', 'может', 'можно', 'мой',
    'моя', 'мы', 'на', 'над', 'надо', 'наконец', 'нас', 'не',
    'него', 'нее', 'ней', 'нельзя', 'нет', 'ни', 'нибудь',
    'никогда', 'ним', 'них', 'ничего', 'но', 'ну', 'о', 'об',
    'один', 'он', 'она', 'они', 'опять', 'от', 'перед', 'по',
    'под', 'после', 'потом', 'потому', 'почти', 'при', 'про',
    'раз', 'разве', 'с', 'сам', 'свою', 'себе', 'себя',
    'сейчас', 'со', 'совсем', 'так', 'такой', 'там', 'тебя',
    'тем', 'теперь', 'то', 'тогда', 'того', 'тоже', 'только',
    'том', 'тот', 'три', 'тут', 'ты', 'у', 'уж', 'уже',
    'хорошо', 'хоть', 'чего', 'чем', 'через', 'что', 'чтоб',
    'чтобы', 'чуть', 'эти', 'этого', 'этой', 'этом', 'этот',
    'эту', 'я'
}


class EasyPath(Path):
    """
    `pathlib.Path` subclass with `glob` and `rglob` methods
    modified for a file.
    """

    _flavour = type(Path())._flavour

    def _glob(self, glob_name, pattern):
        # standard case
        if self.is_dir():
            glob_method = getattr(super(), glob_name)
            yield from glob_method(pattern)

        # special case
        if self.is_file():
            yield self

    glob = partialmethod(_glob, 'glob')
    rglob = partialmethod(_glob, 'rglob')


def public_api(obj: Any) -> list[str]:
    """Return object's public API members. """
    return [name for name in dir(obj) if not name.startswith('_')]
