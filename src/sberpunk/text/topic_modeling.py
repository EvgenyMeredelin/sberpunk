__all__ = ['LDATopicModeler', 'VALID_POS']


# imports: standard library
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# imports: 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from pymorphy3 import MorphAnalyzer
from scipy.stats import entropy

# imports: user modules
from sberpunk import NLTK_RUS_STOPWORDS, RUNTIME_CONFIG


# settings
pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_rows', 1000)
plt.rcParams.update(RUNTIME_CONFIG)
plt.style.use('seaborn-v0_8-ticks')


# https://pymorphy2.readthedocs.io/en/stable/user/grammemes.html
VALID_POS = {
    'NOUN',  # имя существительное
    'VERB',  # глагол (личная форма)
    'ADJF',  # имя прилагательное (полное)
    'ADJS',  # имя прилагательное (краткое)
    'ADVB'   # наречие
}


# def nltk_rus_stopwords_factory() -> set[str]:
#     """
#     `nltk` russian stopwords.
#     """
#     import nltk
#     from nltk.corpus import stopwords
#     nltk.download('stopwords')
#     return set(stopwords.words('russian'))


@dataclass(kw_only=True)
class LDATopicModeler:
    """
    Topic modeling with `gensim` Latent Dirichlet Allocation (LDA) model.

    Attributes:
        docs (pandas.Series):
            Collection of documents.

        num_topics (int):
            Number of topics to model.

        num_docs (int):
            Number of most relevant documents to get for each modeled topic.

        token_pattern (str):
            Regular expression pattern to split a document into tokens.
            Defaults to `r'[а-яё]{3,}'`

        tokens_min_count (int):
            Minimal number of tokens a document must generate. Defaults to 10.

        lda_kwargs (dict[str, Any]):
            `gensim.models.ldamodel.LdaModel` model initialization kwargs.
            Defaults to an empty dict.

        stopwords (set[str]):
            Set of stopwords. Defaults to `nltk` russian stopwords.

        valid_pos (set[str]):
            Set of parts of speech a lemmatized token must belong to.
            Defaults to `VALID_POS`.

        repl_mapping (dict[str, str]):
            Dict to fix typos and/or reduce synonyms. Defaults to an empty dict.

        labels (pandas.Series | None):
            True labels of documents for modeling quality evaluation.
            Defaults to None.

        report_dir (str | pathlib.Path):
            Directory to write modeling report and applied parameters to.
            Defaults to the current working directory.
    """

    docs: pd.Series
    num_topics: int
    num_docs: int
    token_pattern: str = r'[а-яё]{3,}'
    tokens_min_count: int = 10
    lda_kwargs: dict[str, Any] = field(default_factory=dict)
    stopwords: set[str] = field(default_factory=lambda: NLTK_RUS_STOPWORDS)
    valid_pos: set[str] = field(default_factory=lambda: VALID_POS)
    repl_mapping: dict[str, str] = field(default_factory=dict)
    labels: pd.Series | None = None
    report_dir: str | Path = Path.cwd()

    def __post_init__(self) -> None:
        if not self.valid_pos.issubset(VALID_POS):
            raise ValueError(f'`valid_pos` must be a subset of {VALID_POS}')

        self._morph = MorphAnalyzer()
        self.tokens = list(map(self._tokenize, self.docs))
        id2word = Dictionary(self.tokens)
        corpus = list(map(id2word.doc2bow, self.tokens))

        self.lda = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=self.num_topics,
            **self.lda_kwargs
        )

        topics = self.lda.show_topics(
            num_topics=self.num_topics,
            formatted=False
        )

        self.keywords = pd.Series([
            self._ith_elem(topic, index=0)
            for topic in self._ith_elem(topics, index=1)
        ], name='keywords')

        self.keywords.index.name = 'topic_id'

        all_docs_topics, indices = [], []

        for doc_id, doc_bow in enumerate(corpus):
            if doc_bow:  # if document has generated tokens
                indices.append(doc_id)
                single_doc_topics = self.lda.get_document_topics(
                    doc_bow,
                    minimum_probability=0  # it's important to set 0
                )
                all_docs_topics.append(single_doc_topics)

        self.doc_topic_distr = np.array([
            self._ith_elem(single_doc_topics, index=1)
            for single_doc_topics in all_docs_topics
        ])

        self.docs_ = self.docs.reset_index(drop=True).loc[indices]

        self.results = {
            'doc': self.docs_,
            'topic_id': np.argmax(self.doc_topic_distr, axis=1),
            'entropy': np.apply_along_axis(
                entropy,
                axis=1,
                arr=self.doc_topic_distr
            )
        }

        grouping = ['keywords', 'doc', 'entropy']

        if self.labels is not None:
            labels = self.labels.reset_index(drop=True).loc[indices]
            self.results['label'] = labels
            grouping.insert(1, 'label')

        self.results = pd.merge(
            pd.DataFrame(self.results),
            self.keywords,
            left_on='topic_id',
            right_index=True
        )

        self.results = (
            self.results
            .groupby('topic_id')[grouping]
            .apply(lambda g: g.nsmallest(self.num_docs, columns='entropy'))
            .reset_index()
            .drop(columns=['level_1', 'entropy'])
        )

    def _tokenize(self, doc: str) -> list[str]:
        """
        Tokenizer with lemmatization, part of speech verification
        and stopwords removal.
        """
        tokens = []
        pattern = re.compile(self.token_pattern)

        for word in pattern.findall(doc.lower()):
            parse_obj = self._morph.parse(word)[0]
            if parse_obj.tag.POS not in self.valid_pos:
                continue

            norm = parse_obj.normal_form
            if norm in self.stopwords:
                continue

            repl = self.repl_mapping.get(norm, norm)
            tokens.append(repl)

        if len(tokens) < self.tokens_min_count:
            tokens.clear()

        return tokens

    @staticmethod
    def _ith_elem(tuples: list[tuple[Any, ...]], index: int) -> list[Any]:
        """Pick `index`-th items from tuples in list `tuples`. """
        return [t[index] for t in tuples]

    def plot_decisive_proba(self) -> None:
        """Decisive (voting) probabilities distribution. """
        # decisive (voting) is the largest probability that defines
        # which topic a document belongs to
        kwargs = dict(bins=16, range=(0.2, 1), edgecolor='w')
        plt.figure(figsize=(8, 4))
        plt.hist(self.doc_topic_distr.max(axis=1), **kwargs)
        plt.xlabel('probability')
        plt.ylabel('count')
        plt.title(self.plot_decisive_proba.__doc__.strip('.\n '))

    def plot_average_doc(self) -> None:
        """Topic profile of the average document. """
        x = range(self.num_topics)
        plt.figure(figsize=(8, 4))
        plt.bar(x, self.doc_topic_distr.mean(axis=0))
        plt.xticks(x)
        plt.yticks(np.arange(0, 0.3, 0.05))
        plt.xlabel('topic')
        plt.ylabel('probability')
        plt.title(self.plot_average_doc.__doc__.strip('.\n '))

    def to_json(self) -> None:
        """Write topic modeling results and model parameters to json. """
        import json
        from datetime import datetime
        from sberpunk import DATETIME_DEFAULT_FORMAT
        keys = [
            'num_topics', 'num_docs', 'tokens_min_count', 'lda_kwargs',
            'token_pattern', 'valid_pos'
        ]
        params = {key: self.__dict__[key] for key in keys}
        params['valid_pos'] = list(self.valid_pos)  # set is not serializable
        now = datetime.now().strftime(DATETIME_DEFAULT_FORMAT)
        base = f'{self.report_dir}/{now}'

        with open(base + '_params.json', 'w', encoding='utf-8') as file:
            json.dump(params, file, ensure_ascii=False, indent=4)

        self.results.to_json(
            base + '.json', orient='records', force_ascii=False, indent=4
        )
