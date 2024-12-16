__all__ = ['Rag']


# imports: standard library
import re
import textwrap
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

# imports: 3rd party libraries
import black
import numpy as np
from multimethod import multimethod as singledispatchmethod
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

# imports: user modules
from sberpunk import EasyPath


METRICS = {
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
    'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
    'sokalsneath', 'sqeuclidean', 'yule'
}


@dataclass
class TextChunk:
    """Chunk of text. """

    content: str  # text
    source: str   # path to the source file

    def __len__(self) -> int:
        return len(self.content)

    def __str__(self) -> str:
        attrs = {}
        attrs['source'] = self.source
        attrs['content'] = textwrap.wrap(self.content, width=60)
        return black.format_str(repr(attrs), mode=black.Mode())

    __repr__ = __str__


@dataclass(kw_only=True)
class Rag:
    """
    Retrieval Augmented Generation (RAG).

    Attributes:
        source (str):
            TXT source file / directory with TXT files.

        model (str):
            Directory with embedding model or model name on Hugging Face.

        model_max_tokens (int):
            Maximal length of the embedding model's input sequence.
    """

    source: str
    model: str
    model_max_tokens: int

    def __post_init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._sentinel_pattern = re.compile(r'\[__.+__\]')
        self.chunks = self._split_text_into_chunks()
        if not self.chunks:
            raise ValueError('Split produced no chunks')

        self._embedder = SentenceTransformer(self.model)
        self._embedder.max_seq_length = self.model_max_tokens
        self._embeddings = self._compute_embeddings(self.chunks)

    def _split_text_into_chunks(self) -> list[TextChunk]:
        """Split `source` files into `TextChunk` chunks. """
        chunks = []
        for path in EasyPath(self.source).rglob('*'):
            if path.suffix.lstrip('.').lower() != 'txt':
                continue
            with path.open(encoding='utf-8') as file:
                text = file.read()
            # if `source` is a file, `removeprefix` returns an empty string
            src = str(path).removeprefix(self.source + '\\') or self.source
            chunks.extend(self._assemble_chunks(src, text, sep=r'\n+'))
        return chunks

    def _assemble_chunks(
        self, src: str, text: str, *, sep: str
    ) -> list[TextChunk]:
        """
        Assemble text chunk from lines/single words until embedding model's max
        tokens threshold is reached.
        """

        chunks, temp = [], []  # `temp`: buffer for lines/single words of chunk
        n_tokens = 0  # counter of tokens representing current `temp` contents

        sep_pattern = re.compile(sep)
        lines = filter(bool, map(self._clean_line, sep_pattern.split(text)))
        line = next(lines, None)

        while line is not None:
            # peek the total number of tokens considering new line
            n_tokens += len(self._tokenizer.tokenize(line))

            if n_tokens > self.model_max_tokens:
                n_tokens = 0

                # if empty buffer overflowed by the very first line
                # we go down to the line's single words
                if not temp:
                    chunks.extend(self._assemble_chunks(src, line, sep=r'\s+'))
                    line = next(lines, None)

                # no `lines` push:
                # cleared buffer begins with a line that caused overflow
                else:
                    content = ' '.join(temp)
                    yet_another_chunk = TextChunk(content, src)
                    chunks.append(yet_another_chunk)
                    temp.clear()
            else:
                temp.append(line)
                line = next(lines, None)

        if temp:
            content = ' '.join(temp)
            final_chunk = TextChunk(content, src)
            chunks.append(final_chunk)

        return chunks

    def _clean_line(self, line) -> str:
        """Clean a text line. """
        line = self._sentinel_pattern.sub('', line)
        return ''.join(char for char in line if char.isprintable())

    @singledispatchmethod
    def _compute_embeddings(self, arg):
        msg = f'Cannot compute emdedding for {type(arg)}'
        raise NotImplementedError(msg)

    @_compute_embeddings.register
    def _(self, prompt: str) -> np.ndarray:
        """Compute embedding for a prompt. """
        return self._embedder.encode(prompt, normalize_embeddings=True)

    @_compute_embeddings.register
    def _(self, chunks: list[TextChunk]) -> np.ndarray:
        """Compute embeddings for some `TextChunk` chunks. """
        content = [chunk.content for chunk in chunks]
        return self._embedder.encode(content, normalize_embeddings=True)

    def get_top_chunks(
        self, prompt: str, *, metric: str = 'cosine', n_chunks: int = 5,
        chunk_min_len: int = 0
    ) -> list[TextChunk]:
        """
        Top `n_chunks` closest chunks with metric `metric` for prompt `prompt`.

        Args:
            prompt (str):
                Prompt (query or passage).

            metric (str):
                The distance metric to use. Defaults to 'cosine'.
                See `scipy.spatial.distance.cdist` for valid options.

            n_chunks (int):
                Number of chunks to keep. Defaults to 5.

            chunk_min_len (int):
                Chunk minimal length. Defaults to 0.

        Returns:
            list[TextChunk]: list of `TextChunk` chunks.
        """

        if metric not in METRICS:
            msg = (
                'See `scipy.spatial.distance.cdist` for valid `metric` options'
            )
            raise ValueError(msg)

        prompt_embed = self._compute_embeddings(prompt).reshape(1,-1)
        distance = cdist(prompt_embed, self._embeddings, metric).squeeze()

        # without `chunk_min_len`
        # indices = np.argsort(distance)[:n_chunks]
        # return [self.chunks[i] for i in indices]

        indices = iter(np.argsort(distance))
        n_chunks = min(n_chunks, len(self.chunks))
        chunks = []

        while n_chunks:
            index = next(indices)
            candidate = self.chunks[index]
            if len(candidate) >= chunk_min_len:
                chunks.append(candidate)
                n_chunks -= 1

        return chunks
