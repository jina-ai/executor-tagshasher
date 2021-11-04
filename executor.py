import hashlib
import json

import numpy as np

from jina import Executor, DocumentArray, requests


class TagsHasher(Executor):
    """Convert an arbitrary set of tags into a fixed-dimensional matrix using the hashing trick.

    Unlike FeatureHashser, you should only use Jaccard/Hamming distance when searching documents
    embedded with TagsHasher. This is because the closeness of the value of each feature is meaningless
    it is basically the result of a hash function. Hence, only identity value matters.

    More info: https://en.wikipedia.org/wiki/Feature_hashing
    """

    def __init__(self, n_dim: int = 256, max_val: int = 65536, sparse: bool = False, **kwargs):
        """
        :param n_dim: the dimensionality of each document in the output embedding.
            Small numbers of features are likely to cause hash collisions,
            but large numbers will cause larger overall parameter dimensions.
        :param sparse: whether the resulting feature matrix should be a sparse csr_matrix or dense ndarray.
            Note that this feature requires ``scipy``
        :param text_attrs: which attributes to be considered as text attributes.
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.max_val = max_val
        self.hash = hashlib.md5
        self.sparse = sparse

    def _any_hash(self, v):
        try:
            return int(v)  # parse int parameter
        except ValueError:
            try:
                return float(v)  # parse float parameter
            except ValueError:
                if not v:
                    # ignore it when the parameter is empty
                    return 0
                if isinstance(v, str):
                    v = v.strip()
                    if v.lower() in {'true', 'yes'}:  # parse boolean parameter
                        return 1
                    if v.lower() in {'false', 'no'}:
                        return 0
                if isinstance(v, (tuple, dict, list)):
                    v = json.dumps(v, sort_keys=True)

        return int(self.hash(str(v).encode('utf-8')).hexdigest(), base=16)

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        if self.sparse:
            from scipy.sparse import csr_matrix

        for idx, doc in enumerate(docs):
            if doc.tags:
                idxs, data = [], []  # sparse
                table = np.zeros(self.n_dim)  # dense
                for k, v in doc.tags.items():
                    h = self._any_hash(k)
                    sign_h = np.sign(h)
                    col = h % self.n_dim
                    val = self._any_hash(v)
                    sign_v = np.sign(val)
                    val = val % self.max_val
                    idxs.append((0, col))
                    val = sign_h * sign_v * val
                    data.append(val)
                    table[col] += val

                if self.sparse:
                    doc.embedding = csr_matrix(
                        (data, zip(*idxs)), shape=(1, self.n_dim)
                    )
                else:
                    doc.embedding = table
