"""
Offline stand-ins for real models, used by the tests and `scripts/evaluate.py --offline`.

BagOfWordsEmbeddings hashes words into a fixed-size vector. It knows nothing about
meaning, but it is deterministic, free and fast, and similar wording gives similar
vectors - enough to test that the retrieval plumbing returns the right chunks.
"""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its may must of on or should "
    "the their them then there these this to use used was what when where which who why will with "
    "you your only not no all any each".split()
)


class BagOfWordsEmbeddings(Embeddings):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in STOPWORDS:
                continue
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)
