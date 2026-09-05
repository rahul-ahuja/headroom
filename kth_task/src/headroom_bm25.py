from __future__ import annotations

from typing import Any

from headroom.integrations.langchain.retriever import (
    HeadroomDocumentCompressor,
)

from bm25_retriever import BM25Retriever


class HeadroomBM25DocumentCompressor(HeadroomDocumentCompressor):
    """Headroom compressor using corpus-wide BM25Okapi scoring."""

    def __init__(
        self,
        corpus_texts: list[str] | None = None,
        bm25_retriever: BM25Retriever | None = None,
        max_documents: int = 10,
        min_relevance: float = 0.0,
        prefer_diverse: bool = False,
        k1: float = 1.2,
        b: float = 0.75,
        **kwargs: Any,
    ):
        super().__init__(
            max_documents=max_documents,
            min_relevance=min_relevance,
            prefer_diverse=prefer_diverse,
            **kwargs,
        )

        if bm25_retriever is not None:
            self._bm25 = bm25_retriever

        elif corpus_texts is not None:
            self._bm25 = BM25Retriever(k1=k1, b=b)
            self._bm25.build_index(corpus_texts)

        else:
            raise ValueError(
                "Provide either corpus_texts or bm25_retriever."
            )

        if self._bm25.bm25 is None:
            raise ValueError("BM25 index has not been built.")

        self._last_query: str | None = None
        self._last_scores = None

    def _get_scores(self, query: str):
        """Calculate corpus-wide BM25 scores once per query."""

        if query != self._last_query:
            tokenized_query = query.lower().split()

            self._last_scores = self._bm25.bm25.get_scores(
                tokenized_query
            )
            self._last_query = query

        return self._last_scores

    def _score_document(
        self,
        doc: Any,
        query: str,
    ) -> float:
        """Return normalized corpus-wide BM25 score."""

        doc_id = doc.metadata.get("doc_id")

        if doc_id is None:
            raise ValueError(
                "Every document must contain metadata['doc_id'] "
                "mapping it to the BM25 corpus."
            )

        scores = self._get_scores(query)

        if not 0 <= doc_id < len(scores):
            raise ValueError(
                f"doc_id={doc_id} is outside the BM25 corpus "
                f"(size={len(scores)})."
            )

        raw_score = float(scores[doc_id])
        max_score = float(scores.max())

        if max_score <= 0:
            return 0.0

        return min(1.0, raw_score / max_score)