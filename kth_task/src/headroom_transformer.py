from __future__ import annotations

from typing import Any

import numpy as np

from headroom.integrations.langchain.retriever import (
    HeadroomDocumentCompressor,
)

from transformer_retriever import TransformerRetriever


class HeadroomTransformerDocumentCompressor(
    HeadroomDocumentCompressor
):
    """
    HeadroomDocumentCompressor using corpus-wide semantic
    similarity from the project's TransformerRetriever.

    Each LangChain Document must contain:

        metadata={"doc_id": <index into corpus_texts>}

    The transformer model embeds the entire corpus once. For each
    query, the query is embedded and cosine similarity is calculated
    against the corpus. Headroom then uses those scores for its
    existing compression/selection logic.
    """

    def __init__(
        self,
        corpus_texts: list[str] | None = None,
        transformer_retriever: TransformerRetriever | None = None,
        max_documents: int = 10,
        min_relevance: float = 0.0,
        prefer_diverse: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ):
        super().__init__(
            max_documents=max_documents,
            min_relevance=min_relevance,
            prefer_diverse=prefer_diverse,
            **kwargs,
        )

        if transformer_retriever is not None:
            self._transformer = transformer_retriever

        elif corpus_texts is not None:
            self._transformer = TransformerRetriever(
                model_name=model_name
            )
            self._transformer.build_index(corpus_texts)

        else:
            raise ValueError(
                "Provide either corpus_texts or transformer_retriever."
            )

        if self._transformer.corpus_embeddings is None:
            raise ValueError(
                "Transformer index has not been built."
            )

        self._last_query: str | None = None
        self._last_scores: np.ndarray | None = None

    def _get_scores(self, query: str) -> np.ndarray:
        """
        Calculate semantic similarity against the entire corpus once
        for each unique query.
        """

        if query == self._last_query and self._last_scores is not None:
            return self._last_scores

        # Encode query using the same SentenceTransformer model
        query_embedding = self._transformer.model.encode(
            [query],
            convert_to_numpy=True,
        )

        # Normalize embeddings
        query_embedding = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True)
            + 1e-12
        )

        corpus_embeddings = self._transformer.corpus_embeddings

        corpus_embeddings = corpus_embeddings / (
            np.linalg.norm(
                corpus_embeddings,
                axis=1,
                keepdims=True,
            )
            + 1e-12
        )

        # Cosine similarity because embeddings are normalized
        scores = np.dot(
            corpus_embeddings,
            query_embedding[0],
        )

        # Convert [-1, 1] → [0, 1]
        scores = (scores + 1.0) / 2.0

        self._last_query = query
        self._last_scores = scores

        return scores

    def _score_document(
        self,
        doc: Any,
        query: str,
    ) -> float:
        """
        Return semantic similarity score for one candidate document.
        """

        doc_id = doc.metadata.get("doc_id")

        if doc_id is None:
            raise ValueError(
                "Every document must contain metadata['doc_id'] "
                "mapping it to the transformer corpus."
            )

        scores = self._get_scores(query)

        if not 0 <= doc_id < len(scores):
            raise ValueError(
                f"doc_id={doc_id} is outside the transformer corpus "
                f"(size={len(scores)})."
            )

        return float(scores[doc_id])