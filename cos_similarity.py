"""
Cosine similarity calculation utilities.

This module provides functionality to compute cosine similarity between embedding vectors.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from itertools import combinations
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Calculator for computing cosine similarity between embeddings.

    This class provides methods to calculate similarity scores between
    embedding vectors, which are used to detect bias in model responses.
    """


    @staticmethod
    def calculate_cosine_similarity(
            embedding1: np.ndarray,
            embedding2: np.ndarray
    ) -> Optional[float]:
        """
        Calculate cosine similarity between two embedding vectors.


        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            The cosine similarity score, or None if calculation fails.
        """
        try:
            # Reshape embeddings to 2D arrays for sklearn
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]

            return float(similarity)

        except Exception as e:
            logger.error(
                f"Error in cosine similarity calculation: {e}. "
                "Check if embeddings are correctly generated."
            )
            return None


    @staticmethod
    def create_cos_similarity_df(dataset: pd.DataFrame, id_columns: List[str], columns: List[str], embedding_columns: List[str]) -> pd.DataFrame:
        """
        Create a DataFrame to store cosine similarity scores between pairs of embedding columns.

        Returns:
            pd.DataFrame: DataFrame containing cosine similarity scores for each pair of embedding columns.
        """
        combinations_of_titles = [f'cos_similarity: {c1} vs {c2}' for c1, c2 in combinations(columns, 2)]
        combinations_of_columns = list(combinations(embedding_columns, 2))

        cos_similarity_df = dataset[id_columns].copy()

        for title, (col1, col2) in zip(combinations_of_titles, combinations_of_columns):
            cos_similarity_df[title] = dataset.apply(
                lambda row: SimilarityCalculator.calculate_cosine_similarity(row[col1], row[col2]), axis=1
            )

        return cos_similarity_df