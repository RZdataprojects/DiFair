from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(column1, column2):
    try:
        return cosine_similarity(np.array(column1).reshape(1, -1), np.array(column2).reshape(1, -1))[0][0]
    except Exception:
        logger.error("Error in cosine similarity calculation. Check if embeddings are correctly generated.")
        return None

def create_cos_similarity_df(dataset, id_columns, columns, embedding_columns) -> pd.DataFrame:
    """
    Create a DataFrame to store cosine similarity scores between pairs of embedding columns.

    Returns:
        pd.DataFrame: DataFrame containing cosine similarity scores for each pair of embedding columns.
    """
    combinations_of_columns_titles = []
    for col1, col2 in combinations(columns, 2):  
        combinations_of_columns_titles.append(f'cos_similarity: {col1} vs {col2}')

    cos_similarity_df = pd.DataFrame(index=dataset.index, columns=combinations_of_columns_titles)
    cos_similarity_df[id_columns] = dataset[id_columns]
        
    combinations_of_columns = []
    for col1, col2 in combinations(embedding_columns, 2):  
    # Create new column name
        combinations_of_columns.append([col1, col2])

    for index, row in dataset.iterrows():
        for column, embeddings_col in zip(combinations_of_columns_titles, combinations_of_columns):
            score = calculate_cosine_similarity(row[embeddings_col[0]], row[embeddings_col[1]])
            cos_similarity_df.at[index, column] = score

    return cos_similarity_df