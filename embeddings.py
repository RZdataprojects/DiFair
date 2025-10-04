from typing import Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_embedding(text, client, model="text-embedding-3-large") -> Union[np.ndarray, None]:
    try:
        if not text:
            logger.warning("No model response provided, it is possible either model did not respond or removing stopwords failed. Manually check the output dataset when run is finished.")
            return None
        text = text.replace("\n", " ")
        return np.array(client.embeddings.create(input=[text], model=model).data[0].embedding).reshape(1, -1)[0]
    except Exception as e:
        logger.warning('Error in OpenAI embedding retrieval: ' + str(e))
    return None


def create_embeddings_df(*, filtered_response_dataset, id_columns, columns, filtered_response_columns, client) -> pd.DataFrame:
    """
    Retrieve embeddings for each filtered response column.

    Returns:
        pd.DataFrame: DataFrame containing embeddings.
    """
    postfix = '_embeddings'
    embedding_columns = [col + postfix for col in columns]
    embedding_df = pd.DataFrame()
    embedding_df[id_columns] = filtered_response_dataset[id_columns]

    for filtered_response_column, embeddings_column in zip(filtered_response_columns, embedding_columns):
        embedding_df[embeddings_column] = filtered_response_dataset[filtered_response_column].apply(get_embedding, args=(client, "text-embedding-3-large"))
    return embedding_df