import os
from typing import Union
from openai import OpenAI
import numpy as np
import pandas as pd
import logging
from google import genai

logger = logging.getLogger(__name__)


class Embeddings:
    """Class for handling embeddings operations."""
    def __init__(self, open_ai_api_key:str, model_name:str="text-embedding-3-large"):
        self.client = OpenAI(api_key=open_ai_api_key)
        self.model_name = model_name


    def get_embedding(self, text:str) -> Union[np.ndarray, None]:
        """
        Retrieve the embedding for a given text using the OpenAI API.

        Args:
            text: The input text to be embedded.
        Returns:
            np.array: The embedding vector, None if an error occurs or text is empty.
        """
        try:
            if not text:
                logger.warning("No model response provided, it is possible either model did not respond or removing stopwords failed. Manually check the output dataset when run is finished.")
                return None
            text = text.replace("\n", " ")
            return np.array(self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding).reshape(1, -1)[0]
        except Exception as e:
            logger.warning('Error in OpenAI embedding retrieval: ' + str(e))
        return None


    def create_embeddings_df(self, filtered_response_dataset, id_columns, columns, filtered_response_columns) -> pd.DataFrame:
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
            embedding_df[embeddings_column] = filtered_response_dataset[filtered_response_column].apply(self.get_embedding)
        return embedding_df