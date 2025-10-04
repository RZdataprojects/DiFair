from responses import TextProcessor
from embeddings import Embeddings
from cos_similarity import SimilarityCalculator
import pandas as pd
import logging
from typing import List, Optional, Tuple

from models.open_ai import OpenAIModelAdapter
from models.anthropic import AnthropicModelAdapter
from models.google import GoogleModelAdapter
from models.llama_2_3 import LlamaAdapter
from models.mistral import MistralAdapter
from models.gemma import GemmaAdapter
from models.yi import YiAdapter

logger = logging.getLogger(__name__)


class Pipeline:
    @staticmethod
    def run(dataset: pd.DataFrame,
            bias: str,
            title_comment: str,
            id_columns: List[str],
            columns: List[str],
            saving_path: str,
            open_ai_api_key:str,
            google_api_key:Optional[str] = None,
            hugging_face_api_key:Optional[str] = None,
            anthropic_api_key:Optional[str] = None,
            temperature: float = 0.5,
            max_tokens: int = 1000,
            model_name: Optional[str] = None
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        A comprehensive pipeline for generating model responses, converting them into embeddings,
        and computing cosine similarities for stereotype-based bias detection and analysis.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - df_responses: Model-generated responses.
                - embedding_df: Embeddings for responses.
                - cos_similarity_df: Cosine similarity results.
        Raises:
            ValueError: If the model is not supported or if required API keys are missing.
        """
        logger.info("Pipeline started | Model: %s | Bias: %s | Dataset type: %s", model_name, bias, title_comment)
        logger.info("Saving directory: %s", saving_path)

        # Model initialization
        init_dispatch = {
            "claude-3-opus-20240229": lambda: AnthropicModelAdapter(anthropic_api_key=anthropic_api_key, temperature=temperature, max_tokens=max_tokens),
            "gemini-2.5-flash-lite": lambda: GoogleModelAdapter(google_api_key=google_api_key, temperature=temperature, max_tokens=max_tokens),
            "gemini-1.0-pro": lambda: GoogleModelAdapter(google_api_key=google_api_key, temperature=temperature, max_tokens=max_tokens),
            "gpt-4o-mini-2024-07-18": lambda: OpenAIModelAdapter(open_ai_api_key=open_ai_api_key, temperature=temperature, max_tokens=max_tokens),
            "llama-2": lambda: LlamaAdapter(version=2, hugging_face_api_key=hugging_face_api_key, temperature=temperature, max_tokens=max_tokens),
            "llama-3": lambda: LlamaAdapter(version=3, hugging_face_api_key=hugging_face_api_key, temperature=temperature, max_tokens=max_tokens),
            "mistral": lambda: MistralAdapter(hugging_face_api_key=hugging_face_api_key, temperature=temperature, max_tokens=max_tokens),
            "gemma": lambda: GemmaAdapter(hugging_face_api_key=hugging_face_api_key, temperature=temperature, max_tokens=max_tokens),
            "yi": lambda: YiAdapter(hugging_face_api_key=hugging_face_api_key, temperature=temperature, max_tokens=max_tokens),
        }

        model = init_dispatch[model_name]()

        # --- Step 1: Generate responses ---
        df_responses = TextProcessor(bias=bias).create_responses_df(
            dataset=dataset,
            id_columns=id_columns,
            columns=columns,
            model=model
        )
        df_responses.to_csv(saving_path + model.model_name + ' ' + bias + ' ' + title_comment + ' - ' + 'responses.csv', index=False)

        logger.info('50% - of the process is finished. '
                    'The responses dataframe has been saved. '
                    'Converting filtered outputs to embeddings.')

        # --- Step 2: Create embeddings ---
        # Filtered response columns contain '_filtered' in their names and were processed to remove certain content (see responses.py)
        filtered_response_columns = [col for col in df_responses.columns.str.lower() if "_filtered" in col]
        embedding_client = Embeddings(open_ai_api_key=open_ai_api_key)  # Embeddings client, can be replaced with another embedding model if needed
        embedding_df = embedding_client.create_embeddings_df(
            filtered_response_dataset=df_responses,
            id_columns=id_columns,
            columns=columns,
            filtered_response_columns=filtered_response_columns
        )
        embedding_df.to_parquet(saving_path + model.model_name + ' ' + bias + ' ' + title_comment + ' - ' + 'embeddings.parquet', index=False)

        logger.info('75% - of the process is finished. '
                    'Getting Cosine similarities from the embeddings.')

        # --- Step 3: Compute cosine similarities ---
        cos_similarity_df = SimilarityCalculator().create_cos_similarity_df(
            dataset=embedding_df,
            id_columns=id_columns,
            columns=columns,
            embedding_columns= [col for col in embedding_df.columns if not col.endswith('_id')]
        )
        cos_similarity_df.to_csv(saving_path + model.model_name + ' ' + bias + ' ' + title_comment + ' - ' + 'cos_similarity.csv', index=False)

        logger.info('100% complete - pipeline finished running, tables created successfully.')
        return df_responses, embedding_df, cos_similarity_df