import responses
import embeddings
import cos_similarity
import pandas as pd
from openai import OpenAI
import logging
from typing import List, Optional, Tuple

from DiFair.models.open_ai import OpenAIModelAdapter
from DiFair.models.anthropic import AnthropicModelAdapter
from DiFair.models.google import GoogleModelAdapter
from DiFair.models.llama_2_3 import LlamaAdapter
from DiFair.models.mistral import MistralAdapter
from DiFair.models.gemma import GemmaAdapter
from DiFair.models.yi import YiAdapter

logger = logging.getLogger(__name__)


def pipeline(dataset: pd.DataFrame,
             bias: str,
             dataset_type: str,
             id_columns: List[str],
             columns: List[str],
             saving_path: str,
             open_ai_api_key: str,
             anthropic_api_key: Optional[str] = None,
             google_api_key: Optional[str] = None,
             hugging_face_api_key: Optional[str] = None,
             temperature: float = 0.5,
             max_tokens: int = 1000,
             model_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    A comprehensive pipeline for generating model responses, converting them into embeddings,
    and computing cosine similarities for stereotype-based bias detection and analysis.

    Args:
        dataset (pd.DataFrame): The dataset containing prompts or inputs for the model.
        bias (str): The type of bias being analyzed.
        dataset_type (str): Dataset type (used in file naming, e.g., ["YYYY-MM-DD", "calibration"]).
        id_columns (List[str]): Column names representing identifiers in the dataset.
        columns (List[str]): Column names containing prompts for generating responses.
        saving_path (str): Path to save the generated CSV files.
        open_ai_api_key (str): API key for OpenAI embeddings (required).
        anthropic_api_key (Optional[str]): API key for Anthropic models. Default is None.
        google_api_key (Optional[str]): API key for Google models. Default is None.
        hugging_face_api_key (Optional[str]): API key for Hugging Face models. Default is None.
        temperature (float): Sampling temperature for response generation. Default is 0.5.
        max_tokens (int): Maximum tokens for response generation. Default is 1000.
        model_name (Optional[str]): The model to use for generating responses. Default is None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_responses: Model-generated responses.
            - embedding_df: Embeddings for responses.
            - cos_similarity_df: Cosine similarity results.
    Raises:
        ValueError: If the model is not supported or if required API keys are missing.
    """

    bias = bias.lower()
    dataset_type = dataset_type.lower()

    logger.info("Pipeline started | Model: %s | Bias: %s | Dataset type: %s", model, bias, dataset_type)
    logger.info("Saving directory: %s", saving_path)

    # Model initialization
    init_dispatch = {
        "claude-3-opus-20240229": lambda: AnthropicModelAdapter(anthropic_api_key, temperature=temperature, max_tokens=max_tokens),
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
    df_responses = responses.create_responses_df(
        dataset=dataset,
        bias=bias,
        id_columns=id_columns,
        columns=columns,
        model=model
    )
    df_responses.to_csv(saving_path + model.model_name + ' ' +  bias + ' ' + dataset_type + ' - ' + 'responses.csv', index=False)

    logger.info('50% - of the process is finished. '
                'The responses dataframe has been saved. '
                'Converting filtered outputs to embeddings.')

    # --- Step 2: Create embeddings ---
    # Filtered response columns contain '_filtered' in their names and were processed to remove certain content (see responses.py)
    filtered_response_columns = [col for col in df_responses.columns.str.lower() if "_filtered" in col]
    embedding_client = OpenAI(api_key=open_ai_api_key)  # Embeddings client, can be replaced with another embedding model if needed
    embedding_df = embeddings.create_embeddings_df(
        filtered_response_dataset=df_responses,
        id_columns=id_columns,
        columns=columns,
        filtered_response_columns=filtered_response_columns,
        client=embedding_client
    )
    embedding_df.to_parquet(saving_path + model.model_name + ' ' +  bias + ' ' + dataset_type + ' - ' + 'embeddings.parquet', index=False)

    logger.info('75% - of the process is finished. '
                'Getting Cosine similarities from the embeddings:')

    # --- Step 3: Compute cosine similarities ---
    cos_similarity_df = cos_similarity.create_cos_similarity_df(
        dataset=embedding_df,
        id_columns=id_columns,
        columns=columns,
        embedding_columns= [col for col in embedding_df.columns if not col.endswith('_id')]
    )
    cos_similarity_df.to_csv(saving_path + model.model_name + ' ' +  bias + ' ' + dataset_type + ' - ' + 'cos_similarity.csv', index=False)

    logger.info('100% complete - pipeline finished running, tables created successfully.')
    return df_responses, embedding_df, cos_similarity_df