import responses 
import embeddings
import cos_similarity
import initialize_models
import pandas as pd
from openai import OpenAI
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def pipeline(dataset: pd.DataFrame,
             bias: str,
             dataset_type: str,
             id_columns: List[str],
             columns: List[str],
             saving_path: str,
             open_ai_key: str,
             anthropic_key: Optional[str] = None,
             google_key: Optional[str] = None,
             hugging_face_key: Optional[str] = None,
             model: Optional[str] = None,
             verbose: bool = True) -> pd.DataFrame:
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
        open_ai_key (str): API key for OpenAI embeddings (required).
        anthropic_key (Optional[str]): API key for Anthropic models. Default is None.
        google_key (Optional[str]): API key for Google models. Default is None.
        hugging_face_key (Optional[str]): API key for Hugging Face models. Default is None.
        model (Optional[str]): The model to use for generating responses. Default is None.
        verbose (bool): Whether to log progress. Default is True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_responses: Model-generated responses.
            - embedding_df: Embeddings for responses.
            - cos_similarity_df: Cosine similarity results.
    Raises:
        ValueError: If the model is not supported or if required API keys are missing.
    """

    if model is None:
        raise ValueError("You must specify a model.")

    model = model.lower()
    supported_models = {
        "claude-3-opus-20240229",
        "gpt-4o-mini-2024-07-18",
        "gemini-1.0-pro",
        "gemma",
        "llama-2",
        "llama-3",
        "mistral",
        "yi",
    }

    if model not in supported_models:
        raise ValueError(
            f"Model '{model}' is not supported.\n"
            f"Supported models: {', '.join(sorted(supported_models))}"
        )
    
    if not open_ai_key:  # OpenAI key is mandatory for embeddings
        raise RuntimeError("OpenAI API key is required for embeddings.\nTry again with a key or implement another embedding model.")

    # Initialize logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    bias = bias.lower()
    dataset_type = dataset_type.lower()

    logger.info("Pipeline started | Model: %s | Bias: %s | Dataset type: %s", model, bias, dataset_type)
    logger.info("Saving directory: %s", saving_path)

    # Model initialization
    init_dispatch = {
        "claude-3-opus-20240229": lambda: (initialize_models.initialize_anthropic(anthropic_key), None, None),
        "gpt-4o-mini-2024-07-18": lambda: (initialize_models.initialize_open_ai(open_ai_key), None, None),
        "llama-2": lambda: (None, *initialize_models.initialize_llama2(hugging_face_key)),
        "llama-3": lambda: (None, *initialize_models.initialize_llama3(hugging_face_key)),
        "mistral": lambda: (None, *initialize_models.initialize_mistral(hugging_face_key)),
        "gemma": lambda: (None, *initialize_models.initialize_gemma(hugging_face_key)),
        "yi": lambda: (None, *initialize_models.initialize_yi(hugging_face_key)),
        "gemini-1.0-pro": lambda: (initialize_models.initialize_gemini_1_pro(model, google_key), None, None),
    }

    client, hugging_face_model, tokenizer = init_dispatch[model]()

    # --- Step 1: Generate responses ---
    df_responses = responses.create_responses_df(
        dataset=dataset,
        bias=bias,
        id_columns=id_columns,
        columns=columns,
        model=model,
        client=client,
        hugging_face_model=hugging_face_model,
        tokenizer=tokenizer
    )
    df_responses.to_csv(saving_path + model + ' ' +  bias + ' ' + dataset_type + ' - ' + 'responses.csv', index=False)

    logger.info('50% - of the process is finished.\nThe responses dataframe has been saved.\nConverting filtered outputs to embeddings:')


    logger.info("Response columns standardized. Filtered columns: %s", filtered_response_columns)
    # --- Step 2: Create embeddings ---
    # Filtered response columns contain '_filtered' in their names and were processed to remove certain content (see responses.py)
    filtered_response_columns = [col for col in df_responses.columns.str.lower() if "_filtered" in col]
    embedding_client = OpenAI(api_key=open_ai_key)  # Embeddings client, can be replaced with another embedding model if needed
    embedding_df = embeddings.create_embeddings_df(
        filtered_response_dataset=df_responses,
        id_columns=id_columns,
        columns=columns,
        filtered_response_columns=filtered_response_columns,
        client=embedding_client,
        model=model,
    )
    embedding_df.to_parquet(saving_path + model + ' ' +  bias + ' ' + dataset_type + ' - ' + 'embeddings.parquet', index=False)

    logger.info('75% - of the process is finished.\nGetting Cosine similarities from the embeddings:')

    # --- Step 3: Compute cosine similarities ---
    cos_similarity_df = cos_similarity.create_cos_similarity_df(
        dataset=embedding_df,
        id_columns=id_columns,
        columns=columns,
        embedding_columns= [col for col in embedding_df.columns if not col.endswith('_id')]
    )
    cos_similarity_df.to_csv(saving_path + model + ' ' +  bias + ' ' + dataset_type + ' - ' + 'cos_similarity.csv', index=False)

    logger.info('100% complete - pipeline finished running, tables created successfully.')
    return df_responses, embedding_df, cos_similarity_df