import responses 
import embeddings
import cos_similarity
import initialize_models
import pandas as pd
from openai import OpenAI


def pipeline(dataset: pd.DataFrame,
             bias: str,
             dataset_type: str,
             id_columns: list,
             columns: list,
             saving_path: str,
             open_ai_key: str,
             stereotype_dict: dict = None,
             anthropic_key: str = None,
             google_key: str = None,
             hugging_face_key: str = None,
             model: str = None,
             verbose=1):
    """
    A comprehensive pipeline for generating model responses, converting them into embeddings, 
    and computing cosine similarities for strerotype-based biases detection and analysis.

    Args:
        dataset (pd.DataFrame): The dataset containing prompts or inputs for the model.
        bias (str): The type of bias being analyzed.
        dataset_type (str): Dataset type - meant for comments that you wish to save in the file's name ["YYYY-MM-DD", "calibration"].
        id_columns (list): List of column names that represent identifiers in the dataset.
        columns (list): List of column names containing prompts for which responses are generated.
        stereotype_dict (dict): A dictionary mapping stereotypes to their corresponding prompt_id.
        saving_path (str): Path to save the generated CSV files.
        open_ai_key (str): API key for OpenAI, necessary for embedding retrieval..
        anthropic_key (str, optional): API key for Anthropic models. Default is None.
        google_key (str, optional): API key for Google models. Default is None.
        hugging_face_key (str, optional): API key for Hugging Face models. Default is None.
        model (str, optional): The model to be used for generating responses. Default is None.
        verbose (int, optional): Verbosity level for logging progress. Default is 1.

    Returns:
        pd.DataFrame: The DataFrame containing the final embeddings.
    """
    model = model.lower()    
    assert model in ['claude-3-opus-20240229',
                     'gpt-4o-mini-2024-07-18',
                     'gemini-1.0-pro',
                     'gemma',
                     'llama-2',
                     'llama-3',
                     'mistral',
                     'yi'], """This model is not supported.\nThe supported models are 'claude-3-opus-20240229',
                     'gpt-4o-mini-2024-07-18',
                     'gemini-1.0-pro',
                     'gemma',
                     'llama-2',
                     'llama-3',
                     'mistral',
                     'yi'."""
    
    assert open_ai_key != ' ', "API key for OpenAI embeddings is missing. Try again with a key or implement another embedding model."
    
    bias = bias.lower()
    dataset_type = dataset_type.lower()
    client = OpenAI(api_key=open_ai_key)
    
    if verbose:
        print('getting model outputs: ', model)
        print('saving directory: ', saving_path)
        print('bias: ', bias)
        print('dataset type: ', dataset_type)
        
    if model == 'claude-3-opus-20240229':
        anthropic_client = initialize_models.initialize_anthropic(anthropic_key)
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias,
                                                     dataset_type=dataset_type, 
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path, 
                                                     model=model, 
                                                     client=anthropic_client, 
                                                     hugging_face_model=None, 
                                                     tokenizer=None, 
                                                     verbose=1)
    if model == 'llama-2':
        llama_model, tokenizer = initialize_models.initialize_llama2(hugging_face_key)
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias, 
                                                     dataset_type=dataset_type, 
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path, 
                                                     model=model, 
                                                     client=None, 
                                                     hugging_face_model=llama_model, 
                                                     tokenizer=tokenizer, 
                                                     verbose=1)
    if model == 'llama-3':
        llama_model, tokenizer = initialize_models.initialize_llama3(hugging_face_key)
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias, 
                                                     dataset_type=dataset_type,
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path, 
                                                     model=model, 
                                                     client=None,
                                                     hugging_face_model=llama_model, 
                                                     tokenizer=tokenizer, 
                                                     verbose=1)
    if model == 'mistral':
        mistral_model, tokenizer = initialize_models.initialize_mistral(hugging_face_key)
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias, 
                                                     dataset_type=dataset_type,
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path,
                                                     model=model,
                                                     client=None, 
                                                     hugging_face_model=mistral_model,
                                                     tokenizer=tokenizer,
                                                     verbose=1)
    if model == 'gpt-4o-mini-2024-07-18':
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias, 
                                                     dataset_type=dataset_type, 
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path, 
                                                     model=model, 
                                                     client=client,
                                                     hugging_face_model=None,
                                                     tokenizer=None, 
                                                     verbose=1)
    if model == 'gemma':
        gemma_model, tokenizer = initialize_models.initialize_gemma(hugging_face_key)
        df_responses = responses.create_responses_df(dataset=dataset, 
                                                     bias=bias, 
                                                     dataset_type=dataset_type, 
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path, 
                                                     model=model,
                                                     client=None, 
                                                     hugging_face_model=gemma_model,
                                                     tokenizer=tokenizer, 
                                                     verbose=1)
    if model == 'yi':
        yi_model, tokenizer = initialize_models.initialize_yi(hugging_face_key)
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias, 
                                                     dataset_type=dataset_type, 
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path,
                                                     model=model, 
                                                     client=None, 
                                                     hugging_face_model=yi_model, 
                                                     tokenizer=tokenizer, 
                                                     verbose=1)
    if model == 'gemini-1.0-pro':
        gemini_1_pro_client = initialize_models.initialize_gemini_1_pro(model, google_key)
        df_responses = responses.create_responses_df(dataset=dataset,
                                                     bias=bias, 
                                                     dataset_type=dataset_type, 
                                                     id_columns=id_columns, 
                                                     columns=columns, 
                                                     saving_path=saving_path, 
                                                     model=model, 
                                                     client=gemini_1_pro_client,
                                                     hugging_face_model=None,
                                                     tokenizer=None,
                                                     verbose=verbose)

    if verbose:
        print('50% - of the process is finished.\nThe responses dataframe has been saved. \nConverting filtered outputs to embeddings:')
        
    df_responses.columns = df_responses.columns.str.lower()
    filtered_response_columns = [item for item in df_responses.columns if '_filtered' in item]
    embedding_df = embeddings.create_embeddings_df(
                                                  filtered_response_dataset=df_responses, 
                                                  bias=bias,
                                                  dataset_type=dataset_type,
                                                  id_columns=id_columns,
                                                  columns=columns,
                                                  filtered_response_columns=filtered_response_columns,
                                                  client=client,
                                                  model=model,
                                                  saving_path=saving_path,
                                                  verbose=verbose)
    
    if verbose:
        print('75% - of the process is finished.\nGetting Cosine similarities from the embeddings:')
    
    cos_similarity.create_cos_similarity_df(dataset=embedding_df,
                                            bias=bias, 
                                            dataset_type=dataset_type,
                                            id_columns=id_columns, 
                                            columns=columns,
                                            embedding_columns=embedding_df.columns,
                                            model=model, 
                                            saving_path=saving_path, 
                                            verbose=1)
    print('100% complete - pipeline finished running, CSVs created succesfully.')
    return embedding_df