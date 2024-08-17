import numpy as np
import pandas as pd


def get_embedding(text, client, model="text-embedding-3-large", verbose=1):
    try:
        text = text.replace("\n", " ")
        return np.array(client.embeddings.create(input=[text], model=model).data[0].embedding).reshape(1, -1)[0]
    except Exception:
        print('None value has been detected, please fill all missing values manually.')
    return None


def create_embeddings_df(*, filtered_response_dataset, bias, dataset_type, id_columns, columns, filtered_response_columns, client, model, saving_path, verbose=1):
    postfix = '_embeddings'
    embedding_columns = [col + postfix for col in columns if not col.endswith('_id')]
    embedding_df = pd.DataFrame()
    embedding_df[id_columns] = filtered_response_dataset[id_columns] 
    print(model)
    
    for filtered_response_column, embeddings_column in zip(filtered_response_columns, embedding_columns):
        embedding_df[embeddings_column] = filtered_response_dataset[filtered_response_column].apply(get_embedding, args=(client, "text-embedding-3-large", verbose))
    embedding_df.to_parquet(saving_path + model + ' ' +  bias + ' ' + dataset_type + ' - ' + 'embeddings.parquet', index=False)
    if verbose==1:
        print('Done.')
    return embedding_df