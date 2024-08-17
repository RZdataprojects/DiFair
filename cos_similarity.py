from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from itertools import combinations

def calculate_cosine_similarity(column1, column2):
    try:
        return cosine_similarity(np.array(column1).reshape(1, -1), np.array(column2).reshape(1, -1))[0][0]
    except:
        return None

def create_cos_similarity_df(dataset, bias, dataset_type, id_columns, columns, embedding_columns, model, saving_path, verbose=0):
    postfix = '_cos_similarity'
    embedding_columns = [col for col in embedding_columns if not col.endswith('_id')]
    cos_similarity_columns = [col + postfix for col in columns if not col.endswith('_id')]
    
    combinations_of_columns_titles = []
    for col1, col2 in combinations(columns, 2):  
        combinations_of_columns_titles.append(f'cos_similarity: {col1} vs {col2}')

    cos_similarity_df = pd.DataFrame(index=dataset.index, columns=combinations_of_columns_titles)
   
    for col in id_columns:
        cos_similarity_df[col] = dataset[col] 
        
    combinations_of_columns = []
    for col1, col2 in combinations(embedding_columns, 2):  
    # Create new column name
        combinations_of_columns.append([col1, col2])

    for index, row in dataset.iterrows():
        for column, embeddings_col in zip(combinations_of_columns_titles, combinations_of_columns):
            score = calculate_cosine_similarity(row[embeddings_col[0]], row[embeddings_col[1]])
            cos_similarity_df.at[index, column] = score
    if verbose==1:
        print('Done.')
    cos_similarity_df.to_csv(saving_path + model + ' ' +  bias + ' ' + dataset_type + ' - ' + 'cos_similarity.csv', index=False)
    return cos_similarity_df