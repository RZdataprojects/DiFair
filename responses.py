import pandas as pd
import torch
import gc
import re
from time import sleep
import logging
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.WARNING)


def clear_gpu_memory():
    """
    Clears GPU memory by emptying the CUDA cache and collecting garbage.
    """
    torch.cuda.empty_cache()
    gc.collect()


def get_response_google_gemini_1_pro(user_prompt: str, model="gemini-1-pro", client=None, verbose=1):
    """
    Generates a response using Google's models.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is "gemini-1-pro".
        client: The API client for communication with the model.
        verbose (int): The verbosity level for logging. Default is 1.

    Returns:
        str: The generated response. If an error is received from the API, returns None.
    """
    sleep(5)
    if verbose == 1:
        print(model + ': ', user_prompt, end=' ')
    try:
        raw_response = client.generate_content(f"""You are a helpful assistant. 
        Answer the question without asking for additional information. User's question: {user_prompt}""")
        # print(raw_response)
        response = raw_response.candidates[0].content.parts[0].text.replace('\n\n', '').replace('\n', '')
        # print(response)  # Checking if a response was received from the API.
        if response:
            # Returning the content of the first choice as the generated response.
            if verbose == 1:
                print(response, 'Done.')
            return response
        else:
            #
            return get_response_google_gemini_1_pro(user_prompt=user_prompt, model=model, client=client, verbose=verbose)
    except Exception as e:
        print('Error: ' + str(e))
        return None


def get_response_anthropic(user_prompt: str, model="claude-3-opus-20240229", client=None, verbose=1):
    """
    Generates a response using Anthropic's models.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is "claude-3-opus-20240229".
        client: The API client for communication with the model.
        verbose (int): The verbosity level for logging. Default is 1.

    Returns:
        str: The generated response. If an error is received from the API, returns None.
    """
    sleep(5)
    if verbose == 1:
        print(model + ': ', user_prompt, end=' ')
    try:
        response = client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=1000,
            temperature=0.5,
            system="You are a helpful assistant. Answer the question without asking for additional information.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )
        
        # Checking if a response was received from the API.
        if response.content:
            # Returning the content of the first choice as the generated response.
            if verbose == 1:
                print('Done.')
            return response.content[0].text
        else:
            return None

    except Exception as e:
        print("Error in Anthropic's API: " + str(e))
    return None


def get_response_open_ai(user_prompt: str, client, model='gpt-4o-mini-2024-07-18', verbose=1):
    """
    Generates a response using OpenAI's GPT model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        client: The API client for communication with the model.
        model (str): The model to be used. Default is 'gpt-4o-mini-2024-07-18'.
        verbose (int): The verbosity level for logging. Default is 1.

    Returns:
        str: The generated response. If an error is received from the API, returns None.
    """
    if verbose == 1:
        print(model + ': ', user_prompt, end=' ')
    try:
        response = client.chat.completions.create(
          model=model,
          messages=[
            {
              "role": "system",
              "content": "You are a helpful assistant. Answer the question without asking for additional information."
            },
            {
              "role": "user",
              "content": user_prompt
            }
          ],
          temperature=0.5,
          max_tokens=1000
        )
        # Checking if a response was received from the API.
        if response.choices:
            # Returning the content of the first choice as the generated response.
            if verbose == 1:
                print('Done.')
            return response.choices[0].message.content
    except Exception as e:
        print("An unexpected error occurred while fetching responses from OpenAI:", str(e))
    return None


def get_response_meta_llama(user_prompt: str, model='llama-2', verbose=1, hugging_face_model=None, tokenizer=None):
    """
    Generates a response using Meta's LLaMAs models.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is 'llama-2'.
        verbose (int): The verbosity level for logging. Default is 1.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    if verbose:
        print(model + ': ', user_prompt, end=' ')

    if model == 'llama-2':
        # Input text
        input_text = """<s>[INST] <<SYS>>
        You are a helpful assistant. Answer the question without asking for additional information. 
        User's question: {BODY}
        [/INST]""".format(BODY=user_prompt)
        # In your answer avoid relying on sterotypes and stay inclusive towards all groups.
        # Tokenize input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate text
        output = hugging_face_model.generate(input_ids, temperature=0.5, do_sample=True, max_new_tokens=1000, num_return_sequences=1)
                
        # Decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).split("[/INST]")[1].strip()
        
    elif model == 'llama-3':
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question without asking for additional information."},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt"
                                                ).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = hugging_face_model.generate(
            input_ids,
            max_new_tokens=1000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5
        )
        
        # Decode generated text
        response = output[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(response, skip_special_tokens=True)
    # Clear CUDA memory
    clear_gpu_memory()

    if verbose == 1:
        print('Done.')
    return generated_text


def get_response_google_gemma(user_prompt: str, model, hugging_face_model, tokenizer, verbose=1):
    """
    Generates a response using Google's Gemma model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model: The model to be used.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.
        verbose (int): The verbosity level for logging. Default is 1.

    Returns:
        str: The generated response.
    """
    if verbose:
        print(model + ': ', user_prompt, end=' ')

    # Input text
    input_text = """<start_of_turn>user
    You are a helpful assistant. Answer the question without asking for additional information. 
    User's question: {BODY}<end_of_turn>
    <start_of_turn>model
    """.format(BODY=user_prompt)

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate text
    output = hugging_face_model.generate(input_ids, temperature=0.5, do_sample=True, max_new_tokens=1000, num_return_sequences=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).split('model\n')[1].strip()

    # Clear CUDA memory
    clear_gpu_memory()

    if verbose==1:
        print('Done.')
    return generated_text


def get_response_mistral_ai_mistral(user_prompt: str, model='mistarl', verbose=1, hugging_face_model=None, tokenizer=None):
    """
    Generates a response using Mistral_AI's Mistral model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is 'mistarl'.
        verbose (int): The verbosity level for logging. Default is 1.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    if verbose:
        print(model + ': ', user_prompt, end=' ')

    # Input text
    messages = [
        {"role": "user", "content": "You are a helpful assistant. Answer the question without asking for additional information. User's question: {BODY}".format(BODY=user_prompt)},
        {"role": "assistant", "content": ""}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    outputs = hugging_face_model.generate(inputs, temperature=0.5, do_sample=True, max_new_tokens=1000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1].strip()

    # Clear CUDA memory
    clear_gpu_memory()

    if verbose==1:
        print('Done.')
    return generated_text


def get_response_01_ai_yi(user_prompt: str, model='yi-6b-chat', verbose=1, hugging_face_model=None, tokenizer=None):
    """
    Generates a response using 01.AI's Yi model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is 'mistarl'.
        verbose (int): The verbosity level for logging. Default is 1.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    if verbose:
        print(model + ': ', user_prompt, end=' ')

    # Input text
    messages = [
              {"role": "system", "content": "You are a helpful assistant. Answer the question without asking for additional information."},
              {"role": "user", "content": user_prompt}  # User's prompt for generating a relevant response.
        ]

    # Tokenize input text
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)
    
    # Generate text
    output = hugging_face_model.generate(input_ids, temperature=0.5, do_sample=True, max_new_tokens=1000, num_return_sequences=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)#.split("[/INST]")[1].strip()

    # Clear CUDA memory
    clear_gpu_memory()

    if verbose==1:
        print('Done.')
    return generated_text


def remove_stopwords(text, bias=None):
    """
    Cleans input text by removing unwanted characters, pronouns, and excessive whitespaces.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    try:
        if bias == 'gender':
            custom_stopwords = ['Jane', 'John', 'actor', 'actress', 'aunt', 'bachelor', 'bachelorette', 'boy', 
                                'boyfriend', 'bride', 'brother', 'child', 'daughter', 'duchess', 'duke', 'emcee', 
                                'father', 'firefighter', 'fireman', 'firewoman', 'flight attendant', 'flight-attendant', 
                                'friend', 'gentleman', 'girl', 'girlfriend', 'grandchild', 'granddaughter', 'grandfather', 
                                'grandmother', 'grandparent', 'grandson', 'groom', 'he', 'him', 'her', 'they', 'them', 
                                'hero', 'heroine', 'heros', 'host', 'hostess', 'husband', 'kid', 'king', 'lady', 'lord', 
                                'madam', 'mailman', 'mailwoman', 'man', 'men', 'missis', 'mister', 'monarch', 'mother', 
                                'mr.', 'ms.', 'nephew', 'newlywed', 'niece', 'parent', 'partner', 'people', 'performer', 
                                'person', 'police officer', 'policeman', 'policewoman', 'postal worker', 'postal-worker',
                                'worker', 'prince', 'princess', 'queen', 'royal', 'salesman', 'salesperson', 'saleswoman',
                                'server', 'she', 'sibling', 'single person', 'sir', 'sister', 'son', 'spokesman',
                                'spokesperson', 'spokeswoman', 'spouse', 'stepchild', 'stepdad', 'stepdaughter', 'stepmom', 
                                'stepparent', 'stepson', 'steward', 'stewardess', 'uncle', 'waiter', 'waitress',
                                'who lost a spouse', 'who-lost-a-spouse', 'widow', 'widower', 'wife', 'woman', 'women']
        elif bias == 'ageism':
            custom_stopwords = ['person', 'man', 'woman', 'student', 'teenager', 'young', 'boy', 'girl', '15 year old', 
                                '20 year old', '30 year old', '40 year old', '15-year-old', '20-year-old', '30-year-old', 
                                '40-year-old', 'year-old', 'youngster', 'adult', 'employee', 'middle-aged', 'middle aged', 
                                'father', 'mother', 'year', 'breadwinner', 'senior', 'pensioner', 'elderly', 'old',
                                'grandpa', 'grandma', 'elder', 'geezer', 'old-timer', 'oldtimer', 'old timer']
        elif bias == 'ethnicity':
            custom_stopwords = ['man', 'woman', 'person', 'american', 'white', 'asian', 'black', 'african', 'latino',
                                'latin', 'native', 'cherokee', 'arab', 'middle', 'eastern', 'middle-eastern',
                                'white-american', 'asian-american', 'black-american', 'african-american', 'latino-american',
                                'latina', 'latina-american', 'latin-american', 'native-american', 'cherokee-american',
                                'hispanic-american', 'middle-eastern-american', 'arab-american', 'hispanic',
                                'Brad', 'Smith', 'David', 'Miller',	'Li', 'Chen', 'Malik', 'Williams', 'Xavier', 
                                'Rodriguez', 'Ricardo', 'Lopez']
        else:
            print(bias, " is not supported, please add custom stopwords if you are implementing a custom dataset.")
            custom_stopwords = []

        stop_words = set(stopwords.words('english'))
        if bias:
            stop_words.update(custom_stopwords)  # Add custom stopwords to the set

        # Remove multi-word stopwords using regex
        for phrase in custom_stopwords:
            phrase_pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            text = phrase_pattern.sub('', text)

        tokens = nltk.word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        filtered_text = ' '.join(filtered_tokens)
        
        # Remove extra spaces around punctuation
        filtered_text = re.sub(r'\s([,\.!?\"])', r'\1', filtered_text)
        return filtered_text

    except TypeError as e:
        print("Unable to remove stopwords, TypeError:", str(e))
    except Exception as e:
        print("An unexpected error occurred while removing stopwords:", str(e))
    return None

    
def get_response(user_prompt: str, model: str, client=None, hugging_face_model=None, tokenizer=None, verbose=1):
    """
    This method calls a get_response_<org>_<model name> according to the model parameter based on the model selected. 

    user_prompt (str): The input prompt.
    model (str): The model to be used for generating a response. Default is set to 'gpt-4-0613'.
    key (str): API key for authentication (optional).
    client: Client object for API communication (optional).
    verbose (int): Verbosity level for logging. Default is 1.
    hugging_face_model: Hugging Face model object (optional).
    tokenizer: Tokenizer object for processing input (optional).
    """
    if model == 'claude-3':
        return get_response_anthropic(user_prompt=user_prompt, model=model, client=client, verbose=verbose)
    if ((model == 'llama-2') | (model == 'llama-3')):
        return get_response_meta_llama(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer, verbose=verbose)
    if model in ['gpt-4o-mini-2024-07-18']:
        return get_response_open_ai(user_prompt=user_prompt, model=model, client=client, verbose=verbose)
    if model == 'gemma':
        return get_response_google_gemma(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer, verbose=verbose)
    if model == 'yi':
        return get_response_01_ai_yi(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer, verbose=verbose)
    if model == 'gemini-1.0-pro':
        return get_response_google_gemini_1_pro(user_prompt=user_prompt, model=model, client=client, verbose=verbose)
    if model == 'mistral':
        return get_response_mistral_ai_mistral(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer, verbose=verbose)
    print(f'error, model "{model}" not in built-in options.')


def create_responses_df(dataset,
                        bias,
                        dataset_type,
                        id_columns,
                        columns,
                        saving_path,
                        model,
                        client=None,
                        hugging_face_model=None,
                        tokenizer=None, 
                        verbose=1):
    """
    Creates a DataFrame containing the responses generated by a model for each entry in the dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing the prompts to which responses will be generated.
        bias (str): The type of bias to be considered during stopwords removal.
        dataset_type (str): The type of the dataset, e.g., ["YYYY-MM-DD", "calibration"].
        id_columns (list): A list of column names that represent identifiers in the dataset.
        columns (list): A list of column names that contain the prompts for which responses need to be generated.
        saving_path (str): The path where the resulting DataFrame with responses will be saved as a CSV file.
        model (str): The model to be used for generating responses.
        client: The API client used for communicating with the model (if applicable).
        hugging_face_model: The Hugging Face model object used for generating responses (if applicable).
        tokenizer: The tokenizer object used for processing input text (if applicable).
        verbose (int): The verbosity level for logging progress. Default is 1.

    Returns:
        pd.DataFrame: The DataFrame containing the generated responses and filtered responses.
    """
    postfix = '_response'
    response_columns = [col + postfix for col in columns if not col.endswith('_id')]
    responses_df = pd.DataFrame(index=dataset.index, columns=response_columns)
    responses_df[id_columns] = dataset[id_columns]

    for index, row in dataset.iterrows():
        for column in columns:
            if client is None:
                response = get_response(user_prompt=row[column], model=model, verbose=verbose,
                                        hugging_face_model=hugging_face_model, tokenizer=tokenizer)
            else:
                response = get_response(user_prompt=row[column], model=model, client=client, verbose=verbose)
            responses_df.at[index, column + postfix] = response
    for col in columns:
        filtered_col = col + postfix + '_filtered'  # remove stopwords and custom stopwords
        responses_df[filtered_col] = responses_df[col + postfix].apply(remove_stopwords, args=(bias,))

    responses_df.to_csv(saving_path + model + ' ' +  bias + ' ' + dataset_type + ' - ' + 'responses.csv', index=False)
    return responses_df