import pandas as pd
import torch
import gc
import re
from time import sleep
import logging
import nltk
from nltk.corpus import stopwords
from typing import Optional, Union, List, Set
from functools import partial

# --- Setup device and logger ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clears GPU memory by emptying the CUDA cache and collecting garbage."""
    torch.cuda.empty_cache()
    gc.collect()
    logger.debug("GPU memory cleared.")


def get_response_google_gemini_1_pro(user_prompt: str, model="gemini-1-pro", client=None) -> Optional[str]:
    """
    Generates a response using Google's models.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is "gemini-1-pro".
        client: The API client for communication with the model.

    Returns:
        str: The generated response. If an error is received from the API, returns None.
    """
    sleep(5)
    logger.debug(model + ': ', user_prompt)
    try:
        raw_response = client.generate_content(f"""You are a helpful assistant."
        "Answer the question without asking for additional information."
        "User's question: {user_prompt}""")
        response = raw_response.candidates[0].content.parts[0].text.replace('\n\n', '').replace('\n', '')
        if response:
            return response
        else:
            logger.warning('Empty response received from Gemini.')
            return None
    except Exception as e:
        logger.error('Error: ' + str(e))
        return None


def get_response_anthropic(user_prompt: str, model: str="claude-3-opus-20240229", client=None) -> Optional[str]:
    """
    Generates a response using Anthropic's models.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used. Default is "claude-3-opus-20240229".
        client: The API client for communication with the model.

    Returns:
        str: The generated response. If an error is received from the API, returns None.
    """
    sleep(5)
    logger.debug(model + ': ', user_prompt)
    try:
        response = client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=1000,
            temperature=0.5,
            system="You are a helpful assistant."
                   "Answer the question without asking for additional information.",
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
            return response.content[0].text
        else:
            logger.warning('Empty response received from Anthropic.')
            return None

    except Exception as e:
        logger.error("Error in Anthropic's API: " + str(e))
    return None


def get_response_open_ai(user_prompt: str, client, model='gpt-4o-mini-2024-07-18') -> Optional[str]:
    """
    Generates a response using OpenAI's GPT model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        client: The API client for communication with the model.
        model (str): The model to be used. Default is 'gpt-4o-mini-2024-07-18'.

    Returns:
        str: The generated response. If an error is received from the API, returns None.
    """
    logger.debug(model + ': ', user_prompt)
    try:
        response = client.chat.completions.create(
          model=model,
          messages=[
            {
              "role": "system",
              "content": "You are a helpful assistant."
                         "Answer the question without asking for additional information."
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
            return response.choices[0].message.content
        else:
            logger.warning('Empty response received from OpenAI.')
            return None
    except Exception as e:
        print("Error in OpenAI API:", str(e))
    return None


def get_response_meta_llama(user_prompt: str, model:str, hugging_face_model, tokenizer) -> str:
    """
    Generates a response using Meta's LLaMAs models.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    logger.debug(model + ': ', user_prompt)
    generated_text = ""

    if model == 'llama-2':
        # Input text
        input_text = """<s>[INST] <<SYS>>
        You are a helpful assistant. Answer the question without asking for additional information. 
        User's question: {BODY}
        [/INST]""".format(BODY=user_prompt)
        # Tokenize input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate text
        output = hugging_face_model.generate(input_ids, temperature=0.5, do_sample=True, max_new_tokens=1000, num_return_sequences=1)
                
        # Decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).split("[/INST]")[1].strip()
        
    if model == 'llama-3':
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

    return generated_text


def get_response_google_gemma(user_prompt: str, model:str, hugging_face_model, tokenizer) -> str:
    """
    Generates a response using Google's Gemma model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    logger.debug(model + ': ', user_prompt)

    # Input text
    input_text = """<start_of_turn>user
    You are a helpful assistant.
    Answer the question without asking for additional information. 
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

    return generated_text


def get_response_mistral_ai_mistral(user_prompt: str, model: str, hugging_face_model, tokenizer) -> str:
    """
    Generates a response using Mistral_AI's Mistral model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    logger.debug(model + ': ', user_prompt)

    # Input text
    messages = [
        {"role": "user", "content": "You are a helpful assistant."
                                    "Answer the question without asking for additional information."
                                    "User's question: {BODY}".format(BODY=user_prompt)},
        {"role": "assistant", "content": ""}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    outputs = hugging_face_model.generate(inputs, temperature=0.5, do_sample=True, max_new_tokens=1000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1].strip()

    # Clear CUDA memory
    clear_gpu_memory()

    return generated_text


def get_response_01_ai_yi(user_prompt: str, model:str, hugging_face_model, tokenizer) -> str:
    """
    Generates a response using 01.AI's Yi model.

    Args:
        user_prompt (str): The input prompt for generating a response.
        model (str): The model to be used.
        hugging_face_model: The Hugging Face model object.
        tokenizer: The tokenizer object for processing input.

    Returns:
        str: The generated response.
    """
    logger.debug(model + ': ', user_prompt)

    # Input text
    messages = [
              {"role": "system", "content": "You are a helpful assistant."
                                            "Answer the question without asking for additional information."},
              {"role": "user", "content": user_prompt}
        ]

    # Tokenize input text
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)
    
    # Generate text
    output = hugging_face_model.generate(input_ids, temperature=0.5, do_sample=True, max_new_tokens=1000, num_return_sequences=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)#.split("[/INST]")[1].strip()

    # Clear CUDA memory
    clear_gpu_memory()

    return generated_text


def get_stopwords_list(bias: Optional[str] = None) -> Set[str]:
    """
    Returns a list of stopwords including custom stopwords based on the specified bias type.

    Args:
        bias (Optional[str]): The type of bias, used to determine custom stopwords to be removed.
    Returns:
        Set[str]: The list of stopwords.
    """
    # custom_stopwords are nouns related to the bias type from the dataset.
    # This ensures any disparity detected is not a results of the presence of these words.
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
                            'Brad', 'Smith', 'David', 'Miller', 'Li', 'Chen', 'Malik', 'Williams', 'Xavier',
                            'Rodriguez', 'Ricardo', 'Lopez']
    else:
        logger.warning(bias, " is not supported, add custom stopwords when implementing a custom dataset.")
        custom_stopwords = []

    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)  # Add custom stopwords to the set
    else:
        logger.warning(
            "No custom stopwords were added, continued with default NLTK stopwords. Results will be influenced by the presence of bias-related words.")

    return stop_words


def remove_stopwords(text: str, custom_stopwords: Set[str]) -> Optional[str]:
    """
    Cleans input text by removing unwanted characters, pronouns, and excessive whitespaces.

    Args:
        text (str): The input text to clean.
        custom_stopwords (Set[str]): Custom stopwords to be removed.
    Returns:
        str: The cleaned text.
    """
    try:
        # Remove multi-word stopwords using regex
        # This step is needed for multi-word stopwords (e.g., "flight attendant")
        for phrase in custom_stopwords:
            phrase_pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            text = phrase_pattern.sub('', text)

        # Token filtering ensures any remaining single-word stopwords
        # (missed by regex due to punctuation or tokenization) are removed
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in custom_stopwords]
        filtered_text = ' '.join(filtered_tokens)
        
        # Remove extra spaces around punctuation
        filtered_text = re.sub(r'\s([,\.!?\"])', r'\1', filtered_text)

    except TypeError as e:
        logger.error("Unable to remove stopwords, TypeError:", str(e))
        return None  # Return None if issue arises to prevent run from stopping without saving progress
    except Exception as e:
        logger.error("An unexpected error occurred while removing stopwords:", str(e))
        return None  # Return None if issue arises to prevent run from stopping without saving progress

    logger.debug("Removed stopwords.")
    return filtered_text

def get_response(user_prompt: str, model: str, client=None, hugging_face_model=None, tokenizer=None) -> Union[str, None]:
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
        return get_response_anthropic(user_prompt=user_prompt, model=model, client=client)
    if model in ['llama-2', 'llama-3']:
        return get_response_meta_llama(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer)
    if model == 'gpt-4o-mini-2024-07-18':
        return get_response_open_ai(user_prompt=user_prompt, model=model, client=client)
    if model == 'gemma':
        return get_response_google_gemma(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer)
    if model == 'yi':
        return get_response_01_ai_yi(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer)
    if model in ['gemini-1.0-pro', 'gemini-2.5-flash-lite']:
        return get_response_google_gemini_1_pro(user_prompt=user_prompt, model=model, client=client)
    if model == 'mistral':
        return get_response_mistral_ai_mistral(user_prompt=user_prompt, model=model, hugging_face_model=hugging_face_model, tokenizer=tokenizer)
    raise ValueError(f'Model "{model}" not supported.')


def create_responses_df(
    dataset: pd.DataFrame,
    bias: str,
    id_columns: List[str],
    columns: List[str],
    model: str,
    client=None,
    hugging_face_model=None,
    tokenizer=None
) -> pd.DataFrame:

    """
    Creates a DataFrame containing the responses generated by a model for each entry in the dataset.

    Returns:
        pd.DataFrame: The DataFrame containing the generated responses and filtered responses.
    """
    postfix = '_response'
    response_columns = [col + postfix for col in columns]
    responses_df = pd.DataFrame(index=dataset.index, columns=response_columns)
    responses_df[id_columns] = dataset[id_columns]

    for idx, row in dataset.iterrows():
        for column in columns:
            if client is None:
                response = get_response(user_prompt=row[column],
                                        model=model,
                                        hugging_face_model=hugging_face_model,
                                        tokenizer=tokenizer
                            )
            else:
                response = get_response(user_prompt=row[column], model=model, client=client)
            responses_df.at[idx, column + postfix] = response

    custom_stopwords = get_stopwords_list(bias)
    remove_stopwords_with_custom = partial(remove_stopwords, custom_stopwords=custom_stopwords)

    for col in columns:
        filtered_col = col + postfix + '_filtered'  # remove stopwords and custom stopwords
        responses_df[filtered_col] = responses_df[col + postfix].apply(remove_stopwords_with_custom)

    return responses_df
