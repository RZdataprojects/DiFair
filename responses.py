import pandas as pd
import re
import logging
import nltk
from nltk.corpus import stopwords
from typing import Optional, List, Set, Any
from functools import partial

# --- Setup device and logger ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles text processing operations.

    This class removes stopwords and specific terms from text to ensure
    that detected disparities are not simply due to the presence of these words.

    Attributes:
        custom_stopwords: Bias-specific terms to remove.
    """

    # Custom Stopwords are nouns related to the bias type from the dataset.
    # This ensures any disparity detected is not a results of the presence of these words.
    BIAS_STOPWORDS = {
        'gender': ['Jane', 'John', 'actor', 'actress', 'aunt', 'bachelor', 'bachelorette', 'boy',
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
                   'who lost a spouse', 'who-lost-a-spouse', 'widow', 'widower', 'wife', 'woman', 'women'
        ],
        'ageism': [
            'person', 'man', 'woman', 'student', 'teenager', 'young', 'boy', 'girl', '15 year old',
            '20 year old', '30 year old', '40 year old', '15-year-old', '20-year-old', '30-year-old',
            '40-year-old', 'year-old', 'youngster', 'adult', 'employee', 'middle-aged', 'middle aged',
            'father', 'mother', 'year', 'breadwinner', 'senior', 'pensioner', 'elderly', 'old',
            'grandpa', 'grandma', 'elder', 'geezer', 'old-timer', 'oldtimer', 'old timer'
        ],
        'ethnicity': [
            'man', 'woman', 'person', 'american', 'white', 'asian', 'black', 'african', 'latino',
            'latin', 'native', 'cherokee', 'arab', 'middle', 'eastern', 'middle-eastern',
            'white-american', 'asian-american', 'black-american', 'african-american', 'latino-american',
            'latina', 'latina-american', 'latin-american', 'native-american', 'cherokee-american',
            'hispanic-american', 'middle-eastern-american', 'arab-american', 'hispanic',
            'Brad', 'Smith', 'David', 'Miller', 'Li', 'Chen', 'Malik', 'Williams', 'Xavier',
            'Rodriguez', 'Ricardo', 'Lopez'
        ]
    }

    def __init__(self, bias: Optional[str] = None):
        self.custom_stopwords = self.BIAS_STOPWORDS.get(bias, [])
        if not self.custom_stopwords:
            logger.warning(f"{bias} is not supported, add custom stopwords when implementing a custom dataset.")
        self.stop_words = set(stopwords.words('english')).union(set(self.custom_stopwords))


    def remove_stopwords(self, text: str) -> Optional[str]:
        """
        Cleans input text by removing unwanted characters, pronouns, and excessive whitespaces.

        Args:
            text (str): The input text to clean.
        Returns:
            str: The cleaned text, None if an error occurs.
        """
        try:
            # Remove multi-word stopwords using regex
            # This step is needed for multi-word stopwords (e.g., "flight attendant")
            for phrase in self.custom_stopwords:
                phrase_pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                text = phrase_pattern.sub('', text)

            # Token filtering ensures any remaining single-word stopwords
            # (missed by regex due to punctuation or tokenization) are removed
            tokens = nltk.word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in self.custom_stopwords]
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


    def create_responses_df(
            self,
            dataset: pd.DataFrame,
            id_columns: List[str],
            columns: List[str],
            model: Any) -> pd.DataFrame:
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
                response = model.generate(prompt=row[column])
                responses_df.at[idx, column + postfix] = response

        remove_stopwords_with_custom = partial(self.remove_stopwords)

        for col in columns:
            filtered_col = col + postfix + '_filtered'  # remove stopwords and custom stopwords
            responses_df[filtered_col] = responses_df[col + postfix].apply(remove_stopwords_with_custom)

        return responses_df
