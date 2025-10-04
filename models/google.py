from .base_model import BaseModelAdapter
import logging
import os
import google.generativeai as genai
from time import sleep
from typing import Optional

logger = logging.getLogger(__name__)


class GoogleModelAdapter(BaseModelAdapter):
    def __init__(self,
                 google_api_key: str,
                 temperature: float = 0.5,
                 max_tokens: int = 1000):
        os.environ['GOOGLE_API_KEY'] = google_api_key
        model_name = "gemini-2.5-flash-lite"#"gemini-1.0-pro"
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.model = self.initialize_model(google_api_key)


    def initialize_model(self, google_api_key: str):
        """
        Initialize the Google Generative AI Client.
        Args:
            google_api_key (str): Google API key.

        Returns:
            Model Client
        """
        genai.configure(api_key=google_api_key)

        generation_config = {
            "temperature": self.temperature,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": self.max_tokens,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logger.debug("Initialized Gemini client")
        return client

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generates a response using Google's models.

        Args:
            prompt (str): The input prompt for generating a response.

        Returns:
            Optional[str]: The generated response, None if an error occurs.
        """
        sleep(5)
        logger.debug(self.model_name + ': ', prompt)
        try:
            raw_response = self.model.generate_content(f"""{self._system_prompt()}
                "User's question: {prompt}""")
            response = raw_response.candidates[0].content.parts[0].text.replace('\n\n', '').replace('\n', '')
            if response:
                return response
            else:
                logger.warning('Empty response received from Gemini.')
                return None
        except Exception as e:
            logger.error('Error: ' + str(e))
            return None