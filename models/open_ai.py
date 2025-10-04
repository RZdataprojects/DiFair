from base_model import BaseModelAdapter
import logging
import os
from openai import OpenAI
from typing import Optional

logger = logging.getLogger(__name__)


class OpenAIModelAdapter(BaseModelAdapter):
    def __init__(self,
                 open_ai_api_key: str,
                 temperature: float = 0.5,
                 max_tokens: int = 1000):
        model_name = "gpt-4o-mini-2024-07-18"
        os.environ['OPENAI_API_KEY'] = open_ai_api_key
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.model = self.initialize_model(open_ai_api_key)


    def initialize_model(self, open_ai_api_key: str):
        """
        Initialize OpenAI client.

        Args:
            open_ai_api_key (str): OpenAI API key.

        Returns:
            Any: OpenAI client instance.
        """
        client = OpenAI(api_key=open_ai_api_key)
        logger.debug("Initialized OpenAI client")
        return client


    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generates a response using OpenAI's GPT model.

        Args:
            prompt (str): The input prompt for generating a response.

        Returns:
            str: The generated response, None if an error occurs.
        """
        logger.debug(self.model_name + ': ', prompt)
        try:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # Checking if a response was received from the API.
            if response.choices:
                # Returning the content of the first choice as the generated response.
                return response.choices[0].message.content
            else:
                logger.warning('Empty response received from OpenAI.')
                return None
        except Exception as e:
            logger.warning("Error in OpenAI API:", str(e))
        return None
