from base_model import BaseModelAdapter
import logging
import os
from anthropic import Anthropic
from time import sleep
from typing import Optional

logger = logging.getLogger(__name__)


class AnthropicModelAdapter(BaseModelAdapter):
    def __init__(self,
                 anthropic_api_key: str,
                 temperature: float = 0.5,
                 max_tokens: int = 1000):
        model_name = "claude-3-opus-20240229"
        os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.model = self.initialize_model(anthropic_api_key)


    def initialize_model(self, anthropic_api_key: str):
        """
        Initialize the Anthropic Client.
        Args:
            anthropic_api_key (str): Google API key.

        Returns:
            Model Client
        """
        client = Anthropic(key=anthropic_api_key)
        logger.debug("Initialized Anthropic client")
        return client


    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generates a response using Anthropic's models.

        Args:
            prompt (str): The input prompt for generating a response.

        Returns:
            Optional[str]: The generated response, None if an error occurs.
        """
        sleep(5)
        logger.debug(self.model_name + ': ', prompt)
        try:
            response = self.model.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
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
