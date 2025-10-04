from typing import Optional

from .hugging_face_models import HuggingFaceModelAdapter
import logging

logger = logging.getLogger(__name__)


class GemmaAdapter(HuggingFaceModelAdapter):
    """
    Adapter for Gemma model from Hugging Face.

    This class extends the HuggingFaceModel to provide specific configurations
    and methods for interacting with LLaMA 2 models.

    Attributes:
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        hugging_face_key (Optional[str]): API key for accessing private models.
    """
    def __init__(self,
                 temperature: float = 0.5,
                 max_tokens: int = 1000,
                 hugging_face_api_key: Optional[str] = None):
        """
        Initialize the Gemma model adapter.

        Args:
            temperature: Sampling temperature for generation (default: 0.5).
            max_tokens: Maximum number of tokens to generate (default: 1000).
            hugging_face_api_key: Optional API key for accessing private models (default: None).
        """
        model_name = "gemma-gpt/gemma-1.5b"
        super().__init__(model_name, temperature, max_tokens, hugging_face_api_key)

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate a response for the given user prompt.
        Adapts the prompt format based on the LLaMA version.
        Args:
            prompt: The input prompt to generate a response for.
        Returns:
            model generated responses.
        """
        # Input text
        input_text = f"""<start_of_turn>user
            {self._system_prompt()}
            User's question: {prompt}<end_of_turn>
            <start_of_turn>model
            """

        output, input_tokens = super().generate(input_text)

        # Decode generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True).split('model\n')[1].strip()

        return generated_text
