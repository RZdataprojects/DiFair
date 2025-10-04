from typing import Optional

from .hugging_face_models import HuggingFaceModelAdapter
import logging

logger = logging.getLogger(__name__)


class MistralAdapter(HuggingFaceModelAdapter):
    """
    Adapter for Gemma model from Hugging Face.

    This class extends the HuggingFaceModel to provide specific configurations
    and methods for interacting with LLaMA 2 models.

    Attributes:
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        hugging_face_api_key (Optional[str]): API key for accessing private models.
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
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         hugging_face_api_key=hugging_face_api_key)

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate a response for the given user prompt.
        Adapts the prompt format based on the LLaMA version.
        Args:
            prompt: The input prompt to generate a response for.
        Returns:
            model generated responses.
        """
        logger.debug(self.model_name + ': ', prompt)

        # Input text
        messages = [
            {"role": "user", "content": "You are a helpful assistant."
                                        "Answer the question without asking for additional information."
                                        "User's question: {user_prompt}".format(user_prompt=prompt)},
            {"role": "assistant", "content": ""}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)

        outputs = self.model.generate(inputs, temperature=0.5, do_sample=True, max_new_tokens=1000)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1].strip()

        # Clear CUDA memory
        self.clear_gpu_memory()

        return generated_text