from typing import Optional

from hugging_face_models import HuggingFaceModelAdapter
import logging

logger = logging.getLogger(__name__)


class YiAdapter(HuggingFaceModelAdapter):
    """Adapter for 01.AI's Yi models."""

    def __init__(
        self,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        hugging_face_api_key: Optional[str] = None
    ):
        """
        Initialize the Yi adapter.

        Args:
            temperature: Sampling temperature for generation (default: 0.5).
            max_tokens: Maximum number of tokens to generate (default: 1000).
            hugging_face_api_key: Optional API key for accessing private models (default: None).
        """
        model_name = "01-ai/Yi-6B-Chat"
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         hugging_face_api_key=hugging_face_api_key)
        self.model.eval()

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate a response using 01.AI's Yi model.

        Args:
            prompt: The input prompt to generate a response for.

        Returns:
            The generated response text.
        """
        logger.debug(self.model_name + ': ', prompt)

        # Input text
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt}
        ]

        # Tokenize input text
        input_ids = self.tokenizer.apply_chat_template(conversation=messages,
                                                       tokenize=True,
                                                       add_generation_prompt=True,
                                                       return_tensors='pt').to(self.device)

        # Generate text
        output = self.model.generate(input_ids, temperature=0.5, do_sample=True, max_new_tokens=1000,
                                             num_return_sequences=1)

        # Decode generated text
        generated_text = self.tokenizer.decode(output[0][input_ids.shape[1]:],
                                               skip_special_tokens=True)  # .split("[/INST]")[1].strip()

        # Clear CUDA memory
        self.clear_gpu_memory()

        return generated_text