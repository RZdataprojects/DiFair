from base_model import BaseModelAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import Optional, Tuple, Any, List
import os
import gc

logger = logging.getLogger(__name__)


class HuggingFaceModelAdapter(BaseModelAdapter):
    """
    This class provides an interface to interact with Hugging Face models,

    Attributes:
        model_name (str): The name/identifier of the Hugging Face model.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        hugging_face_api_key: Optional[str] = None
    ):
        """
        Initialize the Hugging Face model adapter.

        Args:
            model_name: The name/identifier of the Hugging Face model.
            temperature: Sampling temperature for generation (default: 0.5).
            max_tokens: Maximum number of tokens to generate (default: 1000).
            hugging_face_api_key: Optional API key for accessing private models (default: None).
        """
        super().__init__(model_name, temperature, max_tokens)

        if hugging_face_api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self.initialize_model()
        logger.debug("Hugging Face model adapter initialized with model '%s' on device '%s'.", self.model_name, self.device)

    def initialize_model(self) -> Tuple[Any, Any]:
        """
        Initialize the Hugging Face model and tokenizer.

        Returns:
            [model, tokenizer]
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        logger.debug("Hugging Face model '%s' loaded successfully.", self.model_name)
        return self.model, self.tokenizer

    def generate(self, prompt: str, terminators: Optional[Any] = None) -> tuple[Any, Any]:
        """
        Generate a response for the given prompt using the Hugging Face model.

        Args:
            prompt: The input prompt string.
            terminators: end of sequence token.
        Returns:
            The generated response string, or None if generation fails.
        """
        logger.debug(self.model_name + ': ', prompt)

        # Tokenize input text
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate text
        output = self.model.generate(input_tokens,
                                     temperature=0.5,
                                     do_sample=True,
                                     eos_token_id=terminators,
                                     max_new_tokens=1000,
                                     num_return_sequences=1)

        input_tokens = input_tokens.cpu().numpy()
        output = output.cpu().numpy()

        # Clear CUDA memory
        self.clear_gpu_memory()

        return output, input_tokens

    @staticmethod
    def clear_gpu_memory():
        """Clears GPU memory by emptying the CUDA cache and collecting garbage."""
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPU memory cleared.")