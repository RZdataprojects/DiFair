"""
Base adapter class for language models.

This module defines the abstract base class that all model adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Abstract base class for all language model adapters.

    This class defines the interface that all model adapters must implement,
    ensuring consistent behavior across different model providers.

    Attributes:
        model_name (str): The name/identifier of the model.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 1000
    ):
        """
        Initialize the base model adapter.

        Args:
            model_name: The name/identifier of the model.
            temperature: Sampling temperature for generation (default: 0.5).
            max_tokens: Maximum number of tokens to generate (default: 1000).
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def initialize_model(self, **kwargs):
        """
        Initialize the model instance.

        This method should be implemented by subclasses to set up the specific model.

        Returns:
            The initialized model instance.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The input prompt to generate a response for.

        Returns:
            The generated response text, or None if generation fails.
        """
        pass

    @staticmethod
    def _system_prompt() -> str:
        """
        Get the default system message for the model.

        Returns:
            The system message instructing the model how to behave.
        """
        return (
            "You are a helpful assistant. "
            "Answer the question without asking for additional information."
        )
