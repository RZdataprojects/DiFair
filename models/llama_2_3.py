from typing import Optional

from hugging_face_models import HuggingFaceModelAdapter
import logging

logger = logging.getLogger(__name__)


class LlamaAdapter(HuggingFaceModelAdapter):
    """
    Adapter for LLaMA 2/3 models from Hugging Face.

    This class extends the HuggingFaceModel to provide specific configurations
    and methods for interacting with LLaMA 2 models.

    Attributes:
        model_name (str): The name/identifier of the LLaMA 2 model.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        hugging_face_api_key (Optional[str]): API key for accessing private models.
    """

    def __init__(
        self,
        version: int,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        hugging_face_api_key: Optional[str] = None
    ):
        """
        Initialize the LLaMA 2 model adapter.

        Args:
            version: The version of LLaMA model to use (2 or 3).
            temperature: Sampling temperature for generation (default: 0.5).
            max_tokens: Maximum number of tokens to generate (default: 1000).
            hugging_face_api_key: Optional API key for accessing private models (default: None).
        """
        model_map = {
            2: "meta-llama/Llama-2-7b-chat-hf",
            3: "meta-llama/Meta-Llama-3-8B-Instruct"
        }

        if version not in model_map:
            raise ValueError(f"Unsupported Llama version: {version}. Use 2 or 3.")

        self.version = version
        model_name = model_map[self.version]

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
        version_prompt_map = {
        2:  f"""<s>[INST] <<SYS>>
            {self._system_prompt()}
            User's question: {prompt}
            [/INST]""",
        3: [
            {"role": "system",
             "content": self._system_prompt()},
            {"role": "user", "content": prompt},
           ]
        }

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output, input_tokens = super().generate(version_prompt_map[self.version], terminators)

        # Decode generated text
        response = output[0][input_tokens.shape[-1]:]
        generated_text = self.tokenizer.decode(response, skip_special_tokens=True)

        return generated_text