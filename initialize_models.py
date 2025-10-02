"""
Module to initialize various language models including Hugging Face models,
Anthropic, and Google Gemini.
ChatGPT is initialized in the main pipeline file due to its role in retrieving embeddings.
"""

import logging
from typing import Tuple, Any
from anthropic import Anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


def initialize_hugging_face_models(model_name: str, hugging_face_token: str) -> Tuple[Any, Any]:
    """
    Initialize a Hugging Face model and tokenizer.

    Args:
        model_name (str): Name of the model.
        hugging_face_token (str): Hugging Face API token.

    Returns:
        Tuple[Any, Any]: Model and tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hugging_face_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hugging_face_token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Loaded Hugging Face model: {model_name} on {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face model {model_name}: {e}")
        raise


def initialize_anthropic(anthropic_key: str) -> Anthropic:
    """
    Initialize Anthropic client.

    Args:
        anthropic_key (str): Anthropic API key.

    Returns:
        Anthropic: Anthropic client instance.
    """
    client = Anthropic(key=anthropic_key)
    logger.info("Initialized Anthropic client")
    return client


def initialize_llama2(hugging_face_token: str) -> Tuple[Any, Any]:
    """Initialize Llama 2 model."""
    return initialize_hugging_face_models(model_name="meta-llama/Llama-2-7b-chat-hf", hugging_face_token=hugging_face_token)


def initialize_llama3(hugging_face_token: str) -> Tuple[Any, Any]:
    """Initialize Llama 3 model."""
    return initialize_hugging_face_models( model_name="meta-llama/Meta-Llama-3-8B-Instruct", hugging_face_token=hugging_face_token)


def initialize_mistral(hugging_face_token: str) -> Tuple[Any, Any]:
    """Initialize Mistral model."""
    return initialize_hugging_face_models(model_name="mistralai/Mistral-7B-Instruct-v0.2", hugging_face_token=hugging_face_token)


def initialize_gemma(hugging_face_token: str) -> Tuple[Any, Any]:
    """Initialize Gemma model."""
    return initialize_hugging_face_models(model_name="google/gemma-7b-it", hugging_face_token=hugging_face_token)


def initialize_yi(hugging_face_token: str) -> Tuple[Any, Any]:
    """Initialize Yi model."""
    model, tokenizer = initialize_hugging_face_models(model_name="01-ai/Yi-6B-Chat", hugging_face_token=hugging_face_token)
    model.eval()
    logger.info("Initialized Yi model")
    return model, tokenizer


def initialize_gemini_1_pro(model: str, google_key: str) -> Any:
    """
    Initialize Gemini 1 Pro model.

    Args:
        model (str): Model name.
        google_key (str): Google API key.

    Returns:
        Any: Gemini GenerativeModel instance.
    """
    genai.configure(api_key=google_key)
    generation_config = {
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 1000,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    client = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logger.info("Initialized Gemini client")
    return client