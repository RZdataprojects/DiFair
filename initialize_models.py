from anthropic import Anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def initialize_hugging_face_models(model_name: str, hugging_face_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hugging_face_token)
    hugging_face_model = AutoModelForCausalLM.from_pretrained(model_name, token=hugging_face_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hugging_face_model.to(device)
    return hugging_face_model, tokenizer


def initialize_anthropic(anthropic_key: str):
    anthropic_client = Anthropic(api_key=anthropic_key)
    return anthropic_client


def initialize_llama2(hugging_face_token: str):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def initialize_llama3(hugging_face_token: str):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def initialize_mistral(hugging_face_token: str):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def initialize_gemma(hugging_face_token: str):
    model_name = "google/gemma-7b-it"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def initialize_yi(hugging_face_token: str):
    model_name = "01-ai/Yi-6B-Chat"
    yi_model, tokenizer = initialize_hugging_face_models(model_name, hugging_face_token)
    yi_model.eval()
    return yi_model, tokenizer


def initialize_gemini_1_pro(model: str, google_key: str):
    genai.configure(api_key=google_key)

    # Set up the model
    generation_config = {"temperature": 0.5, "top_p": 1, "top_k": 1, "max_output_tokens": 1000, }

    safety_settings = [
      {"category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"},]

    gemini_1_pro_client = genai.GenerativeModel(model_name=model,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    return gemini_1_pro_client