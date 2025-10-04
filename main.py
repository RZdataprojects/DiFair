import pandas as pd
import os
from dotenv import load_dotenv
import argparse
import logging

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the pipeline with the specified parameters.')
    parser.add_argument('--model_name', type=str, required=True, help="""Model name. Supported models: ['claude-3-opus-20240229', 'gemini-2.5-flash-lite', 'gpt-4o-mini-2024-07-18','gemini-1.0-pro', 'gemma', 'llama-2', 'llama-3','mistral', 'yi']""")
    parser.add_argument('--title_comment', type=str, required=False, default='', help="""Comments that you wish to save in the file's name (e.g. "YYYY-MM-DD", "calibration").""")
    parser.add_argument('--bias', type=str, required=True, help='Bias type to be analyzed.')
    parser.add_argument('--id_columns', nargs='+', required=True, help='List of ID columns.')
    parser.add_argument('--columns', nargs='+', required=True, help='List of lowercase column headers to compare, e.g., ["male", "female", "neutral"].')
    parser.add_argument('--temperature', type=float, required=False, default=0.5, help='Temperature for text generation.')
    parser.add_argument('--max_tokens', type=int, required=False, default=1000, help='The maximum number of tokens allowed for text generation.')
    parser.add_argument('--open_ai_api_key', type=str, required=True, help="Key for OpenAI, necessary for embedding retrieval.")
    parser.add_argument('--anthropic_api_key', type=str, required=False, help="Key for Anthropic's models.")
    parser.add_argument('--google_api_key', type=str, required=False, help="Key for Google's models.")
    parser.add_argument('--hugging_face_api_key', type=str, required=False, help="Key for Hugging Face's models.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset's TSV file with the prompts.")
    parser.add_argument('--saving_path', type=str, required=False, default="./output", help='Path to save the results.')
    parser.add_argument('--log_level', type=str, required=False, default='INFO',
                        help='Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.')

    args = parser.parse_args()
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Import pipeline here to ensure logging config is synced
    from pipeline import Pipeline

    # Read the dataset
    df = pd.read_csv(args.dataset_path).iloc[:3]

    # Ensure the saving path exists
    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path)

    # Run the pipeline
    Pipeline().run(
        dataset=df,
        bias=args.bias,
        title_comment=args.title_comment,
        id_columns=args.id_columns, 
        columns=args.columns, 
        saving_path=args.saving_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        open_ai_api_key=args.open_ai_api_key or os.getenv('OPENAI_API_KEY'),
        anthropic_api_key=args.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY'),
        google_api_key=args.google_api_key or os.getenv('GOOGLE_API_KEY'),
        hugging_face_api_key=args.hugging_face_api_key or os.getenv('HUGGING_FACE_TOKEN'),
        model_name=args.model_name
    )

if __name__ == '__main__':
    main()