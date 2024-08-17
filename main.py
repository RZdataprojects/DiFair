import pipeline
import pandas as pd
import os
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the pipeline with the specified parameters.')
    parser.add_argument('--model', type=str, required=True, help="""Model name. Supported models: ['claude-3-opus-20240229',   'gpt-4o-mini-2024-07-18','gemini-1.0-pro', 'gemma', 'llama-2', 'llama-3','mistral', 'yi']""")
    parser.add_argument('--dataset_type', type=str, required=True, help="""Dataset type - meant for comments that you wish to save in the file's name ["YYYY-MM-DD", "calibration"].""")
    parser.add_argument('--bias', type=str, required=True, help='Bias type to be analyzed.')
    parser.add_argument('--id_columns', nargs='+', required=True, help='List of ID columns.')
    parser.add_argument('--columns', nargs='+', required=True, help='List of lowercase column headers to compare, e.g., ["male", "female", "neutral"].')
    parser.add_argument('--open_ai_key', type=str, required=True, help="Key for OpenAI, necessary for embedding retrieval.")
    parser.add_argument('--anthropic_key', type=str, required=False, help="Key for Anthropic's models.")
    parser.add_argument('--google_key', type=str, required=False, help="Key for Google's models.")
    parser.add_argument('--hugging_face_key', type=str, required=False, help="Key for Hugging Face's models.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset's TSV file with the prompts.")
    parser.add_argument('--saving_path', type=str, required=True, help='Path to save the results.')
    parser.add_argument('--verbose', type=int, required=False, default=1, help='Verbosity level.')

    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(args.dataset_path, sep='\t')

    # Ensure the saving path exists
    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path)

    print(args.saving_path, " ", args.model)
    # Run the pipeline
    pipeline.pipeline(
        dataset=df,
        bias=args.bias,
        dataset_type=args.dataset_type, 
        id_columns=args.id_columns, 
        columns=args.columns, 
        saving_path=args.saving_path, 
        open_ai_key=args.open_ai_key, 
        anthropic_key=args.anthropic_key,  
        google_key=args.google_key, 
        hugging_face_key=args.hugging_face_key, 
        model=args.model, 
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()