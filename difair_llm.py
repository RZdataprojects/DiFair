# DiFair/difair_llm.py

import os
from .initialize_models import initialize_all_models
from .embeddings import generate_embeddings
from .responses import generate_responses
from .cos_similarity import compute_cosine_similarity

class DiFairLLM:
    """
    Main class to run the DiFair-LLM pipeline.
    """

    def __init__(self, config):
        """
        Initialize the pipeline with a configuration dictionary.
        :param config: dict containing paths and parameters for the pipeline
        """
        self.config = config
        self.models = None
        self.embeddings = None
        self.responses = None
        self.similarities = None

    def setup_models(self):
        """
        Initialize and load all required language models.
        """
        self.models = initialize_all_models(self.config.get('model_params', {}))

    def run_responses(self):
        """
        Generate model responses for the datasets.
        """
        data_paths = self.config.get('data_paths', {})
        self.responses = generate_responses(self.models, data_paths)

    def run_embeddings(self):
        """
        Generate embeddings for the model responses.
        """
        self.embeddings = generate_embeddings(self.responses, self.config.get('embedding_params', {}))

    def run_cosine_similarity(self):
        """
        Compute cosine similarity between embeddings and calibration data.
        """
        calibration_path = self.config.get('calibration_path')
        self.similarities = compute_cosine_similarity(self.embeddings, calibration_path)

    def save_outputs(self):
        """
        Save responses, embeddings, and similarity results to disk.
        """
        output_dir = self.config.get('output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        # Save responses
        if self.responses is not None:
            self.responses.to_csv(os.path.join(output_dir, 'responses.csv'), index=False)
        # Save embeddings
        if self.embeddings is not None:
            self.embeddings.to_parquet(os.path.join(output_dir, 'embeddings.parquet'), index=False)
        # Save similarities
        if self.similarities is not None:
            self.similarities.to_csv(os.path.join(output_dir, 'cos_similarity.csv'), index=False)

    def run_full_pipeline(self):
        """
        Run the entire DiFair-LLM pipeline.
        """
        self.setup_models()
        self.run_responses()
        self.run_embeddings()
        self.run_cosine_similarity()
        self.save_outputs()