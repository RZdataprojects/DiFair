# DiFair-LLM: Evaluating Fairness Disparities in LLMs Toward Demographic Groups

![DiFair](https://github.com/RZdataprojects/DiFair/blob/main/DiFair-LLM%20Flow.png)
This repository contains the code and pipeline for SEiLLM - *Stereotype evaluation in Large Language Models (LLMs)* with respect to demographic attributes and groups.

## Overview

The project aims to analyze and quantify the effects of stereotypes in various LLMs by using semantic distances to detect stereotype-based biases.

## Features

- Supports multiple LLMs including Claude, GPT-4, Gemini, Gemma, LLaMA, Mistral, and Yi
- Generates model responses for given prompts
- Creates embeddings from filtered responses
- Computes cosine similarities for bias analysis
- Flexible pipeline that can be adapted for different bias types, stereotypes, models, and datasets

## Prerequisites

- **Python**: Ensure you have Python 3.10 or later installed.
- **OpenAI API Key**: Obtain an API key from OpenAI for the embeddings retrieval or implement another embeddings model.
- **Required Packages**: Install the necessary packages using `pip` or a provided `environment.yaml` file.

## Installation

**Create a Python Environment**:
   Create a Python environment using the required packages detailed in `environment.yaml`:
   ```
   conda env create -f environment.yaml
   ```
   Or manually install the necessary package:
   ```
   pip install openai
   ```

## Usage

1. **Run the Main Script**:
  Chose a combination of model and dataset (see Main.py help for more info.).
   Example Usage:

   
### Gender

```
python main.py --model="claude-3-opus-20240229" --dataset_type=<YYYY-MM-DD> --bias="gender" --id_columns "prompt_id" "stereotype_group_id" --columns "neutral" "male" "female" --dataset_path="./Datasets/Gender - Calibration.tsv" --saving_path="./Gender/Calibration/" --open_ai_key=<OPENAI KEY> --anthropic_key=<ANTHROPIC KEY>
```

#### - Calibration

```
python main.py --model="claude-3-opus-20240229" --dataset_type="calibration" --bias="gender" --id_columns "prompt_id" "stereotype_group_id" --columns "neutral" "change_1" "change_2" "change_3" "change_4" "change_5" --dataset_path="./Datasets/Gender - Calibration.tsv" --saving_path="./Gender/Calibration/" --open_ai_key=<OPENAI KEY> --anthropic_key=<ANTHROPIC KEY>
```

### Ageism

```
python main.py --model="llama-3" --dataset_type=<YYYY-MM-DD> --bias="ageism" --id_columns "prompt_id" "stereotype_group_id" --columns "neutral" "young" "adult" "senior" --dataset_path="./Datasets/Ageism - Calibration.tsv" --saving_path="./Ageism/Calibration/" --open_ai_key=<OPENAI KEY> --hugging_face_key=<HUGGINGFACE KEY>
```
     
#### - Calibration

```
python main.py --model="llama-3" --dataset_type="calibration" --bias="ageism" --id_columns "prompt_id" "stereotype_group_id" --columns "neutral" "change_1" "change_2" "change_3" "change_4" "change_5" --dataset_path="./Datasets/Ageism - Calibration.tsv" --saving_path="./Ageism/Calibration/" --hugging_face_key=<HUGGINGFACE KEY>
```

### Ethnicity

```
python main.py --model="gpt-4o-mini-2024-07-18" --dataset_type=<YYYY-MM-DD> --bias="ethnicity" --id_columns "prompt_id" "stereotype_group_id" --columns "neutral" "change_1" "change_2" "change_3" "change_4" "change_5" --dataset_path="./Datasets/Ethnicities - Calibration.tsv" --saving_path="./Ethnicities/Calibration/" --open_ai_key=<OPENAI KEY>    
```

#### - Calibration

```
python main.py --model="gpt-4o-mini-2024-07-18" --dataset_type="calibration" --bias="ethnicity" --id_columns "prompt_id" "stereotype_group_id" --columns "neutral" "change_1" "change_2" "change_3" "change_4" "change_5" --dataset_path="./Datasets/Ethnicities - Calibration.tsv" --saving_path="./Ethnicities/Calibration/" --open_ai_key=<OPENAI KEY> 
```

## Project Structure

```
├───── Ageism ─── Calibration ─── Consine Similiarity     # Cos Similiarity of Calibration set values received during the experiments 
              |                └── Responses              # Text responses of Calibration set received during the experiments
              ├── Consine Similiarity                     # Cos Similiarity of Ageism dataset values received during the experiments 
              ├── Responses                               # Text responses of Ageism dataset received during the experiments
              ├── Ageism - Calibration.tsv                # Ageism calibration dataset
              └── Ageism.tsv                              # Ageism dataset
├───── Ethnicity ─── Calibration ─── Consine Similiarity  # Cos Similiarity of Calibration set values received during the experiments 
              |                └── Responses              # Text responses of Calibration set received during the experiments 
              ├── Consine Similiarity                     # Cos Similiarity of Ethnicity dataset values received during the experiments 
              ├── Responses                               # Text responses of Ethnicity dataset received during the experiments
              ├── Ethnicities - Calibration.tsv           # Ethnicity calibration dataset
              └── Ethnicities - Dataset.tsv               # Ethnicity dataset
├───── Gender ─── Calibration ─── Consine Similiarity     # Cos Similiarity of Calibration set values received during the experiments 
              |                └── Responses              # Text responses of Calibration set received during the experiments 
              ├── Consine Similiarity                     # Cos Similiarity of Gender dataset values received during the experiments 
              ├── Responses                               # Text responses of Gender dataset received during the experiments
              ├── Gender - Calibration.tsv                # Gender calibration dataset
              └── Gender Dataset.tsv                      # Gender dataset
├── main.py               
├── pipeline.py           
├── responses.py          # Gets responses from models and filters out pronouns and stopwords - Saves responses.csv
├── embeddings.py         # Retrieves embeddings - Saves an embeddings.parquet
├── cos_similarity.py     # Calculates semantic distances - Saves cos_similarity.csv
├── environment.yaml      # Conda environment configuration file
├── SEiLLM.ipynb          # Statistical analysis of results
└── README.md             # Project documentation
```

## License
Creative Commons Attribution-ShareAlike 4.0 International

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss proposed changes.

## Acknowledgements
Paper is under review.
