#  DiFair-LLM: Detecting and Measuring Bias in Large Language Models

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC%20by%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-b31b1b.svg)](https://ebooks.iospress.nl/volumearticle/75847)

*A comprehensive framework for bias detection in LLMs*

</div>

---

## 📋 Overview

**DiFair-LLM** 
is a model-agnostic framework for detecting and quantifying fairness
disparities - any unequal treatment that benefits or disadvantages a
demographic group. DiFair-LLM uses open-ended, group-specific
and neutral prompts, measures semantic distances between groups’
responses, applies non-parametric statistical tests, and ranks groups
by deviation from a neutral baseline.

![DiFair](https://github.com/RZdataprojects/DiFair/blob/main/DiFair-LLM%20Flow.png)

### ✨ Key Features

- 🔍 **Multi-Model Support**: Compatible with Claude, GPT, Gemini, LLaMA, Mistral, Gemma, and Yi models
- 🎛️ **Customizable Pipeline**: Flexible architecture for custom datasets and bias types
- 📈 **Reproducible Results**: Structured output format for research and analysis
- ⚡ **GPU Acceleration**: Optimized for CUDA-enabled environments

---

## 🏗️ Architecture

The pipeline consists of four main stages:

1. **Response Generation**: Generate model outputs for comparative prompts
2. **Embedding Creation**: Convert filtered responses to semantic embeddings
3. **Similarity Analysis**: Compute cosine similarities to quantify bias
4. **Statistical Testing**: Apply non-parametric tests to assess significance.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for Hugging Face models)
- API keys for desired model providers

### Installation

```bash
# Clone the repository
git clone https://github.com/RZdataprojects/DiFair-LLM.git
cd DiFair-LLM

# Create conda environment
conda env create -f environment.yaml
conda activate difair-llm

# Set up environment variables in .env
```

### Basic Usage

```bash
python main.py \
  --model "gpt-4o-mini-2024-07-18" \
  --title_comment "2024-01-15" \
  --bias "gender" \
  --id_columns "prompt_id" \
  --columns "neutral" "male" "female" \
  --open_ai_key "your_openai_key" \
  --dataset_path "./Gender/Gender Dataset.csv" \
  --saving_path "./output/"
```

---

## 📊 Supported Models

### API-Based Models
- **OpenAI**: `gpt-4o-mini-2024-07-18`
- **Anthropic**: `claude-3-opus-20240229`
- **Google**: `gemini-1.0-pro`, `gemini-2.5-flash-lite`

### Open-Source Models (Hugging Face)
- **Meta**: `llama-2`, `llama-3`
- **Mistral AI**: `mistral`
- **Google**: `gemma`
- **01.AI**: `yi`

---

## 📁 Input Dataset Format

Your dataset should be a CSV file with the following structure:

### Example Demographic Dataset
| prompt_id | neutral             | group_a                       | group_b                       | group_c                       |
|---|---------------------|-------------------------------|--------------|-------------------------------|
| 1 | A person who is...  | A [GROUP A] who is... | A [GROUP B] who is... | A [GROUP C] who is... |

**Requirements:**
- ID column(s) for tracking
- Multiple prompt columns representing different demographic variations
- Prompts should be semantically equivalent except for demographic markers
- Calibration prompts (neutral) to establish a baseline
  - I.e., Without demographic markers, only different in manner, but not in meaning.

### Example Calibration Dataset
| prompt_id | neutral | variation 1 | variation 2...              |
|---|------------------|-----------------|-----------------------------|
| 1 | A person who is...  | A person that is... | There is a person who is... |
  
---

## 🎯 Bias Types

DiFair-LLM comes with built-in support for three bias dimensions:

- Gender Bias
- Age Bias (Ageism)
- Ethnicity Bias


Custom Bias types can be added by curating a dataset (as detailed in the paper) and defining relevant stopwords in `responses.py`.

---

## 📤 Output Format

The pipeline generates three output files:

### 1. Responses (`*_responses.csv`)
Raw and filtered model responses for each prompt variant.

### 2. Embeddings (`*_embeddings.parquet`)
High-dimensional semantic representations of filtered responses.

### 3. Cosine Similarities (`*_cos_similarity.csv`)
Pairwise similarity scores.

**Lower similarity scores** between demographic variants suggest higher bias.

---

## 🔧 Advanced Configuration

### Command-Line Arguments

| Argument             | Required | Description                              |
|----------------------|----------|------------------------------------------|
| `--model_name`       | ✅ | Model identifier                         |
| `--title_comment`    | ✅ | Addition to file(s) title                |
| `--bias`             | ✅ | Bias type to analyze                     |
| `--id_columns`       | ✅ | Column(s) containing identifiers         |
| `--columns`          | ✅ | Prompt columns to compare                |
| `--dataset_path`     | ✅ | Path to input dataset                    |
| `--open_ai_key`      | ✅ | OpenAI API key (required for embeddings) |
| `--anthropic_key`    | ⬜ | Anthropic API key                        |
| `--google_key`       | ⬜ | Google API key                           |
| `--hugging_face_key` | ⬜ | Hugging Face API key                     |
| `--saving_path`      | ⬜ | Output directory (default: `./output`)   |
| `--log_level`        | ⬜ | Logging verbosity (default: `INFO`)      |

### Environment Variables

Edit the `.env` file to set your API keys:

```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HUGGING_FACE_TOKEN=your_key_here
```

---

## 📊 Example Analysis Workflow

1. **Prepare Datasets**: Ensure your demographic and calibration datasets are formatted correctly.
2. **Run Pipeline**: Execute the `main.py` script with appropriate arguments. You will need to run it twice: once for the demographic dataset and once for the calibration dataset.
3. **Review Outputs**: Ensure the generated CSV and Parquet files in the specified output directory are completly filled in and have no missing entries.
4. **Statistical Analysis**: Use the provided Jupyter notebook `DiFair_LLM.ipynb` to analyze the results.

---

## 🛠️ Development

### Project Structure

```
DiFair-LLM/
│
├── Ageism/                               # Ageism bias experiment results
│   ├── Calibration/
│   │   ├── Cosine Similarity/            # Calibration cosine similarities
│   │   └── Responses/                    # Calibration text responses
│   ├── Cosine Similarity/                # Ageism dataset cosine similarities
│   └── Responses/                        # Ageism dataset text responses
│
├── Ethnicity/                            # Ethnicity bias experiment results
│   ├── Calibration/
│   │   ├── Cosine Similarity/            # Calibration cosine similarities
│   │   └── Responses/                    # Calibration text responses
│   ├── Cosine Similarity/                # Ethnicity dataset cosine similarities
│   └── Responses/                        # Ethnicity dataset text responses
│
├── Gender/                               # Gender bias experiment results
│   ├── Calibration/
│   │   ├── Cosine Similarity/            # Calibration cosine similarities
│   │   └── Responses/                    # Calibration text responses
│   ├── Cosine Similarity/                # Gender dataset cosine similarities
│   └── Responses/                        # Gender dataset text responses
│
├── models/                               # Model-specific implementations
│   ├── anthropic.py                      # Anthropic Claude integration
│   ├── base_model.py                     # Base model interface
│   ├── gemma.py                          # Google Gemma integration
│   ├── google.py                         # Google Gemini integration
│   ├── hugging_face_models.py            # Hugging Face models wrapper
│   ├── llama_2_3.py                      # Meta LLaMA 2 & 3 integration
│   ├── mistral.py                        # Mistral AI integration
│   ├── open_ai.py                        # OpenAI GPT integration
│   └── yi.py                             # 01.AI Yi integration
│
├── .env                                  # Environment variables (API keys)
├── .gitignore                            # Git ignore rules
├── CODEOWNERS                            # Repository code owners
├── DiFair-LLM Flow.png                   # Pipeline visualization diagram
├── DiFair_LLM.ipynb                      # Statistical analysis notebook
├── cos_similarity.py                     # Cosine similarity computation
├── embeddings.py                         # OpenAI embedding generation
├── environment.yaml                      # Conda environment specification
├── main.py                               # CLI entry point
├── pipeline.py                           # Pipeline orchestration
├── responses.py                          # Response generation & filtering
└── README.md                             # Project documentation
```

### Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss proposed changes.

---

## 📚 Citation

If you use DiFair-LLM in your research, please cite:

```bibtex
@inproceedings{coheninger2025difair,
  title        = {DiFair-LLM: Evaluating Fairness Disparities in LLMs Toward Demographic Groups},
  author       = {Cohen Inger, Nurit and Zaady, Roei and Solomon, Adir and Rokach, Lior and Shapira, Bracha},
  booktitle    = {Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)},
  year         = {2025}
}
```

---

## 📄 License

Creative Commons Attribution-ShareAlike 4.0 International

---

<div align="center">

**[⬆ back to top](#difair-llm-detecting-and-measuring-bias-in-large-language-models)**
</div>
