# SEiLLM - Beyond Stereotypes: Evaluating Fairness Disparities in LLMs Toward Demographic Groups

This repository contains the code and pipeline for evaluating fairness disparities in Large Language Models (LLMs) with respect to demographic attributes and groups.

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
- **OpenAI API Key**: Obtain an API key from OpenAI.
- **Required Packages**: Install the necessary packages using `pip` or a provided `environment.yaml` file.

## Installation

1. Clone this repository:
