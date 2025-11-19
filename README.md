<p align="center">
<img src="./assets/project_banner.gif" alt="Project banner" height="300"/>
<!-- ffmpeg -i unused/banner_video.mp4 -vf "fps=15,scale=800:-1:flags=lanczos" -loop 0 assets/project_banner.gif -->
</p>

# ShortLang: Compressed Text for efficient LLMs
## *The future of text representation and processing*

[![AI](https://img.shields.io/badge/AI-C21B00?style=for-the-badge&logo=openaigym&logoColor=white)]()
[![LLMs](https://img.shields.io/badge/LLMs-1A535C?style=for-the-badge&logo=openai&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-darkgreen.svg?style=for-the-badge&logo=github&logoColor=white)](./LICENSE.md)

## Overview

ShortLang is a minimal-length, semantically-preserving textual representation framework designed to optimize language model reasoning, training efficiency, and storage requirements. It compresses natural language into a concise symbolic form while retaining core meaning as measured by embedding similarity.

## Features

- **Rule-Based Compression**: Deterministic methods to remove stopwords, abbreviate entities, and eliminate redundancy.
- **Model-Based Compression**: Uses fine-tuned language models for nuanced semantic compression.
- **Hybrid Approach**: Combines rule-based preprocessing with model-based compression.
- **Embedding Validation**: Objective assessment of semantic retention using cosine similarity.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pro-GenAI/ShortLang.git
   cd ShortLang
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in ".env" based on ".env.example".

## Usage

Run "short_lang/shortlang.py".

## Applications

- Reasoning Optimization
- Training Data Compression
- Efficient Chunking for Vector Embedding
- Vector Database Storage and Retrieval
- Multi-Agent and Multi-Step Systems
