# ShortLang: Compressed Text for efficient LLMs
# Implementation based on the paper

import re
from typing import Dict
import string

from sklearn.metrics.pairwise import cosine_similarity

from short_lang.constants import ABBREVIATIONS, STOPWORDS
from short_lang.llm_functions import ask_llm, get_embedding


def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between embeddings of two texts."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return similarity


def rule_based_compress(text: str) -> str:
    """Apply rule-based compression to text."""
    # Convert to lowercase for processing
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split into words
    words = text.split()

    # Remove stopwords
    filtered_words = [word for word in words if word not in STOPWORDS]

    # Join back
    compressed = " ".join(filtered_words)

    # Apply abbreviations (case insensitive)
    for full, abbr in ABBREVIATIONS.items():
        compressed = re.sub(re.escape(full.lower()), abbr.lower(), compressed)

    return compressed


def llm_based_compress(text: str, max_tokens: int = 100) -> str:
    """Apply model-based compression using OpenAI."""
    prompt = (
        "Compress the following text into a minimal, semantically-preserving "
        "representation optimized for machine processing. Retain core meaning "
        "but remove redundancy and unnecessary details. Keep it concise:\n\n"
        "Example Input: 'The quick brown fox jumps over the lazy dog several times in a playful manner.'\n"
        "Example Compressed Output: 'quick brown fox jumps lazy dog playfully.'\n\n"
        f"Input text: {text}\n\nCompressed version:"
    )
    compressed = ask_llm(prompt)
    return compressed


# def hybrid_compress(text: str, max_tokens: int = 100) -> str:
#     """Apply hybrid compression: rule-based first, then model-based."""
#     # Step 1: Rule-based preprocessing
#     preprocessed = rule_based_compress(text)

#     # Step 2: Model-based compression on preprocessed text
#     compressed = llm_based_compress(preprocessed, max_tokens)
#     return compressed


def validate_compression(original: str, compressed: str) -> Dict[str, float]:
    """Validate compression by computing similarity and compression ratio."""
    similarity = compute_similarity(original, compressed)
    original_words = len(original.split())
    compressed_words = len(compressed.split())
    compression_ratio = (
        compressed_words / original_words if original_words > 0 else 0
    )
    return {
        "similarity": similarity,
        "compression_ratio": compression_ratio,
        "original_words": original_words,
        "compressed_words": compressed_words,
    }


if __name__ == "__main__":
    sample_text = (
        "The rapid increase in the size and complexity of modern language models "
        "has generated renewed interest in methods that can reduce computational "
        "requirements without compromising semantic fidelity."
    )

    print("Original:", sample_text)
    rule_compressed = rule_based_compress(sample_text)
    print("Rule-based:", rule_compressed)
    llm_compressed = llm_based_compress(sample_text)
    print("Model-based:", llm_compressed)
    print("=" * 50)

    # Validation
    validation = validate_compression(sample_text, llm_compressed)
    for key, value in validation.items():
        print(f"{key}: {value}")
