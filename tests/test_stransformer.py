# test_stransformer.py

from sentence_transformers import SentenceTransformer
import numpy as np
import json

def read_input_sentences(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def read_output_embeddings(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def compare_embeddings(generated, expected, tolerance=1e-4):
    return np.allclose(generated, expected, atol=tolerance)

def test_model_embeddings():
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read inputs and expected outputs
    input_sentences = read_input_sentences("tests/input_sentences.txt")
    expected_embeddings = read_output_embeddings("tests/output_embeddings.txt")

    assert len(input_sentences) == len(expected_embeddings), \
        f"Mismatch: {len(input_sentences)} inputs vs {len(expected_embeddings)} expected embeddings"

    for idx, sentence in enumerate(input_sentences):
        generated = model.encode(sentence)
        expected = np.array(expected_embeddings[idx])

        assert compare_embeddings(generated, expected), \
            f"Embedding mismatch at index {idx}: \nInput: {sentence}"

if __name__ == "__main__":
    test_model_embeddings()
    print("✅ All embedding tests passed.")
