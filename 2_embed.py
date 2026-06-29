import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# ── Configuration ─────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH    = "/path/to/finetuned_model"          # path to fine-tuned S-BERT model
SOURCE_DIR    = "/path/to/earnings_call_transcripts" # directory containing CEO transcript JSON files
TARGET_DIR    = "/path/to/output_embeddings"         # output directory for firm-quarter embeddings
MAX_TOKENS    = 510
OVERLAP_SENTS = 2
# ──────────────────────────────────────────────────────────────────

model     = SentenceTransformer(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

os.makedirs(TARGET_DIR, exist_ok=True)


def get_chunks(text, max_tokens=510, overlap_sents=2):
    """Split text into overlapping chunks that do not exceed max_tokens."""
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = []
        for j in range(i, len(sentences)):
            test_text = " ".join(current_chunk + [sentences[j]])
            tokens = tokenizer.encode(test_text, add_special_tokens=True)
            if len(tokens) > max_tokens:
                break
            current_chunk.append(sentences[j])

        if not current_chunk and i < len(sentences):
            current_chunk.append(sentences[i])

        chunks.append(" ".join(current_chunk))
        i += max(1, len(current_chunk) - overlap_sents)
        if i >= len(sentences):
            break
    return chunks


def embed_transcripts():
    file_list = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]

    for file_name in tqdm(file_list, desc="Embedding transcripts"):
        with open(os.path.join(SOURCE_DIR, file_name), 'r') as f:
            data = json.load(f)

        result_data = {}
        for year, quarters in data.items():
            result_data[year] = {}
            for q, content in quarters.items():
                text = content['text']
                if not text.strip():
                    continue

                chunks = get_chunks(text, max_tokens=MAX_TOKENS,
                                    overlap_sents=OVERLAP_SENTS)
                chunk_embeddings = model.encode(chunks, batch_size=8,
                                                show_progress_bar=False)
                avg_embedding = np.mean(chunk_embeddings, axis=0)

                result_data[year][q] = {
                    "text"     : text,
                    "embedding": avg_embedding.tolist(),
                }

        with open(os.path.join(TARGET_DIR, file_name), 'w') as f:
            json.dump(result_data, f)

    print(f"Embeddings saved to {TARGET_DIR}")


if __name__ == "__main__":
    embed_transcripts()
