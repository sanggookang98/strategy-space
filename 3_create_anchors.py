import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────
ANCHOR_CSV  = "data/axis_definition.csv"            # CSV file with anchor texts
OUTPUT_PATH = "anchors/anchors_embedded.json"       # output path for anchor vectors
MODEL_PATH  = "/path/to/finetuned_model"            # path to fine-tuned S-BERT model
# ──────────────────────────────────────────────────────────────────

POSITIVE_POLE_MAP = {
    'exploration_exploitation'      : 'exploration',
    'international_domestic'        : 'international',
    'diversification_specialization': 'diversification',
    'longterm_shortterm'            : 'longterm',
}


def get_pos_pole(axis_name, poles):
    poles = list(poles)
    if axis_name in POSITIVE_POLE_MAP:
        mapped = POSITIVE_POLE_MAP[axis_name]
        for p in poles:
            if p.lower() == mapped.lower():
                return p
    first_token = axis_name.split('_')[0].lower()
    for p in poles:
        if p.lower() == first_token or p.lower().startswith(first_token):
            return p
    pole_list = sorted(poles)
    print(f"  Warning: could not determine positive pole for '{axis_name}'. "
          f"Using fallback: '{pole_list[1]}'")
    return pole_list[1]


def create_anchors(csv_path, output_path, model_path):
    print(f"Loading model from {model_path} ...")
    model = SentenceTransformer(model_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep='|')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    embedded_anchors = []

    for axis_name, group in df.groupby('axis'):
        poles = group['pole'].unique()
        print(f"Processing '{axis_name}' ... (poles: {list(poles)})")

        if len(poles) == 1:
            texts = group['definition'].tolist()
            embeddings = model.encode(texts)
            embedded_anchors.append({
                'name'         : axis_name,
                'type'         : 'unipolar',
                'definitions'  : [{'text': t, 'vector': v.tolist()}
                                   for t, v in zip(texts, embeddings)],
                'anchor_vector': np.mean(embeddings, axis=0).tolist(),
            })

        elif len(poles) >= 2:
            pos_pole = get_pos_pole(axis_name, poles)
            neg_pole = [p for p in poles if p != pos_pole][0]
            print(f"  pos='{pos_pole}'  neg='{neg_pole}'")

            pos_texts = group[group['pole'] == pos_pole]['definition'].tolist()
            neg_texts = group[group['pole'] == neg_pole]['definition'].tolist()

            pos_emb = model.encode(pos_texts)
            neg_emb = model.encode(neg_texts)

            anchor_vector = (np.mean(pos_emb, axis=0) - np.mean(neg_emb, axis=0)).tolist()

            embedded_anchors.append({
                'name'           : axis_name,
                'type'           : 'bipolar',
                'pos_definitions': [{'text': t, 'vector': v.tolist()}
                                     for t, v in zip(pos_texts, pos_emb)],
                'neg_definitions': [{'text': t, 'vector': v.tolist()}
                                     for t, v in zip(neg_texts, neg_emb)],
                'anchor_vector'  : anchor_vector,
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'anchors': embedded_anchors}, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(embedded_anchors)} anchors to {output_path}")


if __name__ == "__main__":
    create_anchors(ANCHOR_CSV, OUTPUT_PATH, MODEL_PATH)
