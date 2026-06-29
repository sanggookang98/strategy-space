import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────
JSON_DIR    = "/path/to/output_embeddings"          # directory containing firm-quarter embedding JSON files
ANCHOR_FILE = "anchors/anchors_embedded.json"       # anchor vectors produced by 3_create_anchors.py
SP500_FILE  = "/path/to/SP500.csv"                  # S&P 500 ticker list (Symbol column required)
OUTPUT_FILE = "outputs/axis_scores_all.csv"         # output CSV with per-firm-quarter axis scores

EXCLUDED = ['BF.B', 'BRK.B', 'DELL', 'ED', 'EXPD', 'FOX', 'GOOG',
            'MPWR', 'NVR', 'NWS', 'PSKY', 'Q', 'SCHW']
MAPPING  = {'BFb': None, 'FOXA': 'FOXA', 'GOOGL': 'GOOGL', 'NWSA': 'NWSA'}

AXIS_DIRECTIONS = {
    'diversification_specialization': {'reversed': False},
    'international_domestic'        : {'reversed': False},
    'exploration_exploitation'      : {'reversed': False},
    'longterm_shortterm'            : {'reversed': False},
}
# ──────────────────────────────────────────────────────────────────


def load_anchors(anchor_file):
    with open(anchor_file, 'r') as f:
        anchor_data = json.load(f)

    anchors = {}
    for anchor in anchor_data['anchors']:
        name = anchor['name']
        if anchor['type'] == 'bipolar' and name in AXIS_DIRECTIONS:
            pos_mean = np.mean([np.array(d['vector'])
                                for d in anchor['pos_definitions']], axis=0)
            neg_mean = np.mean([np.array(d['vector'])
                                for d in anchor['neg_definitions']], axis=0)
            vec = pos_mean - neg_mean
            vec = vec / np.linalg.norm(vec)
            anchors[name] = vec

    print(f"Loaded {len(anchors)} anchor vectors.")
    return anchors


def cosine_similarity(embedding, anchor_vector):
    norm_emb    = np.linalg.norm(embedding)
    norm_anchor = np.linalg.norm(anchor_vector)
    if norm_emb == 0 or norm_anchor == 0:
        return np.nan
    return np.dot(embedding, anchor_vector) / (norm_emb * norm_anchor)


def compute_scores(json_dir, anchor_file, sp500_file, output_file):
    anchors = load_anchors(anchor_file)

    sp500 = pd.read_csv(sp500_file, encoding='utf-8-sig')
    sp500_tickers = set(sp500[~sp500['Symbol'].isin(EXCLUDED)]['Symbol'].tolist())

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    records = []
    json_files = list(Path(json_dir).glob("*.json"))

    for json_file in tqdm(json_files, desc="Computing scores"):
        ticker = json_file.stem

        if ticker in MAPPING:
            ticker = MAPPING[ticker]
            if ticker is None:
                continue

        if ticker not in sp500_tickers:
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        for year, quarters in data.items():
            for quarter, content in quarters.items():
                if 'embedding' not in content:
                    continue

                emb = np.array(content['embedding'])
                record = {
                    'ticker' : ticker,
                    'year'   : int(year),
                    'quarter': int(quarter),
                }

                for axis_name, axis_vec in anchors.items():
                    score = cosine_similarity(emb, axis_vec)
                    if AXIS_DIRECTIONS[axis_name]['reversed']:
                        score = -score
                    record[axis_name] = score

                records.append(record)

    df_scores = (pd.DataFrame(records)
                   .sort_values(['ticker', 'year', 'quarter'])
                   .reset_index(drop=True))

    df_scores.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDone. {len(df_scores):,} firm-quarters saved to {output_file}")
    print(df_scores.head(10))


if __name__ == "__main__":
    compute_scores(JSON_DIR, ANCHOR_FILE, SP500_FILE, OUTPUT_FILE)
