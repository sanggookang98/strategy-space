# Strategy Space

Code for **"Strategy Space: A unified corporate strategy space for managerial decision support"**

> Sanggoo Kang, Doeun Kim, Oh-Hyun Kwon, Woo-Sung Jung, Jaehyuk Park

---

## Overview

Strategy Space is a continuous, multidimensional framework that maps the strategic orientations of S&P 500 firms by embedding CEO earnings call transcripts into a semantic space via a domain-adapted Sentence-BERT model. Each firm-quarter is projected onto four theoretically grounded axes:

- **Exploitation–Exploration (EE)**
- **Domestic–International (DI)**
- **Short–Long-term (SL)**
- **Specialization–Diversification (SD)**

---

## Requirements

```bash
pip install sentence-transformers transformers datasets spacy nltk tqdm pandas numpy
python -m spacy download en_core_web_sm
```

---

## Data

Earnings call transcripts and financial data were obtained from LSEG and are subject to licensing restrictions. These data are not publicly available.

The file `data/axis_definition.csv` contains the reference texts used to construct the four semantic axes. `data/SP500.csv` contains the S&P 500 ticker list with GICS sector information.

---

## Usage

Run the scripts in order. Update the path variables at the top of each script before running.

**Step 1: Domain-adaptive pre-training**
```bash
python 1_finetune.py
```
Fine-tunes `all-mpnet-base-v2` on CEO earnings call transcripts using selective masked language modeling (nouns, verbs, and adjectives only; ORG and PERSON entities excluded).

**Step 2: Transcript embedding**
```bash
python 2_embed.py
```
Encodes each CEO transcript into a 768-dimensional firm-quarter embedding.

**Step 3: Anchor vector construction**
```bash
python 3_create_anchors.py
```
Embeds the reference texts in `data/axis_definition.csv` and computes bipolar anchor vectors for each strategic axis.

**Step 4: Axis score computation**
```bash
python 4_compute_scores.py
```
Projects each firm-quarter embedding onto the four semantic axes via cosine similarity and saves the results to `outputs/axis_scores_all.csv`.

---

