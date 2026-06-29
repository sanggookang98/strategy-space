import os
import json
import torch
import spacy
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ── Configuration ─────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
SOURCE_DIR = "/path/to/earnings_call_transcripts"   # directory containing CEO transcript JSON files
OUTPUT_DIR = "/path/to/finetuned_model"             # output directory for fine-tuned model
# ──────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_and_chunk():
    """Segment transcripts into chunks of up to 256 tokens with sentence boundaries."""
    all_chunks = []
    file_list = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]

    for file_name in tqdm(file_list, desc="Loading transcripts"):
        with open(os.path.join(SOURCE_DIR, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for year, quarters in data.items():
                for q, content in quarters.items():
                    text = content.get('text', "")
                    if len(text) < 100:
                        continue

                    sentences = sent_tokenize(text)
                    current_chunk = []
                    current_length = 0

                    for sent in sentences:
                        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
                        if current_length + len(sent_tokens) > 256:
                            if current_chunk:
                                all_chunks.append(
                                    tokenizer.decode(
                                        tokenizer.encode(" ".join(current_chunk),
                                                         add_special_tokens=False)
                                    )
                                )
                            current_chunk = [sent]
                            current_length = len(sent_tokens)
                        else:
                            current_chunk.append(sent)
                            current_length += len(sent_tokens)

                    if current_chunk:
                        all_chunks.append(" ".join(current_chunk))

    return Dataset.from_dict({"text": all_chunks})


class StrategicDataCollator(DataCollatorForLanguageModeling):
    """Custom MLM collator that masks only nouns, verbs, and adjectives,
    excluding ORG and PERSON named entities."""

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.0)

        for i, sentence_tensor in enumerate(inputs):
            text = self.tokenizer.decode(sentence_tensor, skip_special_tokens=True)
            doc = nlp(text)

            target_words = [
                token.text for token in doc
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']
                and token.ent_type_ not in ['ORG', 'PERSON']
                and len(token.text) > 1
            ]

            for word in target_words:
                word_ids = self.tokenizer.encode(word, add_special_tokens=False)
                for w_id in word_ids:
                    probability_matrix[i][inputs[i] == w_id] = self.mlm_probability

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def main():
    dataset = load_and_chunk()
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        save_steps=1000,
        fp16=True,
        learning_rate=2e-5,
        logging_steps=100,
        weight_decay=0.01,
        push_to_hub=False,
    )

    data_collator = StrategicDataCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting domain-adaptive pre-training ...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
