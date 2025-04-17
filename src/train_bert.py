import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

def tokenize_and_prepare(df, tokenizer):
    """
    Tokenizes and prepares a Hugging Face Dataset for training.
    Uses 'clean_text' as input and 'label' as target.
    """
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean_text"], df["label"].astype(int), test_size=0.2, random_state=42
    )

    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

    train_dataset = Dataset.from_dict({**train_encodings, "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({**test_encodings, "label": test_labels.tolist()})

    return train_dataset, test_dataset, test_labels


def train_and_evaluate(df, model_tag="default"):
    """
    Trains and evaluates BERT on a dataset and saves performance metrics.
    """
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset, test_dataset, y_true = tokenize_and_prepare(df, tokenizer)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"./results/bert_{model_tag}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="no",                # disable automatic eval
        logging_dir=f"./results/logs_{model_tag}",
        logging_steps=10,                        # match partner logging
        save_strategy="no",                      # don’t save checkpoints
        report_to="none",                        # suppress W&B or others
        disable_tqdm=False                       # enable progress bar
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    os.makedirs("results", exist_ok=True)
    with open(f"results/metrics_{model_tag}.json", "w") as f:
        json.dump({
            "accuracy": round(acc, 4),
            "f1_phishing": round(report["1"]["f1-score"], 4),
            "precision": round(report["1"]["precision"], 4),
            "recall": round(report["1"]["recall"], 4),
        }, f, indent=2)
