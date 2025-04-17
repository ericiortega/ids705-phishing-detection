import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

def tokenize_and_prepare(df, tokenizer, max_length=128):
    """
    Splits the dataset into train/test and tokenizes the text.
    """
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["body"], df["label"].astype(int), test_size=0.2, random_state=42
    )

    train_enc = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length)
    test_enc = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=max_length)

    train_dataset = Dataset.from_dict({**train_enc, "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({**test_enc, "label": test_labels.tolist()})

    return train_dataset, test_dataset, test_labels

def train_and_evaluate(df, model_tag="default"):
    """
    Trains and evaluates BERT on a given dataset and saves performance metrics locally.
    """
    # smaller portion because I am testing locally
    if len(df) > 2000:
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds, test_ds, y_true = tokenize_and_prepare(df, tokenizer)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # using Metal (MPS) if available (Apple Silicon), fallback to CPU, allows to run faster on local
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        output_dir=f"./results/bert_{model_tag}",
        per_device_train_batch_size=4,   # low batch size to reduce load
        per_device_eval_batch_size=4,
        num_train_epochs=1,              # fewer epochs for faster run
        logging_dir=f"./results/logs_{model_tag}",
        logging_steps=10,
        save_strategy="no",
        report_to="none",                # disable wandb and other loggers
        disable_tqdm=False               # see progress
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = preds.predictions.argmax(-1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open(f"results/metrics_{model_tag}.json", "w") as f:
        json.dump(
            {
                "accuracy": round(acc, 4),
                "f1_phishing": round(report["1"]["f1-score"], 4),
                "precision": round(report["1"]["precision"], 4),
                "recall": round(report["1"]["recall"], 4),
            },
            f,
            indent=2,
        )
