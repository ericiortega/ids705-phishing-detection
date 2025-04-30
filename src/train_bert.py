from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report, accuracy_score
from datasets import Dataset
import torch
import os
import json
from sklearn.model_selection import train_test_split


def train_and_evaluate(df, model_tag="default", text_col="clean_text"):
    """
    Trains and evaluates a BERT model using the specified text column.
    Optimized for fast experimentation and runs in just a few minutes.
    """
    # smaller dataset 
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    # Tokenize
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[text_col], df["label"].astype(int), test_size=0.2, random_state=42
    )
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

    train_dataset = Dataset.from_dict(
        {**train_encodings, "label": train_labels.tolist()}
    )
    test_dataset = Dataset.from_dict({**test_encodings, "label": test_labels.tolist()})

    #  model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training arguments 
    training_args = TrainingArguments(
        output_dir=f"./results/bert_{model_tag}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"./results/logs_{model_tag}",
        logging_steps=2000,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train + Evaluate
    trainer.train()
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)

    # Metrics
    acc = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred, output_dict=True)

    #  results
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
