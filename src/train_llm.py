import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from src.config import CFG

MODEL_NAME = "google/flan-t5-small"
OUT_DIR = "runs/flan_t5_boolq"


def preprocess(examples, tokenizer, max_input_len=384, max_target_len=16):
    inputs = [
        f"question: {q} passage: {p} answer yes or no"
        for q, p in zip(examples["question"], examples["passage"])
    ]

    targets = ["yes" if a else "no" for a in examples["answer"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        targets,
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
    )["input_ids"]

    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset("boolq")

    print("Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized = ds.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    train_ds = tokenized["train"].shuffle(seed=0).select(range(CFG["llm_train_n"]))
    eval_ds = tokenized["validation"].select(range(CFG["llm_eval_n"]))

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        learning_rate=2e-4,
        per_device_train_batch_size=CFG["batch_train"],
        per_device_eval_batch_size=CFG["batch_eval"],
        num_train_epochs=CFG["epochs"],
        weight_decay=0.01,
        fp16=CFG["fp16"],
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"Training complete. Model saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

