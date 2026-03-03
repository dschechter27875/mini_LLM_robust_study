import os

PRESET = os.environ.get("PRESET", "cpu").lower()

CFG = {
    "cpu": {
        "llm_train_n": 600,
        "llm_eval_n": 200,
        "batch_train": 4,
        "batch_eval": 4,
        "epochs": 1,
        "fp16": False,
    },
    "gpu": {
        "llm_train_n": 3000,
        "llm_eval_n": 500,
        "batch_train": 16,
        "batch_eval": 16,
        "epochs": 2,
        "fp16": True,
    },
}[PRESET]

