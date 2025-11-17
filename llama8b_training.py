import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ======================================================
# 0. Environment Setup
# ======================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE = "data.json"

# ======================================================
# 1. Load Dataset
# ======================================================
required_cols = [
    "Input text",
    "input text emotion",
    "response_gender",
    "response_age",
    "response_persona",
    "actual_responses",
]

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
for col in required_cols:
    if col not in dataset.column_names:
        raise ValueError(f"Missing column: {col}")

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

print("Train:", len(train_dataset), "Val:", len(val_dataset))

# ======================================================
# 2. Tokenizer
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


# ======================================================
# 3. Prompt + Label Masking
# ======================================================
def format_prompt(example):
    system = (
        "You are an empathetic AI assistant with the following persona:\n"
        f"- Personality: {example['response_persona']}\n"
        f"- Gender: {example['response_gender']}\n"
        f"- Age: {example['response_age']}\n"
        f"- Emotion: {example['input text emotion']}\n"
    )

    return (
        "<|system|>\n" + system + "\n"
        "<|user|>\n" + example["Input text"] + "\n"
        "<|assistant|>\n"
    )


def tokenize(example):
    prefix = format_prompt(example)
    response = example["actual_responses"]

    full_text = prefix + response

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=4096,
        add_special_tokens=False,
    )

    # Label masking
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    labels = [-100] * (len(tokenized["input_ids"]) - len(response_ids)) + response_ids
    labels = labels[: len(tokenized["input_ids"])]

    tokenized["labels"] = labels
    return tokenized


train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# ======================================================
# 4. Load Base Model
# ======================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto",
)

model.config.pad_token_id = tokenizer.eos_token_id
model.gradient_checkpointing_enable()

# ======================================================
# 5. Apply LoRA
# ======================================================
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ======================================================
# 6. TrainingArguments (p4d-24xlarge optimized)
# ======================================================
training_args = TrainingArguments(
    output_dir="lora-llama3.1-8b-empathetic",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    warmup_ratio=0.03,
    bf16=True,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    fsdp="full_shard auto_wrap",
    fsdp_config={"min_num_params": int(1e8), "use_orig_params": True},
    ddp_find_unused_parameters=False,
    logging_steps=30,
    report_to=None,
)

# ======================================================
# 7. Trainer
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ======================================================
# 8. Train
# ======================================================
trainer.train()

# ======================================================
# 9. Final Evaluation
# ======================================================
metrics = trainer.evaluate()
loss = metrics["eval_loss"]
metrics["perplexity"] = float(np.exp(loss))
print(metrics)

with open("final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ======================================================
# 10. Save Model
# ======================================================
out_dir = "final-empathetic-lora-llama3.1-8b"
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)

# ======================================================
# 11. Inference Example
# ======================================================
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
)
lora_model = PeftModel.from_pretrained(base_model, out_dir, device_map="auto")

prompt = (
    "<|system|>\nYou are an empathetic assistant.\n"
    "<|user|>\nI feel stressed about exam deadlines.\n"
    "<|assistant|>\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
output = lora_model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
