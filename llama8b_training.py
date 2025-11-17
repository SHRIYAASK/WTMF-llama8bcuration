# fine_tune_llama3_lora_p4d.py
import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# -------------------------
# Environment / performance
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# -------------------------
# 1. Load dataset
# -------------------------
DATA_FILE = "data.json"
required_columns = [
    "Input text",
    "input text emotion",
    "response_gender",
    "response_age",
    "response_persona",
    "actual_responses",
]

try:
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Missing required column: {col}")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    print(f"Train samples: {len(train_dataset)}; Val samples: {len(val_dataset)}")
except Exception as e:
    print("Error loading dataset:", e)
    raise

# -------------------------
# 2. Tokenizer
# -------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # safe padding / eos settings
    # ensure pad_token exists and is same as eos (common for causal models)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    print("Tokenizer loaded. Vocab size:", getattr(tokenizer, "vocab_size", "unknown"))
except Exception as e:
    print("Error loading tokenizer:", e)
    raise


# -------------------------
# 3. Prompt formatting (robust) + Label masking
# -------------------------
def make_prompt(example):
    system = (
        "You are an empathetic AI assistant with the following persona:\n"
        f"- Personality: {example['response_persona']}\n"
        f"- Gender: {example['response_gender']}\n"
        f"- Age: {example['response_age']}\n"
        f"- Emotional context: {example['input text emotion']}\n\n"
        "Provide a helpful, empathetic response to the user's input.\n"
    )
    prompt = (
        "<|system|>\n"
        f"{system}\n"
        "<|user|>\n"
        f"{example['Input text']}\n"
        "<|assistant|>\n"
    )
    return prompt


def tokenize_and_mask_labels(example, max_length=4096):
    """
    Tokenize the whole prompt and create labels that are -100 for everything
    except the assistant response tokens. This prevents the model from learning
    to reproduce system/user text and focuses supervision on the assistant reply.
    """
    assistant_text = example["actual_responses"]
    prompt_prefix = make_prompt(example)

    # Tokenize full input (prompt prefix + assistant text) together so positions match
    full_text = prompt_prefix + assistant_text
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    # Tokenize assistant_text separately to know its length (no special tokens)
    assistant_tokens = tokenizer(
        assistant_text, add_special_tokens=False
    )["input_ids"]

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full.get("attention_mask", [1] * len(input_ids))

    # Ensure assistant_tokens fits at the end; if truncation removed part of assistant,
    # fall back to labeling only what's present
    if len(assistant_tokens) > len(input_ids):
        # extreme truncation: put labels for last N tokens equal to assistant length clipped
        assistant_tokens = assistant_tokens[-len(input_ids) :]

    labels = [-100] * (len(input_ids) - len(assistant_tokens)) + assistant_tokens

    # If lengths mismatch, align by trimming front (safer than throwing)
    if len(labels) != len(input_ids):
        min_len = min(len(labels), len(input_ids))
        input_ids = input_ids[:min_len]
        attention_mask = attention_mask[:min_len]
        labels = labels[:min_len]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Map and filter datasets
train_dataset = train_dataset.map(
    tokenize_and_mask_labels, remove_columns=train_dataset.column_names, batched=False
)
val_dataset = val_dataset.map(
    tokenize_and_mask_labels, remove_columns=val_dataset.column_names, batched=False
)
# Keep only non-empty examples
train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)
print("After tokenization - Train:", len(train_dataset), "Val:", len(val_dataset))

# -------------------------
# 4. Load base model
# -------------------------
try:
    # On multi-GPU nodes it's safer to load with device_map="auto" (lets Transformers place submodules)
    # and specify torch_dtype=bfloat16 for A100-backed p4d. If you prefer Trainer-managed FSDP only,
    # you can remove device_map and load on CPU with low_cpu_mem_usage=True — but that can be slow.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",  # helps place modules across the 8 GPUs on p4d
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    # enable gradient checkpointing explicitly
    model.gradient_checkpointing_enable()
    print("Model loaded.")
except Exception as e:
    print("Error loading model:", e)
    raise

# -------------------------
# 5. Apply LoRA (PEFT)
# -------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
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
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# 6. Data collator (we already supply labels)
# -------------------------
# For causal LM where labels are precomputed, default_data_collator works well
data_collator = default_data_collator

# -------------------------
# 7. TrainingArguments tuned for p4d.24xlarge
# -------------------------
training_args = TrainingArguments(
    output_dir="lora-llama3.1-8b-empathetic",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    bf16=True,  # enable bf16 at training time (p4d with A100 supports bf16)
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    report_to=None,
    remove_unused_columns=False,
    logging_dir="./logs",
    # FSDP settings (use_orig_params required for LoRA / PEFT compatibility)
    fsdp="full_shard auto_wrap",
    fsdp_config={"min_num_params": int(1e8), "use_orig_params": True},
    ddp_find_unused_parameters=False,
)

# -------------------------
# 8. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# -------------------------
# 9. Train
# -------------------------
try:
    print("Starting training on", torch.cuda.device_count(), "GPUs")
    train_result = trainer.train()
    with open("training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    print("Training finished.")
except Exception as e:
    print("Error during training:", e)
    import traceback

    traceback.print_exc()
    try:
        trainer.save_model("partial-empathetic-lora-llama3.1")
        tokenizer.save_pretrained("partial-empathetic-lora-llama3.1")
        print("Saved partial model & tokenizer.")
    except Exception as se:
        print("Failed to save partial model:", se)
    raise

# -------------------------
# 10. Final evaluation + perplexity
# -------------------------
print("Running final evaluation...")
final_metrics = trainer.evaluate()
eval_loss = final_metrics.get("eval_loss", None)
if eval_loss is not None:
    try:
        perplexity = float(np.exp(eval_loss))
    except OverflowError:
        perplexity = float("inf")
    final_metrics["perplexity"] = perplexity
    print(f"Eval loss: {eval_loss:.4f}  Perplexity: {perplexity:.4f}")
else:
    print("eval_loss not returned by trainer.evaluate(); metrics:", final_metrics)

with open("final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

# -------------------------
# 11. Save final LoRA + tokenizer
# -------------------------
output_dir = "final-empathetic-lora-llama3.1-8b"
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
training_args.save_to_json(os.path.join(output_dir, "training_args.json"))
print(f"Saved final model & tokenizer to {output_dir}")

# -------------------------
# 12. Inference example (load base + PEFT weights correctly)
# -------------------------
try:
    # Load base model with device_map for inference (auto places modules across GPUs)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    inference_model = PeftModel.from_pretrained(base_model, output_dir, device_map="auto")
    inference_model.eval()

    test_prompt = (
        "<|system|>\nYou are a helpful, empathetic assistant.\n"
        "<|user|>\nI'm feeling really stressed about my work deadlines.\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(test_prompt, return_tensors="pt").to(inference_model.device)
    with torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Inference response:\n", resp)
except Exception as e:
    print("Inference example failed:", e)
    import traceback

    traceback.print_exc()
