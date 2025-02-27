from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset, load_metric
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
import os
import numpy as np
from datetime import datetime

# Initialize TensorBoard
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/{timestamp}"
writer = SummaryWriter(log_dir)

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset with padding and truncation
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length", 
        max_length=128,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load the pre-trained BERT model for token classification
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)

# Enhanced Training Arguments optimized for GPU
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Increased for GPU
    per_device_eval_batch_size=16,   # Increased for GPU
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id="bniladridas/token-classification-ai-fine-tune",
    logging_dir=log_dir,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    warmup_steps=500,
    gradient_accumulation_steps=4,  # Added for GPU optimization
    report_to="tensorboard",
    fp16=True  # Enable mixed precision training
)

# Define the Trainer with TensorBoard callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[TensorBoardCallback(writer)]
)

# Train the model with progress monitoring
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# Save model locally
model_save_path = "./ner_model"
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved locally at {model_save_path}")

# Create inference pipeline
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)

# Example batch processing function
def process_batch(texts):
    return ner_pipeline(texts, batch_size=16)

# Push the model to Hugging Face Model Hub
trainer.push_to_hub()

# Close TensorBoard writer
writer.close()
print("Training complete!")
