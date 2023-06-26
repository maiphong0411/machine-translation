from transformers import (
    MBartForConditionalGeneration, MBartTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
)
from accelerate import Accelerator
import torch
from torch.utils.data import random_split
import datasets
import pandas as pd
import numpy as np
from datasets import load_metric
import gc
import torch
from data_process import prepare_dataset


CHECKPOINT = 'facebook/mbart-large-50-many-to-many-mmt'
PATH = '/kaggle/input/multidomain/'
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "vi"

def load_model(CHECKPOINT):
    model = MBartForConditionalGeneration.from_pretrained(CHECKPOINT)
    return model

def load_tokenizer(CHECKPOINT):
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    return tokenizer


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    # no need this line
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
def main():
    dataset = prepare_dataset(PATH)
    tokenizer = load_tokenizer(CHECKPOINT)
    model = load_model(CHECKPOINT)

    tokenized_train_set = dataset['train'].map(preprocess_function, batched=True)
    tokenized_test_set = dataset['test'].map(preprocess_function, batched=True)
    tokenized_val_set = dataset['val'].map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = load_metric("sacrebleu")
    accelerator = Accelerator()

    args = Seq2SeqTrainingArguments(
        output_dir="./mbart_EnglistToVietnamese/",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=2,
        predict_with_generate=True,
        logging_dir="/logs",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="none"
    )


    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_test_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    tokenized_datasets, tokenized_test_set, trainer = accelerator.prepare(
        tokenized_datasets, tokenized_test_set, trainer
    )

    trainer.train()
if __name__ == '__main__':
    main()