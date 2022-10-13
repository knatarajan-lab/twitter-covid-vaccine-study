# coding=utf-8
# Python imports
# Third-party imports
import numpy as np
import torch
from datasets import Dataset, load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler)

# Project imports
from common_utils import load_covid_df
from resources.constants import data_path

MODEL = "distilbert-base-uncased"
CUDA = True


def train_model(device, model, model_path, train_dataloader):
    learning_rate = 5e-5
    num_epochs = 5

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_batches = len(train_dataloader)
    num_training_steps = num_epochs * num_training_batches
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Move model to device
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update()

    model.save_pretrained(model_path)


def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenized_batch = tokenizer(batch['text'], padding=True, truncation=True, max_length=350)
    return tokenized_batch


def predict_labels(device, model, dataloader, predict_path):
    model.eval()
    # predictions = torch.empty((0), dtype=torch.int64).to(device)
    # ids = torch.empty((0), dtype=torch.int64).to(device)
    # id_predictions = torch.empty((0, 2), dtype=torch.int64).to(device)
    with predict_path.open('a+') as f:
        for batch in dataloader:
            ids = batch.pop('id')
            batch = {k: v.to(device) for k, v in batch.items()}
            # ids = torch.cat((ids, batch.pop('id')), 0)
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            # predictions = torch.cat((predictions, torch.argmax(logits, dim=-1)), 0)
            predictions = torch.argmax(logits, dim=-1)
            ids = ids.to(device)
            # id_predictions = torch.cat((id_predictions, torch.stack((ids, predictions), 1)), 0)
            id_predictions = torch.stack((ids, predictions), 1)
            id_pred_array = id_predictions.cpu().numpy()
            np.savetxt(f, id_pred_array, fmt='%s')
    return


def calculate_metrics(metric, batch_metrics):
    weighted = metric.compute(predictions=batch_metrics["pred"],
                              references=batch_metrics["ref"],
                              average="weighted")
    macro = metric.compute(predictions=batch_metrics["pred"],
                           references=batch_metrics["ref"],
                           average="macro")
    micro = metric.compute(predictions=batch_metrics["pred"],
                           references=batch_metrics["ref"],
                           average="micro")
    return weighted, macro, micro


def evaluate_model(device, model, eval_dataloader):
    f1_metric = load_metric("f1")
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    batch_metrics = {'pred': torch.empty((0), dtype=torch.int64).to(device),
                     'ref': torch.empty((0), dtype=torch.int64).to(device)}

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        batch_metrics['pred'] = torch.cat((batch_metrics['pred'], predictions), 0)
        batch_metrics['ref'] = torch.cat((batch_metrics['ref'], batch["labels"]), 0)
        # metric.add_batch(predictions=predictions, references=batch["labels"])

    weighted_f1, macro_f1, micro_f1 = calculate_metrics(f1_metric, batch_metrics)
    weighted_p, macro_p, micro_p = calculate_metrics(precision_metric, batch_metrics)
    weighted_r, macro_r, micro_r = calculate_metrics(recall_metric, batch_metrics)

    results = '\n'.join([f'f1: weighted: {weighted_f1}, macro: {macro_f1}, micro: {micro_f1}',
                         f'precision: weighted: {weighted_p}, macro: {macro_p}, micro: {micro_p}',
                         f'recall: weighted: {weighted_r}, macro: {macro_r}, micro: {micro_r}'])
    return results


def prepare_dataset_train(df):
    dataset = Dataset.from_pandas(df).train_test_split(train_size=0.7,
                                                       seed=123)

    dataset = dataset.class_encode_column("label")
    keep_cols = ["label"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cols_to_remove = [col for col in dataset["train"].column_names if col not in keep_cols]

    dataset_enc = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove, num_proc=8)

    dataset_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset_enc["train"], shuffle=True, batch_size=64, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        dataset_enc["test"], batch_size=64, collate_fn=data_collator
    )
    return dataset, train_dataloader, eval_dataloader


def prepare_dataset_validate(df):
    dataset = Dataset.from_pandas(df)

    dataset = dataset.class_encode_column("label")
    keep_cols = ["label"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cols_to_remove = [col for col in dataset.column_names if col not in keep_cols]

    dataset_enc = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove, num_proc=8)

    dataset_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    val_dataloader = DataLoader(
        dataset_enc, shuffle=True, batch_size=64, collate_fn=data_collator
    )
    return dataset, val_dataloader


def prepare_dataset_infer(df):
    dataset = Dataset.from_pandas(df)

    keep_cols = ["id"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cols_to_remove = [col for col in dataset.column_names if col not in keep_cols]

    dataset_enc = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove, num_proc=16)

    dataset_enc.set_format('torch', columns=['id', 'input_ids', 'attention_mask'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset_enc, shuffle=True, batch_size=600, collate_fn=data_collator
    )
    return dataset, dataloader


def main():
    df = load_covid_df('train')
    dataset, train_dataloader, eval_dataloader = prepare_dataset_train(df)

    num_labels = dataset["train"].features["label"].num_classes
    print(f"Number of labels: {num_labels}")

    device = torch.device("cuda:0" if CUDA else "cpu")

    # Infer
    df = load_covid_df('predict')
    dataset, dataloader = prepare_dataset_infer(df)

    monitor_path = data_path / 'predicted' / 'run_monitor.txt'
    with monitor_path.open('a+') as f:
        for model_name in [f"{MODEL}-tweets", f"{MODEL}-tweets-v1", f"{MODEL}-tweets-v2", f"{MODEL}-tweets-v3"]:
            # Load model from checkpoint
            f.write(f'Running model {model_name}')
            model_path = data_path / 'models' / model_name
            # model = AutoModelForSequenceClassification.from_pretrained(MODEL,
            #                                                            num_labels=num_labels)
            # TODO use config
            model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                       num_labels=num_labels)

            model = model.to(device)

            # Train
            # train_model(device, model, model_path, train_dataloader)

            # Evaluate
            results = evaluate_model(device, model, eval_dataloader)
            f.write(results)

            # Validate
            f.write('Do not trust below since already trained on this set')
            df = load_covid_df('validate')
            dataset, val_dataloader = prepare_dataset_validate(df)
            results = evaluate_model(device, model, val_dataloader)
            f.write(results)

            predict_path = data_path / f"predicted" / f"{model_name.replace('tweets', 'labels')}.csv"
            predict_labels(device, model, dataloader, predict_path)


if __name__ == '__main__':
    main()
