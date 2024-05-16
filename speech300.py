from datasets import Dataset, DatasetDict, load_dataset, Audio
from transformers import AutoProcessor, AutoFeatureExtractor, AutoModelForCTC, TrainingArguments, Trainer
import json
import os
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
import numpy as np

jsonl_path = "data/asr.jsonl"
audio_dir = "data/audio"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():
    # Load the JSONL data
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    # print(data[0])

    # Create a dataset from the list of dictionaries
    dataset = Dataset.from_dict({
        "audio": [os.path.join(audio_dir, item["audio"]) for item in data],
        "transcript": [item["transcript"].upper() for item in data]
    })
    # print(dataset[0])

    # Cast the 'audio' column to Audio type
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    print(dataset[0])

    return dataset

def preprocess_function(row):
    audio_arrays = [x["array"] for x in row["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=200000, # find out the longest length of audio
        truncation=True,
    )
    return inputs

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcript"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

dataset = load_dataset()

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

dataset = dataset.train_test_split(test_size=0.1)
processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["test"])

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

wer = evaluate.load("wer")

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

model = model.to(device)

training_args = TrainingArguments(
    output_dir="winner_asr_mind_model",
    per_device_train_batch_size=20,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=20,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()