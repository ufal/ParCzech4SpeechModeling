from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class BatchPreparer:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.processor.tokenizer(batch["sentence"]).input_ids
        return batch


class MetricComputer:
    def __init__(self, processor):
        self.processor = processor
        self.metric = evaluate.load("wer")

    def __call__(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


def get_dataset(dataset_name, sr, processor, num_proc):
    if dataset_name == "mozilla-foundation/common_voice_11_0":
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(dataset_name, "cs", split="train+validation", trust_remote_code=True)
        common_voice["test"] = load_dataset(dataset_name, "cs", split="test", trust_remote_code=True)
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sr))
        batch_preparer = BatchPreparer(processor)
        common_voice = common_voice.map(
            batch_preparer,
            remove_columns=common_voice.column_names["train"],
            num_proc=num_proc
        )

        return common_voice
    raise ValueError(f"Unknown dataset: {dataset_name}")
