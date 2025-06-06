import math
import pickle
from pathlib import Path

import hydra
import pandas as pd
import torch
import torchaudio
import transformers
from torch.amp import autocast
from tqdm import tqdm


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, target_sr, processor):
        self.df = pd.read_parquet(df_path).reset_index(drop=True)
        self.target_sr = target_sr
        self.processor = processor

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["audio_path"]
        waveform, sr = torchaudio.load(path)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.mean(dim=0)  # convert to mono
        return waveform.numpy()


    def collate_fn(self, items):
        return self.processor(
            items,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding="longest",
        )

    def __len__(self):
        return len(self.df)


class Wav2VecInfer:
    def __init__(self, device):
        self.device = device
        self.processor = transformers.AutoProcessor.from_pretrained("arampacha/wav2vec2-large-xlsr-czech")
        self.model = transformers.Wav2Vec2ForCTC.from_pretrained("arampacha/wav2vec2-large-xlsr-czech").eval().to(device)

    def __call__(self, batch):
        with autocast(device_type=self.device):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)

class WhisperWithNumbersInfer:
    def __init__(self, device):
        self.device = device
        self.processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = transformers.WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").eval().to(device)

    def __call__(self, batch):
        with autocast(device_type=self.device):
            batch = batch.to(self.device)
            with torch.no_grad():
                predict_ids = self.model.generate(**batch, language="czech")
        return self.processor.batch_decode(
            predict_ids,
            skip_special_tokens=True,
        )


@hydra.main(config_path="../configs", config_name="infer_asr")
def main(cfg):
    model = hydra.utils.instantiate(cfg.model)
    output_dir = Path(cfg.output_dir, model.__class__.__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df_path = Path(output_dir, Path(cfg.df_path).name)

    if Path(output_df_path).exists():
        print(f"Output file {output_df_path} already exists. Done.")
        return

    dataset = AudioDataset(cfg.df_path, 16_000, model.processor)

    available_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    scaler = available_mem / 24

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=math.floor(cfg.batch_size * scaler),
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    predictions = []

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        predictions.extend(model(batch))

    dataset.df[model.__class__.__name__] = predictions
    dataset.df.to_parquet(output_df_path, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
