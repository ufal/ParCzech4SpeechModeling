from pathlib import Path
import pickle
import hydra
import torchaudio
import whisperx
from whisperx.asr import load_model
from parczech.util import (
    set_segment_len, 
    get_model_inputs, 
    get_batched_inputs,
    find_undesired_symbol_tokens
)
from tqdm import tqdm
import pandas as pd


@hydra.main(config_path="../configs", config_name="whisper_infer")
def main(cfg):
    tmp = load_model("large-v2", device=cfg.device, language="cs")
    undesired_symbols = find_undesired_symbol_tokens(tmp.tokenizer) + [-1]
    results = []
    cnt = 0
    for suppress_numerals in [True, False]:

            
        model = load_model(
            cfg.model_name,
            device=cfg.device,
            language="cs",
            asr_options=dict(
                suppress_numerals=suppress_numerals,
                word_timestamps=True,
                without_timestamps=False,
                suppress_tokens=undesired_symbols if suppress_numerals else None
            )
        )
        model_a, metadata = whisperx.load_align_model(language_code="cs", device=cfg.device)
        for i, mp3_source in tqdm(enumerate(Path(cfg.mp3_source).rglob('*.mp3'))):
            if cfg.debug and cnt == 2:
                break
            mp3_source = str(mp3_source)            
            audio = whisperx.load_audio(mp3_source)
            result = model.transcribe(
                audio, 
                batch_size=cfg.batch_size,
            )
            aligned = whisperx.align(result["segments"], model_a, metadata, audio, cfg.device, return_char_alignments=False)
            print(aligned)
            results.append(dict(
                mp3_source=mp3_source,
                recognized=result,
                aligned=aligned["words"],
                suppress_tokens=suppress_numerals,

            ))
            cnt += 1

            # segment_len_sec = set_segment_len(27, 33, mp3_tensor.shape[1])
            # inputs = get_model_inputs(mp3_tensor, segment_len_sec, cfg.overlap_len_sec, sr)
            # batches = get_batched_inputs(inputs, cfg.batch_size, cfg.num_workers)
    
    pd.DataFrame(results).to_csv(Path(cfg.mp3_source, "recognized.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
