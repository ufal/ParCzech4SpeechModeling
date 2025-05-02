from parczech.util import find_undesired_symbol_tokens, get_batch_size
from whisperx import align, load_align_model
from whisperx.asr import load_audio, load_model


class DefaultWhisperRecognizer:
    def __init__(self, name, device, suppress_numerals, model_name, num_workers):
        self.name = name
        self.device = device
        self.batch_size = get_batch_size()
        self.num_workers = num_workers
        unwanted_symbols = None
        if suppress_numerals:
            tmp = load_model(model_name, device=self.device, language="cs")
            unwanted_symbols = find_undesired_symbol_tokens(tmp.tokenizer) + [-1]

        self.model = load_model(
            model_name,
            device=device,
            language="cs",
            asr_options=dict(
                suppress_numerals=suppress_numerals,
                word_timestamps=True,
                without_timestamps=False,
                suppress_tokens=unwanted_symbols if suppress_numerals else None,
            ),
        )
        self.model_a, self.metadata = load_align_model(language_code="cs", device=device)


    def __call__(self, file_path):
        try:
            audio = load_audio(file_path.as_posix())
        except Exception as e:
            if "Failed to load audio" in str(e):
                print(f"Skipping corrupt/invalid audio file: {file_path}")
                return None
        segment_result = self.model.transcribe(audio, batch_size=self.batch_size, num_workers=self.num_workers)
        aligned = align(
            segment_result["segments"],
            self.model_a,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False
        )
        return aligned["word_segments"]
