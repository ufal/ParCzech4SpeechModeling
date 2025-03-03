import whisperx
from whisperx.asr import load_model


def find_undesired_symbol_tokens(tokenizer, undesired_symbols='!"%&\'()*+,-./:;=?@\\|§¨°´·\u200e’…'):
    udesired_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_udesired_symbol = any(c in undesired_symbols for c in token)
        if has_udesired_symbol:
            udesired_symbol_tokens.append(i)
    return udesired_symbol_tokens


device = "cuda"
# audio_file = "/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/parczech-3.0-asr-speakers.dev/2020012312281242/31/2020012312281242.wav"
# audio_file = "/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/parczech-3.0-asr-other/2018012410381052/03/2018012410381052.wav"
# audio_file = "/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/parczech-3.0-asr-other/2019102309080922/01/2019102309080922.wav"
audio_file = "/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/parczech-3.0-asr-other/2018030711081122/01/2018030711081122.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

tmp = load_model("large-v2", device=device, language="cs")
# 1. Transcribe with original whisper (batched)
model = load_model(
    "large-v2", 
    device=device,
    language="cs", 
    asr_options=dict(
        suppress_numerals=True,
        word_timestamps=True,
        without_timestamps=False,
        suppress_tokens=find_undesired_symbol_tokens(tmp.tokenizer) + [-1]
    ))

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
# print(result["words"]) # after alignment