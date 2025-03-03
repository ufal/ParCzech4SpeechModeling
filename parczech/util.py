
# choose best segment len, so the last input is not too short
def set_segment_len(start_range, end_range, mp3_frames_cnt, sample_rate, overlap_len_sec):
    best_segment_len = start_range
    best_segment_len_reminder = 0
    if mp3_frames_cnt / sample_rate < start_range:
        return start_range

    for segment_len in range(start_range, end_range):
        total_offset = 0
        segments = []

        while True:
            if total_offset >= mp3_frames_cnt:
                break
            segments.append(min(mp3_frames_cnt, total_offset + segment_len * sample_rate) - total_offset)
            total_offset = min(mp3_frames_cnt, total_offset + (segment_len - overlap_len_sec) * sample_rate)

        segments = [s / sample_rate for s in segments]

        if min(segments) > best_segment_len_reminder:
            best_segment_len = segment_len
            best_segment_len_reminder = min(segments)

    return best_segment_len

def get_model_inputs(source_tensor, segment_len_sec, overlap_len_sec, sr):
    total_offset = 0
    samples = []

    while True:
        if total_offset >= source_tensor.shape[1]:
            break

        x = source_tensor[0, total_offset: min(source_tensor.shape[1], total_offset + segment_len_sec * sr)]
        total_offset = total_offset + min((segment_len_sec - overlap_len_sec) * sr, source_tensor.shape[1] - total_offset)
        samples.append(x)

    return samples

def get_batched_inputs(samples, batch_size):
    batches = []
    batch = []
    for i in range(0, len(samples)):
        batch.append(samples[i])
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return batches

def find_undesired_symbol_tokens(tokenizer, undesired_symbols='!"%&\'()*+,-./:;=?@\\|§¨°´·\u200e’…'):
    udesired_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_udesired_symbol = any(c in undesired_symbols for c in token)
        if has_udesired_symbol:
            udesired_symbol_tokens.append(i)
    return udesired_symbol_tokens