from functools import lru_cache
from pathlib import Path

import pandas as pd
from Bio import pairwise2
from Levenshtein import distance


@lru_cache(maxsize=None)
def normalized_levenshtein_similarity(s1, s2):
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1  # Both strings are empty, so they are identical
    edit_distance = distance(s1, s2)
    return 1 - (edit_distance / max_len)


def strip_punctuations(word: str, punctuations: set[str]) -> str:
    """
    Strips punctuations from the word based on the provided set of punctuations.
    """
    if word[0] in punctuations and word[-1] in punctuations and len(word) > 2:
        return word[1:-1]
    elif word[-1] in punctuations and len(word) > 1:
        return word[:-1]
    elif word[0] in punctuations and len(word) > 1:
        return word[1:]
    else:
        return word

def align(
    vert_path: str | None = None,
    recognized_path: str | None = None,
    vert_columns: list[str] | None = None,
    gap_char: str | None = None,
    punctuations: set[str] | None = None,
    default_edit_distance: int | None = None
):
    vert_df = pd.read_csv(
        vert_path,
        sep="\t",
        names=vert_columns,
        header=None,
        quoting=3,
    ).dropna(subset=["token_str"])
    
    try:
        rec_df = pd.read_csv(recognized_path, sep="\t").dropna(subset=["word"])
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty file encountered at {recognized_path}")
        return None

    alignment = pairwise2.align.globalcx(
        vert_df["token_str"].str.lower().tolist(),
        rec_df["word"].str.lower().tolist(),
        normalized_levenshtein_similarity,
        gap_char=[gap_char],
    )[0]

    aligned_rows = []
    vert_idx = 0
    rec_idx = 0

    for i, (true_word, rec_word) in enumerate(zip(alignment.seqA, alignment.seqB)):
        common_row = {"token_str": gap_char, "word": gap_char, "vert_file": Path(vert_path).name}

        if true_word != gap_char:
            common_row.update(vert_df.iloc[vert_idx].to_dict())
            vert_idx += 1
        if rec_word != gap_char:
            rec_idx += 1
            common_row.update({
                k: v for k, v in rec_df.iloc[rec_idx-1].to_dict().items()
            })
        if true_word != gap_char and rec_word != gap_char:
            if true_word in punctuations:
                common_row["edit_distance"] = 0
            else:
                common_row["edit_distance"] = distance(true_word, strip_punctuations(rec_word, punctuations))
        elif true_word == gap_char and rec_word != gap_char:
            common_row["edit_distance"] = len(rec_word)
        elif true_word != gap_char and rec_word == gap_char:
            if true_word in punctuations:
                common_row["edit_distance"] = 0
            else:
                common_row["edit_distance"] = len(true_word)
        else:
            common_row["edit_distance"] = default_edit_distance
        aligned_rows.append(common_row)
    return pd.DataFrame(aligned_rows)
