import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import jiwer
import pandas as pd
from Levenshtein import distance
from tqdm import tqdm

jiwer_transform = jiwer.Compose([
    jiwer.RemoveEmptyStrings(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


def contains_num(token):
    return any(char.isdigit() for char in token)


def normalize_segment(segment, punc_list):
    norm_segment = []
    for token in segment.strip().lower():
        if token not in punc_list:
            norm_segment.append(token)
    return " ".join(norm_segment)


@dataclass
class Segmenter(ABC):
    """
    Abstract class for to create segments and compute their statistics.
    """
    gap_word: str
    vert_col: str
    token_col: str
    token_id_col: str
    recognized_col: str
    start_col: str
    end_col: str
    speakers_col: str
    edit_distance_col: str
    no_ws_after_col: str
    score_col: str | None
    punc_list: list[str]
    transform=None

    def __post_init__(self):
        if self.transform is None:
            self.transform = jiwer_transform

    def get_wer(self, true_text, recognized_text):
        try:
            wer = jiwer.wer(
                true_text,
                recognized_text,
                truth_transform=self.transform,
                hypothesis_transform=self.transform
            )
        except ValueError:
            print(f"Error calculating WER for segment: {true_text} vs {recognized_text}")
            wer = None
        return wer

    def get_true_text(self, df):
        true_text = ""
        for i, row in df.iterrows():
            if row[self.token_col] != self.gap_word:
                true_text += row[self.token_col]
                if not row[self.no_ws_after_col]:
                    true_text += " "
        return true_text.strip()

    @abstractmethod
    def get_segment_starts_ends(self, df) -> list[dict[str, int]]:
        """Starts and ends are defined as indexes of the first and last rows of the segment."""
        pass

    @abstractmethod
    def extract_segments(self, file) -> pd.DataFrame:
        """For the input dataframe create a new dataframe with segments and their statistics."""
        pass


@dataclass
class SenteceSegmenter(Segmenter):
    """
    This class defines segment as a sentence.
    """
    segment_id_col: str = "segment_id"

    def get_segment_starts_ends(self, df):
        segments = []

        for vert in df[self.vert_col].unique():
            subset_start, subset_end = df[df[self.vert_col] == vert].iloc[[0, -1]].index
            true_subset_start = -1
            for i, row in df.iloc[subset_start:subset_end].iterrows():
                if row[self.segment_id_col] is not None:
                    true_subset_start = i
                    break
            segment = {"start_id": true_subset_start}
            prev_segment_id = df.iloc[true_subset_start][self.segment_id_col]
            for i, row in df.iloc[true_subset_start: subset_end].iterrows():
                if row[self.segment_id_col] is not None:
                    if row[self.segment_id_col] != prev_segment_id:
                        segment["end_id"] = i
                        segments.append(segment)
                        segment = {"start_id": i}
                    prev_segment_id = row[self.segment_id_col]

        return segments

    def _find_segment_end(self, segment_df):
        correct_end = False
        end = None
        true_word_id = -1
        for i, row in segment_df.iloc[::-1].iterrows():
            if (
                row[self.token_col] != self.gap_word
                and row[self.token_col] not in self.punc_list
                and true_word_id == -1
            ):
                true_word_id = i

            if (
                row[self.recognized_col] != self.gap_word
                and row[self.end_col] is not None
                and not math.isnan(row[self.end_col])
            ):
                end = row[self.end_col]
                correct_end = i == true_word_id
                break
        return end, correct_end

    def extract_segments(self, file):
        file = Path(file)
        df = pd.read_parquet(file)
        df[self.segment_id_col] = None
        df.loc[~df[self.token_id_col].isna(), self.segment_id_col] = (
            df.loc[~df[self.token_id_col].isna(), self.vert_col]
            + df.loc[~df[self.token_id_col].isna(), self.token_id_col].str.split(".").str[:-1].str.join(".")
        )
        segment_indices = self.get_segment_starts_ends(df)
        resulting_segments = []

        for j, seg_indices in tqdm(enumerate(segment_indices), desc=f"Processing {file.stem}", total=len(segment_indices)):
            segment_df = df.iloc[seg_indices["start_id"]:seg_indices["end_id"]]
            vert_files = segment_df.dropna(subset=self.vert_col)[self.vert_col].unique().tolist()
            if len(vert_files) > 1:
                print(f"File: {file}, segment {j}: {seg_indices} has more than one vert file: {vert_files}. Skipping.")
                continue
            true_text = ""
            recognized_words = []

            correct_start, correct_end = False, False
            start, end = None, None
            n_true_words, n_true_chars = 0, 0
            n_rec_words, n_rec_chars = 0, 0
            n_missed_true_words, n_missed_rec_words = 0, 0
            n_numbers = 0
            for i, row in segment_df.iterrows():
                if row[self.token_col] != self.gap_word:
                    true_text += row[self.token_col]
                    if not row[self.no_ws_after_col]:
                        true_text += " "

                    if row[self.recognized_col] == self.gap_word:
                        n_missed_true_words += 1
                    n_true_words += row[self.token_col] not in self.punc_list
                    n_true_chars += len(row[self.token_col]) * (row[self.token_col] not in self.punc_list)
                    n_numbers += 1 if contains_num(row[self.token_col]) else 0

                if row[self.recognized_col] != self.gap_word:
                    recognized_words.append(row[self.recognized_col])
                    if row[self.token_col] == self.gap_word:
                        n_missed_rec_words += 1

                    if (
                        start is None
                        and row[self.start_col] is not None
                        and not math.isnan(row[self.start_col])
                    ):
                        start = row[self.start_col]
                        correct_start = n_true_words == 1

                    n_rec_words += row[self.recognized_col] not in self.punc_list
                    n_rec_chars += len(row[self.recognized_col]) * (row[self.recognized_col] not in self.punc_list)

            if recognized_words:
                end, correct_end = self.find_segment_end(segment_df)

            recognized_text = " ".join(recognized_words)

            segment_dict = dict(
                vert=vert_files[0],
                seg_id=j,
                seg_start=seg_indices["start_id"],
                seg_end=seg_indices["end_id"],
                start_token_id=segment_df.dropna(subset=[self.token_id_col]).iloc[0][self.token_id_col],
                end_token_id=segment_df.dropna(subset=[self.token_id_col]).iloc[-1][self.token_id_col],
                true_text=true_text.strip(),
                is_correct_start=correct_start,
                is_correct_end=correct_end,
                speakers=",".join(segment_df.dropna(subset=[self.speakers_col])[self.speakers_col].unique()),
                n_speakers=len(segment_df.dropna(subset=[self.speakers_col])[self.speakers_col].unique().tolist()),
                n_true_words=n_true_words,
                n_true_chars=n_true_chars,
                n_rec_words=n_rec_words,
                n_rec_chars=n_rec_chars,
                n_numbers=n_numbers,
                n_missed_true_words=n_missed_true_words,
                n_missed_rec_words=n_missed_rec_words,
            )

            if recognized_words:
                scores = segment_df[segment_df[self.recognized_col] != self.gap_word][self.score_col].tolist()
                aligened_edit_distances = (
                    segment_df[
                        (segment_df[self.token_col] != self.gap_word) 
                        & (segment_df[self.recognized_col] != self.gap_word)
                    ][self.edit_distance_col].tolist()
                )

                segment_dict["start"] = start
                segment_dict["end"] = end
                segment_dict["dur"] = end - start if end is not None and start is not None else None
                segment_dict["rec_text"] = recognized_text.strip()
                segment_dict["min_score"] = min(scores)
                segment_dict["max_score"] = max(scores)
                segment_dict["avg_score"] = sum(scores) / len(scores)
                segment_dict["sum_score"] = sum(scores)
                if aligened_edit_distances:
                    segment_dict["seg_edit_dist"] = distance(
                        normalize_segment(recognized_text, self.punc_list),
                        normalize_segment(true_text, self.punc_list)
                    )
                    segment_dict["align_edit_dist_sum"] = sum(aligened_edit_distances)
                    segment_dict["align_edit_dist_avg"] = sum(aligened_edit_distances) / len(aligened_edit_distances)
                    segment_dict["align_edit_dist_max"] = max(aligened_edit_distances)
                    segment_dict["seg_wer"] = self.get_wer(true_text, recognized_text)

            resulting_segments.append(segment_dict)

        return pd.DataFrame(resulting_segments)

@dataclass
class ContinuousSegmenter(Segmenter):
    """
    This class defines segment as a continuous sequence of words. Meaning that
    semgnet can span across sentence boundaries, or be inside a sentence.
    """
    min_word_count: int
    is_token_word_col: str
    avg_true_char_dur_lb: float
    avg_true_char_dur_ub: float
    norm_edit_dist_ub: float
    special_symbols: list[str]
    allowed_ending_punct: list[str]
    contains_num_col: str = "contains_num"
    avg_true_char_dur_col: str = "true_char_avg_dur"
    true_norm_edit_dist_col: str = "norm_edit_dist"
    dur_col: str = "dur"
    seed: int = 42
    max_segment_duration: float = 30

    def is_continuous(self, prev_row, cur_row, prev_id, cur_id):
        if prev_row[self.speakers_col] != cur_row[self.speakers_col]:
            return False
        if prev_id + 1 == cur_id:
            return True
        return False

    def get_correct_start_end(self, df, start_id, end_id):
        """
        Sometimes the first or last word does not have a timestamp.
        """
        correct_start_idx = end_id
        for i, row in df.loc[start_id: end_id].iterrows():
            if not math.isnan(row[self.start_col]) and row[self.is_token_word_col]:
                correct_start_idx = i
                break
        if correct_start_idx != end_id:
            correct_end_idx = correct_start_idx
            for i, row in df.loc[end_id: correct_start_idx: -1].iterrows():
                # i am puct but previous word is has a timestamp
                if (
                    i - 1 in df.index 
                    and not row[self.is_token_word_col]
                    and row[self.token_col] in self.allowed_ending_punct
                    and df.loc[i - 1, self.is_token_word_col]
                    and not math.isnan(df.loc[i - 1, self.end_col])
                ):
                    correct_end_idx = i
                    break
                # i am token and have a timestamp
                if row[self.is_token_word_col] and not math.isnan(row[self.end_col]):
                    correct_end_idx = i
                    break
            return correct_start_idx, correct_end_idx

        return end_id, end_id

    def get_segment_starts_ends(self, df):
        segment_ids = []
        verticals = df[self.vert_col].unique().tolist()

        for vert in tqdm(verticals, desc="Finding start and end indices"):
            subset_df = df[df[self.vert_col] == vert]
            segment_start = subset_df.index[0]

            prev_row = subset_df.loc[segment_start]
            prev_idx = segment_start

            for i, row in subset_df.iterrows():
                if segment_start < i:
                    if not self.is_continuous(prev_row, row, prev_idx, i):
                        true_start_idx, true_end_idx = self.get_correct_start_end(subset_df, segment_start, prev_idx)
                        if true_end_idx - true_start_idx > 0:
                            segment_ids.append(dict(
                                start_id=true_start_idx,
                                # Critical: in this segmenter we do NOT do prev_idx + 1
                                # since we use loc instead of iloc, which includes both ends
                                end_id=true_end_idx,
                            ))
                        segment_start = i
                prev_row = row
                prev_idx = i

        return segment_ids

    def get_segment_duration(self, df, start_id, end_id):
        subset = df.loc[start_id: end_id]
        start_time = subset[self.start_col].min()
        end_time = subset[self.end_col].max()
        return end_time - start_time

    def split_one_too_long_segment(self, df, start, end, depth=0):
        """
        The function tries to split the segment into two segments.
        If it fails it returns the original segment.
        """
        n_words = end - start
        # to avoid too short segments, each should have at least
        # 2 * self.min_word_count words from both sides
        offset = 2 * self.min_word_count

        n_words = n_words - 2 * offset
        if n_words < 2:
            return [dict(
                start_id=start,
                end_id=end,
            )]

        indices = list(range(1, n_words))
        random.shuffle(indices)

        # now we can create two segments
        # the first one will be from start to split_point
        # the second one will be from split_point to end
        # we need to check if the next word is a token word, otherwise
        # the second segment will start with some punctuation or number
        split_point = indices[0]

        for idx in indices:
            split_point = idx
            if not math.isnan(df.loc[start + split_point + offset][self.start_col]) and df.loc[start + split_point + offset][self.is_token_word_col]:
                break

        if not math.isnan(df.loc[start + split_point + offset][self.start_col]) and df.loc[start + split_point + offset][self.is_token_word_col]:
            result = []

            if self.get_segment_duration(df, start, start + split_point + offset - 1) > self.max_segment_duration:
                # if the segment is too long, we need to split it again
                new_segments = self.split_one_too_long_segment(df, start, start + split_point + offset - 1, depth + 1)
                result.extend(new_segments)
            else:
                result.append(dict(
                    start_id=start,
                    end_id=start + split_point + offset - 1,
                ))

            if self.get_segment_duration(df, start + split_point + offset, end) > self.max_segment_duration:
                # if the segment is too long, we need to split it again
                new_segments = self.split_one_too_long_segment(df, start + split_point + offset, end, depth + 1)
                result.extend(new_segments)
            else:
                result.append(dict(
                    start_id=start + split_point + offset,
                    end_id=end,
                ))

            return result

        return [dict(
            start_id=start,
            end_id=end,
        )]

    def split_too_long_segments(self, df, segment_ids):
        random.seed(self.seed)
        resulting_segments = []
        for s in tqdm(segment_ids, desc="Splitting too long segments"):
            start, end = s["start_id"], s["end_id"]
            duration = self.get_segment_duration(df, start, end)

            if duration < self.max_segment_duration:
                resulting_segments.append(s)
                continue
            divided_segments = self.split_one_too_long_segment(df, start, end)
            resulting_segments.extend(divided_segments)

        return resulting_segments

    def extract_segments(self, file):
        file = Path(file)
        df = pd.read_parquet(file)
        df[self.contains_num_col] = df[self.token_col].apply(contains_num)

        df.loc[~df[self.recognized_col].isna(), self.dur_col] = (
            df.loc[~df[self.recognized_col].isna(), self.end_col] 
            - df.loc[~df[self.recognized_col].isna(), self.start_col]
        )

        df.loc[~df[self.recognized_col].isna(), self.avg_true_char_dur_col] = (
            df.loc[~df[self.recognized_col].isna(), self.dur_col]
            / df.loc[~df[self.recognized_col].isna(), self.token_col].str.len()
        )

        df.loc[(df[self.token_col] != self.gap_word) & (df[self.recognized_col] != self.gap_word), self.true_norm_edit_dist_col] = (
            df.loc[(df[self.recognized_col] != self.gap_word) & (df[self.recognized_col] != self.gap_word), self.edit_distance_col]
            / df.loc[(df[self.recognized_col] != self.gap_word) & (df[self.recognized_col] != self.gap_word), self.token_col].str.len()
        )

        # here we do not reset index since we want to keep the original index and use it later
        clean_df = df[
            (df[self.token_col] != self.gap_word)
            & (
                (df[self.recognized_col] != self.gap_word) 
                | (df[self.is_token_word_col].notna() & ~df[self.is_token_word_col].fillna(False))
            ) & (
                (df[self.avg_true_char_dur_col] >= self.avg_true_char_dur_lb) 
                | (df[self.is_token_word_col].notna() & ~df[self.is_token_word_col].fillna(False)) | df[self.contains_num_col]
            ) & (
                (df[self.avg_true_char_dur_col] <= self.avg_true_char_dur_ub) 
                | (df[self.is_token_word_col].notna() & ~df[self.is_token_word_col].fillna(False)) | df[self.contains_num_col]
            )
            & (
                df[self.is_token_word_col].notna()
                & (
                    ((df[self.true_norm_edit_dist_col] <= self.norm_edit_dist_ub) & df[self.is_token_word_col])
                    | (~df[self.is_token_word_col].fillna(False) & df[self.true_norm_edit_dist_col].isna())
                )
            )
        ].copy(deep=True)

        seg_indices = self.get_segment_starts_ends(clean_df)

        long_enough_segments = []
        for s in seg_indices:
            subset = clean_df.loc[s["start_id"]: s["end_id"]]
            if len(subset[subset[self.is_token_word_col]]) >= self.min_word_count:
                long_enough_segments.append(s)

        seg_indices = self.split_too_long_segments(clean_df, long_enough_segments)

        segments_with_stats = []
        for j, s in tqdm(enumerate(seg_indices), desc="Calculating segment statistics", total=len(seg_indices)):
            start, end = s["start_id"], s["end_id"]
            segment_df = df.loc[start: end]
            vert_files = segment_df.dropna(subset=self.vert_col)[self.vert_col].unique().tolist()
            if len(vert_files) > 1:
                print(f"File: {file}, segment {j}: {seg_indices} has more than one vert file: {vert_files}. Skipping.")
                continue

            true_words = segment_df[segment_df[self.is_token_word_col]][self.token_col].tolist()
            recognized_words = segment_df[segment_df[self.recognized_col] != self.gap_word][self.recognized_col].tolist()
            scores = segment_df[segment_df[self.recognized_col] != self.gap_word][self.score_col].tolist()
            aligened_edit_distances = segment_df[
                (segment_df[self.token_col] != self.gap_word)
                & (segment_df[self.recognized_col] != self.gap_word)
            ][self.edit_distance_col].tolist()

            true_text = self.get_true_text(segment_df).strip()
            recognized_text = " ".join(recognized_words).strip()

            start_time = segment_df[segment_df[self.recognized_col] != self.gap_word][self.start_col].min()
            end_time = segment_df[segment_df[self.recognized_col] != self.gap_word][self.end_col].max()

            start_time_oldrich = segment_df[segment_df[self.recognized_col] != self.gap_word]["start_time"].min()
            end_time_oldrich = segment_df[segment_df[self.recognized_col] != self.gap_word]["start_time"].max()

            segment_dict = dict(
                vert=vert_files[0],
                seg_id=j,
                seg_start=start,
                seg_end=end,
                start=start_time,
                end=end_time,
                start_oldrich=start_time_oldrich,
                end_oldrich=end_time_oldrich,
                dur=end_time - start_time,
                start_token_id=segment_df.dropna(subset=[self.token_id_col]).iloc[0][self.token_id_col],
                end_token_id=segment_df.dropna(subset=[self.token_id_col]).iloc[-1][self.token_id_col],
                true_text=true_text,
                rec_text=recognized_text,
                speakers=",".join(segment_df.dropna(subset=[self.speakers_col])[self.speakers_col].unique()),
                n_true_words=len(true_words),
                n_true_chars=sum(len(w) for w in true_words),
                n_rec_words=len(recognized_words),
                n_rec_chars=sum(len(w) for w in recognized_words),
                n_numbers=len([w for w in true_words if contains_num(w)]),
                n_spec_symbols=len([w for w in true_words if w in self.special_symbols]),
                seg_edit_dist=distance(
                    normalize_segment(recognized_text, self.punc_list),
                    normalize_segment(true_text, self.punc_list)
                ),
                align_edit_dist_sum=sum(aligened_edit_distances) if aligened_edit_distances else None,
                align_edit_dist_avg=sum(aligened_edit_distances) / len(aligened_edit_distances) if aligened_edit_distances else None,
                align_edit_dist_max=max(aligened_edit_distances) if aligened_edit_distances else None,
                min_score=min(scores) if scores else None,
                max_score=max(scores) if scores else None,
                avg_score=sum(scores) / len(scores) if scores else None,
                sum_score=sum(scores) if scores else None,
                wer=self.get_wer(true_text, recognized_text),
            )
            segments_with_stats.append(segment_dict)

        return pd.DataFrame(segments_with_stats)

