import pickle
from pathlib import Path

import hydra
import pandas as pd
from tqdm import tqdm


@hydra.main(config_path="../configs", config_name="create_vocabulary")
def main(cfg):
    vocab2files = {}
    vocab_counter = {}
    non_token_vocab_counter = {}
    files = list(Path(cfg.vertical_dir).rglob("*.tsv"))
    for f in tqdm(files, desc="Reading vertical files", total=len(files)):
        vert_df = pd.read_csv(
            f,
            sep="\t",
            names=cfg.vert_columns,
            header=None,
            quoting=3,
        )

        for i, row in vert_df.iterrows():
            w = row[cfg.token_column]
            is_token = row[cfg.is_token_word_column]

            if w not in vocab_counter:
                vocab_counter[w] = 0
            if w not in vocab2files:
                vocab2files[w] = set()

            vocab2files[w].add(f.as_posix())
            if is_token:
                vocab_counter[w] += 1
            else:
                non_token_vocab_counter[w] = non_token_vocab_counter.get(w, 0) + 1


    with open(cfg.vocab2files, "wb") as f:
        pickle.dump(vocab2files, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(cfg.non_token_vocab_counter, "wb") as f:
        pickle.dump(non_token_vocab_counter, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(cfg.vocab_counter, "wb") as f:
        pickle.dump(vocab_counter, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
