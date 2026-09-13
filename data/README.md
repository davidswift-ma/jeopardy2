# Dataset directory

Intentionally empty in git. The Jeopardy clue data is **not redistributed
here**: the source dataset asks that it not be used in a public-facing site,
app, or product, and this repository is public.

To populate it:

1. Download the dataset from
   <https://github.com/jwolle1/jeopardy_clue_dataset/releases>
2. Either point `DATASET_PATH` in `.env` at the full
   `combined_season1-42.tsv`, or generate a small local sample:

   ```bash
   make sample     # writes data/jeopardy_sample.tsv from your download
   ```

`*.tsv` here is gitignored, so a generated sample stays local.

The agent's general question-answering does not need this data at all --
it is for the dataset-backed phase.
