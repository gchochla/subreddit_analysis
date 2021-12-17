# Subreddit Analysis

This repo includes tools for Subreddit analysis, originally developed for our class project of PSYC 626 in USC, titled "Powered by the Will?: Themes in online discussions of Fitness".

## Installation and Requirements

You need to use Python 3.9, R 4.1.0 and git basically to run the scripts provided in this repo. For Ubuntu, to install essential dependencies:

```bash
sudo apt update
sudo apt install git python3.9 python3-pip
pip3 install virtualenv
```

Now clone this repo:

```bash
git clone https://github.com/gchochla/subreddit-analysis
cd subreddit-analysis
```

Create and activate a python environment to download the python requirements for the scripts:

```bash
~/.local/bin/virtualenv .venv
source .venv/bin/activate
pip install .
```

## Usage

1. Download a subreddit into a JSON that preserves the hierarchical structure of the posts by running:

```bash
python subreddit_analysis/subreddit_forest.py -r <SUBREDDIT_NAME>
```

where `<SUBREDDIT_NAME>` is the name of the subreddit after `r/`. You can also limit the number of submissions returned by setting `-l <LIMIT>`. The result can be found in the file `<SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift.json`.

2. Transform this JSON to a rectangle (CSV), you can use:

```bash
python subreddit_analysis/json_forest_to_csv.py -fn <SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift.json
```

which creates `<SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift.csv`.

3. To have a background corpus for control, you can download posts from the redditors that have posted in your desired subreddit from other subreddits:

```bash
python subreddit_analysis/user_baseline.py -fn <SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift.json -pl 200
```

where `-pl` specifies the number of posts per redditor to fetch (before filtering the desired subreddit). The file is saved as `<SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift-baseline-<pl>.json`

4. Transform that as well to a CSV:

```bash
python subreddit_analysis/json_baseline_to_csv.py -fn <SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift-baseline-<pl>.json
```

which creates `<SUBREDDIT_NAME>-<NUM_OF_POSTS>-pushshift-baseline-<pl>.csv`.

5. Create a folder, `<ROOT>`, move the subreddit CSV to it, and create another folder inside it named `dictionaries` that includes a file (note: the filename -- with a possible extensions -- will be used as the header of the loading) per distributed dictionary with space-separated words:

```
positive joy happy excited
```

6. Tokenize CSVs using the `r_scripts`.

7. Compute each post's loadings and write it into the CSV:

```bash
python subreddit_analysis/submission_loadings.py -d <ROOT> -doc <CSV_FILENAME>
```

where `<CSV_FILENAME>` is relative to `<ROOT>`.

8. If annotations are available, which should be in a CSV with (at least) a column for the labels themselves and the ID of the post with a `post_id` header, you can use these to design a data-driven distributed dictionary. You can first train an RNN to create another annotation file with a predicted label for each post with:

```bash
python subreddit_analysis/rnn.py --doc_filename <SUBREDDIT_CSV> --label_filename <ANNOTATION_CSV> --label_column <LABEL_HEADER_1> <LABEL_HEADER_2> ... <LABEL_HEADER_N> --out_filename <NEW_ANNOTATION_CSV>
```

where you can provide multiple labels for multitasking, thought the model provides predictions only for the first specified label for now. Finally, if annotations are ordinal, you can get learned coefficients from Ridge Regression for each word in the vocabulary of all posts (in descending order of importance) using a `tf-idf` model to represent each document using:

```bash
python subreddit_analysis/bow_model.py --doc_filename <SUBREDDIT_CSV> --label_filename <ANY_ANNOTATION_CSV> --label_column <LABEL_HEADER> --out_filename <IMPORTANCE_CSV>
```

9. Run analyses using `r_scripts`.
