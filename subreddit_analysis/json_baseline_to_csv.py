"""Parses a JSON downloaded by `user_baseline.py` into a CSV."""

import argparse
import csv
import json
import os


# desired new columns and how to create them for each entry
# from each post's dictionary
CSV_MAPPING = {
    "post_id": lambda d: d["url"].split("/")[
        6 if len(d["url"].split("/")) == 8 else 4
    ],
    "parent_post_id": lambda d: d["url"].split("/")[4],
    "author": lambda d: d["author"],
    "subreddit": lambda d: d["subreddit"],
    "title": lambda d: d.get("title", "N/A"),
    "text": lambda d: d["text"],
    "n_comments": lambda d: d.get("n_comments", "N/A"),
    "score": lambda d: d["score"],
    "url": lambda d: d["url"],
    "timestamp": lambda d: d["timestamp"],
    "comment": lambda d: int(len(d["url"].split("/")) == 7),  # bool
}


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fn", "--filename", type=str, required=True, help="input JSON filename"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    json_fn = args.filename

    with open(json_fn) as fp:
        baseline = json.load(fp)

    rows = [list(CSV_MAPPING)]
    for user in baseline:
        for post in baseline[user]:
            rows.append([fun(post) for fun in CSV_MAPPING.values()])

    # save on same filename (with proper extension)
    csv_fn = os.path.splitext(json_fn)[0] + ".csv"

    with open(csv_fn, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)
