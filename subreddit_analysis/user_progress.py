"""Track progression of posts of individual redditors."""

import json
import os
import argparse


def parse_args():
    """Parses args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fn", "--filename", type=str, required=True, help="input JSON filename"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.filename) as fp:
        subreddit_data = json.load(fp)

    user_data = {
        user: []
        for user in set([val["author"] for val in subreddit_data.values()])
    }
    for submission_id, submission_info in subreddit_data.items():
        submission_info["id"] = submission_id
        user_data[submission_info["author"]].append(submission_info)

    new_fn = os.path.splitext(args.filename)[0] + "-users.json"

    with open(new_fn, "w") as fp:
        json.dump(user_data, fp)
