"""Download a background corpus from the users of a particular subreddit."""

import praw
import json
import argparse
import os

from subreddit_analysis.subreddit_forest_praw import (
    parse_submission,
    parse_comment,
)


def parse_args():
    """Parses args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fn", "--filename", type=str, required=True, help="input JSON filename"
    )
    parser.add_argument(
        "-pl",
        "--per_user_limit",
        type=int,
        default=50,
        help="how many submission per user to fetch",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    reddit = praw.Reddit(
        client_id="nzzROBOXXsVHoEOqY6SBlw",
        client_secret="vXJ0NWRVaTXnfZE7QCjf6u6RelakwA",
        user_agent="gchochla",
    )

    with open(args.filename) as fp:
        subreddit_data = json.load(fp)

    subreddit = args.filename.split("-")[0]

    user_baseline = {}
    users = set([val["author"] for val in subreddit_data.values()])
    unresolved = []

    for i, user in enumerate(users):

        print(f"{i+1}/{len(users)}: {user}")

        redditor = reddit.redditor(user)
        baseline = redditor.new(limit=args.per_user_limit)
        try:
            for post in baseline:
                if post.subreddit.display_name != subreddit:
                    try:
                        post_data = parse_submission(post)
                    except AttributeError:
                        post_data = parse_comment(post)

                    post_data.update({"subreddit": post.subreddit.display_name})

                    user_baseline.setdefault(user, []).append(post_data)

        except Exception as e:
            print("\t", e)
            unresolved.append(user)

    basename = os.path.splitext(args.filename)[0]
    with open(f"{basename}-{args.per_user_limit}-baseline.json", "w") as fp:
        json.dump(user_baseline, fp)

    with open(f"{basename}-{args.per_user_limit}-unresolved.txt", "w") as fp:
        fp.write(",".join(unresolved))
