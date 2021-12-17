"""Downloads subreddit posts using PRAW.
This is limited to the latest 1000 posts."""

import json
import praw
import argparse
from typing import Dict, Any

from subreddit_analysis.subreddit_forest import transform_timestamp


def parse_args():
    """Parses args."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--subreddit",
        type=str,
        default="C25K",
        help="subreddit to download from",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=3000,
        help="number of submissions/posts",
    )
    args = parser.parse_args()

    return args


def parse_submission(submission: praw.reddit.Submission) -> Dict[str, Any]:
    """Parses submission into convenient serializable struct"""

    return {
        "author": submission.author.name,
        "title": submission.title,
        "text": submission.selftext,
        "score": submission.score,
        "upvotes": submission.ups,
        "downvotes": submission.downs,
        "upvote_ratio": submission.upvote_ratio,
        "n_comments": submission.num_comments,
        "url": "www.reddit.com" + submission.permalink,
        "timestamp": transform_timestamp(int(submission.created_utc)),
        "comments": {},
    }


def parse_comment(comment: praw.reddit.Comment) -> Dict[str, Any]:
    """Parses comment into convenient serializable struct"""

    return {
        "author": comment.author.name,
        "text": comment.body,
        "score": comment.score,
        "upvotes": comment.ups,
        "downvotes": comment.downs,
        "url": "www.reddit.com" + comment.permalink,
        "timestamp": transform_timestamp(int(comment.created_utc)),
        "post_id": comment.parent_id.split("_")[1],
        "link_id": comment.link_id.split("_")[1],
        "comments": {},
    }


def append_comment(comment: praw.reddit.Comment, node: Dict[str, Any]):
    """Recursive function that builds tree of each post.

    Args:
        comment: comment to append to node.
        node: current tree node's "comment" property.
    """
    if comment.author is None:
        return
    node[comment.id] = parse_comment(comment)
    for reply in comment.replies:
        append_comment(reply, node[comment.id]["comments"])


if __name__ == "__main__":
    args = parse_args()

    reddit = praw.Reddit(
        client_id="nzzROBOXXsVHoEOqY6SBlw",
        client_secret="vXJ0NWRVaTXnfZE7QCjf6u6RelakwA",
        user_agent="gchochla",
    )
    subreddit = reddit.subreddit(args.subreddit)
    submissions = subreddit.new(limit=args.limit)

    forest = {}

    for i, submission in enumerate(submissions):

        print(f"Submission {i+1}/{args.limit}")

        if submission.author is None:
            continue

        submission.comments.replace_more(limit=None)

        forest[submission.id] = parse_submission(submission)

        for comment in submission.comments:
            append_comment(comment, forest[submission.id]["comments"])

    with open(f"{args.subreddit}-{i+1}.json", "w") as fp:
        json.dump(forest, fp)
