"""Parses a JSON downloaded by `subreddit_forest.py` into a CSV."""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


# keep csv labels and how to compute from id (x), value of node (y), OP (z)
CSV_MAPPING = {
    "post_id": lambda x, y, z, w: x,
    "parent_post_id": lambda x, y, z, w: y.get("post_id", x),
    "direct_parent_id": lambda x, y, z, w: w if w else "N/A",
    "author": lambda x, y, z, w: y["author"],
    "OP": lambda x, y, z, w: str(int(y["author"] == z)),
    "title": lambda x, y, z, w: y.get("title", "N/A"),
    "text": lambda x, y, z, w: y["text"],
    "n_comments": lambda x, y, z, w: str(y.get("n_comments", "N/A")),
    "n_awards": lambda x, y, z, w: str(y.get("n_awards", "N/A")),
    "score": lambda x, y, z, w: str(y["score"]),
    "upvote_ratio": lambda x, y, z, w: str(y.get("upvote_ratio", "N/A")),
    "url": lambda x, y, z, w: y["url"],
    "thumbnail": lambda x, y, z, w: y.get("thumbnail", "N/A"),
    "timestamp": lambda x, y, z, w: str(y["timestamp"]),
    "comment": lambda x, y, z, w: str(int(y.get("post_id", x) != x)),
}


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fn", "--filename", type=str, required=True, help="input JSON filename"
    )
    args = parser.parse_args()

    return args


def tree_traversal(
    node: str,
    tree: Dict[str, Any],
    root_author: str,
    direct_parent: Optional[str] = None,
) -> List[List[str]]:
    """Traverses a tree represented by dictionaries (preorder traversal).

    Args:
        node: node ID.
        tree: root node info (which of course
            includes children).
        root_author: OP of post (application specific info).
        direct_parent: id of direct parent of node in tree.

    Returns:
        List of rows with info.
    """

    rows = [
        [
            fun(node, tree, root_author, direct_parent)
            for fun in CSV_MAPPING.values()
        ]
    ]

    for child, info in tree["comments"].items():
        rows.extend(tree_traversal(child, info, root_author, node))

    return rows


if __name__ == "__main__":
    args = parse_args()
    json_fn = args.filename

    with open(json_fn) as fp:
        forest = json.load(fp)

    rows = [list(CSV_MAPPING)]  # first row is labels
    for post in forest:
        rows.extend(tree_traversal(post, forest[post], forest[post]["author"]))

    # save on same filename (with proper extension)
    csv_fn = os.path.splitext(json_fn)[0] + ".csv"

    with open(csv_fn, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)
