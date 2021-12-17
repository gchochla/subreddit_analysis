"""Downloads subreddit posts using using Pushshift"""

from typing import Any, Dict
import requests
import json
import argparse
from datetime import datetime


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
        type=float,
        default=3000,
        help="number of submissions/posts",
    )
    args = parser.parse_args()

    return args


def transform_timestamp(timestamp: int, format="%Y-%m-%d %H:%M:%S") -> str:
    """Transforms UTC timestamp to readable date."""
    return datetime.utcfromtimestamp(timestamp).strftime(format)


def get_pushshift_data(data_type: str, max_retries: int = 20, **kwargs) -> Dict:
    """Gets data from the pushshift.io API. Read more:
    `https://github.com/pushshift/api`.

    Args:
        data_type: specific path of API. Choices
            are "search/submission", "search/comment"
            and "submission/comment_ids/{post_id}".
        kwargs: params of the get request.

    Returns:
        JSONified response.

    """

    def aux(s):
        print(f"Received {len(response)} {s}(s)")

    base_url = f"https://api.pushshift.io/reddit/{data_type}/"
    print(f"Querying {base_url}")

    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(base_url, params=kwargs).json()["data"]
            break
        except:
            print("Retrying...")
            retries += 1

    if retries >= max_retries:
        return

    if data_type == "search/submission":
        items = "post"
        aux(items)
    elif data_type.startswith("submission/comment_ids"):
        items = "comment"
        aux(items)

    return response


def parse_posts(resp: Dict) -> Dict[str, Dict[str, Any]]:
    """Parses posts to a format with necessary
    info to run our studies.

    Args:
        resp: parsed API response.

    Returns:
        Dict of all posts in input JSON.
    """

    d = {}
    for post in resp:
        try:
            d[post["id"]] = {
                "author": post["author"],
                "title": post["title"],
                "text": post["selftext"],
                "n_comments": post["num_comments"],
                "n_awards": post["total_awards_received"],
                "score": post["score"],
                "upvote_ratio": post["upvote_ratio"],
                "url": "https://www.reddit.com"
                + post["permalink"],  # permalink is "/r/couchto5k/..."
                "thumbnail": post["url"] if post["thumbnail"] != "self" else "",
                "utc": post["created_utc"],
                "timestamp": transform_timestamp(post["created_utc"]),
                "comments": {},  # add in advance
            }
        except:
            pass

    return d


def parse_comments(resp: Dict) -> Dict[str, Dict[str, Any]]:
    """Parses comments to a format with necessary
    info to run our studies.

    Args:
        resp: parsed API response.

    Returns:
        Dict of all comments in input JSON.
    """

    d = {}
    for comment in resp:
        try:
            d[comment["id"]] = {
                "author": comment["author"],
                "text": comment["body"],
                "parent_id": comment["parent_id"].split("_")[1],
                "post_id": comment["link_id"].split("_")[1],
                "score": comment["score"],
                # "upvote_ratio": comment["upvote_ratio"],
                "url": "https://www.reddit.com"
                + comment["permalink"],  # permalink is "/r/couchto5k/..."
                "timestamp": transform_timestamp(comment["created_utc"]),
                "comments": {},  # add in advance
            }
        except:
            pass

    return d


if __name__ == "__main__":

    args = parse_args()

    subreddit_data = {}
    before = None  # previous earlier timestamp
    prev = 0  # previous count of submissions

    while len(subreddit_data) < args.limit:

        subreddit_resp = get_pushshift_data(
            "search/submission",
            size=min(100, args.limit),  # 100 is max acceptable value
            sort_type="created_utc",
            sort="desc",
            subreddit=args.subreddit,
            before=before,
        )

        subreddit_data.update(parse_posts(subreddit_resp))

        # somewhat general check to see if we got anything new
        if prev == len(subreddit_data):
            print("Got all posts from subreddit")
            break

        prev = len(subreddit_data)
        print(f"Received {prev} post(s) so far")

        before = min([info["utc"] for info in subreddit_data.values()])

    post_ids = list(subreddit_data)  # get it separately to avoid confusion
    unresolved = []

    for i, post_id in enumerate(post_ids):
        print(f"Post {i+1}/{prev}:")
        comment_ids = get_pushshift_data(f"submission/comment_ids/{post_id}")

        if comment_ids:  # if comments exist
            comment_resps = get_pushshift_data(
                "search/comment", ids=comment_ids
            )

            if not comment_resps or len(comment_ids) != len(comment_resps):
                unresolved.append(post_id)
                print(f"unresolved {post_id}\n")
                continue

            comment_data = parse_comments(comment_resps)

            comment_ids = set(comment_data)  # only ids
            current_level_ids = {post_id}  # start from root

            # while there are still comments left
            while comment_ids:

                next_level_ids = set()

                for comment_id in comment_ids:

                    # if parent was in previous layer, then we add
                    parent_id = comment_data[comment_id]["parent_id"]
                    if parent_id in current_level_ids:

                        # get path to root so we can retrieve the dictionary
                        parents = [parent_id]
                        while parent_id != post_id:
                            parent_id = comment_data[parent_id]["parent_id"]
                            parents.append(parent_id)

                        # follow path from root to current parent
                        current_node = subreddit_data
                        for id in reversed(parents):
                            current_node = current_node[id]["comments"]

                        # add node to data tree and id to next layers parents
                        current_node[comment_id] = comment_data[comment_id]
                        next_level_ids.add(comment_id)

                if not next_level_ids:
                    print("Comments with weird IDs")
                    unresolved.append(post_id)
                    break

                # set next level's parents to nodes of this layer
                current_level_ids = next_level_ids
                # delete added nodes from pending
                comment_ids.difference_update(next_level_ids)

        print()

    with open(f"{args.subreddit}-{prev}-pushshift.json", "w") as fp:
        json.dump(subreddit_data, fp)

    with open(f"{args.subreddit}-{prev}-pushshift-unresolved.txt", "w") as fp:
        fp.write(",".join(unresolved))
