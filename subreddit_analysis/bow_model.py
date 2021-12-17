"""This script uses annotated posts to estimate importance of words
for the specific task based on a word-count representation of documents."""

import argparse
import csv
import re
import numpy as np
from typing import Optional, List, Union, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from sklearn.linear_model import RidgeCV


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_filename",
        type=str,
        required=True,
        help="CSV to read documents from",
    )
    parser.add_argument(
        "--label_filename",
        type=str,
        required=True,
        help="CSV to read labels from",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="which column of CSV to read document from",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="post_id",
        help="which column of CSV to read post id from",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        nargs="+",
        default=["wp"],
        help="which column of CSV to read label from",
    )
    parser.add_argument(
        "--out_filename", type=str, required=True, help="where to save scores"
    )
    parser.add_argument(
        "--ratio_keep",
        type=float,
        help="what percentage of data to randomly keep",
    )
    args = parser.parse_args()
    return args


def get_documents(
    doc_filename: str,
    id_column: str,
    text_column: str,
    label_filename: Optional[str] = None,
    label_columns: Optional[List[str]] = None,
) -> Union[
    Tuple[List[str], List[str]], Tuple[List[str], List[str], List[List[str]]]
]:
    """Parses CSVs to get documents and potentially their label(s).
    If not label filename or label column(s) are provided, only documents
    are returned. Otherwise, only documents with every label are returned.

    Args:
        doc_filename: path to document CSV.
        id_column: header of document ID column.
        text_column: header of actual text of document.
        label_filename: path to label CSV.
            Default is `None`, aka no labels are used.
        label_columns: headers of label
            columns. Default is `None`, aka no labels are used.

    Returns:
        If label-related arguments are set, a tuple (ids, docs, labels)
        of corresponding document IDs, the document themselves and their
        labels. Otherwise, a tuple (ids, docs) is returned.
    """
    with open(doc_filename) as fp:
        reader = csv.reader(fp)
        headers = next(reader)
        text_idx = headers.index(text_column)
        id_idx = headers.index(id_column)
        documents = {
            row[id_idx]: row[text_idx] for row in reader if row[text_idx]
        }

    if (label_filename is None) or (label_columns is None):
        return list(documents), list(documents.values())

    with open(label_filename) as fp:
        reader = csv.reader(fp)
        headers = next(reader)
        label_inds = [
            headers.index(label_column) for label_column in label_columns
        ]
        id_idx = headers.index(id_column)
        documents = {
            row[id_idx]: [documents[row[id_idx]]]
            + [row[label_idx] for label_idx in label_inds]
            for row in reader
            if all([row[label_idx] != "NA" for label_idx in label_inds])
        }

    return (
        list(documents),
        [doc[0] for doc in documents.values()],
        [doc[1:] for doc in documents.values()],
    )


if __name__ == "__main__":
    args = parse_args()

    _, docs, labels = get_documents(
        doc_filename=args.doc_filename,
        id_column=args.id_column,
        text_column=args.text_column,
        label_filename=args.label_filename,
        label_columns=args.label_column,
    )

    if args.ratio_keep:
        inds = np.random.permutation(len(docs))[
            : int(len(docs) * args.ratio_keep)
        ]
        docs = [docs[i] for i in inds]
        labels = [labels[i] for i in inds]

    labels = np.array(list(map(lambda x: float(x[0]), labels)))

    # consider all w.d. as a single token for this
    docs = [re.sub("w\dd\d", "wd", doc) for doc in docs]

    bow = Vectorizer()
    count_data = bow.fit_transform(docs)

    headers = bow.get_feature_names_out()

    clf = RidgeCV().fit(count_data, labels)

    sort_inds = reversed(clf.coef_.argsort())
    rows = [(headers[idx], clf.coef_[idx]) for idx in sort_inds]

    with open(args.out_filename, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)
