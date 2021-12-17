import argparse
import random
import csv
import os

from willpower.bow_model import get_documents


def parse_args():
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
        "--out_filename", type=str, required=True, help="where to save CSV"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="how many per label to keep"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ids, docs, labels = get_documents(
        doc_filename=args.doc_filename,
        id_column=args.id_column,
        text_column=args.text_column,
        label_filename=args.label_filename,
        label_columns=args.label_column,
    )

    labels = [lbl[0] for lbl in labels]

    unique = list(set(labels))
    inds = {lbl: [] for lbl in unique}

    for i, label in enumerate(labels):
        inds[label].append(i)

    inds = {
        lbl: random.sample(inds[lbl], min(args.k, len(inds[lbl])))
        for lbl in inds
    }

    all_inds = []
    for label in inds:
        all_inds.extend(inds[label])

    perm = random.sample(range(len(all_inds)), len(all_inds))
    all_inds = [all_inds[i] for i in perm]

    with open(args.out_filename, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(
            [[args.id_column, args.text_column, args.label_column[0]]]
            + [[ids[i], docs[i], labels[i]] for i in all_inds]
        )

    base_dir, fn = os.path.split(args.out_filename)
    out_filename = os.path.join(base_dir, "no_labels_" + fn)
    with open(out_filename, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(
            [[args.id_column, args.text_column, args.label_column[0]]]
            + [[ids[i], docs[i], None] for i in all_inds]
        )
