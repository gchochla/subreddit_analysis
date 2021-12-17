"""Scirpt that computes DDR loadings for each row of a CSV."""

import os
import argparse
import csv

import gensim.downloader as api
import ddr


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
        help="root directory of all data, if exists",
    )
    parser.add_argument(
        "-doc",
        "--doc_path",
        type=str,
        required=True,
        help="path to document csv",
    )
    parser.add_argument(
        "-dict",
        "--dict_dir",
        type=str,
        default="dictionaries",
        help="path to dictionaries",
    )
    parser.add_argument(
        "-dict_repr",
        "--dict_repr_path",
        type=str,
        default="dict_vectors.tsv",
        help="tsv to save dictionary representation",
    )
    parser.add_argument(
        "-doc_repr",
        "--doc_repr_path",
        type=str,
        default="doc_vectors.tsv",
        help="tsv to save document representation",
    )
    parser.add_argument(
        "-doc_load",
        "--doc_loadings",
        type=str,
        default="doc_loadings.tsv",
        help="tsv to save document loadings",
    )
    parser.add_argument(
        "-emb",
        "--embeddings",
        type=str,
        default="glove-twitter-200",
        help="which embeddings to use",
    )
    parser.add_argument(
        "-x",
        "--create_new",
        action="store_true",
        default=False,
        help="whether not to overwrite previous csv",
    )
    parser.add_argument(
        "-c",
        "--text_column",
        type=str,
        default="text",
        help="which column of the document csv to read from",
    )
    parser.add_argument(
        "-id",
        "--id_column",
        type=str,
        default="post_id",
        help="which column of the document csv contains "
        "the ids of the documents",
    )

    args = parser.parse_args()
    if args.root_dir:
        args.doc_path = os.path.join(args.root_dir, args.doc_path)
        args.dict_dir = os.path.join(args.root_dir, args.dict_dir)
        args.dict_repr_path = os.path.join(args.root_dir, args.dict_repr_path)
        args.doc_repr_path = os.path.join(args.root_dir, args.doc_repr_path)
        args.doc_loadings = os.path.join(args.root_dir, args.doc_loadings)

    return args


def add_loadings_to_csv(
    loadings_filename: str,
    csv_filename: str,
    id_col: str,
    loadings_id_col: str = "ID",
    overwrite: bool = True,
):
    """Add computed loadings to the CSV they were computed from.

    Args:
        loadings_filename: filename where loadings are stored by DDR.
        csv_filename: filename from which loadings were computed from
            and will be written to.
        id_col: document ID header.
        loadings_id_col: loading of document header. Defaults to "ID".
        overwrite: whether to overwrite previous CSV. Defaults to True.
            .bak is added otherwise to the new file.
    """

    loadings_dict = {}
    csv_list = []

    with open(loadings_filename) as fp:
        reader = csv.reader(fp, delimiter="\t")
        load_headers = next(reader)
        load_id_idx = load_headers.index(loadings_id_col)
        for row in reader:
            post_id = row[load_id_idx]
            loadings_dict[post_id] = row[:load_id_idx] + row[load_id_idx + 1 :]

    # utf8 to work with files from unix systems when read from windows
    with open(csv_filename, encoding="utf8") as fp:
        reader = csv.reader(fp)

        headers = next(reader)
        id_idx = headers.index(id_col)

        csv_list.append(
            headers
            + load_headers[:load_id_idx]
            + load_headers[load_id_idx + 1 :]
        )
        for row in reader:
            post_id = row[id_idx]
            csv_list.append(
                row
                + loadings_dict.get(post_id, (len(load_headers) - 1) * ["N/A"])
            )

    # newline to work properly on windows, utf8 as above
    with open(
        csv_filename + (".bak" if not overwrite else ""),
        "w",
        newline="",
        encoding="utf8",
    ) as fp:
        writer = csv.writer(fp)
        writer.writerows(csv_list)


if __name__ == "__main__":
    args = parse_args()

    doc_path = args.doc_path
    dict_dir = args.dict_dir
    dict_repr_path = args.dict_repr_path
    doc_repr_path = args.doc_repr_path
    doc_loadings = args.doc_loadings

    model = api.load(args.embeddings)
    num_features = model.vector_size
    word_set = set(model.index_to_key)

    dict_terms = ddr.terms_from_txt(input_path=dict_dir)
    aggr_dict_vectors = ddr.dic_vecs(
        dic_terms=dict_terms,
        model=model,
        num_features=num_features,
        model_word_set=word_set,
    )
    ddr.write_dic_vecs(dic_vecs=aggr_dict_vectors, output_path=dict_repr_path)

    ddr.doc_vecs_from_csv(
        delimiter=",",
        input_path=doc_path,
        output_path=doc_repr_path,
        model=model,
        num_features=num_features,
        model_word_set=word_set,
        text_col=args.text_column,
        id_col=args.id_column,
    )

    ddr.get_loadings(
        agg_doc_vecs_path=doc_repr_path,
        agg_dic_vecs_path=dict_repr_path,
        out_path=doc_loadings,
        num_features=num_features,
    )

    add_loadings_to_csv(
        doc_loadings, doc_path, args.id_column, overwrite=(not args.create_new)
    )
