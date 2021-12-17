"""A general-purpose RNN script that can handle multi-tasking,
binary, multilabel or ordinal/regression tasks, load pre-trained
models and save trained models. For the time being, some options
are only available to the end user via direct code manipulation,
e.g. turning an ordinal task into multilabel classification.
Training includes CV."""

from __future__ import annotations

import csv
import argparse
import warnings
from collections import Counter
from typing import List, Union, Tuple, Iterable, Optional, Dict, Mapping

import numpy as np

import gensim.downloader as api
import gensim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

from subreddit_analysis.bow_model import get_documents


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
        help="which column of CSV to read label from, "
        "first will be used for comparison across models",
    )
    parser.add_argument(
        "--out_filename",
        type=str,
        help="where to save predicted labels",
    )
    parser.add_argument("--model_fn", type=str, help="where to save model")
    parser.add_argument(
        "--pretrained_model_fn", type=str, help="pretrained model filename"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=25,
        help="dimensionality of word embeddings",
    )
    args = parser.parse_args()

    if args.out_filename is None and args.model_fn is None:
        warnings.warn(
            "Script won't do anything but show CV scores"
            "since --out_filename and --model_fn were not set"
        )

    return args


class DocumentDataset(Dataset):
    """Dataset class for padding and packing sequences of different
    lengths into a uniform batch for faster processing by Pytorch RNNs.

    Attributes:
        docs (np.ndarray[List[float]]): basically a list of
            sequences of embeddings packed into an array for
            easy indexing.
        lengths (torch.Tensor[int]): length of each sequence.
        labels (torch.Tensor[Union[float, int]]): corresponding labels.
    """

    def __init__(self, documents: List[List[np.ndarray]], labels: np.ndarray):
        """Init.

        Args:
            documents: list of sequence of embeddings.
            labels: corresponding array of labels.
        """

        self.lengths = torch.tensor([len(doc) for doc in documents])
        self.docs = np.array(documents, dtype=object)
        self.labels = torch.tensor(labels)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]:
        """Fetches packed sequence along with corresponding labels.

        Args:
            index: which.
        """
        sort_inds = np.argsort(-self.lengths[index])
        padded = pad_sequence(
            [
                torch.tensor(doc, dtype=torch.float)
                for doc in self.docs[index][sort_inds]
            ]
        )
        return (
            pack_padded_sequence(
                padded, lengths=self.lengths[index][sort_inds]
            ),
            self.labels[index][sort_inds],
        )

    def __len__(self):
        return len(self.docs)


class RNN(nn.Module):
    """Wrapper class around LSTM to utilize PackedSequences.

    Attributes:
        bidirectional (bool): whether RNN is bidirectional.
        hidden_size (int): hidden size including both possible
            directions.
        lstm (nn.Module): LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        """Init.

        Args:
            input_size: input dimensionality of individual samples
                within a sequence.
            hidden_size: hidden size (of each dimension) of LSTM.
            num_layers: number of layers of LSTM.
            dropout: probability of dropout.
            bidirectional: whether to make LSTM bidirectional.
        """

        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size * (1 + int(bidirectional))

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input packed sequence.

        Returns:
            Last timestep of forward and backward LSTM, concatenated.
        """
        outs, lens = pad_packed_sequence(self.lstm(x)[0])
        outs.transpose_(1, 0)

        return self.last_timestep(outs, lens)

    def last_timestep(
        self, outputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Returns the last output of the RNN taking into account
        the zero-padding.

        Args:
            outputs: outputs/hidden states of LSTM.
            lengths: actual length of input sequences.
        """

        if self.bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(
        outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits outputs to forward and backward."""
        direction_size = outputs.size(-1) // 2
        forward = outputs[..., :direction_size]
        backward = outputs[..., direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(
        outputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Fetches indices of the last output for each sequence."""
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()


class RnnSKL(BaseEstimator, ClassifierMixin):
    """Wrapper around nn.Module to use with sklearn, an
    RNN followed by linear layers for the tasks at hand.

    Attributes:
        word_embeddings (gensim.models.keyedvectors.KeyedVectors):
            word embeddings.
        input_size (int): LSTM input size.
        hidden_size (int): LSTM hidden size.
        num_layers (int): LSTM number of layers.
        dropout (float): LSTM dropout probability (internal and output).
        bidirectional (bool): LSTM bidirectionality.
        n_tasks (int): number of ML tasks.
        n_classes (Tuple[int]): the number of classes per task.
            1 for binary, >1 for multiclass and -1 for regression.
        weights (Iterable[torch.Tensor]): weights for cross-entropy
            for each task.
        comp_task (int): index of task actually used for comparison
            between models.
        lr (float): learning rate.
        epochs (int): number of training epochs.
        batch_size (int): batch size.
        device (torch.DeviceObjType): device to run model on.
        verbose (bool): whether to print progress within
            each epoch. Default to False.
        rnn_state_dict (Dict): pre-trained RNN state_dict.
            Can be None.
    """

    def __init__(
        self,
        word_embeddings: gensim.models.keyedvectors.KeyedVectors,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        n_tasks: int,
        n_classes: Tuple[int],
        weights: Iterable[torch.Tensor],
        comp_task: int,
        lr: float,
        epochs: int,
        batch_size: int,
        device: torch.DeviceObjType,
        verbose: Optional[bool] = False,
        rnn_state_dict: Optional[Dict] = None,
    ):
        """Init.

        Args:
            Check class docstring.
        """

        self.rnn_state_dict = rnn_state_dict
        self.verbose = verbose

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.n_tasks = n_tasks
        # for regression it's -1
        self.n_classes = n_classes
        self.weights = weights  # just for sklearn to work

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = device

        self.word_embeddings = word_embeddings

        self.comp_task = comp_task

        # losses for each task
        self.criteria = [
            nn.CrossEntropyLoss(weight=weights[i])
            if nc > 1
            else nn.BCEWithLogitsLoss(weight=weights[i])
            if nc == 1
            else nn.MSELoss()
            for i, nc in enumerate(self.n_classes)
        ]

        # from linear layer outputs to concrete prediction
        self.prediction_mapping = [
            (lambda x: x.detach().to("cpu").argmax(dim=-1).numpy())
            if nc > 1
            else (lambda x: x.detach().to("cpu").round().numpy().int())
            if nc == 1
            else (lambda x: x.detach().to("cpu").numpy())
            for nc in self.n_classes
        ]

        # from predictions and ground-truths
        self.score_mapping = [
            (lambda x, y: (x == y).sum() / len(x))
            if nc >= 1
            else (lambda x, y: -((x - y) ** 2).mean())
            for nc in self.n_classes
        ]

    def init_rnn(self):
        """Initializes RNN based on attributes and stores
        it in base_rnn."""
        self.base_rnn = RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        ).to(self.device)

    def init_linear(self):
        """Initializes linear layers, one for each task, based
        on attributes and stores them in linear_layers as a
        torch.nn.ModuleList."""
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    # apply dropout to rnn reprs
                    nn.Dropout(self.dropout),
                    nn.Linear(
                        self.base_rnn.hidden_size, max(1, self.n_classes[i])
                    ),
                )
                for i in range(self.n_tasks)
            ]
        ).to(self.device)

    def fit(self, X: Iterable[str], y: np.ndarray) -> RnnSKL:
        """Fits base_rnn and linear_layers to X and y.

        Args:
            X: iterable of documents.
            y: corresponding labels, rows must be the
                same length as n_tasks.

        Returns:
            Self.
        """

        self.init_rnn()
        self.init_linear()
        if self.rnn_state_dict:
            self.base_rnn.load_state_dict(self.rnn_state_dict)

        self.base_rnn.train()
        self.linear_layers.train()

        X = self.embed_documents(X)
        # have to create a collate_fn for a dataloader
        dataset = DocumentDataset(X, y)
        n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        optimizer = optim.Adam(
            list(self.base_rnn.parameters())
            + list(self.linear_layers.parameters()),
            lr=self.lr,
        )

        for epoch in range(self.epochs):
            self.print(f"Epoch {epoch+1}")
            for i in range(n_batches):
                self.print(f"\tBatch {i+1}", end=" ")
                x_batch, y_batch = dataset[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(
                    self.device
                )

                repr = self.base_rnn(x_batch)
                task_preds = [lin(repr) for lin in self.linear_layers]

                losses = [
                    crit(y_pred.squeeze(), y_true)
                    for y_pred, y_true, crit in zip(
                        task_preds, y_batch.T, self.criteria
                    )
                ]
                loss = torch.stack(losses).mean()

                self.print(f"Loss {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def print(self, *args, **kwargs):
        """Wrapper around print to only print
        when verbose has been set to True."""
        if self.verbose:
            print(*args, **kwargs)

    def embed_documents(
        self, documents: Iterable[str]
    ) -> List[List[np.ndarray]]:
        """Creates sequences of embeddings from a sequence of documents.

        Args:
            documents: iterable of documents.

        Returns:
            Embedded documents.
        """
        return [
            [
                self.word_embeddings.get_vector(word)
                if word in self.word_embeddings.key_to_index
                else self.word_embeddings.get_vector("<unk>")
                for word in doc.split()
            ]
            for doc in documents
        ]

    def predict(
        self, X: Iterable[str], comp_task: Optional[int] = None
    ) -> np.ndarray:
        """Predicts outcomes of X.

        Args:
            X: iterable of documents.
            comp_task: index of task to actually return the prediction for.
                Defaults to None, aka the class attribute is used.
        """
        if comp_task is None:
            comp_task = self.comp_task

        self.base_rnn.eval()
        self.linear_layers.eval()

        with torch.no_grad():

            X = self.embed_documents(X)
            dummy_y = [0] * len(X)
            dataset = DocumentDataset(X, dummy_y)
            n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

            preds = []

            for i in range(n_batches):
                x_batch, _ = dataset[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                x_batch = x_batch.to(self.device)

                repr = self.base_rnn(x_batch)
                task_preds = [lin(repr) for lin in self.linear_layers]
                preds.append(
                    [
                        mapping(y_pred)
                        for y_pred, mapping in zip(
                            task_preds, self.prediction_mapping
                        )
                    ]
                )

            preds = [
                np.concatenate([pred[i] for pred in preds])
                for i in range(self.n_tasks)
            ]

            return preds[comp_task]

    def score(
        self, X: Iterable[str], y: np.ndarray, comp_task: Optional[int] = None
    ) -> float:
        """Returns score of model for task comp_task.

        Args:
            X: iterable of documents.
            y: correspoinding labels.
            comp_task: index of task we are concerned with.
                Defaults to None.
        """
        if comp_task is None:
            comp_task = self.comp_task

        preds = self.predict(X, comp_task=comp_task)
        score = self.score_mapping[comp_task](preds, y[:, comp_task])

        return score


def handle_labels(
    labels: List[str], ordinal: Optional[bool] = False
) -> Tuple[np.ndarray, List[Mapping], List[int], List[Counter]]:
    """Transforms labels as read from CSV to labels
    usable by Pytorch.

    Args:
        labels: labels as read from CSV.
        ordinal: whether to consider ordinal tasks. Defaults to False.

    Returns:
        A tuple (labels, mappings, n_classes, frequencies), where labels
        are the final, usable labels (integers for classification or floats
        for ordinal tasks), mappings contains the mapping from transformed
        labels to original labels, n_classes is useful for RnnSKL, and
        frequencies includes the counts of each label in each task.
    """

    class LinearMapping:
        """Helper class to map numbers to their
        corresponding strings in a mapping style."""

        def __getitem__(self, index):
            return str(index)

    n_tasks = len(labels[0])

    label_columns = [[label[i] for label in labels] for i in range(n_tasks)]

    mappings = []
    n_classes = []
    frequencies = []

    for i in range(n_tasks):
        unique = list(set(label_columns[i]))
        if len(unique) == 2:
            mapping = {u: i for i, u in enumerate(unique)}
            inv_mapping = {i: u for i, u in enumerate(unique)}
            nc = 1  # convention, see handling in RnnSKL
        elif (
            any([not lbl.replace(".", "", 1).isdigit() for lbl in unique])
            or not ordinal
        ):
            mapping = {u: i for i, u in enumerate(unique)}
            inv_mapping = {i: u for i, u in enumerate(unique)}
            nc = len(unique)
        else:  # ordinal
            mapping = {u: np.float32(u) for u in unique}
            inv_mapping = LinearMapping()
            nc = -1  # convention, see handling in RnnSKL

        label_columns[i] = [mapping[lbl] for lbl in label_columns[i]]
        frequency = Counter(label_columns[i])

        frequencies.append(frequency)
        mappings.append(inv_mapping)
        n_classes.append(nc)

    labels = np.stack([np.array(lbl_col) for lbl_col in label_columns], axis=-1)

    return labels, mappings, n_classes, frequencies


if __name__ == "__main__":
    args = parse_args()

    ids, docs, labels = get_documents(
        doc_filename=args.doc_filename,
        id_column=args.id_column,
        text_column=args.text_column,
        label_filename=args.label_filename,
        label_columns=args.label_column,
    )

    labels, mapping, n_classes, frequencies = handle_labels(labels)

    # keep as list bcs diff tasks may have diff #labels
    inv_frequencies = [
        torch.tensor([1 / freq[i] for i in sorted(freq)])
        for freq in frequencies
    ]
    inv_frequencies = [
        ifreq / ifreq.sum() * len(ifreq) for ifreq in inv_frequencies
    ]

    if args.pretrained_model_fn:
        state_dict = torch.load(args.pretrained_model_fn)
        rnn_state_dict = state_dict["rnn"]
        hidden_size = state_dict["params"]["hidden_size"]
        hidden_sizes = [hidden_size]
        num_layers = state_dict["params"]["num_layers"]

        input_size = state_dict["params"]["input_size"]
        if args.embedding_dim != input_size:
            print(
                "Overwriting embeddings dimension (from "
                f"{args.embedding_dim} to {input_size}) because"
                " pretrained network used the latter"
            )
            args.embedding_dim = input_size
    else:
        # dummy init, will change within the CV loop
        hidden_size, num_layers = 10, 1
        hidden_sizes = [50]
        rnn_state_dict = None

    word_embeddings = api.load(f"glove-twitter-{args.embedding_dim}")
    # add an unknown vector, whoch is the mean of all the embeddings
    # https://groups.google.com/g/globalvectors/c/9w8ZADXJclA/m/hRdn4prm-XUJ
    word_embeddings.add_vectors(
        ["<unk>"], [word_embeddings.vectors.mean(axis=0)]
    )

    model = RnnSKL(
        word_embeddings=word_embeddings,
        input_size=args.embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0,
        bidirectional=True,
        n_tasks=len(n_classes),
        n_classes=n_classes,
        weights=inv_frequencies,
        comp_task=0,
        lr=1e-4,
        epochs=20,
        batch_size=32,
        device="cpu",
        verbose=False,
        rnn_state_dict=rnn_state_dict,
    )

    # NOTE: these seem to work the best for 100-dim embeddings
    # NOTE: change these to alter the grid search
    search_params = {
        "lr": [1e-2],
        "dropout": [0.2],
        "epochs": [50],
        "hidden_size": hidden_sizes,
    }

    if args.pretrained_model_fn:
        # no need to have hidden states as search params
        del search_params["hidden_size"]

    gscv = GridSearchCV(
        model,
        search_params,
        verbose=3,
        cv=5,  # default
    )
    gscv.fit(docs, labels)

    print("Best params", gscv.best_params_)

    # Saving model
    if args.model_fn:
        model_dict = {
            f"linear{i}": gscv.best_estimator_.linear_layers[i]
            for i in range(len(n_classes))
        }
        model_dict["rnn"] = gscv.best_estimator_.base_rnn.state_dict()
        model_dict["params"] = gscv.best_estimator_.get_params()
        torch.save(model_dict, args.model_fn)

    if args.out_filename:
        ids, docs = get_documents(
            doc_filename=args.doc_filename,
            id_column=args.id_column,
            text_column=args.text_column,
        )

        for id, doc in zip(ids, docs):
            if not doc:
                print(id)

        preds = gscv.predict(docs)
        preds = [mapping[0][pred] for pred in preds]

        with open(args.out_filename, "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(
                [[args.id_column, args.label_column[0]]]
                + [[id, lbl] for id, lbl in zip(ids, preds)]
            )
