"""Evaluation utilities."""

from copy import copy
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class Outcome(BaseModel):
    prediction: Union[Dict[str, Any], List[Dict[str, Any]]] = {}
    expected: Union[Dict[str, Any], List[Dict[str, Any]]] = {}
    parameters: Dict[str, Any] = {}
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    by_field: Dict[str, int] = {}
    ixn_by_field: Dict[str, List[str]] = {}

    def calculate_metrics(self):
        self.precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0
        self.recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if self.precision + self.recall > 0
            else 0
        )

    def append_outcomes(self, outcomes: List["Outcome"]) -> None:
        for sub_outcome in outcomes:
            self.tp += sub_outcome.tp
            self.fp += sub_outcome.fp
            self.fn += sub_outcome.fn
            for key, value in sub_outcome.by_field.items():
                self.by_field[key] = self.by_field.get(key, 0) + value
            for key, value in sub_outcome.ixn_by_field.items():
                curr = set(self.ixn_by_field.get(key, []))
                self.ixn_by_field[key] = list(curr.union(value))
        self.calculate_metrics()

    def flatten(self) -> Dict[str, Any]:
        obj = self.model_dump()
        for k, v in copy(obj).items():
            if k == "parameters":
                obj.update(v)
                del obj[k]
            elif isinstance(v, dict):
                del obj[k]
            elif isinstance(v, list):
                obj[k] = [x for x in v if x]
        return obj


def score_prediction(
    predicted: Union[Dict, List], expected: Union[Dict, List], exclude: List = None
) -> Outcome:
    """
    Score the predicted activity.

    >>> outcome = score_prediction({"x": 1}, {"x": 1})
    >>> outcome.tp
    1

    >>> outcome = score_prediction([{"x": 1}], {"x": 1})
    >>> outcome.tp
    1


    >>> outcome = score_prediction({"x": 1}, {"x": 2})
    >>> outcome.tp
    0
    >>> outcome.recall
    0.0

    >>> outcome = score_prediction({"x": 1, "y": 2}, {"x": 1})
    >>> outcome.tp
    1
    >>> outcome.fp
    1

    >>> outcome = score_prediction([{"x": 1}, {"y": 1}], {"x": 1})
    >>> outcome.tp
    1
    >>> outcome.fp
    1


    :param predicted: The predicted activity
    :param expected: The expected activity
    :return: The score
    """
    if exclude is None:
        exclude = ["reference_title", "reference"]
    if isinstance(expected, list) or isinstance(predicted, list):
        if isinstance(expected, dict):
            expected = [expected]
        if isinstance(predicted, dict):
            predicted = [predicted]
        outcomes = best_matches(predicted, expected)
        outcome = Outcome(prediction=predicted, expected=expected)
        for sub_outcome in outcomes:
            outcome.tp += sub_outcome.tp
            outcome.fp += sub_outcome.fp
            outcome.fn += sub_outcome.fn
            for key, value in sub_outcome.by_field.items():
                outcome.by_field[key] = outcome.by_field.get(key, 0) + value
            for key, value in sub_outcome.ixn_by_field.items():
                outcome.ixn_by_field[key] = list(
                    set(outcome.ixn_by_field.get(key, [])).union(value)
                )
        outcome.calculate_metrics()
        return outcome
    outcome = Outcome(prediction=predicted, expected=expected)
    all_keys = set(predicted.keys()).union(expected.keys()).difference(exclude)
    for key in all_keys:
        if key in predicted and key in expected:
            if key == "relationships":
                pred_rels = predicted[key]
                exp_rels = expected[key]
                sub_outcomes = best_matches(pred_rels, exp_rels)
                n_tps = 0
                ixn = set()
                for sub_outcome in sub_outcomes:
                    outcome.tp += sub_outcome.tp
                    outcome.fp += sub_outcome.fp
                    outcome.fn += sub_outcome.fn
                    n_tps += sub_outcome.tp
                    if sub_outcome.precision == 1.0:
                        ixn = ixn.union({str(predicted[key])})
                outcome.by_field[key] = outcome.by_field.get(key, 0) + n_tps
                outcome.ixn_by_field[key] = list(set(outcome.ixn_by_field.get(key, [])).union(ixn))
                continue
            if predicted[key] == expected[key]:
                outcome.tp += 1
                outcome.by_field[key] = outcome.by_field.get(key, 0) + 1
                outcome.ixn_by_field[key] = list(
                    set(outcome.ixn_by_field.get(key, [])).union({predicted[key]})
                )
            else:
                outcome.fp += 1
                outcome.fn += 1
        elif key in predicted:
            outcome.fp += 1
        else:
            outcome.fn += 1
    outcome.calculate_metrics()
    return outcome


def best_matches(pred_rels, exp_rels) -> List[Outcome]:
    """
    Find the best matching pairs of relationships.

    Example:

    >>> outcomes = best_matches([], [])
    >>> len(outcomes)
    1
    >>> outcome = outcomes[0]
    >>> (outcome.tp, outcome.fp, outcome.fn)
    (0, 0, 0)
    >>> best_matches([{"x:": 1}], [])[0].precision
    0.0
    >>> outcome = best_matches([{"x": 1}], [{"x": 1}])[0]
    >>> outcome.precision
    1.0
    >>> outcome = best_matches([{"x": 1}], [{"y": 1}])[0]
    >>> outcome.precision
    0.0
    >>> pred_rels = [{"x":1}, {"y": 2}, {"z": 3}]
    >>> exp_rels = [{"y":2}, {"x": 1}, {"z": 3}]
    >>> outcomes = best_matches(pred_rels, exp_rels)
    >>> [o.precision for o in outcomes]
    [1.0, 1.0, 1.0]
    >>> exp_rels.append({"z": 4})
    >>> outcomes = best_matches(pred_rels, exp_rels)
    >>> sorted([o.precision for o in outcomes])
    [0.0, 1.0, 1.0, 1.0]

    """
    import numpy as np

    if not pred_rels:
        pred_rels = [{}]
    if not exp_rels:
        exp_rels = [{}]

    # Create a matrix to store the scores
    outcome_matrix = np.zeros((len(pred_rels), len(exp_rels)), dtype=object)
    outcome_ix = {}

    # Calculate the scores for each pair of pred_rel and exp_rel
    for i, pred_rel in enumerate(pred_rels):
        for j, exp_rel in enumerate(exp_rels):
            sub_outcome = score_prediction(pred_rel, exp_rel)
            outcome_matrix[i, j] = sub_outcome.tp
            outcome_ix[(i, j)] = sub_outcome

    # Find the best matching pairs
    outcomes = []
    max_row_indices = np.argmax(outcome_matrix, axis=1)
    max_col_indices = np.argmax(outcome_matrix, axis=0)
    best = []
    for i, _pred_rel in enumerate(pred_rels):
        best_j = max_row_indices[i]
        best.append((i, best_j))
        outcomes.append(outcome_ix[(i, best_j)])
    for j, _exp_rel in enumerate(exp_rels):
        best_i = max_col_indices[j]
        if (best_i, j) not in best:
            best.append((best_i, j))
            outcomes.append(outcome_ix[(best_i, j)])
    return outcomes
