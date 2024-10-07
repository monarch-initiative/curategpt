from typing import Any, Iterator, List, Tuple, Union

import yaml

from curategpt.evaluation.evaluation_datamodel import (
    AggregationMethod,
    ClassificationMetrics,
    ClassificationOutcome,
)


def calculate_metrics(
    outcomes: List[Union[ClassificationOutcome, Tuple[ClassificationOutcome, Any]]]
) -> ClassificationMetrics:
    outcomes = [
        outcome if isinstance(outcome, ClassificationOutcome) else outcome[0]
        for outcome in outcomes
    ]
    tp = outcomes.count(ClassificationOutcome.TRUE_POSITIVE)
    tn = outcomes.count(ClassificationOutcome.TRUE_NEGATIVE)
    fp = outcomes.count(ClassificationOutcome.FALSE_POSITIVE)
    fn = outcomes.count(ClassificationOutcome.FALSE_NEGATIVE)

    # Avoid division by zero
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        specificity=specificity,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )


def _normalize(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return yaml.safe_dump(obj, sort_keys=True)
    elif isinstance(obj, list):
        return yaml.safe_dump(obj, sort_keys=True)
    else:
        return str(obj)


def evaluate_predictions(obj1: Any, obj2: Any) -> Iterator[Tuple[ClassificationOutcome, str]]:
    """
    Evaluate a prediction compared to an expected value.

    Where the prediction and the expected value are lists, the results are each
    true positive, true negative, false positive.

    Where the prediction and the expected value are scalars, these are treated as if
    they are lists, thus a correct prediction is a true positive, and no false positives
    or negatives; an incorrect prediction is a false positive and a false negative.

    :param obj1:
    :param obj2:
    :return:
    """
    if isinstance(obj1, list) and isinstance(obj2, list):
        set1 = {_normalize(obj) for obj in obj1}
        set2 = {_normalize(obj) for obj in obj2}
        for x in set1.union(set2):
            if x not in set1:
                yield ClassificationOutcome.FALSE_NEGATIVE, f"{x} in {set2}"
            elif x not in set2:
                yield ClassificationOutcome.FALSE_POSITIVE, f"{x} in {set1}"
            else:
                yield ClassificationOutcome.TRUE_POSITIVE, f"{x} in both"
    else:
        yield from evaluate_predictions([obj1] if obj1 else [], [obj2] if obj2 else [])


def aggregate_metrics(
    metrics_list: List[ClassificationMetrics], method: AggregationMethod = AggregationMethod.MACRO
):
    """
    Aggregate a list of metrics.

    Note that if the evaluation task is for a single labels rather than lists,
    then this is trivially just the proportion of correct predictions.

    :param metrics_list:
    :param method:
    :return:
    """
    if method == AggregationMethod.MACRO:
        return ClassificationMetrics(
            precision=sum(m.precision for m in metrics_list) / len(metrics_list),
            recall=sum(m.recall for m in metrics_list) / len(metrics_list),
            f1_score=sum(m.f1_score for m in metrics_list) / len(metrics_list),
            accuracy=sum(m.accuracy for m in metrics_list) / len(metrics_list),
            specificity=sum(m.specificity for m in metrics_list) / len(metrics_list),
            true_positives=sum(
                m.true_positives for m in metrics_list if m.true_positives is not None
            ),
            true_negatives=sum(
                m.true_negatives for m in metrics_list if m.true_negatives is not None
            ),
            false_positives=sum(
                m.false_positives for m in metrics_list if m.false_positives is not None
            ),
            false_negatives=sum(
                m.false_negatives for m in metrics_list if m.false_negatives is not None
            ),
        )
    elif method == AggregationMethod.MICRO:
        total_tp = sum(m.precision * (m.recall * (m.precision + m.f1_score)) for m in metrics_list)
        total_fp = sum(m.f1_score - m.precision * m.recall for m in metrics_list)
        total_fn = sum((1 - m.recall) * (m.precision + m.f1_score) for m in metrics_list)
        total_tn = sum(
            m.accuracy * (m.precision + m.recall + m.f1_score + 1) - total_tp - total_fp - total_fn
            for m in metrics_list
        )

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        specificity = total_tn / (total_tn + total_fp)

        return ClassificationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            specificity=specificity,
        )
    elif method == AggregationMethod.WEIGHTED:
        total_weight = sum(
            m.precision + m.recall + m.f1_score + m.accuracy + m.specificity for m in metrics_list
        )
        return ClassificationMetrics(
            precision=sum(
                m.precision * (m.precision + m.recall + m.f1_score + m.accuracy + m.specificity)
                for m in metrics_list
            )
            / total_weight,
            recall=sum(
                m.recall * (m.precision + m.recall + m.f1_score + m.accuracy + m.specificity)
                for m in metrics_list
            )
            / total_weight,
            f1_score=sum(
                m.f1_score * (m.precision + m.recall + m.f1_score + m.accuracy + m.specificity)
                for m in metrics_list
            )
            / total_weight,
            accuracy=sum(
                m.accuracy * (m.precision + m.recall + m.f1_score + m.accuracy + m.specificity)
                for m in metrics_list
            )
            / total_weight,
            specificity=sum(
                m.specificity * (m.precision + m.recall + m.f1_score + m.accuracy + m.specificity)
                for m in metrics_list
            )
            / total_weight,
        )
    else:
        raise ValueError("Invalid aggregation method")
