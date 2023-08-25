import pytest

from curate_gpt.evaluation.calc_statistics import (
    aggregate_metrics,
    calculate_metrics,
    evaluate_predictions,
)
from curate_gpt.evaluation.evaluation_datamodel import ClassificationOutcome

all_metrics = []


@pytest.mark.parametrize(
    "obj1, obj2, tps, fps, fns",
    [
        (1, 1, 1, 0, 0),
        (1, 2, 0, 1, 1),
        (["x"], ["x"], 1, 0, 0),
        (["x"], ["y"], 0, 1, 1),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5, 0, 0),
        (["x", "y"], ["y"], 1, 1, 0),
        ({}, {}, 1, 0, 0),
        ({"x": 1}, {}, 0, 1, 1),
        ([{"x": 1}], [{}], 0, 1, 1),
        ([{"x": 1, "y": 2}], [{"y": 2, "x": 1}], 1, 0, 0),
    ],
)
def test_simple_compare(obj1, obj2, tps, fps, fns):
    print(obj1, obj2)
    outcomes = [outcome for outcome, _ in evaluate_predictions(obj1, obj2)]
    atp = outcomes.count(ClassificationOutcome.TRUE_POSITIVE)
    afp = outcomes.count(ClassificationOutcome.FALSE_POSITIVE)
    afn = outcomes.count(ClassificationOutcome.FALSE_NEGATIVE)
    assert atp == tps
    assert afp == fps
    assert afn == fns
    metrics = calculate_metrics(outcomes)
    print(metrics)
    all_metrics.append(metrics)
    aggregated = aggregate_metrics(all_metrics)
    print(aggregated)
