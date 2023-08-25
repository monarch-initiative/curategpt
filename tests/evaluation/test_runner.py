import pytest
import yaml

from curate_gpt.evaluation.evaluation_datamodel import Task
from curate_gpt.evaluation.runner import run_task
from tests import OUTPUT_DIR


@pytest.mark.parametrize(
    "num_training,num_testing,fields_to_predict,fields_to_mask",
    [
        (20, 4, ["label"], ["id"]),
        (20, 4, ["definition"], []),
        (20, 4, ["relationships"], []),
    ],
)
def test_runner(loaded_ontology_db, num_training, num_testing, fields_to_predict, fields_to_mask):
    task = Task(
        source_db_path=loaded_ontology_db.path,
        # target_db_path="test_train",
        source_collection="terms_go",
        num_training=num_training,
        num_testing=num_testing,
        fields_to_predict=fields_to_predict,
        fields_to_mask=fields_to_mask,
        working_directory=str(OUTPUT_DIR / "runner_test"),
    )
    results = run_task(task)
    print(yaml.dump(results.dict(), sort_keys=False))
