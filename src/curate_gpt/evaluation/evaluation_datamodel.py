from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class StratifiedCollection(BaseModel):
    """
    A collection of objects that have been split into training, test, and validation sets.
    """

    source: str = None
    training_set: List[Dict] = None
    testing_set: List[Dict] = None
    validation_set: Optional[List[Dict]] = None
    training_set_collection: Optional[str] = None
    testing_set_collection: Optional[str] = None
    validation_set_collection: Optional[str] = None


class ClassificationOutcome(str, Enum):
    TRUE_POSITIVE = "True Positive"
    TRUE_NEGATIVE = "True Negative"
    FALSE_POSITIVE = "False Positive"
    FALSE_NEGATIVE = "False Negative"


class AggregationMethod(str, Enum):
    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"


class ClassificationMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    true_positives: int = (None,)
    true_negatives: int = (None,)
    false_positives: int = (None,)
    false_negatives: int = (None,)


class Task(BaseModel):
    """
    A task to be run by the evaluation runner.
    """

    model_name: str = "gpt-3.5-turbo"
    embedding_model_name: str = "openai:"
    generate_background: bool = False
    task_started: Optional[str] = None
    task_finished: Optional[str] = None
    executed_on: Optional[str] = None
    agent: Optional[str] = "dae"
    extractor: Optional[str] = "BasicExtractor"
    method: Optional[str] = None
    source_db_path: Optional[str] = None
    target_db_path: Optional[str] = None
    source_collection: Optional[str] = None
    additional_collections: Optional[List[str]] = None
    num_training: int = None
    num_testing: int = None
    num_validation: int = 0
    stratified_collection: Optional[StratifiedCollection] = None
    fields_to_mask: Optional[List[str]] = None
    fields_to_predict: Optional[List[str]] = None
    report_path: Optional[str] = None
    working_directory: Optional[Union[Path, str]] = None
    results: Optional[ClassificationMetrics] = None

    @property
    def id(self) -> str:
        pred = ".".join(self.fields_to_predict) if self.fields_to_predict else ""
        mask = ".".join(self.fields_to_mask) if self.fields_to_mask else ""
        model = self.model_name.replace(":", "")
        em_model = self.embedding_model_name.replace(":", "")
        id = (
            f"{self.source_collection}-P{pred}-M{mask}-BG{self.generate_background}-"
            f"Tr{self.num_training}-Te{self.num_testing}-M{model}-EM{em_model}"
        )
        if self.method:
            id += f"-Mth{self.method}"
        if self.additional_collections:
            id += f"-AC{'-'.join(self.additional_collections)}"
        return id
