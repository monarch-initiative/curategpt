import json
import logging
import sys
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import click
import yaml
from openai import BaseModel

from curate_gpt import BasicExtractor, DBAdapter
from curate_gpt.store import get_store
from curate_gpt.wrappers.bio.gocam_wrapper import GOCAMWrapper

logger = logging.getLogger(__name__)


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


@dataclass
class GOCAMPredictor:

    store: DBAdapter = None
    database_type: str = field(default="chromadb")
    database_path: str = field(default="gocam")
    collection_name: str = field(default="gocam")
    model_name: str = field(default="gpt-4o")
    extractor: BasicExtractor = None
    include_standard_annotations: bool = False
    gocam_wrapper: GOCAMWrapper = field(default=GOCAMWrapper())
    strict: bool = field(default=False)

    def __post_init__(self):
        self.store = get_store(self.database_type, self.database_path)
        self.extractor = BasicExtractor(model_name=self.model_name)
        self.gocam_wrapper = GOCAMWrapper(
            include_standard_annotations=self.include_standard_annotations
        )

    def gocam_by_id(self, id: str) -> Dict[str, Any]:
        return self.gocam_wrapper.object_by_id(id)
        # objs = list(self.store.find({"id": id}, collection=self.collection_name))
        # if len(objs) == 0:
        #    raise ValueError(f"GO-CAM model not found: {id}")
        # return objs[0][0]

    def predict(
        self, gocam: Dict[str, Any], stub: Dict[str, Any], num_examples=0
    ) -> Dict[str, Any]:
        """
        Predict the missing activity in a GO-CAM model.
        :param gocam: The GO-CAM model
        :param stub: The stub to fill in
        :param num_examples: The number of examples to generate
        :return: The predicted activity
        """
        model = self.extractor.model
        gocam_yaml = yaml.dump(gocam, sort_keys=False)
        examples = []
        if num_examples:
            for result, _score, _meta in self.store.search(
                gocam_yaml, limit=num_examples + 1, collection=self.collection_name
            ):
                if result["id"] == gocam["id"]:
                    continue
                examples.append(result)
            logger.info(f"Found {len(examples)} examples")

        stub_yaml = yaml.dump(stub, sort_keys=False)
        system = """
        You are an expert pathway curator. Your job is to provide a single
        additional activity for the provided GO-CAM model.
        """
        if examples:
            system += "Here are some examples other GO-CAM models:\n"
            for example in examples:
                system += f"""
                ```yaml
                {yaml.dump(example, sort_keys=False)}
                ```
                """
        prompt = f"""
        I will first provide the GO-CAM as YAML:

        ```yaml
        {gocam_yaml}
        ```

        Your job is to fill in this stub:

        ```yaml
        {stub_yaml}
        ```

        Provide YAML *ONLY* for this activity. DO NOT provide or correct
        YAML for existing activities. Try and include all relevant fields:
        gene, activity, process, and relationships.

        Example:

        ```yaml
        gene: <GeneName>
        activity: <MolecularActivityName>
        process: <BiologicalProcessName>
        location: <CellularLocationName>
        relationships:
          - type: <PredicateName>
            target_gene: <TargetGeneName>
            target_activity: <TargetMolecularActivityName>
        ```

        In your response, enclose the YAML in triple backticks.
        """
        logger.debug(f"System: {system}")
        logger.debug(f"Prompt: {prompt}")
        if self.model_name.startswith("gemini"):
            response = model.prompt(prompt=system + "\n===\n" + prompt)
        else:
            response = model.prompt(prompt=prompt, system=system)
        text = response.text()
        toks = text.split("```")
        if len(toks) < 3:
            logger.error(f"Invalid response: {text}")
            return {}
        yaml_part = toks[1]
        if yaml_part.startswith("yaml"):
            yaml_part = yaml_part[4:]
        try:
            obj = yaml.safe_load(yaml_part)
        except Exception as e:
            obj = self.fix_yaml(yaml_part, e)
        try:
            Outcome(prediction=obj)
        except:
            logger.error(f"Invalid response: {obj}")
            return {}
        return obj

    def fix_yaml(self, yaml_text: str, e: Exception) -> str:
        """
        Fix the YAML text.

        :param yaml_text: The YAML text
        :return: The fixed YAML text
        """
        model = self.extractor.model
        response = model.prompt(
            system="""You provided this YAML but it doesn't parse. Return in JSON instead.
                       Only return valid JSON in payload, notyhing else""",
            prompt=f"```yaml\n{yaml_text}``` {e}",
        )
        text = response.text()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            if self.strict:
                raise ValueError(f"Invalid JSON response: {text}") from e
            else:
                obj = {}
        return obj


database_type_option = click.option(
    "--database_type", default="chromadb", help="The type of database to use"
)
database_path_option = click.option(
    "--database_path", default="gocam", help="The path to the database"
)
collection_name_option = click.option(
    "--collection_name", default="gocam", help="The name of the collection"
)
model_name_option = click.option("--model-name", default="gpt-4o", help="The name of the model")
num_examples_option = click.option(
    "--num-examples",
    "-N",
    type=click.INT,
    default=0,
    show_default=True,
    help="Number of RAG examples to generate",
)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
def main(verbose: int, quiet: bool):
    """
    CLI for GO-CAM prediction

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    # logger = logging.getLogger()
    logging.basicConfig()
    logger = logging.root
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)
    logger.info(f"Logger {logger.name} set to level {logger.level}")


@main.command()
@database_type_option
@database_path_option
@collection_name_option
@num_examples_option
@model_name_option
@click.option("--gocam-id", "-x", help="The GO-CAM model ID")
@click.option("--pmid", "-P", help="PMID to use for prediction")
@click.option("--output", "-o", help="Output file")
@click.option(
    "--include-standard-annotations/--no-include-standard-annotations",
    default=False,
    show_default=True,
    help="Include standard annotations",
)
@click.option("--prior-results-file", help="Previous results")
def predict(
    gocam_id,
    pmid,
    model_name,
    database_type,
    database_path,
    collection_name,
    num_examples,
    output,
    include_standard_annotations,
    prior_results_file,
):
    already_done = []
    if prior_results_file:
        with open(prior_results_file, "r", encoding="utf-8") as f:
            already_done = [Outcome(**x).parameters["gocam_id"] for x in yaml.safe_load_all(f)]
            logger.info(f"Loaded {len(already_done)} already done")
    if output:
        output_file = open(output, "w", encoding="utf-8")
    else:
        output_file = sys.stdout
    predictor = GOCAMPredictor(
        database_type=database_type,
        database_path=database_path,
        collection_name=collection_name,
        model_name=model_name,
        include_standard_annotations=include_standard_annotations,
    )
    if gocam_id:
        gocam_ids = [gocam_id]
    else:
        if pmid:
            raise ValueError("PMID requires a GO-CAM ID")
        gocam_ids = [
            x["id"]
            for x, _s, _m in predictor.store.find({}, collection=collection_name, limit=9999)
        ]
        logger.info(f"Found {len(gocam_ids)} GO-CAM models")
    outcomes = []
    orig_pmid = pmid
    for gocam_id in gocam_ids:
        if gocam_id in already_done:
            logger.info(f"Skipping {gocam_id}, in already done list")
            continue
        original_gocam = predictor.gocam_by_id(gocam_id)
        title = original_gocam.get("title", None)
        test_gocam = deepcopy(original_gocam)
        if orig_pmid:
            pmids = [orig_pmid]
        else:
            pmids = [a["reference"] for a in test_gocam["activities"]]
        main_outcome = Outcome(prediction=[], expected=[])
        for pmid in pmids:
            refs = [a["reference"] for a in test_gocam["activities"]]
            test_gocam["activities"] = [
                a for a in original_gocam["activities"] if a["reference"] != pmid
            ]
            expected = [a for a in original_gocam["activities"] if a["reference"] == pmid]
            if not expected:
                raise ValueError(
                    f"Activity with reference {pmid} not found in GO-CAM model {gocam_id} // {refs}"
                )
            if len(test_gocam["activities"]) >= len(original_gocam["activities"]):
                raise ValueError(f"Invalid number of activities in GO-CAM model {gocam_id}")
            result = predictor.predict(test_gocam, {"reference": pmid}, num_examples=num_examples)
            print("## PREDICTION")
            print(yaml.dump(result, sort_keys=False))
            print("## EXPECTED")
            print(yaml.dump(expected, sort_keys=False))
            outcome = score_prediction(result, expected)
            outcome.parameters = {
                "gocam_id": gocam_id,
                "title": title,
                "pmid": pmid,
                "num_examples": num_examples,
                "model_name": predictor.model_name,
            }
            print("## OUTCOME")
            print(
                yaml.dump(
                    outcome.model_dump(
                        exclude={"predicted", "expected", "has_standard_annotation"}
                    ),
                    sort_keys=False,
                )
            )
            output_file.write("---\n")
            output_file.write(yaml.dump(outcome.model_dump(), sort_keys=False))
            output_file.write("\n")
            output_file.flush()
            outcomes.append(outcome)
    if len(outcomes) > 1:
        main_outcome.append_outcomes(outcomes)
        print("## MAIN OUTCOME")
        print(yaml.dump(main_outcome.model_dump(), sort_keys=False))
    output_file.close()


@main.command()
@click.option("--output", "-o", help="Output file")
@click.option("--tsv", help="TSV Output file")
@click.option("--add-species/--no-add-species", default=False, help="Add species to the output")
@click.argument("results_file")
def summarize_results(results_file, tsv, output, add_species):
    wrapper = GOCAMWrapper()
    species_by_gocam_id = {}

    def _get_species(id: str):
        if id not in species_by_gocam_id:
            gocam = wrapper.object_by_id(id)
            species = gocam.get("species", None)
            if species:
                species_by_gocam_id[id] = species
            else:
                species_by_gocam_id[id] = "unknown"
            logger.debug(f"Getting species for {id} => {species_by_gocam_id[id]}")
        return species_by_gocam_id[id]

    with open(results_file, "r", encoding="utf-8") as f:

        def _outcome(x: dict):
            outcome = Outcome(**x)
            params = outcome.parameters
            if add_species:
                params["species"] = _get_species(params["gocam_id"])
            logger.debug(f"Loaded outcome: {yaml.dump(outcome.model_dump)}")
            return outcome

        outcomes = [_outcome(x) for x in yaml.safe_load_all(f)]
    logger.info(f"Loaded {len(outcomes)} outcomes")
    rows = [outcome.flatten() for outcome in outcomes]
    if tsv:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(tsv, index=False, sep="\t")
        print(df.describe())
    if output:
        with open(output, "w", encoding="utf-8") as f:
            yaml.dump_all([outcome.model_dump() for outcome in outcomes], f, sort_keys=False)
    main_outcome = Outcome(prediction=[], expected=[])
    for outcome in outcomes:
        outcome.by_field = {}
        outcome.ixn_by_field = {}
    main_outcome.append_outcomes(outcomes)
    print(yaml.dump(main_outcome.model_dump(), sort_keys=False))


# add footer boilerplate for click
if __name__ == "__main__":
    main()
