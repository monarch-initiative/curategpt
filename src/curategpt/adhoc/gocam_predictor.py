"""GO-CAM predictor class."""

import json
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict

import click
import yaml

from curategpt import BasicExtractor, DBAdapter
from curategpt.store import get_store
from curategpt.utils.eval_utils import Outcome, score_prediction
from curategpt.utils.llm_utils import query_model
from curategpt.wrappers.bio.gocam_wrapper import GOCAMWrapper

logger = logging.getLogger(__name__)


@dataclass
class GOCAMPredictor:

    store: DBAdapter = None
    database_type: str = field(default="chromadb")
    database_path: str = field(default="gocam")
    collection_name: str = field(default="gocam")
    model_name: str = field(default="gpt-4o")
    extractor: BasicExtractor = None
    include_standard_annotations: bool = False
    gocam_wrapper: GOCAMWrapper = field(default_factory=lambda: GOCAMWrapper())
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

    def predict_activity_unit(
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
            response = query_model(model, prompt=prompt, system=system)
            # response = model.prompt(prompt=prompt, system=system)
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
    """
    Predict GO-CAMs.

    Example:

        gcpr -v  predict -N 5 --model-name claude-3-opus  -o results/gocam-run-claude3-opus-N5-run1.yaml

    We try and defend against API failures, e.g. implementing exponential backoff, but there is still
    potential for runs to fail, so we allow `--prior-results-file` to start off where we left off.

    Example (initial run):

        gcpr -v  predict -N 5 -o results/gocam-run-N5-run1.yaml

    After failure:

        gcpr -v  predict -N 5 --prior-results-file results/gocam-run-N5-run1.yaml -o results/gocam-run-N5-run1.yaml

    """
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
    # set the list of GO-CAMs to predict; may be a specified singleton or ALL go-cams
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
            # the gocam model used here is simplified in that each activity has a single ref
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
            try:
                result = predictor.predict_activity_unit(
                    test_gocam, {"reference": pmid}, num_examples=num_examples
                )
            except Exception as e:
                logger.error(f"Encountered error: {e}")
                continue
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
                "include_standard_annotations": include_standard_annotations,
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
    """
    Summarize the results of a previous run through gocam-predict.

    Takes the YAML output of the prediction step.
    """
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

    # read in results file YAML, flattening outcomes
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
