"""Command line interface for curate-gpt."""
import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd
import yaml
from click_default_group import DefaultGroup
from llm import UnknownModelError, get_model, get_plugins
from llm.cli import load_conversation
from oaklib import get_adapter
from pydantic import BaseModel

from curate_gpt import ChromaDBAdapter, __version__
from curate_gpt.agents.chat_agent import ChatAgent, ChatResponse
from curate_gpt.agents.dac_agent import DatabaseAugmentedCompletion
from curate_gpt.agents.dase_agent import DatabaseAugmentedStructuredExtraction
from curate_gpt.agents.evidence_agent import EvidenceAgent
from curate_gpt.evaluation.dae_evaluator import DatabaseAugmentedCompletionEvaluator
from curate_gpt.evaluation.evaluation_datamodel import StratifiedCollection, Task
from curate_gpt.evaluation.runner import run_task
from curate_gpt.evaluation.splitter import stratify_collection
from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.wrappers import BaseWrapper, get_wrapper
from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper
from curate_gpt.wrappers.ontology import OntologyWrapper

__all__ = [
    "main",
]

# logger = logging.getLogger(__name__)

path_option = click.option("-p", "--path", help="Path to a file or directory for database.")
model_option = click.option("-m", "--model", help="Model to use for generation, e.g. gpt-4.")
schema_option = click.option("-s", "--schema", help="Path to schema.")
collection_option = click.option("-c", "--collection", help="Collection within the database.")
relevance_factor_option = click.option(
    "--relevance-factor", type=click.FLOAT, help="Relevance factor for search."
)
generate_background_option = click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
limit_option = click.option(
    "-l", "--limit", default=10, show_default=True, help="Number of results to return."
)
replace_option = click.option(
    "--replace/--no-replace",
    default=False,
    show_default=True,
    help="replace the database before indexing.",
)
append_option = click.option(
    "--append/--no-append", default=False, show_default=True, help="Append to the database."
)
object_type_option = click.option(
    "--object-type",
    default="Thing",
    show_default=True,
    help="Type of object in index.",
)
description_option = click.option(
    "--description",
    help="Description of the collection.",
)
init_with_option = click.option(
    "--init-with",
    "-I",
    help="YAML string for initialzation of main wrapper object.",
)
batch_size_option = click.option(
    "--batch-size", default=None, show_default=True, type=click.INT, help="Batch size for indexing."
)


def show_chat_response(response: ChatResponse, show_references: bool = True):
    """Show a chat response."""
    print("# Response:\n")
    click.echo(response.formatted_body)
    print("\n\n# Raw:\n")
    click.echo(response.body)
    if show_references:
        print("\n# References:\n")
        for ref, ref_text in response.references.items():
            print(f"\n## {ref}\n")
            print("```yaml")
            print(ref_text)
            print("```")
        print("# Uncited:")
        for ref, ref_text in response.uncited_references.items():
            print(f"\n## {ref}\n")
            print("```yaml")
            print(ref_text)
            print("```")


@click.group(
    cls=DefaultGroup,
    default="search",
    default_if_no_args=True,
)
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """CLI for curate-gpt.

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
@path_option
@append_option
@collection_option
@model_option
@click.option("--text-field")
@object_type_option
@description_option
@click.option(
    "--view",
    "-V",
    help="View/Proxy to use for the database, e.g. bioc.",
)
@click.option(
    "--glob/--no-glob", default=False, show_default=True, help="Whether to glob the files."
)
@click.option(
    "--collect/--no-collect", default=False, show_default=True, help="Whether to collect files."
)
@batch_size_option
@click.argument("files", nargs=-1)
def index(
    files,
    path,
    append: bool,
    text_field,
    collection,
    model,
    object_type,
    description,
    batch_size,
    glob,
    view,
    collect,
    **kwargs,
):
    """Index files.

    Example:

        curategpt index  -c doc files/*json

    """
    db = ChromaDBAdapter(path, **kwargs)
    db.text_lookup = text_field
    if glob:
        files = [str(gf) for f in files for gf in Path().glob(f)]
    if view:
        wrapper = get_wrapper(view)
        if not object_type:
            object_type = wrapper.default_object_type
        if not description:
            description = f"{object_type} objects loaded from {str(files)[0:30]}"
    else:
        wrapper = None
    if collect:
        raise NotImplementedError
    if not append:
        if collection in db.list_collection_names():
            db.remove_collection(collection)
    for file in files:
        logging.debug(f"Indexing {file}")
        if wrapper:
            wrapper.source_locator = file
            objs = wrapper.objects()  # iterator
        elif file.endswith(".json"):
            objs = json.load(open(file))
        elif file.endswith(".csv"):
            objs = list(csv.DictReader(open(file)))
        elif file.endswith(".tsv.gz"):
            with gzip.open(file, "rt") as f:
                objs = list(csv.DictReader(f, delimiter="\t"))
        elif file.endswith(".tsv"):
            objs = list(csv.DictReader(open(file), delimiter="\t"))
        else:
            objs = yaml.safe_load(open(file))
        if isinstance(objs, (dict, BaseModel)):
            objs = [objs]
        db.insert(objs, model=model, collection=collection, batch_size=batch_size)
    db.update_collection_metadata(
        collection, model=model, object_type=object_type, description=description
    )


@main.command(name="search")
@path_option
@collection_option
@limit_option
@relevance_factor_option
@click.option(
    "--show-documents/--no-show-documents",
    default=False,
    show_default=True,
    help="Whether to show documents/text (e.g. for chromadb).",
)
@click.argument("query")
def search(query, path, collection, show_documents, **kwargs):
    """Query a database."""
    db = ChromaDBAdapter(path)
    results = db.search(query, collection=collection, **kwargs)
    i = 0
    for obj, distance, meta in results:
        i += 1
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))
        if show_documents:
            print("```")
            print(meta)
            print("```")


@main.command()
@path_option
@collection_option
@click.argument("id")
def matches(id, path, collection):
    """Find matches for an ID."""
    db = ChromaDBAdapter(path)
    # TODO: persist this in the database
    db.text_lookup = "label"
    obj = db.lookup(id, collection=collection)
    print(obj)
    results = db.matches(obj, collection=collection)
    i = 0
    for obj, distance in results:
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))


@main.command()
@path_option
@collection_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@schema_option
@click.option(
    "--input",
    "-i",
    default=None,
    help="Input file to extract.",
)
@click.argument("text", nargs=-1)
def extract(
    text,
    input,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    schema,
    **kwargs,
):
    """Extract."""
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    agent = DatabaseAugmentedStructuredExtraction(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        agent.document_adapter = ChromaDBAdapter(docstore_path)
        agent.document_adapter_collection = docstore_collection
    if not text:
        if not input:
            raise ValueError("Must provide either text or input file.")
        text = list(open(input).readlines())
    text = "\n".join(text)
    ao = agent.extract(text, rules=rule, **filtered_kwargs)
    print(yaml.dump(ao.object, sort_keys=False))


@main.command()
@path_option
@collection_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--output-directory",
    "-o",
    required=True,
)
@schema_option
@click.option(
    "--pubmed-id-file",
)
@click.argument("ids", nargs=-1)
def extract_from_pubmed(
    ids,
    pubmed_id_file,
    output_directory,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    schema,
    **kwargs,
):
    """Extract."""
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    agent = DatabaseAugmentedStructuredExtraction(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        agent.document_adapter = ChromaDBAdapter(docstore_path)
        agent.document_adapter_collection = docstore_collection
    if not ids:
        if not pubmed_id_file:
            raise ValueError("Must provide either text or input file.")
        ids = [x.strip() for x in open(pubmed_id_file).readlines()]
    pmw = PubmedWrapper()
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    for pmid in ids:
        pmid_esc = pmid.replace(":", "_")
        text = pmw.fetch_full_text(pmid)
        ao = agent.extract(text, rules=rule, **filtered_kwargs)
        with open(output_directory / f"{pmid_esc}.yaml", "w") as f:
            f.write(yaml.dump(ao.object, sort_keys=False))
        with open(output_directory / f"{pmid_esc}.txt", "w") as f:
            f.write(text)


@main.command()
@path_option
@collection_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@schema_option
@click.argument("query")
def complete(
    query,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    query_property,
    schema,
    **kwargs,
):
    """Generate an entry from a query using object completion.

    Example:

        curategpt generate  -c obo_go "umbelliferose biosynthetic process"
    """
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dac = DatabaseAugmentedCompletion(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dac.document_adapter = ChromaDBAdapter(docstore_path)
        dac.document_adapter_collection = docstore_collection
    if ":" in query:
        query = yaml.safe_load(query)
    ao = dac.complete(query, context_property=query_property, rules=rule, **filtered_kwargs)
    print(yaml.dump(ao.object, sort_keys=False))


@main.command()
@path_option
@collection_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option("--field-to-predict", "-F", help="Field to predict")
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--id-file",
    "-i",
    type=click.File("r"),
    help="File to read ids from.",
)
@click.option(
    "--missing-only/--no-missing-only",
    default=True,
    show_default=True,
    help="Only generate missing values.",
)
@schema_option
def complete_all(
    path,
    collection,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    field_to_predict,
    schema,
    id_file,
    **kwargs,
):
    """Generate missing values for all objects

    Example:

        curategpt generate  -c obo_go TODO
    """
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dae = DatabaseAugmentedCompletion(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dae.document_adapter = ChromaDBAdapter(docstore_path)
        dae.document_adapter_collection = docstore_collection
    object_ids = None
    if id_file:
        object_ids = [line.strip() for line in id_file.readlines()]
    it = dae.generate_all(
        collection=collection,
        field_to_predict=field_to_predict,
        rules=rule,
        object_ids=object_ids,
        **filtered_kwargs,
    )
    for pred in it:
        print(yaml.dump(pred.dict(), sort_keys=False))


@main.command()
@path_option
@collection_option
@click.option("--test-collection", "-T", required=True, help="Collection to use as the test set")
@click.option(
    "--hold-back-fields",
    "-F",
    required=True,
    help="Comma separated list of fields to predict in the test.",
)
@click.option(
    "--mask-fields",
    "-M",
    help="Comma separated list of fields to mask in the test.",
)
@model_option
@limit_option
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--report-file",
    "-o",
    type=click.File("w"),
    help="File to write report to.",
)
@click.option(
    "--num-tests",
    default=None,
    show_default=True,
    help="Number (max) of tests to run.",
)
@schema_option
def generate_evaluate(
    path,
    docstore_path,
    docstore_collection,
    model,
    schema,
    test_collection,
    num_tests,
    hold_back_fields,
    mask_fields,
    rule: List[str],
    **kwargs,
):
    """Evaluate generate using a test set.

    Example:

        curategpt -v generate-evaluate -c cdr_training -T cdr_test -F statements -m gpt-4
    """
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    rage = DatabaseAugmentedCompletion(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        rage.document_adapter = ChromaDBAdapter(docstore_path)
        rage.document_adapter_collection = docstore_collection
    hold_back_fields = hold_back_fields.split(",")
    mask_fields = mask_fields.split(",") if mask_fields else []
    evaluator = DatabaseAugmentedCompletionEvaluator(
        agent=rage, fields_to_predict=hold_back_fields, fields_to_mask=mask_fields
    )
    results = evaluator.evaluate(test_collection, num_tests=num_tests, **kwargs)
    print(yaml.dump(results.dict(), sort_keys=False))


@main.command()
@path_option
@collection_option
@click.option(
    "--hold-back-fields",
    "-F",
    help="Comma separated list of fields to predict in the test.",
)
@click.option(
    "--mask-fields",
    "-M",
    help="Comma separated list of fields to mask in the test.",
)
@model_option
@limit_option
@click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--report-file",
    "-o",
    type=click.File("w"),
    help="File to write report to.",
)
@click.option(
    "--num-testing",
    default=None,
    show_default=True,
    help="Number (max) of tests to run.",
)
@click.option(
    "--working-directory",
    "-W",
    help="Working directory to use.",
)
@click.option(
    "--fresh/--no-fresh",
    default=False,
    show_default=True,
    help="Whether to rebuild test/train collections.",
)
@generate_background_option
@click.argument("tasks", nargs=-1)
def evaluate(
    tasks,
    working_directory,
    path,
    model,
    generate_background,
    num_testing,
    hold_back_fields,
    mask_fields,
    rule: List[str],
    collection,
    **kwargs,
):
    """Evaluate given a task configuration.

    Example:

        curategpt evaluate src/curate_gpt/conf/tasks/bio-ont.tasks.yaml
    """
    normalized_tasks = []
    for task in tasks:
        if ":" in task:
            task = yaml.safe_load(task)
        else:
            task = yaml.safe_load(open(task))
        if isinstance(task, list):
            normalized_tasks.extend(task)
        else:
            normalized_tasks.append(task)
    for task in normalized_tasks:
        task_obj = Task(**task)
        if path:
            task_obj.path = path
        if working_directory:
            task_obj.working_directory = working_directory
        if collection:
            task_obj.source_collection = collection
        if model:
            task_obj.model_name = model
        if hold_back_fields:
            task_obj.hold_back_fields = hold_back_fields.split(",")
        if mask_fields:
            task_obj.mask_fields = mask_fields.split(",")
        if num_testing is not None:
            task_obj.num_testing = int(num_testing)
        if generate_background:
            task_obj.generate_background = generate_background
        if rule:
            # TODO
            task_obj.rules = rule
        result = run_task(task_obj, **kwargs)
        print(yaml.dump(result.dict(), sort_keys=False))


@main.command()
@click.option("--collections", required=True)
@click.option("--models", default="gpt-3.5-turbo")
@click.option("--fields-to-mask", default="id,original_id")
@click.option("--fields-to-predict", required=True)
@click.option("--num-testing", default=50, show_default=True)
@click.option("--background", default="false", show_default=True)
def evaluation_config(collections, models, fields_to_mask, fields_to_predict, background, **kwargs):
    tasks = []
    for collection in collections.split(","):
        for model in models.split(","):
            for fp in fields_to_predict.split(","):
                for bg in background.split(","):
                    tc = Task(
                        source_db_path="db",
                        target_db_path="db",
                        model_name=model,
                        source_collection=collection,
                        fields_to_predict=[fp],
                        fields_to_mask=fields_to_mask.split(","),
                        generate_background=json.loads(bg),
                        stratified_collection=StratifiedCollection(
                            training_set_collection=f"{collection}_training",
                            testing_set_collection=f"{collection}_testing",
                        ),
                        **kwargs,
                    )
                    tasks.append(tc.dict(exclude_unset=True))
    print(yaml.dump(tasks, sort_keys=False))


@main.command()
@click.option(
    "--include-expected/--no-include-expected",
    "-E",
    default=False,
    show_default=True,
)
@click.argument("files", nargs=-1)
def evaluation_compare(files, include_expected=False):
    """Compare evaluation results."""
    dfs = []
    predicted_cols = []
    other_cols = []
    differentia_col = "method"
    for f in files:
        df = pd.read_csv(f, sep="\t", comment="#")
        df[differentia_col] = f
        if include_expected:
            include_expected = False
            base_df = df.copy()
            base_df[differentia_col] = "source"
            for c in base_df.columns:
                if c.startswith("expected_"):
                    new_col = c.replace("expected_", "predicted_")
                    base_df[new_col] = base_df[c]
            dfs.append(base_df)
        dfs.append(df)
        for c in df.columns:
            if c in predicted_cols or c in other_cols:
                continue
            if c.startswith("predicted_"):
                predicted_cols.append(c)
            else:
                other_cols.append(c)
    df = pd.concat(dfs)
    # df = pd.melt(df, id_vars=["masked_id", "file"], value_vars=["predicted_definition"])
    df = pd.melt(df, id_vars=list(other_cols), value_vars=list(predicted_cols))
    df = df.sort_values(by=list(other_cols))
    df.to_csv(sys.stdout, sep="\t", index=False)


@main.command()
@click.option(
    "--system",
    "-s",
    help="System gpt prompt to use.",
)
@click.option(
    "--prompt",
    "-p",
    default="What is the definition of {column}?",
)
@model_option
@click.argument("file")
def multiprompt(file, model, system, prompt):
    if model is None:
        model = "gpt-3.5-turbo"
    model_obj = get_model(model)
    with open(file) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            resp = model_obj.prompt(system=system, prompt=prompt.format(**row)).text()
            resp = resp.replace("\n", " ")
            print("\t".join(list(row.values()) + [resp]))


@main.command()
@collection_option
@path_option
@model_option
@click.option(
    "--show-references/--no-show-references",
    default=True,
    show_default=True,
    help="Whether to show references.",
)
@click.option(
    "_continue",
    "-C",
    "--continue",
    is_flag=True,
    flag_value=-1,
    help="Continue the most recent conversation.",
)
@click.option(
    "conversation_id",
    "--cid",
    "--conversation",
    help="Continue the conversation with the given ID.",
)
@click.argument("query")
def ask(query, path, collection, model, show_references, _continue, conversation_id):
    """Chat with a chatbot."""
    db = ChromaDBAdapter(path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    conversation = None
    if conversation_id or _continue:
        # Load the conversation - loads most recent if no ID provided
        try:
            conversation = load_conversation(conversation_id)
            print(f"CONTINUING CONVERSATION {conversation}")
        except UnknownModelError as ex:
            raise click.ClickException(str(ex))
    chatbot = ChatAgent(path)
    chatbot.extractor = extractor
    chatbot.knowledge_source = db
    response = chatbot.chat(query, collection=collection, conversation=conversation)
    show_chat_response(response, show_references)


@main.command()
@collection_option
@path_option
@model_option
@click.option(
    "--show-references/--no-show-references",
    default=True,
    show_default=True,
    help="Whether to show references.",
)
@click.option(
    "_continue",
    "-C",
    "--continue",
    is_flag=True,
    flag_value=-1,
    help="Continue the most recent conversation.",
)
@click.option(
    "conversation_id",
    "--cid",
    "--conversation",
    help="Continue the conversation with the given ID.",
)
@click.argument("query")
def citeseek(query, path, collection, model, show_references, _continue, conversation_id):
    """Find citations for an object."""
    db = ChromaDBAdapter(path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    chatbot = ChatAgent(db, extractor=extractor, knowledge_source_collection=collection)
    ea = EvidenceAgent(chat_agent=chatbot)
    response = ea.find_evidence(query)
    print("# Response:")
    click.echo(response.formatted_body)
    print("# Raw:")
    click.echo(response.body)
    if show_references:
        print("# References:")
        for ref, ref_text in response.references.items():
            print(f"## {ref}")
            print(ref_text)


@main.command()
def plugins():
    "List installed plugins"
    print(yaml.dump(get_plugins()))


@main.group()
def collections():
    "Operate on collections in the store."


@collections.command(name="list")
@click.option(
    "--minimal/--no-minimal",
    default=False,
    show_default=True,
    help="Whether to show minimal information.",
)
@click.option(
    "--derived/--no-derived",
    default=True,
    show_default=True,
    help="Whether to show derived information.",
)
@click.option(
    "--peek/--no-peek",
    default=False,
    show_default=True,
    help="Whether to peek at the first few entries of the collection.",
)
@path_option
def list_collections(path, peek: bool, minimal: bool, derived: bool):
    """List all collections."""
    db = ChromaDBAdapter(path)
    for cn in db.collections():
        if minimal:
            print(f"## Collection: {cn}")
            continue
        cm = db.collection_metadata(cn, include_derived=derived)
        c = db.client.get_or_create_collection(cn)
        print(f"## Collection: {cn} N={c.count()} meta={c.metadata} // {cm}")
        if peek:
            r = c.peek()
            for id in r["ids"]:
                print(f" - {id}")


@collections.command(name="delete")
@collection_option
@path_option
def delete_collection(path, collection):
    """Delete a collections."""
    db = ChromaDBAdapter(path)
    db.remove_collection(collection)


@collections.command(name="peek")
@collection_option
@limit_option
@path_option
def peek_collection(path, collection, **kwargs):
    """Inspect a collections."""
    logging.info(f"Peeking at {collection} in {path}")
    db = ChromaDBAdapter(path)
    for obj in db.peek(collection, **kwargs):
        print(yaml.dump(obj, sort_keys=False))


@collections.command(name="split")
@collection_option
@click.option(
    "--derived-collection-base",
    help=(
        "Base name for derived collections. Will be suffixed with _train, _test, _val."
        "If not provided, will use the same name as the original collection."
    ),
)
@model_option
@click.option(
    "--num-training",
    type=click.INT,
    help="Number of training examples to keep.",
)
@click.option(
    "--num-testing",
    type=click.INT,
    help="Number of testing examples to keep.",
)
@click.option(
    "--num-validation",
    default=0,
    show_default=True,
    type=click.INT,
    help="Number of validation examples to keep.",
)
@click.option(
    "--test-id-file",
    type=click.File("r"),
    help="File containing IDs of test objects.",
)
@click.option(
    "--ratio",
    type=click.FLOAT,
    help="Ratio of training to testing examples.",
)
@click.option(
    "--fields-to-predict",
    "-F",
    required=True,
    help="Comma separated list of fields to predict in the test. Candidate objects must have these fields.",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    help="Path to write the new store.",
)
@path_option
def split_collection(
    path, collection, derived_collection_base, output_path, model, test_id_file, **kwargs
):
    """
    Split a collection into test/train/validation.

    Example:

        curategpt -v collections split -c hp --num-training 10 --num-testing 20

    The above populates 2 new collections: hp_training and hp_testing.

    This can be run as a pre-processing step for generate-evaluate.
    """
    db = ChromaDBAdapter(path)
    if test_id_file:
        kwargs["testing_identifiers"] = [line.strip().split()[0] for line in test_id_file]
        logging.info(
            f"Using {len(kwargs['testing_identifiers'])} testing identifiers from {test_id_file.name}"
        )
        logging.info(f"First 10: {kwargs['testing_identifiers'][:10]}")
    sc = stratify_collection(db, collection, **kwargs)
    output_db = ChromaDBAdapter(output_path)
    if not derived_collection_base:
        derived_collection_base = collection
    for sn in ["training", "testing", "validation"]:
        cn = f"{derived_collection_base}_{sn}"
        output_db.remove_collection(cn, exists_ok=True)
        objs = getattr(sc, f"{sn}_set", [])
        logging.info(f"Writing {len(objs)} objects to {cn}")
        output_db.insert(objs, collection=cn, model=model)


@collections.command(name="set")
@collection_option
@path_option
@click.argument("metadata_yaml")
def set_collection_metadata(path, collection, metadata_yaml):
    """Delete a collections."""
    db = ChromaDBAdapter(path)
    db.update_collection_metadata(collection, **yaml.safe_load(metadata_yaml))


@main.group()
def ontology():
    "Use the ontology model"


@ontology.command(name="index")
@path_option
@collection_option
@model_option
@append_option
@click.option(
    "--branches",
    "-b",
    help="Comma separated list node IDs representing branches to index.",
)
@click.option(
    "--index-fields",
    help="Fields to index; comma sepatrated",
)
@click.argument("ont")
def index_ontology_command(ont, path, collection, append, model, index_fields, branches, **kwargs):
    """Index an ontology.

    Example:

        curategpt index-ontology  -c obo_hp $db/hp.db

    """
    oak_adapter = get_adapter(ont)
    view = OntologyWrapper(oak_adapter=oak_adapter)
    if branches:
        view.branches = branches.split(",")
    db = ChromaDBAdapter(path, **kwargs)
    db.text_lookup = view.text_field
    if index_fields:
        fields = index_fields.split(",")

        # print(f"Indexing fields: {fields}")
        def _text_lookup(obj: Dict):
            vals = [str(obj.get(f)) for f in fields if f in obj]
            return " ".join(vals)

        db.text_lookup = _text_lookup
    if not append:
        db.remove_collection(collection, exists_ok=True)
    db.insert(view.objects(), collection=collection, model=model)
    db.update_collection_metadata(collection, object_type="OntologyClass")


@main.group()
def view():
    "Virtual store"


@view.command(name="objects")
@click.option("--view", "-V", required=True, help="Name of the wrapper to use.")
@click.option("--source-locator")
@init_with_option
def view_objects(view, init_with, **kwargs):
    """View objects in a virtual store.

    Example:

        curategpt view objects -V filesystem --init-with "root_directory: /path/to/data"

    """
    if init_with:
        for k, v in yaml.safe_load(init_with).items():
            kwargs[k] = v
    vstore = get_wrapper(view, **kwargs)
    for obj in vstore.objects():
        print(yaml.dump(obj, sort_keys=False))


@view.command(name="search")
@click.option("--view", "-V")
@click.option("--source-locator")
@model_option
@limit_option
@init_with_option
@click.argument("query")
def view_search(query, view, model, init_with, limit, **kwargs):
    """Search in a virtual store."""
    if init_with:
        for k, v in yaml.safe_load(init_with).items():
            kwargs[k] = v
    vstore: BaseWrapper = get_wrapper(view, **kwargs)
    vstore.extractor = BasicExtractor(model_name=model)
    for obj, _dist, _ in vstore.search(query, limit=limit):
        print(yaml.dump(obj, sort_keys=False))


@view.command(name="index")
@path_option
@collection_option
@click.option("--view", "-V")
@click.option("--source-locator")
@batch_size_option
@model_option
@init_with_option
@append_option
def view_index(view, path, append, collection, model, init_with, batch_size, **kwargs):
    """Populate an index from a view."""
    if init_with:
        for k, v in yaml.safe_load(init_with).items():
            kwargs[k] = v
    wrapper: BaseWrapper = get_wrapper(view, **kwargs)
    store = ChromaDBAdapter(path)
    if not append:
        if collection in store.list_collection_names():
            store.remove_collection(collection)
    objs = wrapper.objects()
    store.insert(objs, model=model, collection=collection, batch_size=batch_size)


@view.command(name="ask")
@click.option("--view", "-V")
@click.option("--source-locator")
@limit_option
@model_option
@click.argument("query")
def view_ask(query, view, model, limit, **kwargs):
    """Ask a knowledge source wrapper."""
    vstore: BaseWrapper = get_wrapper(view)
    vstore.extractor = BasicExtractor(model_name=model)
    chatbot = ChatAgent(knowledge_source=vstore)
    response = chatbot.chat(query, limit=limit)
    show_chat_response(response, True)


@main.group()
def pubmed():
    "Use pubmed"


@pubmed.command(name="search")
@collection_option
@path_option
@model_option
@click.option(
    "--expand/--no-expand",
    default=True,
    show_default=True,
    help="Whether to expand the search term using an LLM.",
)
@click.argument("query")
def pubmed_search(query, path, model, **kwargs):
    pubmed = PubmedWrapper()
    db = ChromaDBAdapter(path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    pubmed.extractor = extractor
    pubmed.local_store = db
    results = pubmed.search(query, **kwargs)
    i = 0
    for obj, distance, _ in results:
        i += 1
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))


@pubmed.command(name="ask")
@collection_option
@path_option
@model_option
@limit_option
@click.option(
    "--show-references/--no-show-references",
    default=True,
    show_default=True,
    help="Whether to show references.",
)
@click.option(
    "--expand/--no-expand",
    default=True,
    show_default=True,
    help="Whether to expand the search term using an LLM.",
)
@click.argument("query")
def pubmed_ask(query, path, model, show_references, **kwargs):
    pubmed = PubmedWrapper()
    db = ChromaDBAdapter(path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    pubmed.extractor = extractor
    pubmed.local_store = db
    response = pubmed.chat(query, **kwargs)
    click.echo(response.formatted_body)
    if show_references:
        print("# References:")
        for ref, ref_text in response.references.items():
            print(f"## {ref}")
            print(ref_text)


if __name__ == "__main__":
    main()
