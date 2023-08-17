"""Command line interface for curate-gpt."""
import csv
import json
import logging
from pathlib import Path
from typing import List

import click
import yaml
from click_default_group import DefaultGroup
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter, __version__

__all__ = [
    "main",
]

from curate_gpt.agents.chat_agent import ChatAgent, ChatResponse
from curate_gpt.agents.dae_agent import DatabaseAugmentedExtractor
from curate_gpt.agents.evidence_agent import EvidenceAgent
from curate_gpt.evaluation.dae_evaluator import DatabaseAugmentedExtractorEvaluator
from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.utils.metrics import calculate_metrics, evaluate_predictions
from curate_gpt.wrappers import BaseWrapper, OntologyWrapper, get_wrapper
from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper
from llm import UnknownModelError, get_plugins
from llm.cli import load_conversation

# logger = logging.getLogger(__name__)

path_option = click.option("-p", "--path", help="Path to a file or directory for database.")
model_option = click.option("-m", "--model", help="Model to use for generation, e.g. gpt-4.")
schema_option = click.option("-s", "--schema", help="Path to schema.")
collection_option = click.option("-c", "--collection", help="Collection within the database.")
relevance_factor_option = click.option(
    "--relevance-factor", type=click.FLOAT, help="Relevance factor for search."
)
limit_option = click.option(
    "-l", "--limit", default=10, show_default=True, help="Number of results to return."
)
reset_option = click.option(
    "--reset/--no-reset",
    default=False,
    show_default=True,
    help="Reset the database before indexing.",
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


def show_chat_response(response: ChatResponse, show_references: bool = True):
    """Show a chat response."""
    print("# Response:")
    click.echo(response.formatted_body)
    print("# Raw:")
    click.echo(response.body)
    if show_references:
        print("# References:")
        for ref, ref_text in response.references.items():
            print(f"## {ref}")
            print(ref_text)
        print("# Uncited:")
        for ref, ref_text in response.uncited_references.items():
            print(f"## {ref}")
            print(ref_text)


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
@reset_option
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
@click.option(
    "--batch-size", default=None, show_default=True, type=click.INT, help="Batch size for indexing."
)
@click.argument("files", nargs=-1)
def index(
    files,
    path,
    reset: bool,
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
    if reset:
        db.reset()
    if glob:
        files = [str(gf) for f in files for gf in Path().glob(f)]
    if view:
        proxy_object = get_wrapper(view)
        if not object_type:
            object_type = proxy_object.default_object_type
        if not description:
            description = f"{object_type} objects loaded from {str(files)[0:30]}"
    else:
        proxy_object = None
    for file in files:
        logging.debug(f"Indexing {file}")
        if proxy_object:
            proxy_object.source_locator = file
            objs = list(proxy_object.objects())
        elif file.endswith(".json"):
            objs = json.load(open(file))
        elif file.endswith(".csv"):
            objs = list(csv.DictReader(open(file)))
        elif file.endswith(".tsv"):
            objs = list(csv.DictReader(open(file), delimiter="\t"))
        else:
            objs = yaml.safe_load(open(file))
        if not isinstance(objs, list):
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
@click.argument("query")
def search(query, path, collection, **kwargs):
    """Query a database."""
    db = ChromaDBAdapter(path)
    results = db.search(query, collection=collection, **kwargs)
    i = 0
    for obj, distance, _ in results:
        i += 1
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))


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
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
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
@schema_option
@click.argument("query")
def generate(
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
    """Generate an entry from a query using RAGE.

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
    rage = DatabaseAugmentedExtractor(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        rage.document_adapter = ChromaDBAdapter(docstore_path)
        rage.document_adapter_collection = docstore_collection
    ao = rage.generate_extract(
        query, context_property=query_property, rules=rule, **filtered_kwargs
    )
    print(yaml.dump(ao.object, sort_keys=False))


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
    default=10000,
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
    hold_back_fields,
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

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    rage = DatabaseAugmentedExtractor(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        rage.document_adapter = ChromaDBAdapter(docstore_path)
        rage.document_adapter_collection = docstore_collection
    hold_back_fields = hold_back_fields.split(",")
    evaluator = DatabaseAugmentedExtractorEvaluator(agent=rage, hold_back_fields=hold_back_fields)
    results = evaluator.evaluate(test_collection, **kwargs)
    print(yaml.dump(results.dict(), sort_keys=False))
    raise ValueError("STOP")
    # db.find(where={}, collection=test_collection)
    test_objs = db.peek(test_collection, limit=10000)
    for test_obj in test_objs:
        query_obj = {k: v for k, v in test_obj.items() if k not in hold_back_fields}
        print(f"## Query: {query_obj}")
        ao = rage.generate_extract(query_obj, rules=rule, **filtered_kwargs)
        print("## Expected:")
        print(yaml.dump(test_obj, sort_keys=False))
        print("## Prediction:")
        print(yaml.dump(ao.object, sort_keys=False))
        outcomes = []
        for f in hold_back_fields:
            outcomes.extend(
                list(evaluate_predictions(ao.object.get(f, None), test_obj.get(f, None)))
            )
        metrics = calculate_metrics(outcomes)
        print("## Scores")
        print(yaml.dump(metrics.dict(), sort_keys=False))
        # for diff in diffs:
        #    print(f" * Diff: {diff}")


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
    "--peek/--no-peek",
    default=False,
    show_default=True,
    help="Whether to peek at the first few entries of the collection.",
)
@path_option
def list_collections(path, peek: bool):
    """List all collections."""
    db = ChromaDBAdapter(path)
    for cn in db.collections():
        cm = db.collection_metadata(cn, include_derived=True)
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
@reset_option
@collection_option
@model_option
@append_option
@click.option(
    "--index-fields",
    help="Fields to index; comma sepatrated",
)
@click.argument("ont")
def index_ontology_command(
    ont, path, reset: bool, collection, append, model, index_fields, **kwargs
):
    """Index an ontology.

    Example:

        curategpt index-ontology  -c obo_hp $db/hp.db

    """
    oak_adapter = get_adapter(ont)
    view = OntologyWrapper(oak_adapter=oak_adapter)
    db = ChromaDBAdapter(path, **kwargs)
    db.text_lookup = view.text_field
    if index_fields:
        fields = index_fields.split(",")
        db.text_lookup = lambda obj: " ".join([str(getattr(obj, f, "")) for f in fields])
    if reset:
        db.reset()
    if not append:
        db.remove_collection(collection, exists_ok=True)
    db.insert(view.objects(), collection=collection, model=model)
    db.update_collection_metadata(collection, object_type="OntologyClass")


@main.group()
def view():
    "Virtual store"


@view.command(name="objects")
@click.option("--view")
@click.option("--source-locator")
def view_objects(view, **kwargs):
    """View objects in a virtual store."""
    vstore = get_wrapper(view, **kwargs)
    for obj in vstore.objects():
        print(yaml.dump(obj, sort_keys=False))


@view.command(name="search")
@click.option("--view", "-V")
@click.option("--source-locator")
@model_option
@click.argument("query")
def view_search(query, view, model, **kwargs):
    """Search in a virtual store."""
    vstore: BaseWrapper = get_wrapper(view)
    vstore.extractor = BasicExtractor(model_name=model)
    for obj, _dist, _ in vstore.search(query):
        print(yaml.dump(obj, sort_keys=False))


@view.command(name="ask")
@click.option("--view", "-V")
@click.option("--source-locator")
@model_option
@click.argument("query")
def view_ask(query, view, model, **kwargs):
    """Ask a knowledge source wrapper."""
    vstore: BaseWrapper = get_wrapper(view)
    vstore.extractor = BasicExtractor(model_name=model)
    chatbot = ChatAgent(knowledge_source=vstore)
    response = chatbot.chat(query)
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
