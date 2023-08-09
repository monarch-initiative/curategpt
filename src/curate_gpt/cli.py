"""Command line interface for curate-gpt."""
import json
import logging

import click
import yaml
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter, __version__

__all__ = [
    "main",
]

from curate_gpt.agents.rage import RetrievalAugmentedExtractor

from curate_gpt.extract.basic_extractor import BasicExtractor

from curate_gpt.rag.openai_rag import OpenAIRAG
from curate_gpt.store.schema_manager import SchemaManager
from curate_gpt.view.ontology_view import Ontology, OntologyView
from llm import get_plugins

logger = logging.getLogger(__name__)

path_option = click.option("-p", "--path", help="Path to a file or directory for database.")
model_option = click.option("-m", "--model", help="Model to use for generation, e.g. gpt-4.")
schema_option = click.option("-s", "--schema", help="Path to schema.")
collection_option = click.option("-c", "--collection", help="Collection within the database.")
relevance_factor_option = click.option("--relevance-factor", type=click.FLOAT,
                                       help="Relevance factor for search.")
limit_option = click.option("-l", "--limit", default=10, show_default=True, help="Number of results to return.")
reset_option = click.option(
    "--reset/--no-reset",
    default=False,
    show_default=True,
    help="Reset the database before indexing.",
)

@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """CLI for curate-gpt.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    logger = logging.getLogger()
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
@click.argument("ontology")
def index_ontology(ontology, path, reset: bool, collection, model, **kwargs):
    """Index an ontology.

    Example:

        curategpt index-ontology  -c obo_hp $db/hp.db

    """
    oak_adapter = get_adapter(ontology)
    view = OntologyView(oak_adapter)
    db = ChromaDBAdapter(path, **kwargs)
    db.text_lookup = view.text_field
    if model:
        db.model = model
    if reset:
        db.reset()
    db.insert(view.objects(), collection=collection)


@main.command()
@path_option
@reset_option
@collection_option
@model_option
@click.option("--text-field")
@click.argument("files", nargs=-1)
def index(files, path, reset: bool, text_field, collection, model, **kwargs):
    """Index files.

    Example:

        curategpt index  -c doc files/*json

    """
    db = ChromaDBAdapter(path, **kwargs)
    db.text_lookup = text_field
    if model:
        db.model = model
    if reset:
        db.reset()
    for file in files:
        if file.endswith(".json"):
            objs = json.load(open(file))
        else:
            objs = yaml.safe_load(open(file))
        if not isinstance(objs, list):
            objs = [objs]
        db.insert(objs, collection=collection)


@main.command()
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
        c = db.client.get_or_create_collection(cn)
        print(f"## Collection: {cn} N={c.count()} meta={c.metadata}")
        if peek:
            r = c.peek()
            for id in r["ids"]:
                print(f" - {id}")


@main.command()
@collection_option
@path_option
def delete_collection(path, collection):
    """Delete a collections."""
    db = ChromaDBAdapter(path)
    c = db.client.get_collection(collection)
    c.delete()


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
    "-P",
    "--query-property",
    default="label",
    show_default=True,
    help="Property to use for query.")
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.")
@click.option(
    "--docstore-collection",
    default=None,
    help="Collection to use in the docstore.")
@schema_option
@click.argument("query")
def create(query, path, docstore_path, docstore_collection, conversation, model, query_property, schema, **kwargs):
    """Generate an entry from a query using RAGE.

    Example:

        curategpt generate  -c obo_go "umbelliferose biosynthetic process"
    """
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaManager(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_manager = schema
        extractor.schema_manager = schema_manager
    rage = RetrievalAugmentedExtractor(kb_adapter=db, extractor=extractor)
    if docstore_path or docstore_collection:
        rage.document_adapter = ChromaDBAdapter(docstore_path)
        rage.document_adapter_collection = docstore_collection
    ao = rage.generate_extract(query, target_class="OntologyClass", context_property=query_property, **filtered_kwargs)
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
@click.option(
    "--num-examples", default=3, show_default=True, help="Number of examples for few-shot learning."
)
@click.argument("query")
def old_generate(query, path, conversation, collection, num_examples, **kwargs):
    """OLD Generate an entry from a query using RAG.

    Example:

        curategpt generate  -c obo_go "umbelliferose biosynthetic process"
    """
    db = ChromaDBAdapter(path)
    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    rag = OpenAIRAG(
        db_adapter=db, root_class=Ontology, conversation_mode=conversation, **filtered_kwargs
    )
    while True:
        if conversation and (not query or query == "-"):
            query = input("QUERY: ")
        obj = rag.generate(query, collection=collection, num_examples=num_examples)
        print(yaml.dump(obj, sort_keys=False))
        query = None
        if not conversation:
            break


@main.command()
def plugins():
    "List installed plugins"
    print(yaml.dump(get_plugins()))


if __name__ == "__main__":
    main()
