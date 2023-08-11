"""Command line interface for curate-gpt."""
import csv
import json
import logging
from typing import List, Tuple

import click
import yaml
from click_default_group import DefaultGroup
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter, __version__

__all__ = [
    "main",
]

from curate_gpt.agents.chat import ChatEngine

from curate_gpt.agents.dalek import DatabaseAugmentedExtractor
from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.rag.openai_rag import OpenAIRAG
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.view.ontology_view import Ontology, OntologyView
from llm import get_plugins, UnknownModelError
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


@click.group(cls=DefaultGroup,
    default="search",
    default_if_no_args=True,)
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
        elif file.endswith(".csv"):
            objs = list(csv.DictReader(open(file)))
        else:
            objs = yaml.safe_load(open(file))
        if not isinstance(objs, list):
            objs = [objs]
        db.insert(objs, collection=collection)


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
def create(
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
    rage = DatabaseAugmentedExtractor(kb_adapter=db, extractor=extractor)
    if docstore_path or docstore_collection:
        rage.document_adapter = ChromaDBAdapter(docstore_path)
        rage.document_adapter_collection = docstore_collection
    ao = rage.generate_extract(
        query, context_property=query_property, rules=rule, **filtered_kwargs
    )
    print(yaml.dump(ao.object, sort_keys=False))


@main.command()
@collection_option
@path_option
@model_option
@click.option("--show-references/--no-show-references", default=True,
              show_default=True,
              help="Whether to show references.")
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
    chatbot = ChatEngine(path)
    chatbot.extractor = extractor
    chatbot.kb_adapter = db
    response = chatbot.chat(query, collection=collection, conversation=conversation)
    click.echo(response.formatted_response)
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
def index_ontology_command(ont, path, reset: bool, collection, append, model, index_fields, **kwargs):
    """Index an ontology.

    Example:

        curategpt index-ontology  -c obo_hp $db/hp.db

    """
    oak_adapter = get_adapter(ont)
    view = OntologyView(oak_adapter)
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



if __name__ == "__main__":
    main()
