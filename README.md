# CurateGPT

See [notebooks](notebooks) for examples.

## Installation

```
poetry install
```

You can then run the streamlit app with:

```
make app
```

However, you will need some indexes to get started. TODO: provide ready-made downloads

For now you must create your own, see below.

## Building Indexes

CurateGPT depends on vector database indexes of the databases/ontologies you want to curate.

The flagship application is ontology curation, so to build an index for an OBO ontology:

```
make terms-default-cl
```

or if you have an OpenAI API key and want to use OpenAI embeddings:

```
make terms-oai-cl
```

To load the defaults:

```
make all
```

You can load any other files so long as they are json, yaml, or csv. A common use case is
to load background knowledge sources that can complement the KB you want to curate. For
an ontology this may be the GitHub issue tracker.

## Using the command line

```bash
curategpt --help
```

You will see various commands for working with indexes, searching, extracting, generating, etc.

Most of these functions will make their way into the UI