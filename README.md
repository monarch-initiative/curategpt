# CurateGPT

CurateGPT is a prototype web application and framework for performing general purpose AI-guided curation
and curation-related operations over *collections* of objects.


See also the app on [curategpt.io](https://curategpt.io) (note: this is sometimes down, and may only have a
subset of the functionality of the local app)

## Installation

```
poetry install
```

You can then run the streamlit app with:

```
make app
```

However, you will need some indexes to get started. TODO: provide ready-made downloads

For now, you must create your own, see below.

## Building Indexes

CurateGPT depends on vector database indexes of the databases/ontologies you want to curate.

The flagship application is ontology curation, so to build an index for an OBO ontology like CL:

```
make ont-cl
```

This requires an OpenAI key.

(You can build indexes using an open embedding model, modify the command to leave off
the `-m` option, but this is not recommended as currently oai embeddings seem to work best).


To load the default ontologies:

```
make all
```

Note that by default this loads into a collection set stored at `stagedb`, whereas the app works off
of `db`. You can copy the collection set to the db with:

```
cp -r stagedb db
```

You can load an arbitrary json, yaml, or csv file:

```
curategpt view index -c my_foo foo.json
```

To load a GitHub repo of issues:

```
curategpt -v view index -c gh_uberon -m openai:  --view github --init-with "{repo: obophenotype/uberon}"
```

The following are also supported:

- Google Drives
- Google Sheets
- Markdown files
- LinkML Schemas
- HPOA files
- GOCAMs
- MAXOA files
- Many more

## Notebooks

- See [notebooks](notebooks) for examples.


## Using the command line

```bash
curategpt --help
```

You will see various commands for working with indexes, searching, extracting, generating, etc.

These functions are generally available through the UI, and the current priority is documenting these.
