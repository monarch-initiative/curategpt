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

To load different databases:

```
make load-db-hpoa
make load-db-reactome
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

### Chatting with a knowledge base

```
curategpt ask -c ont_cl "What neurotransmitter is released by the hippocampus?"
```

may yield something like:

```
The hippocampus releases gamma-aminobutyric acid (GABA) as a neurotransmitter [1](#ref-1).

...

## 1

id: GammaAminobutyricAcidSecretion_neurotransmission
label: gamma-aminobutyric acid secretion, neurotransmission
definition: The regulated release of gamma-aminobutyric acid by a cell, in which the
  gamma-aminobutyric acid acts as a neurotransmitter.
...
```

### Chatting with pubmed

```
curategpt view ask -V pubmed "what neurons express VIP?"
```

### Chatting with a GitHub issue tracker

```
curategpt ask -c gh_obi "what are some new term requests for electrophysiology terms?"
```

### Term Autocompletion (DRAGON-AI)

```
curategpt complete -c ont_cl  "mesenchymal stem cell of the apical papilla"
```

yields

```yaml
id: MesenchymalStemCellOfTheApicalPapilla
definition: A mesenchymal cell that is part of the apical papilla of a tooth and has
  the ability to self-renew and differentiate into various cell types such as odontoblasts,
  fibroblasts, and osteoblasts.
relationships:
- predicate: PartOf
  target: ApicalPapilla
- predicate: subClassOf
  target: MesenchymalCell
- predicate: subClassOf
  target: StemCell
original_id: CL:0007045
label: mesenchymal stem cell of the apical papilla
```

### All-by-all comparisons

You can compare all objects in one collection 

`curategpt all-by-all --threshold 0.80 -c ont_hp -X ont_mp --ids-only -t csv > ~/tmp/allxall.mp.hp.csv`

This takes 1-2s, as it involves comparison over pre-computed vectors. It reports top hits above a threshold.

Results may vary. You may want to try different texts for embeddings
(the default is the entire json object; for ontologies it is
concatenation of labels, definition, aliases).

sample:

```
HP:5200068,Socially innappropriate questioning,MP:0001361,social withdrawal,0.844015132437909
HP:5200069,Spinning,MP:0001411,spinning,0.9077306606290237
HP:5200071,Delayed Echolalia,MP:0013140,excessive vocalization,0.8153252835818089
HP:5200072,Immediate Echolalia,MP:0001410,head bobbing,0.8348177036912526
HP:5200073,Excessive cleaning,MP:0001412,excessive scratching,0.8699103725005582
HP:5200104,Abnormal play,MP:0020437,abnormal social play behavior,0.8984862078522344
HP:5200105,Reduced imaginative play skills,MP:0001402,decreased locomotor activity,0.85571629684631
HP:5200108,Nonfunctional or atypical use of objects in play,MP:0003908,decreased stereotypic behavior,0.8586700411012859
HP:5200129,Abnormal rituals,MP:0010698,abnormal impulsive behavior control,0.8727804272023427
HP:5200134,Jumping,MP:0001401,jumpy,0.9011393233129765
```

Note that CurateGPT has a separate component for using an LLM to evaluate candidate matches (see also https://arxiv.org/abs/2310.03666); this is
not enabled by default, this would be expensive to run for a whole ontology.

