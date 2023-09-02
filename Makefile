RUN = poetry run
CURATE = $(RUN) curategpt

DB_PATH = stagedb

ONTS = cl uberon obi go envo hp mp mondo po to oba agro fbbt nbo chebi vo peco
TRACKERS = cl uberon obi  envo hp mondo go


all: index_all_ont

## -- Ontology Indexing --


index_all_ont: $(patsubst %,ont-%,$(ONTS))
index_all_issues: $(patsubst %,load-github-%,$(ONTS))

ont-%:
	$(CURATE) ontology index --index-fields label,definition,relationships -p $(DB_PATH) -c ont_$* -m openai: sqlite:obo:$*

## -- Web App --


app:
	$(RUN) streamlit run src/curate_gpt/app/app.py --logger.level=debug

## -- Docs --


apidoc:
	$(RUN) sphinx-apidoc -f -M -o docs/ src/curate_gpt/ && cd docs && $(RUN) make html

## -- Sample Datasets --

load-biosamples_nmdc:
    $(CURATE) index -V nmdc  -c $@ -m openai: API

## -- Annotation Files --

load-hpoa:
	$(CURATE) -v index --batch-size 10 -V hpoa  -c hpoa -m openai: URL

load-hpoa-by-pub:
	$(CURATE) -v index --batch-size 5 -V hpoa_by_pub  -c hpoa_by_pub -m openai: URL

# note: maxoa repo still private, must be downloaded
load-maxoa:
	$(CURATE) -v index --batch-size 10 -V maxoa  -c maxoa -m openai: data/maxoa.tsv

## -- Generate --

load-generic-%:
	$(CURATE) -v view index --view $@ --batch-size 10 -c $* -m openai: WEB


## -- GitHub issues --

# TODO: patternize

load-github-uberon:
	$(CURATE) -v view index  -p $(DB_PATH) -c gh_uberon -m openai:  --view github --init-with "{repo: obophenotype/uberon}"

load-github-hp:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_hp -m openai:  --view github --init-with "{repo: obophenotype/human-phenotype-ontology}"

load-github-go:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_go -m openai:  --view github --init-with "{repo: geneontology/go-ontology}"

load-github-cl:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_cl -m openai:  --view github --init-with "{repo: obophenotype/cell-ontology}"

load-github-envo:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_envo -m openai:  --view github --init-with "{repo: EnvironmentOntology/envo}"

load-github-obi:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_obi -m openai:  --view github --init-with "{repo: obi-ontology/obi}"

load-github-mondo:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_obi -m openai:  --view github --init-with "{repo: monarch-initiative/mondo}"

list:
	$(CURATE) collections list -p $(DB_PATH)


