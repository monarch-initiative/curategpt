RUN = poetry run
CURATE = $(RUN) curategpt

#DB_PATH = stagedb
DB_PATH = db

ONTS = cl uberon obi go envo hp mp mondo po to oba agro fbbt nbo chebi vo peco maxo
TRACKERS = cl uberon obi  envo hp mondo go
DBS = gocam reactome bacdive mediadive alliance_gene maxoa hpoa hpoa_by_pub gocam reactome

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

%-doctest: %
	$(RUN) python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE $<

## -- Sample Datasets --

load-biosamples_nmdc:
    $(CURATE) index -p $(DB_PATH) -V nmdc  -c $@ -m openai: API

## -- Annotation Files --

load-hpoa:
	$(CURATE) -v index -p $(DB_PATH) --batch-size 10 -V hpoa  -c hpoa -m openai: URL

load-hpoa-by-pub:
	$(CURATE) -v index -p $(DB_PATH) --batch-size 5 -V hpoa_by_pub  -c hpoa_by_pub -m openai: URL

# note: maxoa repo still private, must be downloaded
load-maxoa:
	$(CURATE) -v index -p $(DB_PATH) --batch-size 10 -V maxoa  -c maxoa -m openai: data/maxoa.tsv

# note: assumes local checkout in sibling directory;
# in future it should pull this from the web
load-rdp:
	$(CURATE) index -p $(DB_PATH) --view reusabledata -c datasets_rdp -m openai:


## -- Generic --

load-generic-%:
	$(CURATE) -v view index --view $@ --batch-size 10 -c $* -m openai: WEB

load-json-data-%: data/%.json
	$(CURATE) index -c $* -c $* $<

load-db-%:
	$(CURATE) -v view index -p $(DB_PATH) --view $* -c $* -m openai: 

load-linkml-w3id-%:
	$(CURATE) view index --view linkml_schema -c schema_$* -m openai: --source-locator https://w3id.org/$*/$*.yaml

load-mixs: load-linkml-w3id-mixs

load-bacdive:
	$(CURATE) -v view index -p $(DB_PATH) -m openai: -c strain_bacdive -V bacdive --source-locator ~/Downloads/bacdive_strains.json

load-cdr:
	$(CURATE) -v index -p $(DB_PATH) -V bioc -c cdr_test -m openai: data/CDR_TestSet.BioC.xml.gz

load-uniprot-%:
	$(CURATE) -v index -p $(DB_PATH) -V uniprot -c uniprot_$* -m openai: --view-settings "taxon_id: $*"

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

load-github-to:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_to -m openai:  --view github --init-with "{repo: Planteome/plant-trait-ontology}"

load-github-po:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_po -m openai:  --view github --init-with "{repo: Planteome/plant-ontology}"

load-github-envo:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_envo -m openai:  --view github --init-with "{repo: EnvironmentOntology/envo}"

load-github-obi:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_obi -m openai:  --view github --init-with "{repo: obi-ontology/obi}"

load-github-mondo:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_obi -m openai:  --view github --init-with "{repo: monarch-initiative/mondo}"

load-github-maxo:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_maxo -m openai:  --view github --init-with "{repo: monarch-initiative/MAxO}"

list:
	$(CURATE) collections list -p $(DB_PATH)

load-github-mixs:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_mixs -m openai:  --view github --init-with "{repo: GenomicsStandardsConsortium/mixs}"

load-github-nmdc-schema-issues-prs:
	$(CURATE) -v view index -p $(DB_PATH) -c gh_nmdc -m openai:  --view github --init-with "{repo: microbiomedata/nmdc-schema}"
