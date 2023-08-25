RUN = poetry run
CURATE = $(RUN) curategpt

DB_PATH = db

ONTS = cl uberon obi go envo hp mp mondo po

all: index_all_oai

old_index_all_oai: $(patsubst %,terms-oai-%,$(ONTS))
index_all_oai: $(patsubst %,terms_defns-oai-%,$(ONTS))

terms-oai-%:
	$(CURATE) ontology index -p $(DB_PATH) -c terms_$*_oai -m openai: sqlite:obo:$*


terms_defns-oai-%:
	$(CURATE) ontology index --index-fields label,definition,relationships -p $(DB_PATH) -c terms_defns_$*_oai -m openai: sqlite:obo:$*

ont_go_mf:
	$(CURATE) ontology index -b GO:0003674 --index-fields label,definition,relationships -p $(DB_PATH) -c $@ -m openai: sqlite:obo:go

ont_cl_tcell:
	$(CURATE) -v ontology index -b CL:0000084 --index-fields label,definition,relationships -p $(DB_PATH) -c $@ -m openai: sqlite:obo:cl


terms-default-%:
	$(CURATE) ontology index -p $(DB_PATH) -c terms_$* sqlite:obo:$*

load-biosamples_nmdc:
    $(CURATE) index -V nmdc  -c $@ -m openai: API

load-hpoa:
	$(CURATE) -v index --batch-size 10 -V hpoa  -c hpoa -m openai: URL

load-hpoa-by-pub:
	$(CURATE) -v index --batch-size 5 -V hpoa_by_pub  -c hpoa_by_pub -m openai: URL

app:
	$(RUN) streamlit run src/curate_gpt/app/app.py --logger.level=debug


apidoc:
	$(RUN) sphinx-apidoc -f -M -o docs/ src/curate_gpt/ && cd docs && $(RUN) make html
