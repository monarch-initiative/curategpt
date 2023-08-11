RUN = poetry run
CURATE = $(RUN) curategpt

DB_PATH = db

ONTS = cl uberon obi go envo hp mp mondo po

all: index_all_oai

index_all_oai: $(patsubst %,terms-oai-%,$(ONTS))
index_all2_oai: $(patsubst %,terms_defns-oai-%,$(ONTS))

terms-oai-%:
	$(CURATE) ontology index -p $(DB_PATH) -c terms_$*_oai -m openai: sqlite:obo:$*


terms_defns-oai-%:
	$(CURATE) ontology index --index-fields label,definition,relationships -p $(DB_PATH) -c terms_defns_$*_oai -m openai: sqlite:obo:$*


terms-default-%:
	$(CURATE) ontology index -p $(DB_PATH) -c terms_$* sqlite:obo:$*

app:
	$(RUN) streamlit run src/curate_gpt/app/app.py --logger.level=debug
