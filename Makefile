RUN = poetry run
CURATE = $(RUN) curategpt

ONTS = cl uberon obi go envo hp

all: index_all

index_all: $(patsubst %,terms-oai-%,$(ONTS))

terms-oai-%:
	$(CURATE) ontology index -c terms_$*_oai -m openai: sqlite:obo:$*

terms-default-%:
	$(CURATE) ontology index -c terms_$* sqlite:obo:$*

app:
	$(RUN) streamlit run src/curate_gpt/app/app.py --logger.level=debug
