RUN = poetry run
DB_PATH = db

data/nmdc.json:
	$(RUN) python -m curate_gpt.adhoc.nmdc_sample_downloader --no-stream  --format json > $@


index-nmdc: data/nmdc.json
	$(RUN) curategpt -v index -p $(DB_PATH) -m openai: -c biosamples_nmdc_oai --object-type Biosample --description "Samples taken from NMDC database" $<

index-obi-issues:
	$(RUN) curategpt index -c github_issues_obi_oai -m openai: ../formal-ontology-analysis/repo-dirs/metadata/*.json
