RUN = poetry run
DB_PATH = db

data/nmdc.json:
	$(RUN) python -m curate_gpt.adhoc.nmdc_sample_downloader --no-stream  --format json > $@


index-nmdc: data/nmdc.json
	$(RUN) curategpt -v index -p $(DB_PATH) -m openai: -c biosamples_nmdc_oai --object-type Biosample --description "Samples taken from NMDC database" $<

index-%-issues:
	$(RUN) curategpt index -p $(DB_PATH) -c github_issues_$*_oai -m openai: --glob "../formal-ontology-analysis/repo-dirs/metadata/$*-issue-*.json"

#index-all-issues-gh:
#	$(RUN) curategpt index -p $(DB_PATH) -c github_issues_all_oai -m openai: ../formal-ontology-analysis/repo-dirs/metadata/

index-phenopackets:
	$(RUN) curategpt index -p $(DB_PATH) -c phenopackets_384_oai -m openai: --object-type Phenopacket --description "Phenopackets from https://zenodo.org/record/3905420" data/phenopackets/*.json
