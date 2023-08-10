RUN = poetry run

data/nmdc.json:
	$(RUN) python -m curate_gpt.adhoc.nmdc_sample_downloader --no-stream  --format json > $@


index-nmdc: data/nmdc.json
	$(RUN) curategpt -v index -c nmdc $<
