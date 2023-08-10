import json
from typing import Iterator

import click
import requests
import yaml

URL = "https://api.microbiomedata.org/biosamples?per_page={limit}&page={cursor}"


def _get_samples_chunk(cursor=1, limit=200) -> dict:
    """
    Get a chunk of samples from NMDC.

    :param cursor:
    :return:
    """
    url = URL.format(limit=limit, cursor=cursor)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Could not download samples from {url}")


def get_samples(cursor=1, limit=200, maximum: int = None) -> Iterator[dict]:
    """
    Iterate through all samples in NMDC and download them.

    :param cursor:
    :return:
    """
    # note: do NOT do this recursively, as it will cause a stack overflow
    initial_chunk = _get_samples_chunk(cursor=cursor, limit=limit)
    num = initial_chunk["meta"]["count"]
    if maximum is not None:
        num = min(num, maximum)
    yield from initial_chunk["results"]
    while True:
        cursor += 1
        if cursor * limit >= num:
            break
        next_chunk = _get_samples_chunk(cursor=cursor, limit=limit)
        yield from next_chunk["results"]


# CLI using click
@click.command()
@click.option(
    "--cursor",
    default=1,
    show_default=True,
    help="Cursor to start at.",
)
@click.option(
    "--limit",
    default=200,
    show_default=True,
    help="Number of samples to download at a time.",
)
@click.option(
    "--max",
    type=click.INT,
    default=None,
    show_default=True,
    help="Maximum number of samples to download.",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    show_default=True,
    help="Whether to stream the results.",
)
@click.option(
    "--format",
    default="yaml",
    show_default=True,
    help="Format to stream the results in (yaml or json).",
)
def main(cursor, limit, stream, max, format):
    """
    Download all samples from NMDC.
    """
    samples = []

    def _as_str(sample):
        if format == "json":
            return json.dumps(sample)
        elif format == "yaml":
            return yaml.dump(sample)
        else:
            raise ValueError(f"Unknown format: {format}")

    for sample in get_samples(cursor=cursor, limit=limit, maximum=max):
        if stream:
            print(_as_str(sample))
        else:
            samples.append(sample)
    if not stream:
        print(_as_str(samples))


if __name__ == "__main__":
    main()
