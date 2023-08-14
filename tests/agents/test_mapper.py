import re

import pytest

from curate_gpt.agents import Mapper
from curate_gpt.extract import BasicExtractor


@pytest.mark.parametrize(
    "subject_id,object_id",
    [
        ("nuclear membrane", "NuclearMembrane"),
        ("membrane of the nucleus", "NuclearMembrane"),
        ("metabolism", "MetabolicProcess"),
        ("upregulation of metabolism", "PositiveRegulationOfMetabolicProcess"),
        ("UpregulationOfMetabolism", "PositiveRegulationOfMetabolicProcess"),
        ("bacterium", "Bacteria"),
        ("taxon: bacteria", "Bacteria"),
        ("kinase", "KinaseActivity"),
        ("kinase inhibitor", "KinaseInhibitorActivity"),
        ("the membrane surrounding cell", "PlasmaMembrane"),
    ],
)
@pytest.mark.parametrize("randomize_order", [True, False])
# @pytest.mark.parametrize("limit", [10, 100])
@pytest.mark.parametrize("limit", [10])
@pytest.mark.parametrize("fields", [["label"], ["label", "definition"]])
def test_mapper(go_test_chroma_db, subject_id, object_id, randomize_order, limit, fields):
    """Tests mapping selected inputs."""
    mapper = Mapper(kb_adapter=go_test_chroma_db, extractor=BasicExtractor(model_name="gpt-4"))
    result = mapper.match(subject_id, randomize_order=randomize_order, limit=limit, fields=fields)
    print(f"## SUBJECT: {subject_id}")
    print(result.prompt)
    print(f"RESPONSE={result.response_text}")
    for m in result.mappings:
        print(f" -OBJECT: {m.object_id}")
    assert any([m.object_id == object_id for m in result.mappings])
