import pytest
from curate_gpt.agents import MappingAgent
from curate_gpt.agents.mapping_agent import MappingPredicate
from curate_gpt.extract import BasicExtractor

from tests.store.conftest import requires_openai_api_key


@requires_openai_api_key
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
@pytest.mark.parametrize("include_predicates", [False])
@pytest.mark.parametrize("fields", [["label"], ["label", "definition"]])
def test_mapper(
    go_test_chroma_db, subject_id, object_id, include_predicates, randomize_order, limit, fields
):
    """Tests mapping selected inputs."""
    mapper = MappingAgent(
        knowledge_source=go_test_chroma_db, extractor=BasicExtractor(model_name="gpt-4")
    )
    result = mapper.match(
        subject_id,
        include_predicates=include_predicates,
        randomize_order=randomize_order,
        limit=limit,
        fields=fields,
        collection="test",
    )
    print(f"## SUBJECT: {subject_id}")
    print(result.prompt)
    print(f"RESPONSE={result.response_text}")
    for m in result.mappings:
        print(f" -OBJECT: {m.object_id} {m.predicate_id}")
    assert any([m.object_id == object_id for m in result.mappings])
    if include_predicates:
        assert any(
            [
                m.object_id == object_id
                for m in result.mappings
                if m.predicate_id == MappingPredicate.SAME_AS
            ]
        )
