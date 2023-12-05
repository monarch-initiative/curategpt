import pytest
import yaml
from curate_gpt.agents.concept_recognition_agent import AnnotationMethod, ConceptRecognitionAgent
from curate_gpt.extract.basic_extractor import BasicExtractor


@pytest.mark.parametrize(
    "text,categories,prefixes,expected",
    [
        (
            "A metabolic process that results in the breakdown of chemicals in vacuolar structures.",
            ["BiologicalProcess", "SubcellularStructure"],
            ["GO"],
            ["GO:0044237", "GO:0005773"],
        ),
        (
            "A metabolic process that results in the breakdown of chemicals in vacuolar structures.",
            ["BiologicalProcess", "SubcellularStructure"],
            ["FAKE"],
            [],
        ),
        (
            "Protoplasm",
            None,
            None,
            ["GO:0005622"],
        ),
        (
            "The photosynthetic membrane of plants and algae",
            ["BiologicalProcess", "SubcellularStructure", "OrganismTaxon"],
            None,
            ["GO:0005622"],
        ),
    ],
)
@pytest.mark.parametrize(
    "method",
    [AnnotationMethod.CONCEPT_LIST, AnnotationMethod.CONCEPT_LIST, AnnotationMethod.TWO_PASS],
)
def test_concept_recognizer(go_test_chroma_db, text, categories, prefixes, expected, method):
    limit = 50
    if method == AnnotationMethod.TWO_PASS:
        limit = 10
    extractor = BasicExtractor()
    cr = ConceptRecognitionAgent(knowledge_source=go_test_chroma_db, extractor=extractor)
    cr.prefixes = prefixes
    cr.identifier_field = "original_id"
    print(f"## METHOD: {method} CATEGORY: {categories} PREFIXES: {prefixes}")
    ann = cr.annotate(
        text, collection="terms_go", method=method, categories=categories, limit=limit
    )
    print("RESULT:")
    print(yaml.dump(ann.dict(), sort_keys=False))
    overlap = len(set(ann.concepts).intersection(set(expected)))
    print(f"OVERLAP: {overlap} / {len(expected)}")
    if ann.concepts != expected:
        print("MISMATCH")
    if len(expected) == 0:
        assert len(ann.concepts) == 0
