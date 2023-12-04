"""Annotation (Concept Recognition) in texts."""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from curate_gpt.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


CONCEPT = Tuple[str, str]


class Span(BaseModel):

    """An individual span of text containing a single concept."""

    text: str

    start: Optional[int] = None
    end: Optional[int] = None

    concept_id: str = None
    """Concept ID."""

    concept_label: str = None
    """Concept label."""

    is_suspect: bool = False
    """Potential hallucination due to ID/label mismatch."""


class GroundingResult(BaseModel):

    """Result of grounding text."""

    input_text: str
    """Text that is supplied for grounding, assumed to contain a single context."""

    spans: Optional[List[Span]] = []
    """Ordered list of candidate spans."""

    score: Optional[float] = None
    """Score/confidence, from zero to one."""


class AnnotationMethod(str, Enum):

    """Strategy or algorithm used for CR."""

    INLINE = "inline"
    """LLM creates an annotated document"""

    CONCEPT_LIST = "concept_list"
    """LLM creates a list of concepts"""

    TWO_PASS = "two_pass"
    """LLM annotates a document using NER and then grounds the concepts"""


class AnnotatedText(BaseModel):

    """In input text annotated with concept instances."""

    input_text: str
    """Text that is supplied for annotation."""

    concepts: Optional[Dict[str, str]] = {}
    """Dictionary of concepts found in the text. TODO: change to list of spans."""

    annotated_text: Optional[str] = None
    """Text with concepts annotated (not all methods produce this)."""

    spans: Optional[List[Span]] = []

    summary: Optional[str] = None
    """Summary of the results."""

    prompt: Optional[str] = None
    """Prompt used to generate the annotated text."""


GROUND_PROMPT = """
Your role is to assign a concept ID that best matches the supplied text, using
the supplied list of candidate concepts.
Return as a string "CONCEPT NAME // CONCEPT ID".
Only return a result if the input text represents the same or equivalent
concept, in the provided context.
If there is no match, return an empty string.
"""


xxx_GROUND_PROMPT = """
Your role is to assign a concept ID that best matches the supplied text, using
the supplied list of candidate concepts.
Return results as a CSV of ID,label,score triples, where the ID is the concept.
Only use concept IDs from the supplied list of candidate concepts.
Only return a row if the concept ID is a match for the input text
If there is no match, return an empty string.
"""

MENTION_PROMPT = """
Your role is to list all instances of the supplied candidate concepts in the supplied text.
Return the concept instances as a CSV of ID,label,text pairs, where the ID
is the concept ID, label is the concept label, and text is the mention of the
concept in the text.
The concept ID and label should come only from the list of candidate concepts supplied to you.
Only include a row if the meaning of the text section is that same as the concept.
If there are no instances of a concept in the text, return an empty string.
Do not include additional verbiage.
"""

ANNOTATE_PROMPT = """
Your role is to annotate the supplied text with selected concepts.
return the original text with each conceptID in square brackets.
After the occurrence of that concept.
You can use synonyms. For example, if the concept list contains
zucchini // DB:12345
Then for the text 'I love courgettes!' you should return
'I love [courgettes DB:12345]!'
Always try and match the longest span.
he concept ID should come only from the list of candidate concepts supplied to you.
"""


import re


def parse_annotations(text, marker_char: str = None) -> List[CONCEPT]:
    """
    Parse annotations from text.

    >>> text = ("A minimum diagnostic criterion is the combination of either the [skin tumours] or multiple "
    ...        "[odontogenic keratocysts HP:0010603] of the jaw plus a positive [family history HP:0032316] "
    ...        "for this disorder, [bifid ribs HP:0000923], lamellar [calcification of falx cerebri HP:0005462] "
    ...        "or any one of the skeletal abnormalities typical of this syndrome")
    >>> for ann in parse_annotations(text):
    ...    print(ann)
    ('skin tumours', None)
    ('odontogenic keratocysts', 'HP:0010603')
    ('family history', 'HP:0032316')
    ('bifid ribs', 'HP:0000923')
    ('calcification of falx cerebri', 'HP:0005462')

    For texts with marker characters:

    >>> text = "for this disorder, [bifid ribs | HP:0000923], lamellar [calcification of falx cerebri | HP:0005462] "
    >>> for ann in parse_annotations(text, "|"):
    ...    print(ann)
    ('bifid ribs', 'HP:0000923')
    ('calcification of falx cerebri', 'HP:0005462')

    :param text:
    :return:
    """
    # First Pass: Extract text within [ ... ]
    pattern1 = r"\[([^\]]+)\]"
    matches = re.findall(pattern1, text)

    # Second Pass: Parse the last token of each match
    annotations = []
    for match in matches:
        # Split the match into words and check the last token
        if marker_char:
            toks = match.split(marker_char)
            if len(toks) > 1:
                annotation = " ".join(toks[:-1]).strip()
                id = toks[-1].strip()
            else:
                annotation = match
                id = None
        else:
            words = match.split()
            if len(words) > 1 and ":" in words[-1]:
                annotation = " ".join(words[:-1])
                id = words[-1]
            else:
                annotation = match
                id = None

        annotations.append((annotation, id))

    return annotations


def parse_spans(text: str, concept_dict: Dict[str, str] = None) -> List[Span]:
    spans = []
    for line in text.split("\n"):
        logger.debug(f"Line: {line}")
        row = line.split(",")
        if len(row) < 2:
            logger.debug(f"Skipping line: {line}")
            continue
        concept_id = row[0].strip('"')
        if concept_id == "ID":
            continue
        if " " in concept_id:
            continue
        concept_label = row[1].strip('"')
        mention_text = ",".join(row[2:])
        verified_concept_label = concept_dict.get(concept_id, None)
        spans.append(
            Span(
                text=mention_text,
                concept_id=concept_id,
                concept_label=verified_concept_label,
                is_suspect=verified_concept_label != concept_label,
            )
        )
    return spans


@dataclass
class ConceptRecognitionAgent(BaseAgent):
    identifier_field: str = None
    """Field to use as identifier for objects."""

    label_field: str = None
    """Field to use as label for objects."""

    split_input_text: bool = None

    relevance_factor: float = 0.8
    """Relevance factor for diversifying search results using MMR."""

    prefixes: List[str] = None
    """List of prefixes to use for concept IDs."""

    def ground_concept(
        self,
        text: str,
        collection: str = None,
        categories: Optional[List[str]] = None,
        include_category_in_search=True,
        context: str = None,
        **kwargs,
    ) -> GroundingResult:
        system_prompt = GROUND_PROMPT
        query = text
        if include_category_in_search and categories:
            query += " Categories: " + ", ".join(categories)
        concept_pairs, concept_prompt = self._label_id_pairs_prompt_section(
            query, collection, **kwargs
        )
        concept_dict = {c[0]: c[1] for c in concept_pairs}
        system_prompt += concept_prompt
        model = self.extractor.model
        logger.debug(f"Prompting with: {text}")
        if context:
            prompt_text = f"The overall context for this is the sentence '{context}'.\n\n"
            prompt_text += f"Concept to ground: {text}"
        else:
            prompt_text = f"Concept to ground: {text}"
        response = model.prompt(prompt_text, system=system_prompt)
        logger.debug(f"Response: {response.text()}")
        lines = response.text().split("\n")
        spans = []
        for line in lines:
            if "//" in line:
                toks = line.split("//")
                if len(toks) > 2:
                    logger.warning(f"Multiple concepts in one line: {line}")
                concept_label, concept_id = toks[0], toks[1]
                concept_id = concept_id.strip()
                if " " in concept_id:
                    continue
                provided_concept_label = concept_label.strip()
                if concept_id in concept_dict:
                    concept_label = concept_dict[concept_id]
                else:
                    concept_label = None
                span = Span(
                    text=text,
                    concept_id=concept_id,
                    concept_label=concept_label,
                    is_suspect=provided_concept_label != concept_label,
                )
                spans.append(span)
        # spans = parse_spans(response.text(), concept_dict)
        ann = GroundingResult(input_text=text, annotated_text=response.text(), spans=spans)
        return ann

    def annotate(
        self,
        text: str,
        collection: str = None,
        method=AnnotationMethod.INLINE,
        **kwargs,
    ) -> AnnotatedText:
        if method == AnnotationMethod.INLINE:
            return self.annotate_inline(text, collection, **kwargs)
        elif method == AnnotationMethod.CONCEPT_LIST:
            return self.annotate_concept_list(text, collection, **kwargs)
        elif method == AnnotationMethod.TWO_PASS:
            return self.annotate_two_pass(text, collection, **kwargs)
        else:
            raise ValueError(f"Unknown annotation method {method}")

    def annotate_two_pass(
        self,
        text: str,
        collection: str = None,
        categories: List[str] = None,
        **kwargs,
    ) -> AnnotatedText:
        if not categories:
            categories = ["NamedEntity"]
        system_prompt = "Your job is to parse the supplied text, identifying instances of concepts "
        if len(categories) == 1:
            system_prompt += f" that represent some kind of {categories[0]}. "
            system_prompt += (
                "Mark up the concepts in square brackets, "
                "preserving the original text inside the brackets. "
            )
        else:
            system_prompt += " that represent one of the following categories: "
            system_prompt += ", ".join(categories)
            system_prompt += (
                "Mark up the concepts in square brackets, with the category after the pipe symbol, "
            )
            system_prompt += "Using the syntax [ORIGINAL TEXT | CATEGORY]."
        logger.debug(f"Prompting with: {text}")
        model = self.extractor.model
        response = model.prompt(text, system=system_prompt)
        marked_up_text = response.text()
        anns = parse_annotations(marked_up_text, "|")
        spans = []
        for term, category in anns:
            concept = self.ground_concept(
                term,
                collection,
                categories=[category] if category else None,
                context=text,
                **kwargs,
            )
            if not concept.spans:
                logger.debug(f"Unable to ground concept {term} in category {category}")
                continue
            main_span = concept.spans[0]
            spans.append(
                Span(
                    text=term,
                    concept_id=main_span.concept_id,
                    concept_label=main_span.concept_label,
                )
            )
        return AnnotatedText(
            input_text=text,
            annotated_text=marked_up_text,
            spans=spans,
        )

    def annotate_inline(
        self,
        text: str,
        collection: str = None,
        categories: List[str] = None,
        **kwargs,
    ) -> AnnotatedText:
        system_prompt = ANNOTATE_PROMPT
        concept_pairs, concepts_prompt = self._label_id_pairs_prompt_section(
            text, collection, **kwargs
        )
        concept_dict = {c[0]: c[1] for c in concept_pairs}
        system_prompt += concepts_prompt
        model = self.extractor.model
        logger.debug(f"Prompting with: {text}")
        response = model.prompt(text, system=system_prompt)
        anns = parse_annotations(response.text())
        logger.info(f"Anns: {anns}")
        spans = [
            Span(text=ann[0], concept_id=ann[1], concept_label=concept_dict.get(ann[1], None))
            for ann in anns
        ]
        return AnnotatedText(input_text=text, spans=spans, annotated_text=response.text())

    def annotate_concept_list(
        self,
        text: str,
        collection: str = None,
        categories: List[str] = None,
        **kwargs,
    ) -> AnnotatedText:
        system_prompt = MENTION_PROMPT
        concept_pairs, concepts_prompt = self._label_id_pairs_prompt_section(
            text, collection, **kwargs
        )
        concept_dict = {c[0]: c[1] for c in concept_pairs}
        system_prompt += concepts_prompt
        model = self.extractor.model
        logger.debug(f"Prompting with: {text}")
        response = model.prompt(text, system=system_prompt)
        spans = parse_spans(response.text(), concept_dict)
        return AnnotatedText(
            input_text=text, summary=response.text(), spans=spans, prompt=system_prompt
        )

    def _label_id_pairs_prompt_section(
        self,
        text: str,
        collection: str,
        prolog: str = None,
        relevance_factor: float = None,
        **kwargs,
    ) -> Tuple[List[CONCEPT], str]:
        prompt = prolog
        if not prompt:
            prompt = "Here are the candidate concepts, as label // ConceptID pairs:\n"
        id_field = self.identifier_field
        label_field = self.label_field
        if not id_field:
            id_field = "id"
        if not label_field:
            label_field = "label"
        if relevance_factor is None:
            relevance_factor = self.relevance_factor
        logger.debug(f"System prompt = {prompt}")
        concept_pairs = []
        for obj, _, _obj_meta in self.knowledge_source.search(
            text,
            relevance_factor=relevance_factor,
            collection=collection,
            **kwargs,
        ):
            id, label = obj.get(id_field, None), obj.get(label_field, None)
            if self.prefixes:
                if not any(id.startswith(prefix + ":") for prefix in self.prefixes):
                    continue
            if not id:
                raise ValueError(f"Object {obj} has no ID field {id_field}")
            if not label:
                raise ValueError(f"Object {obj} has no label field {label_field}")
            prompt += f"{label} // {id}   \n"
            concept_pairs.append((id, label))
        return concept_pairs, prompt
