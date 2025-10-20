"""Streamlit app for CurateGPT."""

import json
import logging
import os
from typing import List, Union

import streamlit as st
import yaml

from curategpt import BasicExtractor
from curategpt.agents.chat_agent import ChatAgentAlz, ChatResponse
from curategpt.agents.evidence_agent import EvidenceAgent
from curategpt.app.state import get_state
from curategpt.wrappers import BaseWrapper
from curategpt.wrappers.literature import WikipediaWrapper
from curategpt.wrappers.literature.pubmed_wrapper import PubmedWrapper
from curategpt.wrappers.paperqa.paperqawrapper import PaperQAWrapper

PUBMED = "PubMed"
WIKIPEDIA = "Wikipedia"
PAPERQA = "Trusted Alzheimers Corpus (small)"
PAPERQA2 = "Trusted Alzheimers Corpus (medium)"
PAPERQA3 = "Trusted Alzheimers Corpus (large)"

CHAT = "Chat"
SEARCH = "Search"

# Removed other operations
# EXTRACT = "Extract"
# CLUSTER_SEARCH = "Cluster Search"
# MATCH = "Match"
# BOOTSTRAP = "Bootstrap"
# CURATE = "Curate"
# ADD_TO_CART = "Add to Cart"
CITESEEK = "CiteSeek"
# CART = "Cart"
# HELP = "Help"
# EXAMPLES = "Examples"
# ABOUT = "About"

NO_BACKGROUND_SELECTED = "No background collection"

MODELS = [
    "gpt-4o"
]

logger = logging.getLogger(__name__)

state = get_state(st)
db = state.db
cart = state.cart


st.title("Alzheimer's AI Assistant")

# Check if PQA_HOME environment variable is set for PaperQA
if PAPERQA in [PUBMED, PAPERQA, WIKIPEDIA] and os.environ.get("PQA_HOME") is None:
    st.warning(
        "PQA_HOME environment variable is not set. To use the Alzheimer's Papers collection, "
        "you need to set PQA_HOME to the directory containing your indexed papers. "
        "Use 'curategpt paperqa index /path/to/papers' to create an index."
    )

# Check if additional corpora are available
has_second_corpus = os.environ.get("PQA_HOME2") is not None
has_third_corpus = os.environ.get("PQA_HOME3") is not None
if not db.list_collection_names():
    st.warning("No collections found. Please use command line to load one.")

# Include only Chat in PAGES
PAGES = [
    CHAT
]


def _clear_sticky_page():
    logger.error("Clearing sticky page")
    state.page = None


# Always use Chat operation (no sidebar selector needed)
option = CHAT
logger.error(f"Selected Chat; opt={option}")


def filtered_collection_names() -> List[str]:
    return [c for c in db.list_collection_names() if not c.endswith("_cached")]


# Build collection options dynamically
collection_options = [PAPERQA]
if has_second_corpus:
    collection_options.append(PAPERQA2)
if has_third_corpus:
    collection_options.append(PAPERQA3)
collection_options.extend([PUBMED] + filtered_collection_names() + ["No collection"])

collection = st.sidebar.selectbox(
    "Choose collection",
    collection_options,
    index=1,  # Set PAPERQA2 (medium - 1k papers) as default
    help="""
    A collection is a knowledge base that is used for retrieval augmented generation (RAG)
    to support the AI model when answering questions.
    Select 'Trusted Alzheimers Corpus (small)', 'Trusted Alzheimers Corpus (medium)',
    or 'Trusted Alzheimers Corpus (large)' to use corpora of 358, 1,065, and 3,028
    Alzheimer's research papers, respectively, curated by experts at Alzforum,
    U of Washington and Wash U.
    Select 'Pubmed' to use all of Pubmed.
    Select 'kg_alz_humanized' to use KG Alzheimers (beta)
    Select 'No collection' to interact with the model directly without a knowledge base.
    """,
)

# Removed extraction_strategy and background_collection sections

# Default to BasicExtractor with gpt-4o model
extractor = BasicExtractor()
extractor.model_name = "gpt-4o"
state.extractor = extractor


# st.sidebar.markdown(f"Cart: {cart.size} items")


def get_chat_agent() -> Union[ChatAgentAlz, BaseWrapper]:
    if collection == "No collection":
        return ChatAgentAlz(extractor=extractor)
    elif collection == PUBMED:
        source = PubmedWrapper(local_store=db, extractor=extractor)
    elif collection == WIKIPEDIA:
        source = WikipediaWrapper(local_store=db, extractor=extractor)
    elif collection == PAPERQA:
        source = PaperQAWrapper(extractor=extractor)
    elif collection == PAPERQA2:
        source = PaperQAWrapper(extractor=extractor, corpus_id="2")
    elif collection == PAPERQA3:
        source = PaperQAWrapper(extractor=extractor, corpus_id="3")
    else:
        source = db

    agent = ChatAgentAlz(
        knowledge_source=source,
        knowledge_source_collection=collection,
        extractor=extractor,
    )

    if agent.knowledge_source is None:
        raise ValueError(f"Knowledge source is None for collection {collection}")

    return agent


def ask_chatbot(query, expand=False, limit=10) -> ChatResponse:
    agent = get_chat_agent()
    if collection == "No collection":
        response = agent.extractor.model.prompt(query, system="You are a helpful Alzheimer's disease expert.")
        return ChatResponse(
            body=response.text(),
            formatted_body=response.text(),
            prompt=query,
            references={},
            uncited_references={}
        )
    else:
        return agent.chat(query, expand=expand, limit=limit)


def html_table(rows: List[dict]) -> str:
    if len(rows) == 0:
        rows = [{"No data": "No data"}]
    hdr = rows[0].keys()
    html_content = '<table border="1">'
    cols = [f"<th>{h}</th>" for h in hdr]
    html_content += f"<tr>{''.join(cols)}</tr>"
    for row in rows:
        html_content += "<tr>"
        for col in hdr:
            v = row.get(col, "")
            if isinstance(v, dict):
                v = f"<pre>{yaml.dump(v, sort_keys=False)}</pre>"
            html_content += f"<td>{v}</td>"
        html_content += "</tr>"
    html_content += "</table>"
    return html_content


if option == CHAT:
    page_state = state.get_page_state(CHAT)
    if collection == "No collection":
        st.subheader("Chat with the Alzheimer's AI assistant")
        query = st.text_area(
            "Ask me anything about Alzheimer's disease",
            help="Ask questions directly to the AI without using a knowledge base.",
        )
    else:
        query = st.text_area(
            f"Ask me anything about Alzheimer's disease (within the scope of {collection})",
            help="You can query the current knowledge base using natural language.",
        )

    # Only show these controls if using a knowledge base
    if collection != "No collection":
        limit = st.slider(
            "Relevant publications to retrieve",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
            help="""
                                       Behind the scenes, N entries are fetched from the knowledge base,
                                       and these are fed to the LLM. Selecting more examples may give more
                                       complete results, but may also exceed context windows for the model.
                                       """,
        )
    else:
        # Set default values when not using a knowledge base
        limit = 0

    if st.button(CHAT):
        response = ask_chatbot(query, expand=False, limit=limit)
        page_state.chat_response = response

    if page_state.chat_response:
        response = page_state.chat_response
        st.markdown(response.formatted_body)
        # add_button = st.button("Add to your cart")
        # if add_button:
        #     logger.error("Adding to cart")
        #     cart.add(response)
        #     st.write("Added to cart!")

        st.markdown("## References")
        for ref, text in response.references.items():
            st.subheader(f"Reference {ref}", anchor=f"ref-{ref}")
            st.code(text, language="yaml")
            # if st.button(f"Add to cart {ref}"):
            #     # TODO: unpack
            #     cart.add({"text": text, "id": ref})
            #     st.success("Document added to cart!")
        if response.uncited_references:
            st.markdown("## Uncited references")
            st.caption(
                "These references were flagged as potentially relevant, but a citation was not detected."
            )
            for ref, text in response.uncited_references.items():
                st.subheader(f"Reference {ref}", anchor=f"ref-{ref}")
                st.code(text, language="yaml")

elif option == CITESEEK:
    page_state = state.get_page_state(CITESEEK)
    st.subheader("Find citations for a claim")
    v = None
    if page_state.selected is not None:
        v = yaml.dump(page_state.selected, sort_keys=False)
    query = st.text_area(
        f"Enter YAML object to be verified by {collection}",
        value=v,
        help="Copy the YAML from some of the other outputs of this tool.",
    )

    limit = st.slider(
        "Relevant publications to retrieve",
        min_value=0,
        max_value=30,
        value=10,
        step=1,
        help="""
                                   Behind the scenes, N entries are fetched from the knowledge base,
                                   and these are fed to the LLM. Selecting more examples may give more
                                   complete results, but may also exceed context windows for the model.
                                   """,
    )

    if page_state.selected is not None:
        if st.button("Clear"):
            page_state.selected = None
            st.success("Current Selection Cleared!")

    if st.button(CITESEEK):
        chat_agent = get_chat_agent()
        ea = EvidenceAgent(chat_agent=chat_agent)
        try:
            query_obj = yaml.safe_load(query)
        except yaml.YAMLError:
            try:
                query_obj = json.loads(query)
            except json.JSONDecodeError as exc:
                st.warning(f"Invalid YAML or JSON: {exc}")
                query_obj = None
        if query_obj:
            response = ea.find_evidence(query_obj)
            # TODO: reuse code for this
            st.markdown(response.formatted_body)
            st.markdown("## References")
            for ref, text in response.references.items():
                st.subheader(f"Reference {ref}", anchor=f"ref-{ref}")
                st.code(text, language="yaml")
            if response.uncited_references:
                st.markdown("## Uncited references")
                st.caption(
                    "These references were flagged as potentially relevant, but a citation was not detected."
                )
                for ref, text in response.uncited_references.items():
                    st.subheader(f"Reference {ref}", anchor=f"ref-{ref}")
                    st.code(text, language="yaml")
