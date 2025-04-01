"""Streamlit app for CurateGPT."""

import json
import logging
from typing import List, Union

import streamlit as st
import yaml

from curategpt import BasicExtractor
from curategpt.agents.chat_agent import ChatAgent, ChatResponse
from curategpt.agents.evidence_agent import EvidenceAgent
from curategpt.app.helper import get_applicable_examples
from curategpt.app.state import get_state
from curategpt.wrappers import BaseWrapper
from curategpt.wrappers.literature import WikipediaWrapper
from curategpt.wrappers.literature.pubmed_wrapper import PubmedWrapper

PUBMED = "PubMed (via API)"
WIKIPEDIA = "Wikipedia (via API)"
# Removed JGI and ESS-Dive
# JGI = "JGI (via API)"
# ESSDIVE = "ESS-DeepDive (via API)"

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
    "gpt-4o",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4",
    "chatgpt-16k",
    "nous-hermes-13b",
    "llama2",
]

logger = logging.getLogger(__name__)

state = get_state(st)
db = state.db
cart = state.cart


st.title("Alzheimers AI assistant")
if not db.list_collection_names():
    st.warning("No collections found. Please use command line to load one.")

# Include Chat, Search, and CiteSeek in PAGES
PAGES = [
    CHAT,
    SEARCH,
    CITESEEK,
]


def _clear_sticky_page():
    logger.error("Clearing sticky page")
    state.page = None


# Sidebar with operation selection
option_selected = st.sidebar.selectbox(
    "Choose operation",
    PAGES,
    index=0,  # Set Chat as default
    on_change=_clear_sticky_page,
)
option = state.page or option_selected
logger.error(f"Selected {option_selected}; sp={state.page}; opt={option}")
# logger.error(f"State: {state}")


def filtered_collection_names() -> List[str]:
    return [c for c in db.list_collection_names() if not c.endswith("_cached")]


collection = st.sidebar.selectbox(
    "Choose collection",
    [PUBMED, WIKIPEDIA] + filtered_collection_names(),  # Removed JGI and ESSDIVE, put PubMed first
    index=0,  # Set PubMed as default
    help="""
    A collection is a knowledge base. It could be anything, but
    it's likely your instance has some bio-ontologies pre-loaded.
    Select 'About' to see details of each collection
    """,
)

# Set default model and show which model is being used
model_name = "gpt-4o"
st.sidebar.markdown(f"**Using model:** {model_name}")

# Removed extraction_strategy and background_collection sections

# Default to BasicExtractor
extractor = BasicExtractor()
state.extractor = extractor

# Add background_collection for CiteSeek functionality
background_collection = st.sidebar.selectbox(
    "Background knowledge for Search and Citeseek",
    [NO_BACKGROUND_SELECTED, PUBMED, WIKIPEDIA],
    index=0,
    help="""
    Background databases can be used to give additional context to the LLM.
    Select PubMed to search medical literature.
    """,
)

# st.sidebar.markdown(f"Cart: {cart.size} items")

st.sidebar.markdown("Developed by the Monarch Initiative")


def get_chat_agent() -> Union[ChatAgent, BaseWrapper]:
    knowledge_source_collection = None
    if collection == PUBMED:
        source = PubmedWrapper(local_store=db, extractor=extractor)
    elif collection == WIKIPEDIA:
        source = WikipediaWrapper(local_store=db, extractor=extractor)
    # Removed JGI and ESSDIVE cases
    else:
        source = db
        knowledge_source_collection = collection
    return ChatAgent(
        knowledge_source=source,
        knowledge_source_collection=knowledge_source_collection,
        extractor=extractor,
    )


def ask_chatbot(query, expand=False) -> ChatResponse:
    return get_chat_agent().chat(query, expand=expand)


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


# Search operation
if option == SEARCH:
    page_state = state.get_page_state(SEARCH)
    st.subheader(f"Search documents in *{collection}*")
    search_query = st.text_input(
        "Search by text",
        help="Enter any text - embedding similarity will be used to find similar objects.",
    )

    relevance_factor = st.slider(
        "Relevance Factor",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="""
                                 How much to weight the relevance vs diversity of the search query.
                                 If this is set to less than 1.0, then MMR will be used to diversify the results.
                                 (this corresponds to the lambda parameter in the MMR formula)
                                 """,
    )

    if st.button("Search"):
        results = db.search(
            search_query, collection=collection, relevance_factor=relevance_factor, include=["*"]
        )
        page_state.results = list(results)
        st.session_state.add_to_cart_index = None  # Reset

    if page_state.results:
        results = page_state.results
        cm = db.collection_metadata(collection, include_derived=True)
        try:
            if cm:
                st.write(
                    f"Searching over {cm.object_count} objects using embedding model {cm.model}"
                )
            else:
                st.write(f"Dynamic search over {collection}...")
        except AttributeError as e:
            st.write(f"Searching over {collection} but encountered an error: {e}")

        def _flat(obj: dict, limit=40) -> dict:
            if not obj:
                return {}
            return {
                k: str(json.dumps(v) if isinstance(v, (list, dict)) else v)[0:limit]
                for k, v in obj.items()
            }

        rows = [
            {"rank": i + 1, "distance": distance, **_flat(obj), "doc": _flat(doc)}
            for i, (obj, distance, doc) in enumerate(results)
        ]
        html = html_table(rows)
        st.write(html, unsafe_allow_html=True)

        for i, (obj, _distance, _doc) in enumerate(results):
            st.write(f"## Result {i+1}")
            st.code(yaml.dump(obj, sort_keys=False))
            if st.button(f"Add to cart {i+1}"):
                cart.add(obj)
                st.success("Document added to cart!")


elif option == CHAT:
    page_state = state.get_page_state(CHAT)
    st.subheader("Chat with a knowledge base")
    query = st.text_area(
        f"Ask me anything (within the scope of {collection})!",
        help="You can query the current knowledge base using natural language.",
    )

    limit = st.slider(
        "Detail",
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
    expand = st.checkbox(
        "Expand query",
        help="""
                                                If checked, perform query expansion (pubmed only).
                                                """,
    )
    extractor.model_name = model_name
    examples = get_applicable_examples(collection, CHAT)
    st.write("Examples:")
    st.write(f"<details>{html_table(examples)}</details>", unsafe_allow_html=True)

    if st.button(CHAT):
        response = ask_chatbot(query, expand=expand)
        page_state.chat_response = response

    if page_state.chat_response:
        response = page_state.chat_response
        st.markdown(response.formatted_body)
        add_button = st.button("Add to your cart")
        if add_button:
            logger.error("Adding to cart")
            cart.add(response)
            st.write("Added to cart!")

        st.markdown("## References")
        for ref, text in response.references.items():
            st.subheader(f"Reference {ref}", anchor=f"ref-{ref}")
            st.code(text, language="yaml")
            if st.button(f"Add to cart {ref}"):
                # TODO: unpack
                cart.add({"text": text, "id": ref})
                st.success("Document added to cart!")
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
        "Detail",
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
    extractor.model_name = model_name

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
