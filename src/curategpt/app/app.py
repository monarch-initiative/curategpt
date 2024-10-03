"""Streamlit app for CurateGPT."""

import json
import logging
from typing import List, Union

import numpy as np
import streamlit as st
import yaml
from scipy.spatial import distance_matrix

from curategpt import BasicExtractor
from curategpt.agents import MappingAgent
from curategpt.agents.chat_agent import ChatAgent, ChatResponse
from curategpt.agents.dase_agent import DatabaseAugmentedStructuredExtraction
from curategpt.agents.dragon_agent import DragonAgent
from curategpt.agents.evidence_agent import EvidenceAgent
from curategpt.app.components import (DimensionalityReductionOptions,
                                      limit_slider_component, vectors_to_fig)
from curategpt.app.helper import get_applicable_examples, get_case_collection
from curategpt.app.state import get_state
from curategpt.extract import OpenAIExtractor, RecursiveExtractor
from curategpt.wrappers import BaseWrapper
from curategpt.wrappers.investigation.jgi_wrapper import JGIWrapper
from curategpt.wrappers.literature import WikipediaWrapper
from curategpt.wrappers.literature.pubmed_wrapper import PubmedWrapper

PUBMED = "PubMed (via API)"
WIKIPEDIA = "Wikipedia (via API)"
JGI = "JGI (via API)"

CHAT = "Chat"
EXTRACT = "Extract"
SEARCH = "Search"
CLUSTER_SEARCH = "Cluster Search"
MATCH = "Match"
CURATE = "Curate"
ADD_TO_CART = "Add to Cart"
# EXTRACT = "Extract"
CITESEEK = "CiteSeek"
CART = "Cart"
HELP = "Help"
EXAMPLES = "Examples"
ABOUT = "About"

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


st.title("CurateGPT! _alpha_")
if not db.list_collection_names():
    st.warning("No collections found. Please use command line to load one.")

PAGES = [
    SEARCH,
    CLUSTER_SEARCH,
    CHAT,
    CURATE,
    EXTRACT,
    CITESEEK,
    MATCH,
    CART,
    ABOUT,
    HELP,
    EXAMPLES,
]


def _clear_sticky_page():
    logger.error("Clearing sticky page")
    state.page = None


# Sidebar with operation selection
option_selected = st.sidebar.selectbox(
    "Choose operation",
    PAGES,
    on_change=_clear_sticky_page,
)
option = state.page or option_selected
logger.error(f"Selected {option_selected}; sp={state.page}; opt={option}")
# logger.error(f"State: {state}")


def filtered_collection_names() -> List[str]:
    return [c for c in db.list_collection_names() if not c.endswith("_cached")]


collection = st.sidebar.selectbox(
    "Choose collection",
    filtered_collection_names() + [WIKIPEDIA, PUBMED, JGI],
    help="""
    A collection is a knowledge base. It could be anything, but
    it's likely your instance has some bio-ontologies pre-loaded.
    Select 'About' to see details of each collection
    """,
)

model_name = st.sidebar.selectbox(
    "Choose model",
    MODELS,
    help="""
    If this instance allows GPT-4 this likely works the best.
    (be considerate if someone else is paying).
    Open models may not do as well at extraction tasks
    (and they may be very slow).
    Note: if your instance is on EC2 it's likely open models
    that are not API backed will be unavailable or broken.
    """,
)

extraction_strategy = st.sidebar.selectbox(
    "Extraction strategy",
    ["Basic", "OAI Function", "SPIRES"],
    help="""
    The extraction strategy determines how the system will
    extract information from text.
    Note that both OpenAI functions and SPIRES require
    a schema to have been pre-loaded.
    """,
)

if extraction_strategy == "Basic":
    extractor = BasicExtractor()
elif extraction_strategy == "OAI Function":
    extractor = OpenAIExtractor()
elif extraction_strategy == "SPIRES":
    extractor = RecursiveExtractor()
else:
    raise ValueError(f"Unknown extraction strategy {extraction_strategy}")
state.extractor = extractor


background_collection = st.sidebar.selectbox(
    "Background knowledge",
    [NO_BACKGROUND_SELECTED, PUBMED, WIKIPEDIA, JGI] + list(db.list_collection_names()),
    help="""
    Background databases can be used to give additional context to the LLM.
    A standard pattern is to have a structured knowledge base as the main
    collection (this is used to find example records), and an unstructured
    database (e.g. github issues, abstracts, pdfs, ...) as background.
    Note you cannot currently add new databases using the UI. Contact
    the site admin to add new sources.
    """,
)

st.sidebar.markdown(f"Cart: {cart.size} items")

st.sidebar.markdown("Developed by the Monarch Initiative")


def get_chat_agent() -> Union[ChatAgent, BaseWrapper]:
    knowledge_source_collection = None
    if collection == PUBMED:
        source = PubmedWrapper(local_store=db, extractor=extractor)
    elif collection == WIKIPEDIA:
        source = WikipediaWrapper(local_store=db, extractor=extractor)
    elif collection == JGI:
        source = JGIWrapper(local_store=db, extractor=extractor)
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


if option == CURATE:
    page_state = state.get_page_state(CURATE)
    st.subheader(f"Curate new object in {collection}")
    if page_state.selected is None:
        fields = db.field_names(collection=collection)
    else:
        fields = list(page_state.selected.keys())
    input_obj = {}
    for f in fields:
        v = ""
        if page_state.selected is not None:
            v = page_state.selected.get(f, "")
        input_obj[f] = st.text_input(f"{f}", value=v)

    extractor.model_name = model_name

    if page_state.selected is not None:
        if st.button("Clear"):
            page_state.selected = None
            st.success("Current Selection Cleared!")

    generate_background = st.checkbox(
        "Generate background",
        help="""
                                            If checked, a full text description is first generated from the LLM.
                                            This is then used as background knowledge to generate the target object.
                                            """,
    )
    instructions = st.text_input(
        "Additional Instructions",
        help="""
                                        Enter any additional instructions for the model here.
                                        E.g. 'You MUST include a definition field in your answer.
                                        """,
    )

    examples_limit = limit_slider_component()

    if st.button("Suggest"):
        daca = DragonAgent(knowledge_source=db, extractor=extractor)
        if background_collection != NO_BACKGROUND_SELECTED:
            # TODO: DRY
            if background_collection == PUBMED:
                daca.document_adapter = PubmedWrapper(local_store=db, extractor=extractor)
                daca.collection = None
            elif background_collection == WIKIPEDIA:
                daca.document_adapter = WikipediaWrapper(local_store=db, extractor=extractor)
                daca.collection = None
            else:
                daca.document_adapter = db
                daca.document_adapter_collection = background_collection
        st.write(f"Generating using: **{extractor.model_name}** using *{collection}* for examples")
        if background_collection:
            st.write(f"Using background knowledge from: *{background_collection}*")
        rules = [instructions] if instructions else None
        page_state.predicted_object = daca.complete(
            input_obj,
            generate_background=generate_background,
            collection=collection,
            rules=rules,
            limit=examples_limit,
        )

    if page_state.predicted_object:
        created = page_state.predicted_object
        obj = created.object
        st.subheader("Created object")
        # st.write("<pre>" + yaml.dump(obj, sort_keys=False) + "</pre>", unsafe_allow_html=True)
        st.code(yaml.dump(obj, sort_keys=False), language="yaml")
        add_button = st.button(f"Add to {collection}")
        if add_button:
            db.insert([obj], collection=collection)
            st.write("Added!!!")
        st.subheader("Debug info")
        st.write("Prompt:")
        st.code(created.annotations["prompt"])
        # st.write("Property:", property_query)


elif option == MATCH:
    page_state = state.get_page_state(MATCH)
    st.subheader(f"Match to entities in *{collection}*")
    search_query = st.text_input(
        "Match text",
        help="Enter label of concept to match.",
    )
    relevant_fields = st.text_input(
        "Relevant fields",
        help="Comma-separated (e.g. label, definition).",
    )
    limit = st.slider(
        "Max results",
        min_value=0,
        max_value=200,
        value=10,
        step=1,
        help="""
                                       Number of results max
                                       """,
    )

    if st.button("Match"):
        cm = db.collection_metadata(collection, include_derived=True)
        st.write(f"Searching over {cm.object_count} objects using embedding model {cm.model}")
        mapper = MappingAgent(knowledge_source=db, extractor=extractor)
        if not relevant_fields:
            relevant_fields = "label"
        relevant_fields = [f.strip() for f in relevant_fields.split(",")]
        results = mapper.match(
            search_query,
            collection=collection,
            fields=relevant_fields,
            limit=limit,
        )

        rows = [{"subject": search_query, "object": m.object_id} for m in results.mappings]
        html = html_table(rows)
        st.write(html, unsafe_allow_html=True)
        st.subheader("Prompt", help="for debugging")
        st.write(results.prompt)
        st.subheader("Response", help="for debugging")
        st.write(results.response_text)

# Search operation
elif option == SEARCH:
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

    # if not hasattr(st.session_state, "add_to_cart_index"):
    #    st.session_state.add_to_cart_index = None

    # if False and st.session_state.add_to_cart_index is not None:
    #    # add object from first result
    #    obj = page_state.results[st.session_state.add_to_cart_index][0]
    #    cart.add(obj)
    #    page_state.selected = obj
    #    st.session_state.add_to_cart_index = None  # Reset
    #    st.success("Document added to cart!!!!!")

    # elif st.button("Search"):
    if st.button("Search"):
        results = db.search(
            search_query, collection=collection, relevance_factor=relevance_factor, include=["*"]
        )
        page_state.results = list(results)
        st.session_state.add_to_cart_index = None  # Reset

    if page_state.results:
        results = page_state.results
        cm = db.collection_metadata(collection, include_derived=True)
        if cm:
            st.write(f"Searching over {cm.object_count} objects using embedding model {cm.model}")
        else:
            st.write(f"Dynamic search over {collection}...")

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
            if st.button(f"Curate {i+1}"):
                autoadd_page_state = state.get_page_state(CURATE)
                autoadd_page_state.selected = obj
                st.success("Curating!")
                state.page = CURATE
                st.experimental_rerun()
            if st.button(f"Find evidence {i+1}"):
                autoadd_page_state = state.get_page_state(CITESEEK)
                autoadd_page_state.selected = obj
                st.success("OK!")
                state.page = CITESEEK
                st.experimental_rerun()


elif option == CLUSTER_SEARCH:
    page_state = state.get_page_state(CLUSTER_SEARCH)
    st.subheader(f"Cluster Search documents in *{collection}*")
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
        help="""How much to weight the relevance vs diversity of the search query.
                                 If this is set to less than 1.0, then MMR will be used to diversify the results.
                                 (this corresponds to the lambda parameter in the MMR formula)
                                 """,
    )

    limit = st.slider(
        "Max results",
        min_value=0,
        max_value=1000,
        value=50,
        step=1,
        help="""
                                   Number of results max
                                   """,
    )

    method = st.radio(
        "Dimensionality Reduction Method", options=list(DimensionalityReductionOptions), index=0
    )

    if st.button("Search"):
        cm = db.collection_metadata(collection, include_derived=True)
        st.write(f"Searching over {cm.object_count} objects using embedding model {cm.model}")
        include = ["*"]
        results = db.search(
            search_query,
            collection=collection,
            relevance_factor=relevance_factor,
            include=include,
            limit=limit,
        )
        labels = []
        vectors = []
        label_field = db.label_field(collection)
        id_field = db.identifier_field(collection)
        for i, (obj, _distance, doc) in enumerate(results):
            # use label field preferentially, then id field, then numeric
            labels.append(obj.get(label_field, obj.get(id_field, f"Object {i}")))
            vectors.append(np.array(doc["embeddings"]))
        distances = distance_matrix(vectors, vectors)
        fig = vectors_to_fig(labels, np.array(vectors), method=method)
        st.pyplot(fig)

elif option == EXTRACT:
    page_state = state.get_page_state(EXTRACT)
    st.subheader("Extract object", help="Extract a structured object from text.")
    st.write(f"Examples will be drawn from **{collection}**")
    if background_collection != NO_BACKGROUND_SELECTED:
        st.write(f"Background knowledge will be drawn from **{background_collection}**")
    text_query = st.text_input(
        "Text", help="Enter the label or description of the entity type you want to add."
    )
    generate_background = st.checkbox(
        "Generate background",
        help="""
                                        If checked, a full text description is first generated from the LLM.
                                        This is then used as background knowledge to generate the target object.
                                        """,
    )
    instructions = st.text_input(
        "Additional Instructions",
        help="""
                                    Enter any additional instructions for the model here.
                                    E.g. 'You MUST include a definition field in your answer.
                                    """,
    )

    examples_limit = limit_slider_component()

    examples = get_applicable_examples(collection, EXTRACT)
    extractor.model_name = model_name

    if st.button(EXTRACT):
        dase = DatabaseAugmentedStructuredExtraction(knowledge_source=db, extractor=extractor)
        if background_collection != NO_BACKGROUND_SELECTED:
            if background_collection == PUBMED:
                dase.document_adapter = PubmedWrapper(local_store=db, extractor=extractor)
                dase.collection = None
            elif background_collection == WIKIPEDIA:
                dase.document_adapter = WikipediaWrapper(local_store=db, extractor=extractor)
                dase.collection = None
            else:
                dase.document_adapter = db
                dase.document_adapter_collection = background_collection
        st.write(f"Generating using: **{extractor.model_name}** using *{collection}* for examples")
        if background_collection:
            st.write(f"Using background knowledge from: *{background_collection}*")
        rules = [instructions] if instructions else None
        page_state.predicted_object = dase.extract(
            text_query,
            generate_background=generate_background,
            collection=collection,
            rules=rules,
            limit=examples_limit,
        )

    if page_state.predicted_object:
        created = page_state.predicted_object
        obj = created.object
        st.subheader("Created object")
        # st.write("<pre>" + yaml.dump(obj, sort_keys=False) + "</pre>", unsafe_allow_html=True)
        st.code(yaml.dump(obj, sort_keys=False), language="yaml")
        add_button = st.button(f"Add to {collection}")
        if add_button:
            db.insert([obj], collection=collection)
            st.write("Added!!!")
        st.subheader("Debug info")
        st.write("Prompt:")
        st.code(created.annotations["prompt"])


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
    # examples = get_applicable_examples(collection, CITESEEK)
    # st.write("Examples:")
    # st.write(f"<details>{html_table(examples)}</details>", unsafe_allow_html=True)

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

elif option == CART:
    page_state = state.get_page_state(CART)
    st.subheader("Your items")
    for i, item in enumerate(cart.items):
        st.write(f"## Item {i}")
        item_obj = item.object
        st.code(yaml.dump(item_obj, sort_keys=False), language="yaml")
        if st.button(f"Remove {i} from cart"):
            cart.remove(item)

elif option == EXAMPLES:
    page_state = state.get_page_state(EXAMPLES)
    cc = get_case_collection()
    st.subheader("Examples")
    st.code(yaml.dump(cc, sort_keys=False), language="yaml")


elif option == ABOUT:
    page_state = state.get_page_state(ABOUT)
    st.subheader("About this instance")
    st.write(
        f"**DB:** {type(db).__name__} schema: {db.schema_proxy.name if db.schema_proxy else None}"
    )
    st.write("Collections:")
    rows = []
    for cn in db.collections():
        meta = db.collection_metadata(collection_name=cn, include_derived=True)
        rows.append(meta.dict())
    st.table(rows)

elif option == HELP:
    page_state = state.get_page_state(HELP)
    st.subheader("About")
    st.write(
        "CurateGPT is a tool for generating new entries for a knowledge base, assisted by LLMs."
    )
    st.write(
        "It is a highly generic system, but it's likely the instance"
        "you are using now is configured to work with ontologies."
    )
    st.subheader("Issues")
    st.write(
        "If you have any issues, please raise them on the"
        "[GitHub issue tracker](https://github.com/monarch-initiative/curategpt)."
    )
    st.subheader("Warning!")
    st.caption("CurateGPT is pre-alpha, documentation is incomplete!")
    st.caption(
        "If you are using a publicly deployed instance, some operations may be slow, or broken"
    )
    st.subheader("Instructions")
    st.write("Use the sidebar to select the operation you want to perform.")
    st.write(" * Synthesize: the core operation. Generate a new entry for the selected collection.")
    st.write(" * Chat: chat to a structured knowledge base or unstructured source.")
    st.write(" * Search: Search the vector stores.")
    st.write(
        " * Insert: Manually add data (do this responsibly if on public instance - no auth yet!."
    )
    st.write(" * About: View metadata for each instance.")
    st.subheader("FAQ")
    st.write("### Why are there no IDs?")
    st.write("LLMs will hallucinate IDs so we transform to CamelCase labels for demo purposes.")
    st.write("In future versions we will have a more elegant solution.")
    st.write("### What is the PubMed collection?")
    st.write("This is a special *virtual* collections. It is not populated ahead of time.")
    st.write("When this is used as a source, the pubmed API is called with a relevancy search.")
    st.write("These results are then combined with others to answer the query.")
    st.write("### What is the 'background' collection?")
    st.write(f"This is used only by '{EXTRACT}' to provide additional context.")
