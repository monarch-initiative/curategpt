import logging
from typing import List

import streamlit as st
import yaml

from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.chat import ChatEngine
from curate_gpt.agents.dalek import DatabaseAugmentedExtractor
from curate_gpt.extract import BasicExtractor

SEARCH = "Search"
ABOUT = "About"
INSERT = "Insert"
CREATE = "Synthesize"
CHAT = "Chat"
EXTRACT = "Extract"
CART = "Cart"
HELP = "Help"

NO_BACKGROUND_SELECTED = "No background collection"

MODELS = ["gpt-3.5-turbo", "gpt-4", "nous-hermes-13b", "llama2"]

logger = logging.getLogger(__name__)

db = ChromaDBAdapter()
extractor = BasicExtractor()
agent = DatabaseAugmentedExtractor(kb_adapter=db, extractor=extractor)
chatbot = ChatEngine(kb_adapter=db, extractor=extractor)

st.title("CurateGPT! _alpha_")
if not db.list_collection_names():
    st.warning("No collections found. Please use command line to load one.")

# Sidebar with operation selection
option = st.sidebar.selectbox("Choose operation", (CHAT, SEARCH, CREATE, INSERT, CART, ABOUT, HELP, ))


collection = st.sidebar.selectbox(
    "Choose collection",
    list(db.list_collection_names()),
    help="""
    A collection is a knowledge base. It could be anything, but
    it's likely your instance has some bio-ontologies pre-loaded.
    Select 'About' to see details of each collection
    """
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
    """
)

background_collection = st.sidebar.selectbox(
    "Background knowledge",
    [NO_BACKGROUND_SELECTED] + list(db.list_collection_names()),
    help="""
    Background databases can be used to give additional context to the LLM.
    A standard pattern is to have a structured knowledge base as the main
    collection (this is used to find example records), and an unstructured
    database (e.g. githun issues, abstracts, pdfs, ...) as background.
    
    Note you cannot currently add new databases using the UI. Contact
    the site admin to add new sources.
    """
)


def html_table(rows: List[dict]) -> str:
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



# Insert operation
if option == INSERT:
    st.subheader(f"Insert new document in {collection}")
    objs = list(db.peek(collection=collection))
    fields = []
    for obj in objs:
        for f in obj:
            if f not in fields:
                fields.append(f)
    inputs = {}
    for f in fields:
        inputs[f] = st.text_input(f"{f}")

    if st.button("Insert"):
        # data = {"name": name, "age": age, "email": email}
        db.insert([inputs], collection=collection)
        st.success("Document inserted successfully!")

# Search operation
elif option == SEARCH:
    st.subheader(f"Search documents in *{collection}*")
    search_query = st.text_input("Search by text",
                                 help="Enter any text - embedding similarity will be used to find similar objects.")

    relevance_factor = st.slider("Relevance Factor", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                                 help="""
                                 How much to weight the relevance vs diversity of the search query.
                                 If this is set to less than 1.0, then MMR will be used to diversify the results.
                                 (this corresponds to the lambda parameter in the MMR formula)
                                 """)

    if st.button("Search"):
        cm = db.collection_metadata(collection, include_derived=True)
        st.write(f"Searching over {cm.object_count} objects using embedding model {cm.model}")
        results = db.search(search_query, collection=collection, relevance_factor=relevance_factor)
        rows = [
            {"rank": i + 1, "distance": distance, "obj": obj}
            for i, (obj, distance, _) in enumerate(results)
        ]
        html = html_table(rows)
        st.write(html, unsafe_allow_html=True)

elif option == CREATE:
    st.subheader(f"Synthesize object", help="Generate a new object from a seed query.")
    st.write(f"Examples will be drawn from **{collection}**")
    if background_collection != NO_BACKGROUND_SELECTED:
        st.write(f"Background knowledge will be drawn from **{background_collection}**")
    search_query = st.text_input("Seed",
                                 help="Enter the label or description of the entity type you want to add.")
    property_query = st.text_input("Property (e.g. label)",
                                   help="""The value here depends on the data model of the index.
                                           For ontologies, use 'label' if your query if a label,
                                           or 'description' if your query is a description of a term.
                                        """)
    generate_background = st.checkbox("Generate background",
                                        help="""
                                        If checked, a full text description is first generated from the LLM.
                                        This is then used as background knowledge to generate the target object.
                                        """)
    instructions = st.text_input("Additional Instructions",
                                    help="""
                                    Enter any additional instructions for the model here.
                                    E.g. 'You MUST include a definition field in your answer.
                                    """)

    examples_limit = st.slider("Max examples", min_value=0, max_value=20, value=10, step=1,
                               help="""
                               Examples that are similar to your query are picked from the selected
                               knowledge base, and used as context to guide the LLM.
                               If you pick too many examples, it may go beyond the limits of the context window
                               for the model you selected.
                               """)

    extractor.model_name = model_name

    # Check for session state variables
    if "results" not in st.session_state:
        st.session_state.results = []

    if st.button(CREATE):
        if not property_query:
            property_query = "label"
        if background_collection != NO_BACKGROUND_SELECTED:
            agent.document_adapter = db
            agent.document_adapter_collection = background_collection
        st.write(f"Generating using: **{extractor.model_name}** using *{collection}* for examples")
        rules = [instructions] if instructions else None
        st.session_state.results = [
            agent.generate_extract(
                search_query,
                #target_class="OntologyClass",
                context_property=property_query,
                generate_background=generate_background,
                collection=collection,
                rules=rules,
                limit=examples_limit,
            )
        ]

    if st.session_state.results:
        created = st.session_state.results[0]
        obj = created.object
        st.subheader("Created object")
        # st.write("<pre>" + yaml.dump(obj, sort_keys=False) + "</pre>", unsafe_allow_html=True)
        st.code(yaml.dump(obj, sort_keys=False), language="yaml")
        add_button = st.button(f"Add to {collection}")
        if add_button:
            print(f"ADDING!!!!!!!!!")
            db.insert([obj], collection=collection)
            # delete_data(collection, doc['_id'])  # Assuming each document has a unique '_id' field
            st.write(f"Added!!!")
        st.subheader("Debug info")
        st.write("Prompt:")
        st.code(created.annotations["prompt"])
        st.write("Property:", property_query)

elif option == EXTRACT:
    st.subheader(f"Extract from text")
    search_query = st.text_area("Text to parse")
    property_query = st.text_input("Property (e.g. label)")
    generate_background = st.checkbox("Generate background")

    extractor.model_name = model_name

    # Check for session state variables
    if "results" not in st.session_state:
        st.session_state.results = []

    if st.button(EXTRACT):
        if not property_query:
            property_query = "label"
        st.session_state.results = [
            agent.generate_extract(
                search_query,
                target_class="OntologyClass",
                context_property=property_query,
                generate_background=generate_background,
                collection=collection,
            )
        ]

    if st.session_state.results:
        created = st.session_state.results[0]
        obj = created.object
        st.subheader("Created object")
        # st.write("<pre>" + yaml.dump(obj, sort_keys=False) + "</pre>", unsafe_allow_html=True)
        st.code(yaml.dump(obj, sort_keys=False), language="yaml")
        add_button = st.button(f"Add to {collection}")
        if add_button:
            print(f"ADDING!!!!!!!!!")
            db.insert([obj], collection=collection)
            # delete_data(collection, doc['_id'])  # Assuming each document has a unique '_id' field
            st.write(f"Added!!!")
        st.subheader("Debug info")
        st.write("Prompt:")
        st.code(created.annotations["prompt"])
        st.write("Property:", property_query)

elif option == CHAT:
    st.subheader(f"Chat with a knowledge base")
    query = st.text_area(f"Ask me anything (within the scope of {collection})!",
                         help="You can query the current knowledge base using natural language.")

    limit = st.slider("Detail", min_value=0, max_value=30, value=10, step=1,
                               help="""
                                   Behind the scenes, N entries are fetched from the knowledge base,
                                   and these are fed to the LLM. Selecting more examples may give more
                                   complete results, but may also exceed context windows for the model.
                                   """)
    extractor.model_name = model_name

    if st.button(CHAT):
        response = chatbot.chat(query, collection=collection)
        st.markdown(response.formatted_response)
        for ref, text in response.references.items():
            st.subheader(f"Reference {ref}", anchor=f"ref-{ref}")
            st.code(text, language="yaml")

elif option == CART:
    st.subheader("Coming soon!")


elif option == ABOUT:
    st.subheader("About this instance")
    st.write(f"**DB:** {type(db).__name__} schema: {db.schema_proxy.name if db.schema_proxy else None}")
    st.write("Collections:")
    rows = []
    for cn in db.collections():
        meta = db.collection_metadata(collection_name=cn, include_derived=True)
        rows.append(meta.dict())
    st.table(rows)

elif option == HELP:
    st.subheader("About")
    st.write("CurateGPT is a tool for generating new entries for a knowledge base, assisted by LLMs.")
    st.write("It is a highly generic system, but it's likely the instance you are using now is configured to work with ontologies.")
    st.subheader("Warning!")
    st.caption("CurateGPT is pre-alpha, documentation is incomplete!")
    st.caption("If you are using a publicly deployed instance, some operations may be slow, or broken")
    st.subheader("Instructions")
    st.write("Use the sidebar to select the operation you want to perform.")
    st.write(" * Synthesize: the core operation. Generate a new entry for the selected collection.")
    st.write(" * Search: Search the vector stores.")
    st.write(" * Insert: Manually add data (do this responsibly if on public instance - no auth yet!.")
    st.write(" * About: View metadata for each instance.")
    st.subheader("FAQ")
    st.write("### Why are there no IDs")
    st.write("LLMs will hallucinate IDs so we transform to CamelCase labels for demo purposes.")
    st.write("In future versions we will have a more elegant solution.")

