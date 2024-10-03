from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from curategpt import BasicExtractor, ChromaDBAdapter, DBAdapter, Extractor
from curategpt.agents.chat_agent import ChatResponse
from curategpt.app.cart import Cart
from curategpt.extract import AnnotatedObject


@dataclass
class PageState:
    predicted_object: AnnotatedObject = None
    chat_response: ChatResponse = None
    results: List = None
    selected: Any = None


@dataclass
class ApplicationState:
    page: Optional[str] = None
    db: DBAdapter = field(default_factory=ChromaDBAdapter)
    extractor: Extractor = field(default_factory=BasicExtractor)
    cart: Cart = field(default_factory=Cart)
    page_states: Dict[str, PageState] = field(default_factory=dict)
    # selected: Any = None

    def get_page_state(self, page_name: str) -> PageState:
        if page_name not in self.page_states:
            self.page_states[page_name] = PageState()
        return self.page_states[page_name]


def get_state(st) -> ApplicationState:
    """
    Gets the application state from the streamlit session state

    :param st:
    :return:
    """
    if "state" not in st.session_state:
        st.session_state["state"] = ApplicationState()
    return st.session_state["state"]
