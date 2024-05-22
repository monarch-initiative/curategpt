from enum import Enum
from typing import List

import streamlit as st
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEFAULT_LIMIT_SLIDER_HELP = """
Examples that are similar to your query are picked from the selected
knowledge base, and used as context to guide the LLM.
If you pick too many examples, it may go beyond the limits of the context window
for the model you selected.
"""


class DimensionalityReductionOptions(str, Enum):
    PCA = "PCA"
    TSNE = "t-SNE"
    UMAP = "UMAP"


def limit_slider_component(name="Max examples", tooltip=DEFAULT_LIMIT_SLIDER_HELP):
    return st.slider(
        name,
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        help=tooltip,
    )


def vectors_to_fig(labels: List[str], vectors: List, method: DimensionalityReductionOptions = None):
    if method == DimensionalityReductionOptions.PCA:
        reducer = PCA(n_components=2)
    elif method is None or method == DimensionalityReductionOptions.TSNE:
        n_samples = len(vectors)
        perplexity_value = min(
            n_samples - 1, 30
        )  # Default is 30, but should be less than number of samples
        reducer = TSNE(n_components=2, perplexity=perplexity_value)
    elif method == DimensionalityReductionOptions.UMAP:
        # TODO: umap-learn is hard to install on a mac
        raise NotImplementedError("UMAP not yet implemented")
        # reducer = umap.UMAP()
    else:
        raise ValueError(f"Unknown method {method}")

    reduced_data = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50)
    for i, label in enumerate(labels):
        ax.annotate(label, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=9, ha="right")
    return fig
