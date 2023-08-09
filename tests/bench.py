import os
import time
from random import choices

import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions

# list of random english language nouns
words = [
    "hp",
    "house",
    "flower",
    "tree",
    "dog",
    "cat",
    "mouse",
    "car",
    "boat",
    "plane",
    "train",
    "bicycle",
    "motorcycle",
    "truck",
    "bus",
    "bird",
    "fish",
    "horse",
    "cow",
    "pig",
    "sheep",
    "goat",
    "chicken",
    "duck",
    "goose",
    "turkey",
    "apple",
    "banana",
    "orange",
    "pear",
    "grape",
    "strawberry",
    "blueberry",
    "raspberry",
    "blackberry",
    "cherry",
    "peach",
    "plum",
    "apricot",
    "mango",
    "pineapple",
    "watermelon",
    "melon",
    "kiwi",
    "lemon",
    "lime",
    "coconut",
    "avocado",
    "peanut",
    "almond",
    "walnut",
    "cashew",
    "pecan",
    "pistachio",
    "hazelnut",
    "macadamia",
    "acorn",
    "chestnut",
    "pea",
    "bean",
    "lentil",
    "chickpea",
    "soybean",
    "corn",
    "wheat",
    "rice",
    "oat",
    "barley",
    "rye",
    "quinoa",
    "buckwheat",
    "millet",
    "sorghum",
    "cotton",
    "silk",
    "wool",
    "linen",
    "hemp",
    "bamboo",
    "jute",
    "flax",
    "copper",
    "iron",
    "gold",
    "silver",
    "platinum",
    "titanium",
    "aluminum",
    "zinc",
    "nickel",
    "lead",
    "tin",
    "mercury",
    "uranium",
    "radium",
    "hydrogen",
    "helium",
    "lithium",
    "beryllium",
    "boron",
    "carbon",
    "nitrogen",
    "oxygen",
    "fluorine",
    "neon",
    "sodium",
    "magnesium",
    "aluminum",
    "silicon",
    "phosphorus",
    "sulfur",
    "chlorine",
    "argon",
    "potassium",
    "calcium",
    "scandium",
    "titanium",
    "vanadium",
    "chromium",
    "manganese",
    "iron",
    "cobalt",
    "nickel",
    "copper",
    "zinc",
    "gallium",
    "germanium",
    "arsenic",
    "selenium",
]


def random_text(W: int = 3):
    """
    Generate a random text of W words.
    """
    return " ".join(choices(words, k=W))


default_ef = embedding_functions.DefaultEmbeddingFunction()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
para_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-MiniLM-L3-v2"
)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
)

# ef = para_ef
# ef = sentence_transformer_ef
# ef = default_ef
ef = openai_ef


settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="bench-test-db",
)

for W in [1, 2, 10, 100]:
    for N in [10, 100, 100, 1000, 5000]:
        docs = [random_text(W) for _ in range(N)]
        metadatas = [{"foo": 1, "Bar": 7} for _ in range(N)]
        ids = [str(i) for i in range(N)]

        client = chromadb.Client(settings)
        client.reset()
        # toggle this to test using OpenAI
        if ef:
            collection = client.create_collection(name="test", embedding_function=ef)
        else:
            collection = client.create_collection(name="test")
        start_time = time.time()
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
        )

        end_time = time.time()
        t = end_time - start_time
        docs_per_s = N / t
        print(f"W={W} N={N}: Insert operation took {t} seconds ({docs_per_s} docs/s)")
