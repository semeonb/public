# This script is used to generate embeddings for the text data stored in files
# https://platform.openai.com/docs/guides/embeddings/what-are-embeddings

# imports packages
from os import environ, listdir, path
import pandas as pd  # need to install pandas
import pickle  # need to install pickle
import openai  # need to install openai
import json

from openai.embeddings_utils import (
    get_embedding,
    chart_from_components,
    tsne_components_from_embeddings,
)


# write a line that reads the environment variable
openai.api_key = environ.get(
    "OPENAI_API_KEY"
)  # change this to your API key, stored as an environment variable
EMBEDDING_MODEL = "text-embedding-ada-002"
FILES_LOCATION = "/Users/semeonbalagula/Downloads/articles"  # change this to the location of the files

# set path to embedding cache
embedding_cache_path = "/tmp/embeddings_cache.pkl"
embedding_file = "/tmp/embeddings.json"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string: str, model: str = EMBEDDING_MODEL, embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model, api_key=None)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


def generate_embedding(
    files,
    articles,
    model=EMBEDDING_MODEL,
):
    # This function generates embeddings for the text data stored in files
    # and saves the embeddings to a json file
    embeddings = [
        [float(k) for k in embedding_from_string(article, model=model)]
        for article in articles
    ]

    for i, v in enumerate(embeddings):
        article_dict = {
            "file_name": files[i],
            "text_embedding": [
                {"embed": z[0], "code": z[1]} for z in zip(v, list(range(len(v))))
            ],
        }
        with open(embedding_file, "a") as f:
            json.dump(article_dict, f)
            f.write("\n")
    return embeddings


def gen_data(files_dir):
    files = listdir(files_dir)
    articles = []
    for i in listdir(files_dir):
        file = open(path.join(FILES_LOCATION, i), "r")
        articles.append(file.read())
    return files, articles


files, articles = gen_data(FILES_LOCATION)
embeddings = generate_embedding(files, articles)
tsne_components = tsne_components_from_embeddings(embeddings)


# create a chart from the t-SNE components
chart = chart_from_components(
    tsne_components,
    files,
    width=1600,
    height=1500,
    title="articles t-SNE components",
)

chart.show()
