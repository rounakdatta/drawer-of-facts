import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from config import get_qdrant_config

qdrant_config = get_qdrant_config()
client = qdrant_client.QdrantClient(
    host=qdrant_config["host"],
    port=qdrant_config["port"],
    prefer_grpc=False
)

embedding_generator = OpenAIEmbeddings()
qdrant = Qdrant(
    client=client,
    collection_name=qdrant_config["collection"],
    embedding_function=embedding_generator.embed_query
)

def get_qdrant_client():
    return client

def get_qdrant_impl():
    return qdrant
