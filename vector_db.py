import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,
    prefer_grpc=False
)

embedding_generator = OpenAIEmbeddings()
qdrant = Qdrant(
    client=client,
    collection_name="my_test_documents",
    embedding_function=embedding_generator.embed_query
)

def get_qdrant_client():
    return client

def get_qdrant_impl():
    return qdrant
