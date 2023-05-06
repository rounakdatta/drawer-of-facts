import os

qdrant_host = os.getenv("QDRANT_HOST")
qdrant_port = os.getenv("QDRANT_PORT", default=6333)
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")

def get_qdrant_config():
    return {
        "host": qdrant_host,
        "port": qdrant_port,
        "collection": qdrant_collection_name
    }
