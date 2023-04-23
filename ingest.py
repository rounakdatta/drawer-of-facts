from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from vector_db import get_qdrant_impl
from typing import List
from models import Information

def prepare_documents(information: Information) -> List[Document]:
    metadata = {
        "source": information.meta.source,
        "timestamp": information.meta.timestamp,
        "tags": information.meta.tags
        }
    return [Document(page_content=information.info, metadata=metadata)]


def ingest_docs(information: Information):
    # we structure our raw text as a Document, so that it becomes a standard datatype to work with
    structured_documents = prepare_documents(information)

    # we split the document cleverly
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunked_documents = text_splitter.split_documents(structured_documents)

    # we pass on the chunked documents to (1) be generated embeddings out of and (2) stored into qdrant
    qdrant = get_qdrant_impl()
    texts = [d.page_content for d in chunked_documents]
    metadatas = [d.metadata for d in chunked_documents]
    qdrant.add_texts(
        texts=texts,
        metadatas=metadatas
    )

if __name__ == "__main__":
    ingest_docs()
