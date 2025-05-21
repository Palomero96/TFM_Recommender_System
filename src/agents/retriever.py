from langchain_community.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from pymilvus import connections
import os

class Retriever:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        # Conexión a Milvus
        self._connect_to_milvus()
        # Crear función de embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Cargar vectorstore existente desde Milvus
        self.vectorstore = Milvus(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            connection_args={
                "host": os.getenv("MILVUS_HOST", "localhost"),
                "port": os.getenv("MILVUS_PORT", "19530")
            }
        )

        self.retriever = self.vectorstore.as_retriever()

    def _connect_to_milvus(self):
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        connections.connect(host=host, port=port)

    def __call__(self, input):
        return self.retriever.invoke(input)
