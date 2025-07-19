
from pymilvus import connections
from langchain_community.vectorstores import Milvus

from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env

class BaseRetriever:
    def __init__(self, collection_name: str):
        # Definimos la collecion en la que vamos a buscar
        self.collection_name = collection_name
        # Definimos el modelo de embedding que vamos a utilizar
        self.embeddings = OllamaEmbeddings(
                model=os.getenv("OLLAMA_MODEL"),
                base_url=os.getenv("OLLAMA_BASE_URL")
            )
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT")
        )
        # Cargar vectorstore existente desde Milvus
        self.vectorstore = Milvus(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    vector_field="embedding",
                    text_field="text"
        )
    
    def __call__(self, query: str) -> str:
        # Hacemos la busqueda del documento que sea mas similar a lo que pide el usuario
        docs = self.vectorstore.similarity_search(query,10)
        # Se llama al metodo de formato para formatear el contexto
        return self.format_context(docs)

    # Se definira en cada clase de Books y Movies
    def format_context(self, results):
        raise NotImplementedError("Debe implementar format_context en la subclase")


class BookRetriever(BaseRetriever):
    # Formateamos el documento que se ha encontrado
    def format_context(self, results):
        parts = []
        for hit in results:
            # Metadata externa de Milvus
            outer_meta = getattr(hit, "metadata", {}) or {}
            # Metadata interna con los datos reales
            meta = outer_meta.get("metadata", {}) or {}
            # Page_content que es la descripcion y el genero
            description = getattr(hit, "page_content", "") or "" 
            # Añadimos a la lista los atributos encontrados
            parts.append(
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Author: {meta.get('Authors', 'Unknown')}\n"
                f"Year: {meta.get('publication_year', 'Unknown')}\n"
                f"Rating: {meta.get('average_rating', 'N/A')}\n"
                f"{description}\n"
            )
        # Unimos los elementos de la lista y lo devolvemos
        return "\n---\n".join(parts)
    

class MovieRetriever(BaseRetriever):
    def format_context(self, results):
        parts = []
        for hit in results:

            # Metadata externa de Milvus
            outer_meta = getattr(hit, "metadata", {}) or {}
            # Metadata interna con los datos reales
            meta = outer_meta.get("metadata", {}) or {}
            # Page_content que es la descripcion y el genero
            description = getattr(hit, "page_content", "") or "" 
            # Añadimos a la lista los atributos encontrados
            parts.append(
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Rating: {meta.get('Vote_reescaled', 'Unknown')}\n"
                f"Year: {meta.get('release_year', 'Unknown')}\n"
                f"Original_language: {meta.get('original_language', 'Unknown')}\n"
                f"{description}\n"
            )
        # Unimos los elementos de la lista y lo devolvemos
        return "\n---\n".join(parts)