from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import connections, utility
import pandas as pd
import os

HOST = "localhost"
PORT = "19530"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

def embbeddingDataBooks(path):
    df = pd.read_csv(path,sep=";")
    # Lista para almacenar los Document
    documents = []
    for index, row in df.iterrows():
        # Extraer los campos que van al contenido (embedding)
        description = str(row.get("description", ""))
        genre = str(row.get("genre", ""))
        page_content = f"Description: {description}\nGenre: {genre}"
        # Construir metadata con el resto de columnas
        metadata = {
            "isbn": row.get("isbn"),
            "language_code": row.get("language_code"),
            "average_rating": row.get("average_rating"),
            "Authors": row.get("Authors"),
            "num_pages": row.get("num_pages"),
            "publication_year": row.get("publication_year"),
            "title": row.get("title")
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    # Dividir los documentos
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
    splits = text_splitter.split_documents(documents)
    # Crear embeddings y vectorstore
    embeddings = OllamaEmbeddings(
        model="qwen2.5",
        base_url="http://localhost:11434",
    )
    # Obtener directamente los vectores
    vector = embeddings.embed_documents([doc.page_content for doc in splits])
    data_to_milvus("books",vector,splits)
    print("TerminoLibro")

def embeddingDataMovies(path):

    df = pd.read_csv(path,sep=";",encoding="latin-1")
    # Lista para almacenar los Document
    documents = []
    for index, row in df.iterrows():
        # Extraer los campos que van al contenido (embedding)
        description = str(row.get("description", ""))
        genre = str(row.get("genre", ""))
        page_content = f"Description: {description}\nGenre: {genre}"
        # Construir metadata con el resto de columnas
        metadata = {
            "title": row.get("title"),
            "spoken_languages": row.get("spoken_languages"),
            "Vote_reescaled": row.get("Vote_reescaled"),
            "original_language": row.get("original_language"),
            "release_year": row.get("release_year")
            
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    # Dividir los documentos
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
    splits = text_splitter.split_documents(documents)
    # Crear embeddings y vectorstore
    embeddings = OllamaEmbeddings(
        model="qwen2.5",
        base_url="http://localhost:11434",
    )
    vector = embeddings.embed_documents([doc.page_content for doc in splits])
    data_to_milvus("movies",vector,splits)

def data_to_milvus(collection_name,vector,splits):
    connections.connect(host=HOST, port=PORT)
    # Nombre de la colección donde se guardan los vectores
    COLLECTION_NAME = collection_name
    # Extrae los embeddings y documentos
    vectors = vector  # Obtiene todos los vectores
    documents = [doc.page_content for doc in splits]       # Textos originales
    metadatas = [doc.metadata for doc in splits]           # Metadatos asociados

    # Crea la colección en Milvus (si no existe)
    if not utility.has_collection(COLLECTION_NAME):
        dim = len(vectors[0])  # Dimensión de los embeddings
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="Embeddings vectoriales con descripciones y metadatos")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Guarda los datos en Milvus
    data = [
        vectors,                    # Embeddings
        documents,                  # Textos originales
        metadatas                   # Metadatos (opcional)
    ]
    # Inserta los datos
    collection = Collection(COLLECTION_NAME)
    collection.insert(data)
    collection.flush()
    #collection.load()
    #print(f"Vectores insertados: {mr.insert_count}")



if __name__=='__main__':
    
    #Cargamos los datos de los libros
    embbeddingDataBooks("C:\\Users\\palom\\Documents\\GitHub\\TFM_Recommender_System\\data\\books.csv")

    #Cargamos los datos de las peliculas
    embeddingDataMovies("C:\\Users\\palom\\Documents\\GitHub\\TFM_Recommender_System\\data\\\movies.csv")
