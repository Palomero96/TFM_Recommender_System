from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import connections
import os

HOST = "localhost"
PORT = "19530"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")



def embeddingData(path,collection):
    # Cargar y procesar el CSV
    loader = CSVLoader(file_path=path)
    data = loader.load()
            
    # Dividir los documento
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
    splits = text_splitter.split_documents(data)
    # Crear embeddings y vectorstore
    embeddings = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    vector = FAISS.from_documents(splits, embeddings)
    data_to_milvus(collection,vector,splits)

def data_to_milvus(collection,db,texts):
    connections.connect(host=HOST, port=PORT)
    #print(utility.list_collections())  # Muestra las colecciones existentes
    # Nombre de la colección donde guardarás los vectores
    COLLECTION_NAME = collection

    # Extrae los embeddings y documentos de tu FAISS
    vectors = db.index.reconstruct_n(0, db.index.ntotal)  # Obtiene todos los vectores
    documents = [doc.page_content for doc in texts]       # Tus textos originales
    metadatas = [doc.metadata for doc in texts]           # Metadatos asociados

    # Crea la colección en Milvus (si no existe)
    if not utility.has_collection(COLLECTION_NAME):
        dim = len(vectors[0])  # Dimensión de tus embeddings (ej: 384, 768, etc.)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="Índice de embeddings")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Crea un índice HNSW para búsquedas eficientes
        index_params = {
            "metric_type": "L2",  # Distancia euclidiana (usa "IP" para producto interno)
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    # Guarda los datos en Milvus
    data = [
        vectors,                    # Embeddings
        documents,                  # Textos originales
        metadatas                   # Metadatos (opcional)
    ]

    # Inserta los datos
    collection = Collection(COLLECTION_NAME)
    mr = collection.insert(data)
    #print(f"Vectores insertados: {mr.insert_count}")



if __name__=='main':
    #Cargamos los datos de los libros
    embeddingData("books.csv","books")
    #Cargamos los datos de las peliculas
    embeddingData("movies.csv","movies")
