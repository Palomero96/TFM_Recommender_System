from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableLambda

import numpy as np
from typing import List, Dict

class BookRetriever:
    def __init__(self):
        # Modelo de embeddings (ligero pero efectivo)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Base de datos mockeada de libros (puedes reemplazar con tu CSV)
        self.books_db = [
            {
                "title": "Cien años de soledad",
                "author": "Gabriel García Márquez",
                "year": 1967,
                "genres": ["realismo mágico", "ficción literaria"],
                "description": "Una saga familiar en Macondo donde lo mágico y lo real se entrelazan."
            },
            {
                "title": "La casa de los espíritus",
                "author": "Isabel Allende",
                "year": 1982,
                "genres": ["realismo mágico", "drama familiar"],
                "description": "Crónica de la familia Trueba a través de generaciones en Chile."
            },
            {
                "title": "1984",
                "author": "George Orwell",
                "year": 1949,
                "genres": ["distopía", "ficción política"],
                "description": "Una sociedad totalitaria controlada por el Gran Hermano."
            }
        ]
        
        # Precomputar embeddings
        self.book_embeddings = self.encoder.encode(
            [f"{b['title']} {b['description']} {' '.join(b['genres'])}" for b in self.books_db]
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Busca libros similares usando embeddings semánticos."""
        query_embedding = self.encoder.encode(query)
        
        # Calcular similitudes coseno
        similarities = np.dot(self.book_embeddings, query_embedding) / (
            np.linalg.norm(self.book_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Obtener los índices de los más similares
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Formatear resultados
        return [
            {
                "title": self.books_db[i]["title"],
                "author": self.books_db[i]["author"],
                "year": self.books_db[i]["year"],
                "genres": ", ".join(self.books_db[i]["genres"]),
                "similarity_score": float(similarities[i])
            }
            for i in top_indices
        ]


    def as_runnable(self):
        """Convierte el retriever en un Runnable compatible"""
        return RunnableLambda(self.search)