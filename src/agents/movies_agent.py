from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from operator import itemgetter
# Ahora puedes importar
from retriever import Retriever

import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

class MoviesAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,  # Ajusta según tu modelo
            base_url=OLLAMA_BASE_URL,
            temperature=0.3  # Menor aleatoriedad para decisiones críticas
        )

        self.retriever = Retriever(collection_name="movies")

        self.prompt = PromptTemplate.from_template("""
            Eres un critico de cine experto en recomendar libros a los usuarios basandote en sus preferencias.
            Tienes que dar una recomendación basandote en el input que recibas del usuario (recibiras un libro) que te diga el usuario 
            y teniendo en cuenta el contexto que tienes con la información de libros.
            
             
            Contexto: {context}
                                            
    
            Pregunta del usuario: {input}
                                                   
            Genera la recomendación utilizando la siguiente estructura
            
            📚 Pelicula recomendado: [Título exacto del libro]
            ✍️ Autor: [Nombre del autor] (si disponible) 
            📅 Año de publicación: [Año] (si disponible) 
            🌟 Puntuación: [X.X/5] (si disponible)   


            Justifica la recomendación con un breve parrafo. No te extiendas mas de 3 lineas.

            Justificación: [2-3 frases sobre similitudes con el libro de referencia]

            Reglas:
            1. Usa SOLO información de las peliculas proporcionados.
            2. Si no hay libros similares, di: "No encontré coincidencias precisas".
            3. Mantén la respuesta concisa y profesional.
                                         
            """)
        
        self.rag_chain = (
            {
                "context": itemgetter("input") | self.retriever,
                "input": itemgetter("input")
            }
            | self.prompt
            | self.llm
            | StrOutputParser()

        )

    def recommend_movie(self, input):
        """Recomienda un libro"""
        return self.rag_chain.invoke(input)
