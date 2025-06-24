from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from src.agents.retriever import BookRetriever

import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env


class BookAgent:
    def __init__(self):
        # Definimos el modelo LLM que usaremos
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=os.getenv("OLLAMA_TEMPERATURE")
        )
        
        # Definimos el retriever que usaremos indicando la coleccion en la que buscara
        self.retriever = BookRetriever(collection_name="Pruebas") 
        
        # Definimos el prompt que le pasaremos a nuestro modelo LLM para que realize la tarea que se le solicita
        self.prompt = PromptTemplate.from_template("""
            Eres un bibliotecario experto en recomendar libros a los usuarios basandote en sus preferencias.
            Tienes que dar una recomendación basandote en el input que recibas del usuario (recibiras un libro) que te diga el usuario 
            y teniendo en cuenta el contexto que tienes con la información de libros.
            
            Ten en cuenta que el contexto que se te va a dar esta en ingles y el usuario puede preguntar en cualquier idioma. 
            Contexto: {context}
                                            
            La pregunta del usuario puede estar en cualquier idioma. Tienes que responder en el idioma que te pregunte el usuario.
            Pregunta del usuario: {input}
                                                   
            Recomienda un único libro al usuario utilizando la siguiente estructura de salida:

            📚 Libro recomendado: [Título exacto del libro]  
            ✍️ Autor(es): [Nombre del autor o autores]  
            📅 Año de publicación: [Año]  (si disponible) 
            🌐 Género: [Género]  (si disponible) 
            📄 Número de páginas: [Número de páginas]  (si disponible) 
            🌟 Puntuación media: [X.X/5] (si está disponible)

            Justifica la recomendación en un breve párrafo. Habla únicamente del libro que vas a recomendar, basándote en su descripción, género y los datos que el usuario ha proporcionado en su consulta:  
            {input}

            Ten en cuenta si el libro tiene una puntuación alta, si pertenece a un género que encaje con los gustos del usuario, o si su descripción lo hace especialmente relevante para su interés.

            Justificación: [Párrafo de justificación de la recomendación]
                                                   
            Reglas:
            1. Usa SOLO información de los libros proporcionados y habla unicamente del libro que vas a recomendar.
            2. Si no hay libros similares, di: "No encontré coincidencias precisas".
            3. Mantén la respuesta concisa y profesional.
            4. Responde en el idioma en el que ha preguntado el usuario.
            """)

    def recommend_book(self,  state: dict) -> dict:
        # Obtiene el contexto formateado desde Milvus usando el retriever
        context = self.retriever(state['input'])
        # Genera el texto del prompt
        prompt_text = self.prompt.format(context=context, input=state['input'])
        # Pasa el prompt al LLM y devuelve la respuesta
        response = self.llm.invoke(prompt_text)
        # Devolvemos la respuesta
        return {**state, "output": response}



if __name__ == "__main__":
    agent = BookAgent()
    print(agent.recommend_book("Libro similar a como Cien años de soledad"))