from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

class BookAgent:
    def __init__(self, retriever):
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=0.3
        )
        self.retriever = retriever
        
        # Sistema de extracción de títulos
        self.title_extractor = ChatPromptTemplate.from_template("""
            Extrae SOLO el título principal del libro mencionado en esta frase:
            "{user_input}"
            
            Devuelve ÚNICAMENTE el título entre comillas dobles.
            Ejemplo: "Cien años de soledad"
            Respuesta: """)
        
        # Sistema de recomendación
        self.recommendation_prompt = ChatPromptTemplate.from_template("""
            Como experto bibliotecario, genera una recomendación basada en:
            
            Libro de referencia: "{book_title}"
            Consulta original: "{user_input}"
            
            Libros similares encontrados:
            {context}
            
            Formato de respuesta REQUERIDO:
            
            📚 Libro recomendado: {title}
            ✍️ Autor: {author}
            📅 Año: {year}
            🌟 Puntuación: {rating}/5
            
            🔍 Justificación: {justification}
            
            Reglas:
            1. Usa SOLO información de los libros proporcionados
            2. Máximo 2 oraciones de justificación
            3. Si no hay coincidencias, di: "No encontré recomendaciones adecuadas"
            """)

        self.chain = (
            RunnablePassthrough.assign(
                book_title=self.extract_title(),
                context=lambda x: self.format_context(x["book_title"])
            )
            | self.recommendation_prompt
            | self.llm
            | StrOutputParser()
        )

    def extract_title(self):
        """Cadena para extraer títulos de libros"""
        return (
            {"user_input": RunnablePassthrough()}
            | self.title_extractor
            | self.llm
            | StrOutputParser()
            | self.clean_title
        )
    
    def clean_title(self, title: str) -> str:
        """Limpia el título extraído"""
        return title.strip('"').split("\n")[0].strip()
    
    def format_context(self, book_title: str) -> str:
        """Formatea los resultados del retriever"""
        books = self.retriever.search(book_title)
        return "\n".join(
            f"- {b['title']} ({b['author']}, {b.get('year', 'N/A')})"
            for b in books
        )
    
    def recommend_book(self, user_input: str) -> str:
        """Método principal de recomendación"""
        return self.chain.invoke({
            "user_input": user_input,
            "title": "[Título]",
            "author": "[Autor]",
            "year": "[Año]",
            "rating": "[Puntuación]",
            "justification": "[Breve explicación]"
        })

# Uso demo
if __name__ == "__main__":
    from retriever import BookRetriever  # Asegúrate de tener tu retriever
    
    retriever = BookRetriever()  # Inicializa tu retriever
    agent = BookAgent(retriever)
    
    test_phrases = [
        "Quiero libros parecidos a El nombre del viento",
        "Recomiéndame algo similar a '1984' de Orwell",
        "Busco novelas como La sombra del viento pero más cortas"
    ]
    
    for phrase in test_phrases:
        print(f"\n🔍 Consulta: '{phrase}'")
        print(agent.recommend_book(phrase))
        print("-" * 50)