from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env

class GeneralAgent:
    def __init__(self):
        # Definimos el modelo LLM que usaremos
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=os.getenv("OLLAMA_TEMPERATURE")
        )
        self.prompt = PromptTemplate.from_template("""
                                                   
            Eres un agente general que va a recibir una pregunta de un usuario.
            Responde a la pregunta de forma breve y concisa. 
            La respuesta tiene que ocupar 5 frases como mucho.

            La pregunta del usuario puede estar en cualquier idioma. Tienes que responder en el idioma que te pregunte el usuario.                                   
            Pregunta del usuario: {input}
            
            Si no sabes la respuesta di: "Lo siento, no puedo ayudarte con ese problema. Esta fuera de mis conocimientos"
            """)

    def general_response(self, state: dict) -> dict:
        """Responde de forma breve y concisa a la pregunta general plantea el usuario"""
        # Genera el texto del prompt
        prompt_text = self.prompt.format(input=state['input'])
        # Pasa el prompt al LLM y devuelve la respuesta
        response = self.llm.invoke(prompt_text)
        # Devolvemos la respuesta
        return {**state, "output": response}
    