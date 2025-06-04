from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


class GeneralAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
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
        chain = self.prompt | self.llm
        response = chain.invoke({"input": state["input"]}).strip().lower()
        return {"output": response}