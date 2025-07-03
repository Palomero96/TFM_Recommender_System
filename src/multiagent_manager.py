from langgraph.graph import StateGraph, END
from typing import TypedDict
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents import general_agent, director_agent, books_agent, movies_agent


# Definimos el estado que ira de agente en agente
class AgentState(TypedDict):
    input: str
    output: str
    context: str
    decision: str

def create_graph():    
    workflow = StateGraph(AgentState)
    # Definimos los agentes
    DIRECTOR = director_agent.DirectorAgent()
    BOOKS_AGENT = books_agent.BookAgent()
    MOVIES_AGENT = movies_agent.MoviesAgent()
    GENERAL_AGENT = general_agent.GeneralAgent()

    #Definimos los nodos del grafo
    workflow.add_node("analyze", DIRECTOR.analyze_query)
    workflow.add_node("recommend_book", BOOKS_AGENT.recommend_book)
    workflow.add_node("recommend_movie", MOVIES_AGENT.recommend_movie)
    workflow.add_node("general_response", GENERAL_AGENT.general_response)
    
    # Definimos los vertices condicionales del grafo
    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "libro": "recommend_book",
            "pelicula": "recommend_movie",
            "general": "general_response"
        }
    )
    # Definimos el punto de partida del grafo
    workflow.set_entry_point("analyze")
    # Definimos los vertices finales del grafo
    workflow.add_edge("general_response", END)
    workflow.add_edge("recommend_book", END)
    workflow.add_edge("recommend_movie", END)
    #devolvemos la respuesta al usuario
    return workflow.compile()



if __name__ == '__main__':
    app = create_graph()
    initial_state = {
            "input": "que temperatura hace en madrid",
            "output": "",
            "context": "",
            "decision": ""
        }

    final_state = app.invoke(initial_state)
    print(final_state)