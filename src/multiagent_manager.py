from langgraph.graph import StateGraph, END
from typing import TypedDict
import agents.general_agent
import agents.director_agent, agents.books_agent, agents.movies_agent



#MOVIES_AGENT = agents.movies_agent.MoviesAgent()

# Definimos el estado que ira de agente en agente
class AgentState(TypedDict):
    input: str
    output: str
    decision: str

def create_graph():    
    workflow = StateGraph(AgentState)
    # Definimos los agentes
    DIRECTOR = agents.director_agent.DirectorAgent()
    BOOKS_AGENT = agents.books_agent.BookAgent()
    MOVIES_AGENT = agents.movies_agent.MoviesAgent()
    GENERAL_AGENT = agents.general_agent.GeneralAgent()

    #Definimos los nodos del grafo
    workflow.add_node("analyze", DIRECTOR.analyze_question)
    workflow.add_node("recommend_book", BOOKS_AGENT.recommend_book)
    workflow.add_node("recommend_movie", MOVIES_AGENT.recommend_movie)
    workflow.add_node("general_response", GENERAL_AGENT.general_response)

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "libro": "recommend_book",
            "pelicula": "recommend_movie",
            "ninguna": "general_response"
        }
    )

    workflow.set_entry_point("analyze")
    workflow.add_edge("general_response", END)
    workflow.add_edge("recommend_book", END)
    workflow.add_edge("recommend_movie", END)

    return workflow.compile()



if __name__ == 'main':
    app = create_graph()