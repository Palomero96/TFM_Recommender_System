from langgraph.graph import StateGraph, END
from typing import TypedDict
import agents.director_agent, agents.books_agent, agents.movies_agent

DIRECTOR = agents.director_agent.DirectorAgent()
BOOKS_AGENT = agents.books_agent.BookAgent()

#MOVIES_AGENT = agents.movies_agent.MoviesAgent()

# Definimos el estado que ira de agente en agente
class AgentState(TypedDict):
    input: str
    output: str
    decision: str

def create_graph():    
    workflow = StateGraph(AgentState)

    # Definimos los nodos
    workflow.add_node("director", DIRECTOR.analyze_query)
    workflow.add_node("book_agent", BOOKS_AGENT.recommend_book)
    workflow.add_node("book_generic_agent", BOOKS_AGENT.recommend_book_generic)
    #workflow.add_node("movie_agent", MOVIE_AGENT.recommend_movie)
    #workflow.add_node("movie_generic_agent", MOVIE_AGENT.recommend_movie_generic)
    #workflow.add_node("generate_response",RESPONSE_AGENT)

    workflow.add_conditional_edges(
        "director",
        lambda x: x["decision"],
        {
            "books": "books_agent",
            "books_generic": "books_generic_agent",
            "movie": "movie_agent",
            "movie_generic": "movie_generic_agent"
        }
    )

    # Definimos los ultimos vertices para generar respuesta
    workflow.add_edge("books_agent", "generate_response")
    workflow.add_edge("books_generic_agent", "generate_response")
    workflow.add_edge("movies_agent", "generate_response")
    workflow.add_edge("movies_generic_agent", "generate_response")

    # Definimos el nodo de entrada y de salida
    workflow.set_entry_point("director")
    workflow.set_finish_point("generate_response")


    return workflow.compile()



if __name__ == 'main':
    app = create_graph()