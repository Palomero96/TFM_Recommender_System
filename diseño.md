graph TD
    Director[ Director_agent] -->|Book| Nodo1[Book_agent]
    Director -->|BookGenre| Nodo2[Book_generic_agent]
    Director -->|Movie| Nodo3[Movie_agent]
    Director -->|MovieGenre| Nodo4[Movie_generic_agent]
    
    Nodo1 --> Generar[Generate_response_agent]
    Nodo2 --> Generar
    Nodo3 --> Generar
    Nodo4 --> Generar



    graph TD
    Director[ Director_agent] -->|Book| Nodo1[Book_agent]
    Director -->|BookGenre| Nodo2[Book_generic_agent]
    Director -->|Movie| Nodo3[Movie_agent]
    Director -->|MovieGenre| Nodo4[Movie_generic_agent]
    Director -->|General| Nodo4[General_agent]
    
    Nodo1 --> Generar[Generate_response_agent]
    Nodo2 --> Generar
    Nodo3 --> Generar
    Nodo4 --> Generar