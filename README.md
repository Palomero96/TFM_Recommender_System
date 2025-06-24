# TFM_Recommender_System


## Prerequisitos
Para poder ejecutar la aplicacion es necesario tener los siguientes prerequisitos
- Ollama instalado en local
- Docker
- Python 3.10 con las librerias necesarias. Para ello ''' python -r requirements.txt '''
## Instrucciones
1. Levantar los contenedores de la base de datos
   1. Si no se han levantado nunca se dentra que acceder a la carpeta data/RagDocker y lanzar ''' docker-compose up '''
   2. Si ya se ha realizado este primer paso se podra acceder a la interfaz gráfica de Docker
2. Cargar datos de en la base de datos. Para ello habra que ejecutar el archivo src/data_loader.py
3. Crear el fichero .env en la raiz del proyecto con las siguientes lineas:
''' # Conexión a Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Configuración de Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5
OLLAMA_TEMPERATURE=0.3 '''

4. Abrir una powershell y lanzar el siguiente comando para que se ejecute el modelo de lenguaje en local ''' ollama run qwen2.5'''
5. Desde una consola que tengamos el environment con todas las librerias y estando en la unicacion del archivo que es /frontend/app.py lanzar ''' streamlit run app.py ''' 