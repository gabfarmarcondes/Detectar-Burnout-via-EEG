import sys
import os

# Diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sobe dois níveis para chegar na pasta root do projeto - sai de backend e sai de web
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

# # Adiciona a root ao caminho do python para achar a src
sys.path.append(project_root)


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.model import EEGEmbedding



# Inicializa o app
# Ponto principal da interação para criar toda a API do projeto
app = FastAPI(title="NeuroCompute API")

# Configurar o CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Aceita conexões de qualquer lugar
    allow_credentials=True,
    allow_methods=["*"], # Aceita métodos HTTP
    allow_headers=["*"] # pares chave-valor que transportam metadados importantes sobre a requisição ou resposta
)

# Rota de teste
@app.get("/")
def home():
    return {"status: online", "message NeuroCompute API is running"}

