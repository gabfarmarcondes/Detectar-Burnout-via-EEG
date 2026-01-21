from contextlib import asynccontextmanager
import shutil
import sys
import os
import torch
import numpy as np
import pandas as pd
import mne
from scipy import signal
from src import config

# Diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sobe dois níveis para chegar na pasta root do projeto - sai de backend e sai de web
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

# # Adiciona a root ao caminho do python para achar a src
sys.path.append(project_root)


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.inference import BurnoutSystem


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

burnout_system = BurnoutSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Server and Load the AI resources")
    model_path = os.path.join(project_root, 'results', 'saved_models', 'eeg_model.pth')
    data_path = os.path.join(project_root, 'data', 'processed')
    success = burnout_system.load_resources(model_path, data_path)
    if not success:
        print("Warning: The system started but failed to load model. API will answer but inference will fail.")
    else:
        print("System ready and loaded.")
    yield
app = FastAPI(lifespan=lifespan)

# Rotas
@app.get("/")
def home():
    # Rota de verificação da saúde da API.
    model_status = "Online" if burnout_system.is_ready else "Offline"
    return {"status": "API running", "model_status": model_status}

@app.post("/predict")
async def predict(file : UploadFile = File(...)): # File(...) significa obrigatório (Ellipsis). No caso, é obrigatório mandar algum arquivo. Caso contrário, dá erro.
    # Verificação de segurança
    if not burnout_system.is_ready:
        raise HTTPException(status_code=500, detail="AI System is not ready.")
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt is allowed.")
    
    temp_filename = f"temp_{file.filename}"
    try:
        # Copia o fluxo de bits do upload para um arquivo físico no disco
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = burnout_system.predict_patient(temp_filename)

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "details": result,
            "message": "Prototype analysis completed."
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)