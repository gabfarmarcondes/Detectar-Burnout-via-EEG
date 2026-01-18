import sys
import os
import torch

# Diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sobe dois níveis para chegar na pasta root do projeto - sai de backend e sai de web
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

# # Adiciona a root ao caminho do python para achar a src
sys.path.append(project_root)


from fastapi import FastAPI, UploadFile, File, HTTPException
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

# CARREGAMENTO DO MODELO
# Define onde o modelo vai rodar
# Usar a CPU para evitar erro custo ou erros de drivers
device = torch.device("cpu")

# Variável global para guardar o modelo
model = None

try:
    print("Starting to Load the Model")

    # 1. Instanciar a Arquiteturea
    model = EEGEmbedding().to(device)

    # 2. Definir o caminho do .pth
    model_path = os.path.join(project_root, 'results', 'saved_models', 'eeg_model.pth')

    # 3. Carregar os Pesos
    # map_loacation=device é importante para evitar erro se o modelo foi treinado na GPU e esta rodando na CPU
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Modo de Avaliação
    # Desliga o Dropout e BatchNormalization
    # Modo de treino
    model.eval()

    print(f"Model Successfully Loaded: {model_path}")
except FileNotFoundError:
    print(f"File Not Founded in: {model_path}")
    print("Ensure the train_fewshot.py has been run first.")
except Exception as e:
    print(f"Error to Load the Model: {e}")


# Rota de teste
@app.get("/")
def home():
    # Rota de verificação da saúde da API.
    model_status = "Online" if model else "Offline"
    return {"status": "API running", "model_status": model_status}

@app.post("/predict")
async def predict(file : UploadFile = File(...)): # File(...) significa obrigatório (Ellipsis). No caso, é obrigatório mandar algum arquivo. Caso contrário, dá erro.
    # Verificação de segurança
    if model is None:
        raise HTTPException(status_code=500, detail="Model was not Loaded in the Server.")
    return {
        "filename": file.filename,
        "message": "Model Ready for Inference.",
        "device_used": str(device)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)