from contextlib import asynccontextmanager
import shutil
import sys
import os
import tempfile

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
# Inicializa o app
# Ponto principal da interação para criar toda a API do projeto
app = FastAPI(title="NeuroCompute API", lifespan=lifespan)

# Configurar o CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Aceita conexões de qualquer lugar
    allow_credentials=True,
    allow_methods=["*"], # Aceita métodos HTTP
    allow_headers=["*"] # pares chave-valor que transportam metadados importantes sobre a requisição ou resposta
)

# Rotas
@app.get("/")
async def home():
    return FileResponse('web/frontend/home.html')

@app.get("/analysis")
async def read_analysis():
    return FileResponse('web/frontend/analysis.html')

@app.get("/aboutme")
async def aboutme():
    return FileResponse('web/frontend/aboutme.html')

@app.get("/script.js")
async def read_js():
    return FileResponse('web/frontend/script.js')

@app.get("/style.css")
async def read_css():
    return FileResponse('web/frontend/style.css')

@app.get("/how-it-works")
async def how_it_works():
    return FileResponse('web/frontend/how_it_works.html')

@app.post("/predict")
async def predict(file : UploadFile = File(...)): # File(...) significa obrigatório (Ellipsis). No caso, é obrigatório mandar algum arquivo. Caso contrário, dá erro.
    # Verificação de segurança
    if not burnout_system.is_ready:
        raise HTTPException(status_code=500, detail="AI System is not ready.")
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt is allowed.")
    
    temp_filename = f"temp_{file.filename}"
    try:
        suffix = ".txt"
        # Copia o fluxo de bits do upload para um arquivo físico no disco
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_filename = tmp.name

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
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass