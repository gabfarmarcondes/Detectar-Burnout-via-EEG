import os
from pathlib import Path

# Caminhos do Projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "results" / "saved_models"

# Cria diretórios se não existirem
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuração do STEW Dataset (Emotiv EPOC)
SAMPLE_RATE = 128 # O STEW tem 128Hz
DURATION = 150 # O STEW tem ~2.5 minutos

# Os 14 canais exatos do Emotiv EPOC. A ordem dos canais importa
CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# Configurações de Processamento. Filtrar as frequências inúteis
FILTER_LOW = 1.0 # Remove drift lento
FILTER_HIGH = 40.0 # Remove ruído de rede elétrica/muscular alta
EPOCH_LENGTH = 4.0 # tamanho da janela que a IA vai ler 