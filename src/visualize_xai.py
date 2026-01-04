import torch
import numpy as np
import random
from model import EEGEmbedding
from xai_utils import GradCAM, plot_explanation

# 1. Configuração do Ambiente
# Verifica se tem placa de vídeo (GPU) ou vai usar o processador (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting XAI Visualization on device: {device}")

# 2. Carregar o Modelo Treinado
# Instancia a arquitetura vazia (o esqueleto da rede)
model = EEGEmbedding().to(device)

# Carrega os "conhecimentos" (pesos) que a rede aprendeu durante o treino
model.load_state_dict(torch.load('results/saved_models/eeg_model.pth', map_location=device))

# Coloca em modo de avaliação (desliga o aprendizado, agora é só prova)
model.eval()

# 3. Carregar os Dados Processados
# Precisamos das imagens dos cérebros para testar
print("Loading data.")
X = np.load('data/processed/X_stew.npy') # As imagens (espectrogramas)
Y = np.load('data/processed/Y_stew.npy') # Os rótulos (0=Relaxado, 1=Burnout)

# Transforma de Numpy (matemática comum) para Tensor (matemática de IA)
X_tensor = torch.from_numpy(X).float().to(device)
Y_tensor = torch.from_numpy(Y).long().to(device)

# 4. Selecionar um Paciente com Burnout
# Queremos ver o que acontece na cabeça de alguém estressado (Label 1)
print("Searching for a Burnout patient to analyze.")
pacient_idx = -1

# Loop simples para achar o primeiro Burnout da lista (ou sortear um)
while True:
    idx = random.randint(0, len(Y) - 1)
    if Y[idx] == 1: # Se achou Burnout
        pacient_idx = idx
        break

print(f"Patient found: ID {pacient_idx}")

# Prepara a amostra: A rede espera um lote (Batch), então adicionamos uma dimensão extra
# De (14, 33, 17) vira (1, 14, 33, 17)
input_tensor = X_tensor[pacient_idx].unsqueeze(0)

# 5. Configurar o Grad-CAM (O "Detetive")
# Aqui dizemos: "GradCAM, vigie a camada 'conv1' do modelo".
# A 'conv1' é a camada que olha as texturas visuais do espectrograma.
cam = GradCAM(model=model, target_layer=model.conv1)

# 6. Gerar o Mapa de Calor
# O GradCAM roda a imagem pela rede e calcula quais pixels ativaram a decisão
print("Generating explanatory heatmap.")
heatmap = cam(input_tensor)

# 7. Visualizar e Salvar
# Pega a imagem original (para mostrar ao fundo) e o mapa de calor (para mostrar por cima)
# Salva na pasta results para você pegar depois
save_path = f"results/figures/explanation_pacient_{pacient_idx}.png"

plot_explanation(
    original_data=input_tensor, 
    heatmap=heatmap, 
    title=f"AI Attention Focus (Burnout - ID {pacient_idx})",
    save_path=save_path
)

print(f"\nDONE! Image saved at: {save_path}")
print("Open this image to see where the AI focused!")