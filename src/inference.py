import torch
import numpy as np

from model import EEGEmbedding
import utils

# Pegar todos os dados que temos até agora, calcular a média do Burnout e a média do Relaxado. Agora, pegue o paciente X e veja de quem está mais perto.

# 1. Configuração
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Carregar o modelo treinado. Instanciar a arquitetura vazia
model = EEGEmbedding().to(device)

# Carregar os pesos que foi salvado
path_model = 'results/saved_models/eeg_model.pth'

try:
    model.load_state_dict(torch.load(path_model, map_location=device))
except FileExistsError:
    print("ERROR: .NPY FILE NOT FOUND")

# coloca o modelo em modo de prova (desliga o aprendizado).
model.eval()

# 3. Carregar Dados para criar as Referências (protótipos)
# O certo seria ter um banco de dados fixo de pacientes padrão
# Aqui será usado os próprios dados de treino para calibrar
X_numpy = np.load('data/processed/X_stew.npy')
Y_numpy = np.load('data/processed/Y_stew.npy')

X_tensor = torch.from_numpy(X_numpy).float().to(device)
Y_tensor = torch.from_numpy(Y_numpy).long().to(device)

print("\nX-Ray of Loaded Data:")
print(f"Loaded datapath: data/processed/Y_stew.npy")
print(f"Total Samples: {len(Y_tensor)}")
print(f"Funded Classes: {torch.unique(Y_tensor)}")
print(f"Counting Classes: {torch.bincount(Y_tensor)}")
print("-" * 30)

# 4. Gerar o mapa de referência (calibragem)
print("Generating Reference Map")
with torch.no_grad():
    # Passa todos os dados pelo modelo
    all_embeddings = model(X_tensor)

    # calcula os centros (médias) de Relaxado(0) e Burnout(1)
    prototypes = utils.get_prototypes(all_embeddings, Y_tensor, 2)

print(f"Generated Maps. We have: {len(prototypes)} referece profiles")
print("-" *30)

# 5 Teste do Paciente
# Pegar uma amostra aleatória e fingir que é um paciente novo
import random

pacient_id = random.randint(0, len(X_tensor) - 1)

# Dados do paciente fictício
pacient_sample = X_tensor[pacient_id].unsqueeze(0) # adiciona dimensão batch (1, 14, 33, 17)
label_real = Y_tensor[pacient_id].item()

print(f"Analysing Pacient with ID {pacient_id}")

with torch.no_grad():
    # 1. Gera o embedding do paciente
    emb_pacient = model(pacient_sample)

    # 2. Mede a distância para o perfil Relaxado(0)
    relaxing_dist = utils.calc_euclidiean_distance(emb_pacient, prototypes[0].unsqueeze(0))

    # 3.Mede a distância para o perfil Burnout(1)
    burnout_dist = utils.calc_euclidiean_distance(emb_pacient, prototypes[1].unsqueeze(0))

# 6. O veredito
dist_r = relaxing_dist.item()
dist_b = burnout_dist.item()

print(f"\nCalculated Distances:")
print(f"    ->Distance for Relaxed Profile: {dist_r:.4f}")
print(f"    ->Distance for Burnout Profile: {dist_b:.4f}")

diagnostic = 0 if dist_r < dist_b else 1
real_name = "Burnout" if label_real == 1 else "Relaxed"
pred_name = "Burnout" if diagnostic == 1 else "Relaxed"

print("-"*30)
print("Final Results:")
print(f"    IA Prediction: {pred_name}")
print(f"    Real Diagnostic: {real_name}")

if diagnostic == label_real:
    print("CORRECT")
else:
    print("WRONG")
