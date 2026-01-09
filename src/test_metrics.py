import torch
import numpy as np
from model import EEGEmbedding
import utils as utils
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting Metrics on the Device: {device}")

# 1. Carregar Modelo
model = EEGEmbedding().to(device)
model.load_state_dict(torch.load('results/saved_models/eeg_model.pth', map_location=device))
model.eval()

# 2. Carregar Dados
X = np.load('data/processed/X_stew.npy')
Y = np.load('data/processed/Y_stew.npy')
total_samples = len(Y)

# 3. Separar em Treino (Referências) e Teste (Validação)
# Vamos usar 80% para criar os protótipos e testar nos 20% restantes
split_idx = int(total_samples * 0.8)

# Dados para criar os protótipos
X_ref = torch.from_numpy(X[:split_idx]).float().to(device)
Y_ref = torch.from_numpy(Y[:split_idx]).long().to(device)

# Dados para TESTAR (A IA nunca viu esses para criar o mapa)
X_test = torch.from_numpy(X[split_idx:]).float().to(device)
Y_test = torch.from_numpy(Y[split_idx:]).long().to(device)

print(f"Reference Data: {len(Y_ref)} samples")
print(f"Test Data:      {len(Y_test)} samples")

# 4. Gerar Protótipos usando os dados de Referência
print("Generating Map Reference")
with torch.no_grad():
    embeddings_ref = model(X_ref)
    prototypes = utils.get_prototypes(embeddings_ref, Y_ref, 2)

# 5. Rodar o Teste em Lote
print("Running Tests")
y_true = []
y_pred = []

with torch.no_grad():
    # Gera embeddings para todos os dados de teste de uma vez
    embeddings_test = model(X_test)
    
    # Calcula distâncias para Relaxado (0) e Burnout (1)
    dists_0 = utils.calc_euclidiean_distance(embeddings_test, prototypes[0].unsqueeze(0))
    dists_1 = utils.calc_euclidiean_distance(embeddings_test, prototypes[1].unsqueeze(0))
    
    # Compara: Se dist0 < dist1, prediz 0. Senão, prediz 1.
    predictions = (dists_1 < dists_0).long()
    
    # Salva para métricas
    y_pred = predictions.cpu().numpy()
    y_true = Y_test.cpu().numpy()

# 6. Relatório Final
print("\n" + "="*60)
print("TCC Report Performance")
print("="*60)

# Nomes das classes
target_names = ['Relaxed (0)', 'Burnout (1)']

print(classification_report(y_true, y_pred, target_names=target_names))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Salvar imagem da Matriz
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('AI PREDICTION')
    plt.ylabel('Reality')
    plt.title('Confusion Matrix - Burnout Detection')
    plt.savefig('results/figures/confusion_matrix.png')
    print("\nGraphics saved in results/figures/confusion_matrix.png")
except:
    print("\nIt was not possible to generate the graph.")