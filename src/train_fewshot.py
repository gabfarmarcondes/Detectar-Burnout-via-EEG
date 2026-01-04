import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import EEGEmbedding
import utils
from data_loader import get_data_loaders

# Escolhe se usa Placa de Video (GPU) ou o Processador (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = get_data_loaders(batch_size=32)

# O optimizer é o treinador da rede, ele calcula a matemática para ajustar os pesos
""" 
Foi usado o Adam porque ele adapta a velocidade de aprendizado automaticamente e
ele é bom para dados complexos e ruidosos. 

Os parâmetros:
    1. params: diz quais pesos ele tem permissão para mexer. Foi usado o 
        models.parameters() para entregar a lista de todos os pesos da rede.
    2. lr(Learning Rate): é a taxa de aprendizado. 0.01(1e-3) é um valor padrão seguro.
        Se for muito alto, a rede pula a resposta certa. Se for muito baixo,
        ela demora uma eternidade para aprender.
"""

# Cria-se uma variável model para ela receber uma cópia viva da classe e vai para o devive
model = EEGEmbedding().to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

print("Dataset Loaded. Starting training")

num_epochs = 50

model.train()

for epoch in range(num_epochs):
    total_loss = 0
    batches = 0
    # Criar um loop que cria um ciclo de treino que se repete várias vezes
    for inputs, targets in data_loader:
        # limpa a memória do passo anterior (zera os gradientes)
        optimizer.zero_grad()

        # faz a previsão (forward = o modelo trabalha), gera os embeddings (vetores de 64 números)
        embeddings = model(inputs)

        # calcula o erro, função de perda
        """
        O erro é calculado da seguinte forma:
            1. calcular a distância da amostra até o protótipo errado
            2. calcular a distância até o protótipo certo
            3. usar uma função chamada LogSoftmax/CrossEntropy para transformar as distâncias em probabilidade
        """

        # Few-Shot Logic
            # Se o batch for muito pequeno no final (sobra), pula para evitar erro
        if len(targets) < 2: continue
            # Passar os embeddings gerados, os targets verdadeiros e o número de classes
        prototypes = utils.get_prototypes(embeddings, targets, 2)
        
            # Calcular as distâncias
                # Distância de cada amostra até Protótipo 0 (relaxado)
        dist_0 = utils.calc_euclidiean_distance(embeddings, prototypes[0])
                # Distância de cada amostra até  Protótipo 1 (burnout)
        dist_1 = utils.calc_euclidiean_distance(embeddings, prototypes[1])

        # Empilhar as distâncias lado a lado para formar uma tabela (Batch, 2)
        # Coluna 0: distância pro relaxado | Coluna 1: distância pro burnout
        dists = torch.stack([dist_0, dist_1], dim=1)

        # usa-se o -dists pois CrossEntropy gosta de números maiores para a classe certa
        # mas na distância, o melhor é o menor número (mais perto)
        # colocando negativo, a menor distância vira o maior número
        loss = criterion(-dists, targets)

        # Se o gradiente for maior que 1.0, corta ele para 1.0
        # Isso impede que o erro de ir para o infinito (NaN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # descobre quem errou (calcula os gradientes)
        loss.backward()

        # corrige os pesos (atualiza a rede)
        optimizer.step()

        total_loss += loss.item()

        batches += 1

# A cada 5 épocas, imprime o status
if (epoch + 1) % 5 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}] | Error (Loss): {total_loss:.4f}")

print("Training Completed")

# salvar o cérebro treinado num arquivo
torch.save(model.state_dict(), 'results/saved_models/eeg_model.pth')
print("Model saved in results/models/eeg_model.pth")