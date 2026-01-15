import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import EEGEmbedding
import utils

# Escolhe se usa Placa de Video (GPU) ou o Processador (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Full Dataset")
try:
    X_numpy = np.load('data/processed/X_stew.npy')
    Y_numpy = np.load('data/processed/Y_stew.npy')
except FileNotFoundError:
    print("ERROR: .NPY FILE NOT FOUND. Run preprocessing.py first.")
    exit()

# Código sem o Random Split para o estudo do Ablation Study
# Convertendo para tensors e enviando para o device
X_tensor = torch.from_numpy(X_numpy).float().to(device)
Y_tensor = torch.from_numpy(Y_numpy).long().to(device)
full_dataset = TensorDataset(X_tensor, Y_tensor)

# Definindo o ponto de corte 80/20
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Fatiamento Sequencial (Slicing)
# Isso garante que os primeiros 80% fiquem no treino
# E os últimos 20% fiquem no teste
# train_dataset = TensorDataset(X_tensor[:train_size], Y_tensor[:train_size])
# test_dataset = TensorDataset(X_tensor[train_size:], Y_tensor[train_size:])

# --- Mudança aqui para o Ablation Study ---
# Comentar as linhas 22 e 23 e Descomentar a linha abaixo
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Criando loaders finais
# shuffle=True no treino para variar o treinamento dentro do grupo de treino
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# shuffle=False no teste para manter a ordem e consistência
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total samples: {total_size}") 
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")

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

# Inicializar o Modelo
model = EEGEmbedding().to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

print("Dataset Loaded. Starting Training Process")

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    train_batches = 0
    # Criar um loop que cria um ciclo de treino que se repete várias vezes
    for inputs, targets in train_loader:

        # Por segurança
        inputs, targets = inputs.to(device), targets.to(device)

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

        # Cálculo da Acurácia
        # torch.argmin() pois é calculado a menor distância entre os pontos, e a menor distância é retornada
        predicted_class = torch.argmin(dists, dim=1)

        # Comparação para ver se é igual ao gabarito
        # Se a rede previu 0 e era 0 retorna True e vira 1.0
        # Se a rede previu 0 e era 1 retorna True e vira 0.0
        correct_predictions = (predicted_class == targets).float() # Converte Booleano para 1.0/0.0

        # Média de Acertos do Batch
        accuracy = correct_predictions.mean()

        # descobre quem errou (calcula os gradientes)
        loss.backward()

        # Se o gradiente for maior que 1.0, corta ele para 1.0
        # Isso impede que o erro de ir para o infinito (NaN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # corrige os pesos (atualiza a rede)
        optimizer.step()

        # A função do .item() é pegar o valor do tensor para que ele seja um valor operável em Python (int, float)
        total_train_loss += loss.item()
        total_train_acc += accuracy.item()
        train_batches += 1

    # Calcula a média da época inteira
    avg_train_loss = total_train_loss / train_batches
    avg_train_acc = total_train_acc / train_batches

    # A cada 5 épocas, é testado o modelo nos dados que nunca viu (test set) e imprime o status
    if (epoch + 1) % 5 == 0:
        model.eval() # Congela o modelo (desativa dropout e batchnorm se houver)
        total_val_loss = 0
        total_val_acc = 0
        val_batches = 0

        with torch.no_grad(): # Desliga o cálculo pesado de gradientes para economizar na memória na validação
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device) # Garante envio para a CPU

                embeddings = model(inputs)
                if len(targets) < 2: continue # Pula os batches quebrados
                
                # Recalcula protótipos baseados no batch de teste
                prototypes = utils.get_prototypes(embeddings, targets, 2)
                dist_0 = utils.calc_euclidiean_distance(embeddings, prototypes[0])
                dist_1 = utils.calc_euclidiean_distance(embeddings, prototypes[1])
                dists = torch.stack([dist_0, dist_1], dim=1)
                
                loss = criterion(-dists, targets)
                
                # Acurácia de Teste
                predicted_class = torch.argmin(dists, dim=1)
                accuracy = (predicted_class == targets).float().mean()
                
                total_val_loss += loss.item()
                total_val_acc += accuracy.item()
                val_batches += 1
        
        #Médias de Validação
        avg_val_loss = total_val_loss / val_batches
        avg_val_acc = total_val_acc / val_batches

        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"   Train -> Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc*100:.2f}%")
        print(f"   Test  -> Error (Loss): {avg_val_loss:.4f} | Acc: {avg_val_acc*100:.2f}%")
        print("-" * 60)


print("Training Completed")

# salvar o cérebro treinado num arquivo
torch.save(model.state_dict(), 'results/saved_models/eeg_model.pth')
print("Model saved in results/models/eeg_model.pth")