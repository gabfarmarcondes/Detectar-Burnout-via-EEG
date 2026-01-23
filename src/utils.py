import torch

"""
A lógica será:
    1. Pegar exemplos de burnout(suporte).
    2. Calcular a média deles (protótipos).
    3. Medir a distância para ver se o protótipo está mais perdo do burnout ou do normal
"""

# Será preciso calcular a distância entre dois vetores usando a Distância Euclidiana.

def calc_euclidiean_distance(x1, x2):
    return torch.nn.functional.pairwise_distance(x1, x2, p=2)

def get_prototypes(embeddings, targets, num_class):
    # embeddings: Tensor(Batch, 64)
    # targets: Tensor (Batch) com 0(normal) e 1(burnout). Gerando dois protótipos, um para cada situação
    prototypes = [] # lista vazia dos prototypes

    for i in range(num_class):
        # 1. Criar uma máscara para pegar só quem é da classe i
        mask = (targets == i)

        # 2. Aplica a máscara nos embeddings
        class_embeddings = embeddings[mask]
        # proteção se não tiver ninguém da classe no batch
        if class_embeddings.size(0) == 0:
            # cria um vetor de zeros temporários só para não quebrar
            proto = torch.zeros(embeddings.size(1)).to(embeddings.device)
        else:
            # 3. Calcular a média dos embeddings
            proto = torch.mean(class_embeddings, 0)

        prototypes.append(proto)

    # Empilha tudo. Pois o prototype vai ter dois vetores, esta linha pega esses vetores e coloca um em cima do outro, criando um tensor único.
    # Linha 0: Cérebro normal, Linha 1: Cérebro com burnout
    return torch.stack(prototypes)
