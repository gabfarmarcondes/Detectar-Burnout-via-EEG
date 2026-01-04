import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data_loaders(batch_size=5):
    # Função que lê os arquivos .npy e devolve o DataLoader pronto

    # Escolhe se usa Placa de Video (GPU) ou o Processador (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar os arquivos .npy salvos pelo preprocessing.py
    try:
        X_numpy = np.load('data/processed/X_stew.npy')
        Y_numpy = np.load('data/processed/Y_stew.npy')

        # Converter numpy para tensor
        X_tensor = torch.from_numpy(X_numpy).float().to(device)
        Y_tensor = torch.from_numpy(Y_numpy).long().to(device)

        # Criar o dataset e o loader. Ele pega o dataset inteiro e serve batches de 5 amostras por vez
        dataset = TensorDataset(X_tensor, Y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader

    except FileNotFoundError:
        print("ERROR: .NPY FILE NOT FOUND")
        exit()