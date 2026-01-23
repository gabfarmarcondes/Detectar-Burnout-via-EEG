"""
O .txt é ruim para processar rápido e para a rede neural. Portanto irá ser criado um script que irà:
    1- ler o .txt
    2- aplicar o filtro de frequências (1-40Hz)
    3- corta em janelas (epochs) de 4 segundos
    4- gerar o espectograma (STFT) para cada janela
    5- salvar tudo pronto para o treinamento
"""
import shutil
import numpy as np
import pandas as pd
import mne
from scipy import signal
from pathlib import Path
from tqdm import tqdm
import config
import torch

SFREQ = config.SAMPLE_RATE # frequência do dataset

def read_stew_text_file(filepath):
    """
    Lê o arquivo .txt do dataset e converte para MNE RAW. 
    O STEW tem 14 colunas de dados (canais) separados por espaço.
    """

    # Lê o arquivo de texto, ignorando cabeçalhos se houver
    try:
        # Tenta ler separado por espaço
        data = pd.read_csv(filepath, sep=r"\s+", header=None, engine='python')
    except:
        # Fallback se for separado por vírgula
        data = pd.read_csv(filepath, sep=',', header=None)

    # Garante que pegue apenas os números do dataset
    data = data.apply(pd.to_numeric, errors='coerce',).dropna()

    # Transforma em Matriz (Canais x Tempo)
    # O arquivo vem (Tempo x Canais), então fazemos a transposta
    data_np = data.values.T

    # Garante que temos 14 canais. Se tiver mais, corta. Se tiver menos, erro.
    if data_np.shape[0] > 14:
        data_np = data_np[:14, :]
    if data_np.shape[0] < 14:
        return None
    
    # Cria a estrutura MNE
    info = mne.create_info(ch_names=config.CHANNELS, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(data_np, info, verbose=False)

    return raw

def transform_to_spectrogram(epoch_data, sfreq):
    """
    Converte uma janela de EEG (Canais x Tempo) em Imagem (Canais x Frequência x Tempo).
    Usando STFT (Short-Time Fourier Transform)
    """

    # f = frequência, t = tempo, Zxx = complexo da STFT
    f, t, Zxx = signal.stft(epoch_data, fs=sfreq, nperseg=64, noverlap=32)

    # Pegamos apenas a magnitude (abs)
    spectogram = np.abs(Zxx)

    # Normalização logarítmica (para realçar frequências baixas)
    spectogram = np.log1p(spectogram)

    return spectogram

def preprocess_file(filepath, device='cpu'):
    # Função que processa um arquivo
    # Lê -> Filtra -> Janela -> Espectograma -> Tensor Pytorch

    # Leitura
    raw = read_stew_text_file(filepath)

    if raw is None:
        raise ValueError("Error to read file or insufficient (minimun 14 channels).")
    
    # Filtro (1~40Hz)
    raw.filter(config.FILTER_LOW, config.FILTER_HIGH, verbose=False)

    # Janelamento
    if raw.times[-1] < config.EPOCH_LENGTH:
        raise ValueError(f"Audio too short. Minimum {config.EPOCH_LENGTH}")
    
    epochs = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_LENGTH, verbose=False)
    epoch_data = epochs.get_data(copy=True, verbose=False)

    # Espectograma
    processed_list = []
    for window in epoch_data:
        spec = transform_to_spectrogram(window, config.SAMPLE_RATE)
        processed_list.append(spec)
    
    # Converte para Tensor e pronto para a IA
    batch_tensor = torch.tensor(np.array(processed_list), dtype=torch.float32).to(device)

    return batch_tensor

def process_dataset():
    print("Starting the STEW Dataset Processing")

    # Limpeza: Remove a pasta processed antiga para não ter conflito
    processed_path = Path("data/processed")
    if processed_path.exists():
        print("Cleaning the old folder: data/processed")
        shutil.rmtree(processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/raw/STEW_Dataset")
    files = sorted(list(data_path.glob("*.txt")))
    
    if not files:
        print("No file was found in data/raw/STEW.")
        return

    processed_list = []
    labels_list = []

    count_relax = 0
    count_burnout = 0

    for file in tqdm(files, desc="Processing Files"):
        # Loop por cada arquivo com barra de progresso
        filename = file.name
        
        # Define o Label baseado no nome do arquivo
        # 01_lo.txt -> label 0 (relaxado)
        # 01_hi.txt -> label 1 (Burnout/Carga Alta)
        label = -1
        if "lo" in filename.lower():
            label = 0
            count_relax += 1
        elif "hi" in filename.lower():
            label = 1
            count_burnout += 1
        else:
            continue # Pula arquivos que não são lo ou hi
        
        # 1. Carregar
        raw = read_stew_text_file(file)
        if raw is None: continue

        # 2. Filtrar (1-40Hz)
        raw.filter(config.FILTER_LOW, config.FILTER_HIGH, verbose=False)
        
        # 3. Cortar em Janelas (Epochs) e em Janelas de 4 segundos, sem sobreposição
        epochs = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_LENGTH, verbose=False)
        epoch_data = epochs.get_data(copy=True,verbose=False)

        # 4. Converter para Espectograma e Iterara sobre cada janela
        for i in range(len(epoch_data)):
            window = epoch_data[i] # (14 canais, 512 pontos de tempo)

            # Gerar imagem
            spec = transform_to_spectrogram(window, SFREQ) # O shape do spec será algo como (14, 33, 17) -> (Canais, Freqs, Tempo)
            
            processed_list.append(spec)
            labels_list.append(label)

    if len(processed_list) == 0:
        print("No data found")
        return

    # Converter para Tensores Pytorch ou NumPy
    X = np.array(processed_list, dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int64)

    print(f"\nProcessing Completed")
    print(f"Total Samples: (Image):  {X.shape[0]}")
    print(f"Image Shape: {X.shape[1:]} (Channels x Freq x Time)")
    print(f"Relaxing Files founded: {count_relax}")
    print(f"Burnout Files founded:   {count_burnout}")
    print("-" * 30)
    print(f"Total of Windows (Samples): {X.shape[0]}")
    print(f"Image Shape: {X.shape[1:]}")

    unique, counts = np.unique(Y, return_counts=True)
    distribuicao = dict(zip(unique, counts))
    print(f"   -> Class Distribuition: {distribuicao}")
    
    if 1 not in distribuicao:
        print("\nERROR: Still no have Class Number 1 (Burnout).")
    else:
        print("\nSuccess! Data is ready to save.")
        np.save(processed_path / "X_stew.npy", X)
        np.save(processed_path / "Y_stew.npy", Y)
        print(f"Saved in {processed_path}")

if __name__ == "__main__":
    process_dataset()