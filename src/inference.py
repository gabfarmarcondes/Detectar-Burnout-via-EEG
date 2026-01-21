import torch
import numpy as np
import pandas as pd
import mne
from scipy import signal
import os
import sys

curren_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curren_dir)

from model import EEGEmbedding
import utils
import config

SFREQ = config.SAMPLE_RATE
CHANNELS = config.CHANNELS

class BurnoutSystem:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.prototypes = None
        self.is_ready = False
    
    def load_resources(self, model_path, data_path):
        # 1. carrega o modelo .pth
        # 2. carrega o dataset de treino .npy
        # 3. calcula os protótipos para comparação

        print("Starting Inferecence System")
        # Carregar o modelo
        try:
            self.model = EEGEmbedding().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model Loaded")
        except Exception as e:
            print(f"Error to load model: {e}")
            return False
        # Carregar dados para calibragem
        try:
            X_numpy = np.load(data_path + '/X_stew.npy')
            Y_numpy = np.load(data_path + '/Y_stew.npy')

            X_tensor = torch.from_numpy(X_numpy).float().to(self.device)
            Y_tensor = torch.from_numpy(Y_numpy).long().to(self.device)

            with torch.no_grad():
                all_embeddings = self.model(X_tensor)
                self.prototypes = utils.get_prototypes(all_embeddings, Y_tensor, 2)
            
            print(f"Prototypes Calculated: {len(self.prototypes)} references profiles")
            self.is_ready = True
            return True
        except Exception as e:
            print(f"Error to calculate prototypes: {e}")
            return False
    
    def preprocess_txt(self, filepath):
        # Transformar .txt bruto em tensor
        try:
            data = pd.read_csv(filepath, sep=r"\s+", header=None, engine='python')
        except:
            data = pd.read_csv(filepath, sep=',', header=None)
        
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        data_np = data.values.T

        if data_np.shape[0] > 14: data_np = data_np[:14, :]
        elif data_np.shape[0] < 14: raise ValueError("Less than 14 channels found.")

        info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(data_np, info, verbose=False)
        raw.filter(1., 40., verbose=False)

        if raw.times[-1] < 4.0:
            raise ValueError("Audio too short (minimun 4s)")
        
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, verbose=False)
        epoch_data = epochs.get_data(copy=True, verbose=False)

        processed_list = []
        for window in epoch_data:
            f, t, Zxx = signal.stft(window, fs=SFREQ, nperseg=64, noverlap=32)
            spec = np.log1p(np.abs(Zxx))
            processed_list.append(spec)

        batch_tensor = torch.tensor(np.array(processed_list), dtype=torch.float32).to(self.device)

        return batch_tensor
    def predict_patient(self, filepath):
        """
        Faz a inferência usando Distância Euclidiana aos Protótipos
        """
        if not self.is_ready:
            raise Exception("Sistema não inicializado.")

        # 1. Processa o arquivo do paciente
        input_tensor = self.preprocess_txt(filepath)
        num_windows = input_tensor.shape[0]

        with torch.no_grad():
            # 2. Gera Embedding do Paciente
            # O paciente gera várias janelas. Vamos tirar a média delas para ter UM vetor do paciente.
            embeddings = self.model(input_tensor) # (N_janelas, 64)
            patient_profile = torch.mean(embeddings, dim=0).unsqueeze(0) # (1, 64)

            # 3. Calcula Distâncias (A Lógica do seu TCC)
            # Protótipo 0 = Relaxado, Protótipo 1 = Burnout
            dist_relax = utils.calc_euclidiean_distance(patient_profile, self.prototypes[0].unsqueeze(0)).item()
            dist_burnout = utils.calc_euclidiean_distance(patient_profile, self.prototypes[1].unsqueeze(0)).item()

        # 4. Decisão
        # Quanto MENOR a distância, mais parecido é.
        if dist_burnout < dist_relax:
            prediction = "Burnout"
            confidence_raw = dist_relax - dist_burnout # Diferença de distância = confiança
            status_color = "red"
        else:
            prediction = "Relaxed"
            confidence_raw = dist_burnout - dist_relax
            status_color = "green"

        # Normaliza confiança para 50-99% (apenas visual)
        confidence_pct = min(50 + (confidence_raw * 50), 99.9)

        return {
            "prediction": prediction,
            "confidence": f"{confidence_pct:.1f}%",
            "distances": {
                "to_relax": dist_relax,
                "to_burnout": dist_burnout
            },
            "windows_analyzed": num_windows,
            "status_color": status_color
        }
