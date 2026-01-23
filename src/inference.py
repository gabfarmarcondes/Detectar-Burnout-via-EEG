import torch
import numpy as np
import os
import sys

curren_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curren_dir)

from model import EEGEmbedding
import utils
from preprocessing import preprocess_file
import matplotlib
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
            X_numpy = np.load(os.path.join(data_path, 'X_stew.npy'))
            Y_numpy = np.load(os.path.join(data_path, 'Y_stew.npy'))

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

    def predict_patient(self, filepath):
        """
        Faz a inferência usando Distância Euclidiana aos Protótipos
        """
        if not self.is_ready:
            raise Exception("Sistema não inicializado.")

        # 1. Processa o arquivo do paciente
        input_tensor = preprocess_file(filepath, device=self.device)
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

            # Coloca as distâncias em um tensor
            dist_tensor = torch.tensor([dist_relax, dist_burnout])

            # Confianaça da IA
            # 3.0 ~ 5.0 confiança moderada
            # Quanto maior o número, menos confiante
            temperature = 3.0

            # Softmax negativa porque menor distância é melhor
            probs = torch.nn.functional.softmax(-dist_tensor / temperature, dim=0)

            # Pega a probabilidade mais alta
            confidence_score = probs.max().item()
            confidence_pct = confidence_score * 100

        # 4. Decisão
        # Quanto MENOR a distância, mais parecido é.
        if dist_burnout < dist_relax:
            prediction = "Burnout"
            status_color = "red"
        else:
            prediction = "Relaxed"
            status_color = "green"

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