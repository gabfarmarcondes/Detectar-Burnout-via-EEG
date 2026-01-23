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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
from xai_utils import GradCAM

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
        
        plot_base64 = self.generate_spatial_plot(patient_profile)

        xai_base64 = self.generate_xai_plot(input_tensor)

        return {
            "prediction": prediction,
            "confidence": f"{confidence_pct:.1f}%",
            "distances": {
                "to_relax": dist_relax,
                "to_burnout": dist_burnout
            },
            "windows_analyzed": num_windows,
            "status_color": status_color,
            "image_base64": plot_base64,
            "xai_base64": xai_base64
        }
    
    def generate_spatial_plot(self, patient_tensor):
        # Gera um gráfico 2D comparando o paciente com os protótipos
        # Retorna String Base64 da imagem.

        try:
            # Preparar os dados
            # Traz de volta da GPU para a CPU e converte para numpy
            proto_relax = self.prototypes[0].cpu().numpy()
            proto_burnout = self.prototypes[1].cpu().numpy()

            # O paciente vem (1, 64), usa-se flatten para virar vetor (64,)
            patient = patient_tensor.cpu().numpy().flatten()

            # Matriz com 3 linhas (Relax, Burnout, Paciente)
            X = np.array([proto_relax, proto_burnout, patient])

            # PCA: reduz de 64 dimensões para 2
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)

            # Plotagem
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(6,4), dpi=120)
            ax.axis('off')

            colors = ['#10b981', '#ef4444', '#3b82f6']
            labels = ['Relaxed Proto', 'Burnout Proto', 'YOU']
            markers = ['o', 'o', 'X']
            sizes = [150, 150, 250]

            for i in range(3):
                ax.scatter(X_2d[i, 0], X_2d[i, 1], 
                          c=colors[i], 
                          label=labels[i], 
                          s=sizes[i], 
                          marker=markers[i],
                          edgecolors='white', # Borda branca para contraste
                          linewidth=1.5,
                          zorder=5)
                
            # Linha Paciente -> Relaxado
            ax.plot([X_2d[2,0], X_2d[0,0]], [X_2d[2,1], X_2d[0,1]], 
                   linestyle='--', color='#cbd5e1', alpha=0.6, linewidth=1.2)
            
            # Linha Paciente -> Burnout
            ax.plot([X_2d[2,0], X_2d[1,0]], [X_2d[2,1], X_2d[1,1]], 
                   linestyle='--', color='#cbd5e1', alpha=0.6, linewidth=1.2)
            
            # Linha de Base (Relaxado <-> Burnout) para fechar o triângulo
            ax.plot([X_2d[0,0], X_2d[1,0]], [X_2d[0,1], X_2d[1,1]], 
                   linestyle='--', color='#94a3b8', alpha=0.8, linewidth=1.5)

            # Legenda
            legend = ax.legend(loc='upper center', 
                             bbox_to_anchor=(0.5, -0.05),
                             ncol=3,
                             frameon=False,
                             fontsize=10,
                             labelcolor='#cbd5e1')

            plt.tight_layout()
            
            # 4. Salvar
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            # 5. Base64
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return image_base64

        except Exception as e:
            print(f"Plot Error: {e}")
            return None
    
    def generate_xai_plot(self, patient_tensor):
        # Gerar o Mapa de calor (Grad-CAM) mostrand onde a IA focou.
        # Retorna um String Base64 da imagem.
        
        try:
        # Configurar Grad-CAM na camada convolucional 1.
        # O input_tensor precisa ter dimensão (1, 14, 33, 17).
            if patient_tensor.dim() == 2: # Se vier (64, feature) não serve, precisa do tensor original.
                # Precisa passar po tensor original, não o embedding.
                print("Error: Grad-CAM requires the original image tensor, not embedding.")
                return None
            
            cam = GradCAM(model=self.model, target_layer=self.model.conv1)

            # Gerar o HeatMap
            # Precisa garantir que o tensor tenha a dimendsão batch (1, C, H, W).
            # Como o process_file retorna (N_janelas, 14, 33, 17), vamos pegar a média ou a primeira janela representativa.
            # Para XAI ficar bonito, vai ser pego a janela que teve a maior ativção (pior caso) ou a média.
            # Vamos simplificar pegando a média das janelas para representar o paciente como um todo.
            target_input = torch.mean(patient_tensor, dim=0).unsqueeze(0) # (1, 14, 33, 17)
            heatmap = cam(target_input)

            # Preparar a plotagem
            original_img = target_input[0].detach().cpu().numpy()
            avg_spectrogram = np.mean(original_img, axis=0) # Média dos canais para visualizar

            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), dpi=300)

            title_font = {'fontsize': 20, 'fontweight': 'bold', 'color': '#cbd5e1'}
            label_font = {'fontsize': 20, 'fontweight': 'bold', 'color': '#94a3b8'}
            tick_font_size = 16
            cbar_font = 16

            # Plot 1: Espectrograma Original
            im1 = ax1.imshow(avg_spectrogram, aspect='auto', origin='lower', 
                           cmap='viridis', interpolation='nearest')
            ax1.set_title("Input Signal Spectrogram", **title_font, pad=20)
            ax1.set_ylabel("Frequency (Hz)", **label_font)
            ax1.set_xticks([]) 
            ax1.tick_params(axis='y', labelsize=tick_font_size, colors='#94a3b8')

            cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
            cbar1.set_label("Intensity (dB)", fontsize=cbar_font, color='#94a3b8')
            cbar1.ax.tick_params(labelsize=tick_font_size, colors='#94a3b8')

            # Plot 2: Onde a IA Olhou (Heatmap)
            ax2.imshow(avg_spectrogram, aspect='auto', origin='lower', 
                      cmap='gray', interpolation='nearest', alpha=0.3)
            
            im2 = ax2.imshow(heatmap, aspect='auto', origin='lower', 
                           cmap='jet', interpolation='nearest', alpha=0.7)
            
            ax2.set_title("AI Attention (Burnout Pattern)", **title_font, pad=20)
            ax2.set_ylabel("Frequency (Hz)", **label_font)
            ax2.set_xlabel("Time (Epochs)", **label_font)
            ax2.tick_params(labelsize=tick_font_size, colors='#94a3b8')

            cbar2 = plt.colorbar(im2, ax=ax2, pad=0.04)
            cbar2.set_label("Activation Importance", fontsize=cbar_font, color='#94a3b8', labelpad=15)
            cbar2.ax.tick_params(labelsize=10, colors='#94a3b8')

            plt.tight_layout(h_pad=4)

            # 4. Salvar em Base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            return base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"XAI Plot Error: {e}")
            import traceback
            traceback.print_exc()
            return None