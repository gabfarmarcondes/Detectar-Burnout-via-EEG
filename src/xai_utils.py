import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    """
    Técnica Gradient-weighted Class Activation Mapping (Grad-CAM).
    Permite visualizar quais partes da entrada (Espectrograma) foram
    mais importantes para a decisão da Rede Neural.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Ganchos (Hooks) para capturar os dados durante o fluxo
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        self.model.zero_grad()
        output = self.model(x)

        # Como foi usado o Metric Learning (Embeddings) e não Classificação direta,
        # Iŕa ser calculado o gradiente baseado na soma do vetor de embedding.
        # Isso diz: O que na imagem fez esse vetor ser assim
        score = output.sum()
        
        # 2. Backward Pass (Calcula os gradientes)
        score.backward(retain_graph=True)

        # 3. Gerar o Mapa de Calor
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling dos gradientes
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Multiplicação ponderada das ativações
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Aplica ReLU
        cam = F.relu(cam)
        
        # Interpola para o tamanho original da imagem de entrada (33x17)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Normaliza entre 0 e 1 para plotar
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.detach().cpu().numpy()[0, 0]

def plot_explanation(original_data, heatmap, title="XAI Explanation", save_path=None):
    """
    Plota o Espectrograma Original ao lado do Mapa de Calor (Grad-CAM)
    """
    # Prepara a imagem original (tira a média dos 14 canais para visualização 2D)
    # original_data shape: (1, 14, 33, 17) -> pegamos (14, 33, 17)
    original_img = original_data[0].cpu().numpy()
    
    # Média entre os canais para virar uma imagem 2D (Freq x Tempo)
    avg_spectrogram = np.mean(original_img, axis=0) 
    
    fig = plt.figure(figsize=(12, 5))
    
    # 1. Espectrograma Original (Média dos canais)
    plt.subplot(1, 2, 1)
    plt.imshow(avg_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Average Spectogram (Input)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()

    # 2. Heatmap Grad-CAM sobreposto
    plt.subplot(1, 2, 2)
    
    # Plota o fundo
    plt.imshow(avg_spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5)
    
    # Plota o calor por cima
    # Redimensiona o heatmap para bater com o tamanho do imshow se precisar
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.6)
    
    plt.title(f"Grad-CAM: {title}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Explanatory Imagem Saved in: {save_path}")
    
    plt.close(fig)