import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import io
import base64
import mne
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config
CHANNELS = config.CHANNELS
SFREQ = config.SAMPLE_RATE


def generate_topomap_base64(patient_tensor):
    # Recebe o tensor de dados reais do paciente, calcula a energia por canal
    # e retorna uma string base64 da imagem do topomap

    try:
        # Nome dos canais do Dataset STEW
        ch_names = config.CHANNELS
        # Frequência de Amostragem
        sfreq = config.SAMPLE_RATE

        # Processar a energia do sinal
        # O tensor chega da IA. Precisa converter para NumPy
        # Se vier com dimensão de batch, remover o batch
        if torch.is_tensor(patient_tensor):
            if patient_tensor.dim() == 4: # [batch, canal, freq, tempo]
                data = torch.mean(patient_tensor, dim=0).cpu().numpy()
            else:
                data = patient_tensor.cpu().numpy()
        else:
            data = patient_tensor # já é numpy

        # Filtragem Espacial
        # O tensor em formato (14 canais, 33 frequências, X tempo)
        # Assumindo que o eixo 1 é a frequência
        # Queremos apenas o Beta (13 - 30Hz)
        if data.ndim == 3: # (Canais, Freq, Tempo)
            # Fatia-se apenas a banda de interesse (Beta)
            # Se a pessoa estiver relaxada, Beta é baixxo
            # Se a pessoa estiver com burnout, Beta é alto
            beta_band_data = data[:, 13:30, :]

            # Calcula-se a energia (RMS) apenas desta faixa
            channel_energy = np.sqrt(np.mean(beta_band_data**2, axis=(1,2)))
        else:
            # Fallback se o tensor não for espectograma
            channel_energy = np.sqrt(np.mean(data**2, axis=1))
        
        # Configuração MNE
        info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        
        ax.set_title("Beta Band Activity (Stress Marker)\nRed = High Burnout Indicators", 
                    color='#cbd5e1', fontsize=11, fontweight='bold', pad=15)

        # cmap='Reds' -> Vermelho agora vai significar "Muita Onda Beta" (Estresse)
        # vmin/vmax ajuda a travar a escala para comparar melhor
        im, _ = mne.viz.plot_topomap(channel_energy, info, axes=ax, show=False, 
                                     cmap='Reds', # Ou 'inferno' para mais contraste
                                     contours=6,
                                     extrapolate='head',
                                     outlines='head',
                                     sphere=0.11,
                                     image_interp='cubic')
        
        # Barra de Cor
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="5%", pad="5%")
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Beta Power (uV)', color='#94a3b8', fontsize=9)
        cbar.ax.tick_params(colors='#94a3b8', labelsize=8)

        # 6. Salvar
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
        buf.seek(0)
        encoded_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        return encoded_string

    except Exception as e:
        print(f"Erro Topomap: {e}")
        import traceback
        traceback.print_exc()
        return None



# Bloco final para rodar o script
if __name__ == "__main__":
    generate_topomap_base64()