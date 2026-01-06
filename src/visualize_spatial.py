import numpy as np
import matplotlib.pylab as plt
import mne
import os
import config


# Define que será usado dados simulados ou tentar carregar reais
# Será deixado como True para testar o gráfico
USE_MOCK_DATA = True

def generate_spatial_analysis():
    print("Mapping 10-20 Configuration")
    
    # Nome dos canais do Dataset STEW
    ch_names = config.CHANNELS
    # Frequência de Amostragem
    sfreq = config.SAMPLE_RATE

    # Cria o objeto de informações do MNE
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Carrega o mapa padrão de cabeças (Standard 10-20)
    # A montagem 10-20 define as coordenadas X,Y e Z de cada eletrodo numa cabeça padrão
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Criar um paciente que tem burnout. Para validar a visualização, cria-se uma onda senoidal de 22Hz (BETA) apenas nos canais da testa.
    if USE_MOCK_DATA:
        print("Obtaining Data")
        
        # Cria 4 segundos de silêncio
        # 1e-6 um ruído quase 0
        # O objetivo é provar que a visualização funciona. Pois é preciso testar os ruídos reais e não os que provavelmente estariam lá.
        # Para isso, é preciso começar com silêncio absoluto e
            # injtetar estresse apenas na testa garante que que o mapa final mostre a vermelhidão foi culpa do código.
        # Janelamento: o 4 segundos foi definido pelo preprocessing.py e no config.py (Epoch).
        # Tamanho padrão da literatura de EEG e é bom para ter uma boa resolução no Espectograma (STFT).
        n_sample = sfreq * 4
        data = np.random.rand(len(ch_names), n_sample) * 1e-6

        # Injetar estresse (Beta 22Hz) no frontal
        # O lobo frontal foi escolhido baseado na neurociência. O burnout e a ansiedade estão associados a uma superaquecimento do córtex pré-frontal (testa).
        # A função np.linspace() é a função Linear. O computador precisa saber em que momento cada onda acontece. Com os parâmetros:
        """
        start: 0. Começa no segundo 0.
        stop: 4. Termina no segundo 4.
        num: n_sample=512. Dividir esse intervalo de 0 a 4 em 512 pedaços iguais. [0.00 até 4.00]. Isso será útil no gráfico.
        """
        t = np.linspace(0, 4, n_sample)

        # Cria uma onda rápida (sinal de ansiedade)
        # Cérebro relaxado ele pulsa devagar (ALPHA 8-12Hz)
        # Cérebro estressado ele pulsa rápido (BETA 13-30Hz)
        # Foi escolhido 22Hz para simular um estado de alerta/ansiedade claro.
        # A função np.sin(), seno. Ela desenha a onda perfeita (sobe e desce). 
        # A fórmula matemática de uma onda física é: A * sin(2π * f * t). Sendo:
        """
        np.pi: o computador calcula o seno em radianos. Uma volta completa é 2π.
        22: é a frequência de tempo. Dizemos que para a onda completar 22 ciclos a cada segundo.
        t: é a régua de tempo que foi criada antes.
        20e-6: A amplitude. O seno puro vai de -1 a 1 e por conta dos sinais EEG serem tão pequenos, o 20e-6 (20 x 10⁻⁶) ajusta a escala para parecer um sinal elétrico real.
        """
        beta_wave = np.sin(2 * np.pi * 22 * t) * 45e-6 # Hiperatividade Cortical acima do normal(35e-6 a 50e-6) mas abaixo do nível epiléptico (~100µV)

        # Índices dos canais da testa (frontal lobe)
        # AF3, F7, F3, F4, F8, AF4
        # Os números significam a posição de cada canal no array de canais.
        # O loop for passa apenas por essas posições da matriz de dados e soma com as ondas de ansiedade.
        frontal_indices = [0, 1, 2, 11, 12, 13]

        # Soma-se a onda de estresse apenas nos canais citados acima
        for i in frontal_indices:
            data[i] += beta_wave
        
        # Empacota tudo num objeto Raw do MNE
        raw = mne.io.RawArray(data, info)
    else:
        pass

    print("Filtering Frequencies")

    # Aplica o filtro passa-banda: deixar passar apenas só o que é BETA (13 a 30 Hz)
    raw.filter(l_freq=13, h_freq=30, fir_design='firwin')

    # Cálculo de Energia (PSD).
    # Método Welch, uma técnica para reduzir ruído ao dividir dados em segmentos, aplica janelamento e média de periodogramas. 
    # Calcular a densidade Espectral de Potência (PSD).
    # Basicamente, é o volume médio da atividade elétrica em cada eletrodo.
    print("Calculating Energy (PSD)")
    
    # Calcula a potência usando o método Welch.
    spectrum = raw.compute_psd(method='welch')

    # Extrai os dados numéricos (Power Spectral Density)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Tira a média de todas as frequências da banda para ter um número só por canal
    psds_mean = psds.mean(axis=1)

    # Converte para decibéis (dB) para o gráfico ficar mais legível
    psds_db = 10 * np.log10(psds_mean)

    print("Generating Topomap")
    fig, ax = plt.subplots(figsize=(6,5))

    # Plota a cabeça
    im, _ = mne.viz.plot_topomap(
     psds_db, info, 
     axes=ax, 
     show=False,
     cmap='Reds', # Vermelho = Alta energia
     sphere=0.12) # Ajusta da geometrida da cabeça

    # Enfeites do gráfico
    plt.title("Where is the Burnout?\nSpatial Activation(BETA Band)", fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Intensity (dB)')

    # Salvar
    if not os.path.exists(config.FIGURES_DIR):
        os.makedir(config.FIGURES_DIR)
    
    save_path = config.FIGURES_DIR / "spatial_analysis_where.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved in: {save_path}")
    plt.show()


# Bloco final para rodar o script
if __name__ == "__main__":
    generate_spatial_analysis()