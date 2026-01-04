import numpy as np
import mne
import config

def generate_mock_dataset(subject_id=1):
    print(f"Gerando dados 'STEW-like' para Sujeito {subject_id}")
    
    n_channels = len(config.CHANNELS)
    sfreq = config.SAMPLE_RATE
    duration = config.DURATION
    n_samples = int(sfreq * duration)
    
    # 1. Ruído Base (Cérebro normal)
    data = np.random.randn(n_channels, n_samples) * 5e-6 
    times = np.arange(n_samples) / sfreq
    
    # 2. Simular DOIS ESTADOS:
    # Metade 1: REST (Repouso) -> Muita onda Alpha (8-12Hz)
    # Metade 2: BURNOUT (Tarefa Difícil) -> Muita onda Beta (13-30Hz) e Gamma (>30Hz)
    
    half_point = int(n_samples / 2)
    
    # --- Fase de Repouso (0 a 1.25 min) ---
    # Injeta Alpha (10Hz) forte nos canais Occipitais (O1, O2)
    alpha_wave = np.sin(2 * np.pi * 10 * times[:half_point]) * 15e-6
    idx_o1 = config.CHANNELS.index('O1')
    idx_o2 = config.CHANNELS.index('O2')
    data[idx_o1, :half_point] += alpha_wave
    data[idx_o2, :half_point] += alpha_wave
    
    # --- Fase de Burnout/Stress (1.25 a 2.5 min) ---
    # Injeta Beta (20Hz) nos canais Frontais (F3, F4, AF3...) - "Fritando" o córtex
    beta_wave = np.sin(2 * np.pi * 20 * times[half_point:]) * 10e-6
    frontal_channels = ['AF3', 'AF4', 'F3', 'F4']
    for ch in frontal_channels:
        idx = config.CHANNELS.index(ch)
        data[idx, half_point:] += beta_wave

    # 3. Empacotar para MNE
    info = mne.create_info(ch_names=config.CHANNELS, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # Montagem Standard (Posição dos eletrodos na cabeça)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # 4. Salvar
    # No dataset real STEW, os arquivos costumam vir separados (ex: sub01_lo.txt, sub01_hi.txt)
    # Aqui vamos salvar um arquivo unificado para facilitar o teste inicial
    file_name = f"sub{subject_id:02d}_task_mock.fif"
    save_path = config.RAW_DATA_DIR / file_name
    raw.save(save_path, overwrite=True)
    print(f"✅ [MOCK] Arquivo salvo: {file_name}")

if __name__ == "__main__":
    # Gera 3 sujeitos para testarmos o código
    for i in range(1, 4):
        generate_mock_dataset(i)