import matplotlib.pyplot as plt
import numpy as np

# Dados obtidos nos experimentos (Médias e Desvios Padrão)
# Cenário A: Isolamento de Sujeito (O "Correto")
mean_iso = 92.44
std_iso = 0.97

# Cenário B: Random Split (O "Misturado")
mean_rnd = 89.06
std_rnd = 0.40

# Configuração do Gráfico
labels = ['Isolamento de Sujeito\n(Metodologia Correta)', 'Random Split\n(Sem Isolamento)']
means = [mean_iso, mean_rnd]
stds = [std_iso, std_rnd]
colors = ['#2ecc71', '#e74c3c'] # Verde (Bom) e Vermelho (Ruim/Controle)

x_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8, 6))

# Criar barras com erro (capsize coloca o tracinho no topo da linha de erro)
bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, color=colors, ecolor='black', capsize=10, width=0.5)

# Configurações visuais
ax.set_ylabel('Acurácia (%)', fontsize=12)
ax.set_title('Ablation Study: Impacto da Metodologia de Validação', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(80, 100) # Focando a escala entre 80% e 100% para ver a diferença
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adicionar os valores em cima das barras
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 15),  # Deslocamento vertical do texto
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

add_labels(bars)

# Salvar e Mostrar
plt.tight_layout()
plt.savefig('results/ablation_study/ablation_chart_validation.png', dpi=300)
print("Gráfico salvo em: results/ablation_study/ablation_chart_validation.png")
plt.show() # Descomente se quiser ver na tela