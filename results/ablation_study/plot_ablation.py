import matplotlib.pyplot as plt
import numpy as np

# teste A: com filtro
losses_with_filter = [0.3174, 0.3436, 0.3085, 0.3473, 0.2845]

# teste B: sem filtro
losses_without_filter = [0.3134, 0.3312, 0.2948, 0.3285, 0.3211]

# eixo X (número das sessões)
sessions = [1, 2, 3, 4, 5]
 
# configuração do gráfico
plt.figure(figsize=(10,6))

# plotando a linha verde (com filtro)
plt.plot(sessions, losses_with_filter, marker='o', linestyle='-', linewidth=2,
         color='#2ca02c', label='Com Filtro (1-40Hz)')

# plotando a linha vermelha (sem filtro)
plt.plot(sessions, losses_without_filter, marker='s', linestyle='--', linewidth=2,
         color='#d62728', label='Sem Filtro (Raw)')

# Preenchendo a área entre as linhas para destacar a diferença (Opcional, fica bonito)
plt.fill_between(sessions, losses_with_filter, losses_without_filter, color='gray', alpha=0.1)

# Estética
plt.title('Training Stability: Comparison of 5 sessions', fontsize=14, fontweight='bold')
plt.xlabel('Session Number (Execution)', fontsize=12)
plt.ylabel('Final Error (Loss)', fontsize=12)
plt.xticks(sessions) # Garante que mostre 1, 2, 3, 4, 5 inteiro
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Adicionando os valores nos pontos para facilitar a leitura
for i, txt in enumerate(losses_with_filter):
    plt.annotate(f"{txt:.3f}", (sessions[i], losses_with_filter[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', color='#2ca02c', fontweight='bold')

for i, txt in enumerate(losses_without_filter):
    plt.annotate(f"{txt:.3f}", (sessions[i], losses_without_filter[i]), 
                 textcoords="offset points", xytext=(0,-15), ha='center', color='#d62728', fontweight='bold')

plt.tight_layout()
plt.savefig('results/ablation_study/ablation_final_stats.png', dpi=300)
print("Graphics Generated in: ablation_final_stats.png")
plt.show()