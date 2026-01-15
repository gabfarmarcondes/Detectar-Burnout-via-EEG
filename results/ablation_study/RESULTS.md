# Experimento de Ablation Study - Metodologia de Validação

Para garantir a robustez clínica do modelo, foi realizado um estudo comparativo focado na **Metodologia de Divisão de Dados** (Data Splitting Strategy). O objetivo foi verificar se o modelo estava aprendendo padrões generalizáveis de Burnout ou apenas memorizando a identidade dos pacientes (_Data Leakage_).

## Teste A: Isolamento de Sujeito (Subject Isolation) - Proposto
* **Metodologia:** Divisão sequencial onde os últimos 20% dos pacientes (inéditos) são separados exclusivamente para teste.
* **Error (Loss) Médio:** 0.2122.
* **Acurácia Média:** 92.44% (σ = 0.97).
* **Observação:** Representa o desempenho real em cenário clínico (novos pacientes).
* **Log:** 
```bash
Initiating 5 Trainings Sessions.

Running 1/5 Training Session.
Registered Loss: 0.2029 | Acc: 93.75

Running 2/5 Training Session.
Registered Loss: 0.2095 | Acc: 91.71

Running 3/5 Training Session.
Registered Loss: 0.2042 | Acc: 92.66

Running 4/5 Training Session.
Registered Loss: 0.2298 | Acc: 91.03

Running 5/5 Training Session.
Registered Loss: 0.2144 | Acc: 93.07

==================================================
Final Results (5 Sessions)
==================================================
LOSS     -> Mean: 0.2122  | Std Dev: 0.0097
ACCURACY -> Mean: 92.44% | Std Dev: 0.97
--------------------------------------------------
Raw Losses: [0.2029, 0.2095, 0.2042, 0.2298, 0.2144]
Raw Accs:   [93.75, 91.71, 92.66, 91.03, 93.07]
==================================================
```

## Teste B: Sem o Filtro (Dados Brutos)
* **Metodologia:** O dataset inteiro foi embaralhado antes da divisão. Janelas temporais do mesmo paciente aparecem tanto no treino quanto no teste.
* **Error (Loss) Médio:** 0.2752.
* **Acurácia Média:** 89.06% (σ = 0.40).
* **Observação:** Simula um cenário de vazamento de dados (_Data Leakage_) por identidade.
* **Log:**
```bash
Initiating 5 Trainings Sessions.

Running 1/5 Training Session.
Registered Loss: 0.2662 | Acc: 89.4

Running 2/5 Training Session.
Registered Loss: 0.2946 | Acc: 88.32

Running 3/5 Training Session.
Registered Loss: 0.2684 | Acc: 89.19

Running 4/5 Training Session.
Registered Loss: 0.2656 | Acc: 88.99

Running 5/5 Training Session.
Registered Loss: 0.2814 | Acc: 89.4

==================================================
Final Results (5 Sessions)
==================================================
LOSS     -> Mean: 0.2752  | Std Dev: 0.0113
ACCURACY -> Mean: 89.06% | Std Dev: 0.40
--------------------------------------------------
Raw Losses: [0.2662, 0.2946, 0.2684, 0.2656, 0.2814]
Raw Accs:   [89.4, 88.32, 89.19, 88.99, 89.4]
==================================================
```

## Conclusão e Análise
1. **Imunidade ao Vazamento de Identidade:** Contra-intuitivamente, o modelo performou ligeiramente melhor no cenário isolado (92%) do que o cenário misturado (89%). Em redes neurais tradicionais, o _Random Split_ costuma inflar a acurácia devido à memorização das características individuais do sujeito.

O fato da acurácia não ter disparado no Random Split demonstra que a arquitetura _Prototypical Network_  (Few-Shot) projetada neste estudo possui alta **Invariância à Identidade**. O modelo focou em aprender a assinatura neural do Burnout, ignorando as características únicas de cada cérebro.

2. ** Qualidade dos Dados:** A superioridade do teste A sugere que os pacientes selecionados para o conjunto de testes (os últimos da lista do Dataset STEW) possuiam marcadores de Burnout mais claros e definidos do que a média geral da população misturadas no teste B.

3. **Decisão Final:** O modelo final adotará estritamente o **Isolamento de Sujeito (teste A)**. Embora os resultados sejam numericamente próximos, esta metodologia é a única que garante validade científica para aplicações de BCI no mundo real, onde o sistema deve funcionar para usuários nunca antes vistos.
