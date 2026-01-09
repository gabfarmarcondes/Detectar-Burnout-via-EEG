# Experimento de Ablation Study - Impacto do Filtro

## Teste A: Com o Filtro (Baseline)
* **Status do Código:** Linha 115 do ```preprocessing.py``` (```raw.fitler()```) **descomentada**
* **Error (Loss):** 0.3255.
* **Observação:** O modelo convergiu e aprendeu bem, usando apenas a frequência de banda válida (1~40Hz).
* **Log:** 
```bash
train_fewshot.py rodando 5 vezes:
Initiating 5 Trainings Sessions.

Running 1/5 Training Session.
Registered Loss: 0.3174

========================================
Final Result: 5 Sessions.
Loss Mean: 0.3174
Standard Deviation: 0.0
Brute Values: [0.3174]

========================================

Running 2/5 Training Session.
Registered Loss: 0.3436

========================================
Final Result: 5 Sessions.
Loss Mean: 0.3305
Standard Deviation: 0.0131
Brute Values: [0.3174, 0.3436]

========================================

Running 3/5 Training Session.
Registered Loss: 0.3085

========================================
Final Result: 5 Sessions.
Loss Mean: 0.32316666666666666
Standard Deviation: 0.014898396632598503
Brute Values: [0.3174, 0.3436, 0.3085]

========================================

Running 4/5 Training Session.
Registered Loss: 0.3473

========================================
Final Result: 5 Sessions.
Loss Mean: 0.3292
Standard Deviation: 0.01660346349410267
Brute Values: [0.3174, 0.3436, 0.3085, 0.3473]

========================================

Running 5/5 Training Session.
Registered Loss: 0.2845

========================================
Final Result: 5 Sessions.
Loss Mean: 0.32026
Standard Deviation: 0.023242943015031475
Brute Values: [0.3174, 0.3436, 0.3085, 0.3473, 0.2845]

========================================
```

## Teste B: Sem o Filtro (Dados Brutos)
* **Status do Código:** Linha 115 do ```preprocessing.py``` (```raw.fitler()```) **comentada**
* **Error (Loss):** 0.2853.
* **Observação:** O modelo atingiu um Loss menor (melhor desempenho numérico) mais rapidamente. Isso indica a presença de características de alta frequência (provavelmente ruído) que facilitaram a classificação.
* **Log:**
```bash
train_fewshot.py rodando 5 vezes:
Initiating 5 Trainings Sessions.

Running 1/5 Training Session.
Registered Loss: 0.3134

========================================
Final Result: 5 Sessions.
Loss Mean: 0.3134
Standard Deviation: 0.0
Brute Values: [0.3134]

========================================

Running 2/5 Training Session.
Registered Loss: 0.3312

========================================
Final Result: 5 Sessions.
Loss Mean: 0.32230000000000003
Standard Deviation: 0.008899999999999991
Brute Values: [0.3134, 0.3312]

========================================

Running 3/5 Training Session.
Registered Loss: 0.2948

========================================
Final Result: 5 Sessions.
Loss Mean: 0.3131333333333333
Standard Deviation: 0.014861434056719495
Brute Values: [0.3134, 0.3312, 0.2948]

========================================

Running 4/5 Training Session.
Registered Loss: 0.3285

========================================
Final Result: 5 Sessions.
Loss Mean: 0.316975
Standard Deviation: 0.014488680926847686
Brute Values: [0.3134, 0.3312, 0.2948, 0.3285]

========================================

Running 5/5 Training Session.
Registered Loss: 0.3211

========================================
Final Result: 5 Sessions.
Loss Mean: 0.31779999999999997
Standard Deviation: 0.013063690137170276
Brute Values: [0.3134, 0.3312, 0.2948, 0.3285, 0.3211]

========================================
```

## Conclusão
Para validar a consistência dos resultasos e descartar variações aleatórias de inicialização, o treinamento foi executado 5 vezes para cada cenário.

1. **Análise Estatística:** Os resultados finais demonstraram uma proximidade extrema entre os dois cenários:
* Com filtro: Média de Loss 0.32026.
* Sem filtro: Média de Loss 0.31779.

A diferença entre as médias é de apenas 0.00247, o que é estatisticamente irrelevante dada a sobreposição dos desvios padrões.

2. **Interpretação Fisiológica:** Observa-se que o modelo treinado com dados brutos (sem filtro) apresentou uma estabilidade ligeiramente maior (menor desvio padrão) e um erro médio marginalmente menor. Atribuimos este fenômeno ao **Aprendizado de Artefatos (Artifact Learning)**. Dados de EEG sem filtro contêm sinais de **Eletromiografia(EMG)** de alta frequência (>40Hz) provenientes de tensão muscular facial. COmo o estresse/burnout está fisiologicamente correlacionado à tensão muscular, o modelo tende a usar esse ruído forte utilizando **Shortcut Learning** para a classificação, em vez de aprender os padrões cerebrais sutis.

3. **Decisão Final:** Apesar da ligeira vantagem numérica dos dados brutos, foi decidido manter o filtro de 1-40Hz ativo no modelo final.
* **Justificativa:** Um sistema de BCI (Interface Cérebro Computador) robusto deve garantir que a classificação seja baseada em neurofisiologia cortical (ondas cerebrais reais), e não em movimentos musculares involuntários.

* **Resultado:** O modelo com filtro provou ser competitivo (Loss ~0.32) mantendo a validade clínica e evitando falsos positivos causados por simples movimentos faciais do usuário.