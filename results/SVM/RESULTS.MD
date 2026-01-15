# Estudo Comparativo Entre o Modelo SVM Clássico e o Modelo Few-Shot Learning

## 1. Objetivo
O objetivo deste comparativo é validar o modelo clássico de Machine Learning (Support Vector Machine - SVM) e o modelo adotado para este projeto (Few-Shot Learning).

O desafio central deste projeto é a classificação de sinais EEG para a detecção de Burnout. Para garantir a validade clínica, adotou-se o protocolo de Isolamento de Sujeito (Subject Isolation):
* Para o treino foi dividido em 80% dos pacientes.
* Para o teste foi os 20% restantes.

Isso garante que o modelo seja avaliado por sua capacidade de generalização para novos indivíduos. e não pela memorização de padrões específicos dos pacientes de treino.

# 2. Explicação do Modelo Clássico SVM
O SVM é um algoritmo de aprendizado supervisionado clássico, amplamente utilizado em classificação de sinais biológicos devido à sua eficiência em espaços de altas dimensões.

O SVM busca encontrar um **hiperplano ideal** que separe as classes (no caso, "Relaxado" e "Burnout") com a maior margem possível.
* Neste estudo, utilizou-se o Kernel **Linear**. Isso significa que o modelo tenta traçar uma linha reta (ou plano) para separar os dados.
* A entrada para o SVM foi o vetor achatado (flatten) das características do espectograma.

Embora robusto, o SVM Linear possui dificuldades em capturar as não-linearidades complexas e as variações sutis entre os cérebros de diferentes indivíduos (variante inter-sujeito), o que tende a prejudicar sua performance em sujeitos inéditos.

# 3. Explicação do Modelo Few-Shot Learning
O modelo principal do projeto utiliza uma abordagem de **Deep Learning** baseada em _Prototypical Network_(Redes Prototípicas). Diferente da classificação tradicional, este modelo aprende um espaço métrico onde as distâncias correspondem à similaridade semântica.

**Como funciona:**
* **Embedding (CNN):** Uma Rede Neural Convolucional extrai as características profundas do sinal EEG, transformando uma matriz (Canais x Tempo x Frequência) em um vetor compacto.
* **Cálculo de Protótipo:** O modelo calcula o ponto médio (centróide) de cada classe no espaço vetorial. Esse ponto médio é chamado de **Protótipo**.
* **Classificação via Distância:** Para classificar uma nova amostra, o modelo calcula a **Distância Euclidiana** entre a amostra e os protótipos. A amostra é atribuída à classe do protótipo mais próximo.

**Vantagem Teórica**
Ao aprender a geometria das classes em vez de fronteiras fixas, o Few-Shot Learning é, teoricamente, mais capaz de generalizar o conceito de Burnout para novos pacientes, mesmo com poucos exemplos.

# 4. Resultado:
Os testes foram realizados sob as mesmas condições de pré-processamento e divisão de dados (Isolamento de Sujeito).

| Métrica | SVM | Few-Shot | Diferença |
| :---: | :---: | :---: | :---: |
| Acurácia | 75.95% | 89.40% | +13.45% |
| F1-Score | ~0.76 | ~0.89 | +0.13 | 

## Análise da Matriz de Confusão (SVM)
O SVM apresentou o seguinte comportamento nos dados de teste:
```bash
[[257 113] -> Relaxado (113 Falsos Positivos)
 [ 58 283]] -> Burnout (58 falsos Positivos)
```

Observa-se que o SVM tende a gerar muitos **Falsos Positivos** (classificando sujeitos relaxados como burnout), o que indica uma dificuldade em separar o ruído natural do cérebro dos padrões patológicos em pacientes desconhecidos.

# Conclusão
Os resultados demonstram a superioridade estatística do modelo **Few-Shot Learning** para esta tarefa.

Enquanto o SVM sofreu uma degradação de performance ao enfrentar sujeitos inéditos (caindo para ~76%), a Rede Neural Prototípica manteve uma robustez elevada (~89%). Isso confirma a hipótese de que a extração profunda de características (Deep Learning) combinada com a aprendizagem baseada em distância (Few-Shot) é mais adequada para lidar com a alta variabilidade dos sinais de EEG entre diferentes seres humanos.