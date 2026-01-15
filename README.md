# Detecção de Burnout via EEG: Uma Abordagem Independente de Sujeito com Aprendizado Few-Shot e Explicabilidade (XAI).

Detecção de Burnout em Tempo Real usando Few-Shot Learning com uma Camada de Interpretabilidade (XAI) gerando um mapa de calor do cérebro indicando porque a decisão foi tomada e Subject-Independent Classification com Transfer Learning pois será testado de pessoas de outros datasets.

# Dataset

[Link do Dataset Usado](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)

## Resumo do Dataset
É um dataset que contém dados de 48 participantes que estavam em uma carga de trabalho excessivo utilizando o SIMKAP (Vienna Test System: SIMKAP (Simultaneous Capacity/Multi-Tasking). 

Um projeto que testa o indivíduo a fazer várias tarefas ao mesmo tempo, afim de medir a capacidade de lidar com várias tarefas e o estresse nesses momentos). A atividade cerebral do indivíduo foi também registrada e incluída antes do teste. 

O aparelho de epocagem emotiva com amostragem de 128Hz e foi usado 14 canais para obter o dado com 2.5 minutos de gravação de EEG para cada caso. 

Os indivíduos tambémm foram perguntados para avaliar a sua performance mental no trabalho excessivo depois de cada estágio em uma escala de 1 a 9, e essasa avaliações estão registradas em um arquivo separado.

## Instrução do Dataset
O dado de cada indivíduo seguiu uma convenção de nome "subNum_task.txt". Por exemplo, sub01_lo.txt seria um dado cru de EEG para o indivíduo 1 de descanso, enquanto sub23_hi.txt seria um dado cru de EEG indivíduo 23 durante a atividade de multi tarefas. 

As linhas de cada datafile corresponde à amostra das gravações e as colunas corresponde aos 14 canais do aparelho de EEG: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4, respectively. 

As avaliações dos indivíduos é dado em um arquivo separado "rating.txt". São apresentados em um formato de valores separados por vírgulas: número do indivíduo, avaliação do descanso, avaliação do teste. Por exemplo: 1, 2, 8, seria que o indivíduo 1 avaliou como 2 o seu descanso e 8 no teste. Os indivíduos 5, 24 e 42 não possuem avaliação.

### Conceitos Fundamentais para Entendimento do Projeto
Para processar os sinais cerebrais (EEG) de forma eficiente, este projeto utiliza estruturas de dados específicas do **Pytorch**. Abaixo está explicado como os dados são organizados na memória.

1. **Tensor**
Na matemática e na computação, um **Tensor** é uma generalização de matrizes para múltiplas dimensões de um único tipo. ALgumas características do **Tensor**:
* É a estrutura de dados padrão para Deep Learning porque permite operações matemáticas em paralelo na GPU.
* Possui suporte integrado para GPUs habilitadas da NVIDIA-CUDA, habilitando, significantemente, speedups para grandes computações
* Integração de automação de cálculo de gradientes via ```autograd``` do Pytorch, essencial para treinamento de redes neurais.
* Gerenciamento de memória automática com garbage collection.
* Muito similar com o NumPy, facilitando a conversão de dados.
No contexto deste projeto, cada amostra de EEG de 4 segundos não é apenas uma lista de números, são matrizes de 3 dimensões:
* **Shape (Formato) de uma amostra:** [14, 33, 17]:
   * **14 Canais:** Os sensores físicos (eletrodos) na cabeça.
   * **33 Frequências:** A decomposição do sinal (Alpha, Beta, Gamma, etc).
   * **17 Janelas de Tempo:** Como o sinal muda ao longo daqueles 4 segundos.

2. **Batch (Lote):**
O **Batch** é um agrupamento de várias amostras (tensores) para serem processadas simultaneamente pela Rede Neural. Em vez de a rede aprender com um paciente por vez, ela olha para um grupo, no caso, de 32 pacientes de uma vez só.
* **Shape do Batch:** [32, 14, 33, 17].
   * A primeira dimensão (32) representa a quantidade de amostras naquele agrupamento.
Por que usar Batches:
* **Estabilidade Estatística:** No aprendizado Few-Shot (_Prototypical Networks_), precisa-se de várias amostras para calcular uma média(protótipo) confiável da classe. Um único exemplo ruidoso poderia enganar a rede, mas a média de 32 exemplos cancela os ruídos.
* **Eficiência de Hardware:** As GPUs são projetadas para multiplicar matrizes gigantes. Processar 32 exames juntos leva, praticamente, o mesmo tempo que processar 1, acelerando drasticamente o treinamento.
* **Nível Arquitetural:** O código utiliza a função ```utils.get_prototypes```, que opera sobre o batch inteiro para transformar ele em conhecimento. O processo ocorre em três etapas:
   1. **Filtragem (Masking):** O algoritmo separa os embeddings (vetores de características) em dois grupos: os que pertencem à classe "Relaxado" e os que pertencem à classe "Burnout".
   2. **Cálculo da Média (Mean):** Para cada grupo, ele calcula a média artimética de todos os vetores. Esse vetor médio é chamado de protótipo.
   3. **Empilhamento (Stacking):** A função retorna um novo tensor contendo apenas esses dois vetores ideais. É contra esses protótipos que a rede medirá as distâncias para aprender a classificar novos exemplos.

3. 
graph TD
    A[Sinal EEG Bruto] --> B[Tensor 3D<br>(1 Amostra)]
    B --> C{DataLoader}
    C --> D[Tensor 4D<br>(Batch de 32 Amostras)]
    D --> E[Rede Neural<br>(Processamento Paralelo)]

# Estrutura do Projeto
```text
eeg-Burnout-fewshot/
│
├── data/                        # ONDE FICAM OS DADOS
│   ├── raw/                     # Dados originais intocados (ex: SEED-VIG dataset, .edf, .mat)
│   ├── processed/               # Dados limpos e convertidos em tensores/imagens (numpy/torch arrays)
│
├── notebooks/                   # JUPYTER NOTEBOOKS (Para testes rápidos e exploração)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_test.ipynb
│   └── 03_shap_visualization_demo.ipynb
│
├── src/                         # CÓDIGO FONTE OFICIAL
│   ├── __init__.py              # Inicializador do pacote
│   ├── config.py                # Variáveis globais (Canais, Frequências, Caminhos)
│   ├── data_loader.py           # Scripts para carregar e transformar dados (Dataset Class do PyTorch)
│   ├── inference.py             # Script para classificação de novos pacientes
│   ├── make_mock_data.py        # Gerador de dados sintéticos para testes de fluxo
│   ├── models.py                # Definição das classes das Redes Neurais (CNN, EEGEmbedding)
│   ├── preprocessing.py         # Pipeline: Filtro de Banda -> Janelamento -> STFT
│   ├── test_metrics.py          # Geração de Matriz de Confusão e Relatório de Acurácia
│   ├── train_fewshot.py         # Script para o Fine-Tuning (Few-Shot Learning)
│   └── utils.py                 # Funções auxiliares (salvar modelos, plotar gráficos de loss)
│   ├── visualize_xai_utils.py   # Script executável para gerar e salvar imagens do XAI
|   |── visualize_spatial.py     # Script para gerar o mapa topográfico
│   ├── xai_utils.py             # Biblioteca de funções para Grad-CAM e visualização
│
├── results/                     # SAÍDAS DO MODELO
│   ├── saved_models/            # Pesos treinados (.pth)
│   ├── figures/                 # Gráficos gerados (Matrizes, Heatmaps)
│── ablation_study/              # Estudo de Treinamento da Rede Com e Sem o Filtro
│   ├── run_batch.py             # Roda o train_fewshot.py 5 vezes e captura o Loss
│   ├── plot_ablation.py         # Plota o Gráfico de Linha do Loss Com e Sem o Filtro
│
├── README.md                    # Documentação do projeto
└── requirements.txt             # Dependências do Python
```


# Instalação e Configuração

Recomenda-se o uso de um ambiente virtual (venv) para isolar as dependências.

1. Clone o Repositório:
```bash
git clone https://github.com/gabfarmarcondes/eeg-Burnout-fewshot.git
cd eeg-Burnout-fewshot
```

2. Crie e ative o ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Preparação dos Dados: 
Se não tiver o dataset real (STEW), gere dados sintéticos para teste:
```bash
python3 src/make_mock_data.py
```
_Se tiver o dataset real, coloque os arquivos ```.txt``` em ```data/raw/```._

5. Pré-processamento:
Limpa o sinal, aplica filtros de frequência e converte para Espectrogramas (Time-Frequency):
```bash
python3 src/preprocessing.py
```

6. Treinamento do Modelo:
Treina a Rede Neural para aprender a métrica de distância entre Relaxado e Burnout:
```bash
python3 src/train_fewshot.py
```

7. Validação e Métricas:
Gera a Matriz de Confusão e calcula a Acurácia em dados de teste (sujeitos não vistos):
```bash
python3 src/test_metrics.py
```

8. Explicabilidade (XAI):
Gera uma imagem visual mostrando onde a IA observou no cérebro para detectar o Burnout:
```bash
python3 src/visualize_xai_utilis.py
```
_A imagem será salva na pasta ```results/```._

9. Inferência (Simulação Real):
Simula a chegada de um paciente desconhecido e realiza o diagnóstico:
```bash
python3 src/inference.py
```

10. Visualização Espacial:
Mostra o mapa da cabeça onde ocorreu o Burnout, que é identificado pela vermelhidão da área.
```bash
python3 src/visualize_spatial.py
```

11. Rodando o Estudo do Filtro:
Foi estudado como a rede se comportaria com e sem o filtro passa-banda (1-40Hz). Para a execução do estudo:
1. Execute:
```bash
python3 src/preprocessing.py
```
2. Execute:
```bash
cd results/ablation_study
```

```bash
python3 ablation_study/run_batch.py
```
Irá capturar o Loss, com o filtro, 5 vezes no terminal, incluindo a média e o desvio padrão.

12. Gráfico do Estudo:
Foi plotado um gráfico do estudo para mostrar visualmente os dados obtidos.
```bash
python3 ablation_study/plot_ablation.py
```

# Resultado Esperado

O pipeline foi projetado para entregar:

* **Acurácia:** Alta distinção entre estados de repouso e sobrecarga.

* **Interpretabilidade:** O módulo XAI destaca atividades em frequências Beta (13-30Hz) como indicativo de Burnout, corroborando a literatura neurocientífica.

* **Análise Espacial:** Mapeamento topográfico indicando maior ativação nas regiões do Lobo Frontal, corroborando a hipótese de sobrecarga cognitiva.


# Explicação Das Figuras

## 1- Explicação da imagem do Paciente:
* O lado esquerdo:
    * Input: É o espectograma médio. Ele mostra a foto da atividade elétrica do cérebro.
        * Eixo Y: frequências (de 0 a 40Hz).
            * Eixo X: Tempo (janela de 4 segundos).
            * Cores: quanto mais amarelo, mais forte é a onda naquela frequência.

* O lado direito:
    * Grad-CAM: mapa de atenção da IA:
        * As cores vermelhas e laranjas mostram exatamente onde a rede neural olhou para tomar a decisão de que esse paciente tem Burnout.

## 1.1- Interpretação da Imagem da: 
Se olhar para onde estão as imagens vermelhas/laranjas no gráfico da direita:
* Localização no Eixo Y (frequência):
    * As manchas quentes não estão espalhadas aleatoriamente. Elas estão concentradas principalmente na faixa central (entre 10 e o 18 no eixo Y).
            - Considerando que o sinal vai até 40Hz, essa região central corresponde às Ondas Beta (13-30Hz).
* Significado:
    * Ondas Betas estão associadas a: Foco Intenso, Ansiedade, Pensamento Ativo e Estresse.
    * Em um cérebro relaxado (Ondas Alpha/Theta), a energia estaria mais baixa ou em frequências menores.
    * Conclusão: A IA aprendeu sozinha que, para detectar Burnout, ela precisa procurar por picos de atividade na faixa Beta, que indica um cébro que indica que não consegue relaxar ou está em estado de alerta constante.

## 1.2- Resumo:
Figura X: Visualização de Explicabilidade (XAI) utilizando Grad-CAM. À esquerda, o espectrograma de entrada de um paciente diagnosticado com Burnout. À direita, o mapa de calor gerado pela rede neural, onde as regiões em vermelho indicam as features de maior relevância para a classificação. Observa-se que o modelo foca predominantemente nas faixas de frequência intermediárias e altas (correspondentes às bandas Beta), correlacionando-se com a literatura médica que associa essas frequências a estados de ansiedade, estresse cognitivo e alerta sustentado, típicos da síndrome de Burnout.

## 2- Interpretação da Imagem da Matriz de Confusão: 
A estrutura  é um quadrado dividido em 4 quadrantes:
   ### 2.1. Eixo Vertical/Esquerdo:
   O True Label. Representa o estado real do paciente.
   * 0 = Relaxado.
   * 1 = Burnout.
   ### 2.2. Eixo Horizontal/Baixo:
   Predicted Label. Representa o que a IA previu.
   * 0 = IA disse que é Relaxado.
   * 1 = IA disse que é Burnout.
Portanto:
* O quadrante superior esquerdo (0,0):
   * O paciente estava relaxado.
   * A IA disse que estava relaxado.
   * Conclusão: A IA acertou o estado saudável.
* O quadrante inferior direito (1,1):
   * O paciente estava com Burnout.
   * A IA disse que o paciente estava com Burnout.
   * Conclusão: A IA acertou o estado de Burnout.
* O quadrante superior direito (0,1):
   * O paciente estava relaxado.
   * A IA disse que o paciente estava com Burnout.
   * Conclusão: A IA errou em dizer que o paciente estava com Burnout.
* O quadrante inferior esquerdo (1,0):
   * O paciente estava com Burnout.
   * A IA disse que o paciente estava relaxado.
   * Conclusão: A IA errou em dizer que o paciente estava relaxado.

## 3. Explicação da Análise Espacial:
A visualiação apresenta um mapa topográfico da atividade elétrica cerebral de cada paciente. A concentração de cores quentes na região anterior (testa) indica uma hiperativação do Lobo Frontal na faixa de frequência Beta. A barra lateral representa a intensidade do sinal em decibéis (dB). Cores mais escuras indicam maior atividade elétrica, associada a níveis elevados de estresse e Burnout.

## 4. Explicação do Gráfico do Ablation Study:
O gráfico representa o estudo comparativo de impacto do filtro passa-banda (1-40Hz) na convergência da Rede Neural. O experimento consistiu em 5 sessões de treinamento independentes para avaliar a estabilidade do modelo:
* **Linha Verde (Com Filtro):** Representa o modelo final validado. A oscilação observada reflete a complexidade de aprender padrões neurofisiológicos reais (ondas Beta) sem a influência de ruídos.
* **Linha Vermelha (Sem Filtro):** Representa o controle (dados brutos). A menor oscilação sugere que a rede encontrou "atalhos" (overfitting) baseados em artefatos musculares constantes, o que invalida seu uso clínico.

> **Como Criar o Gráfico:** Está explícito a forma de criar o gráfico no tópico **Instalação e Configuração** nos passos 11 e 12.
> **Detalhes Técnicos:** Para a discussão completa sobre a decisão de manter o filtro e a análise estatística dos dados, consulte o relatório técnico em [`results/ablation_study/RESULTS.md`](results/ablation_study/RESULTS.md).

# Autor

**Gabriel Farias Marcondes**

* Curso: Ciência da Computação

* Projeto: Neurocomputação e BCI