# Precisa decidir como a rede neural vai olhar para os dados gerados.
# Os dados são tridimensionais (Canais x Freq x Tempo), então para estruturar o modelo para o PyTorch usa-se:
    # Imagem Multicanal:
        # Transformar o dado em uma imagem onde:
            # Altura: 33 frequências
            # Largura: 17 tempo
            # Profundidade (Canais de Cor): 14 canais
            # Quantidade Amostras: 48
        # No caso, usaria uma convolução 2D. A rede aprenderia padrões "tempo-frequência" simultaneamente em todos os eletrodos.

# O objetivo deste modelo é transformar a matriz (48, 14, 33, 17) em um vetor de características.

# Input: O tensor (Batch, 14, 33, 17).
"""
 Camadas:
    Uma camada convolucional (Conv2d) para extrair as features.
    Uma função de ativação.
    Um MaxPool para reduzir a dimensão.
    Talvez uma segunda camada convolucional.
    Uma camada Flatten para esticar tudo num vetor no final.
"""

import torch
import torch.nn as nn # usado para criar redes neurais.
import torch.nn.functional as F

# # nn depende do autograd (uma biblioteca que provê classes e funções para implementar, automaticamente, diferenciação arbitrária com funções de valores escalares)
#     # é preciso declarar os tensors para que o gradiente seja computado com o requires_grad=True. Só é aceitável floats e números complexos.
#     # backward = computa a soma dos gradientes dos tensors dados, respeitando o grafo das folhas.
#     # grad = computa e retorna a soma dos gradientes dos outputs dos inputs dados.
#     # TODO: estudar depois o autograd.

# """dada uma entrada, irá ocorrer convoluções que geram subamostras que, ao final, 
# gerará várias camadas dessas subamostras que serão conectadas e gerar uma conexão gaussiana.
# Resumindo, um feed-forward network, pega a entrada separa em várias camadas, uma atras da outra, e no final gera uma saída."""

# """
# Um típico treino de rede neural segue:
# 1. define a rede neural para que tenha alguns parâmetros de aprendizagem.
# 2. itera sobre esse dataset de entradas
# 3. processa a entrada pela rede
# 4. computa o loss (o quão correto a saída está)
# 5. propaga gradientes de volta apra os parâmetros da rede
# 6. atualiza os pesos da rede, tipicamente usando uma regra de atualização: weight = weight - learning_rate * gradient

# são esses pesos que treinam a rede neural.
# """

# # Definindo uma rede:
# class Net(nn.Module): # classe da rede neural
#     def __init__(self): # construtor da rede neural
#         super(Net, self).__init__()
#         # 1 canal de entrada, 6 canais de saída, um quadrado de convolução 5x5
#         # kernel = matriz pequena de 5x5
#         # Conv2d lida com imagens.
#         # self.conv1 = nn.Conv2d(entrada, saída, tamanho_do_filtro)
#         """ Camada de Convolução:
#         "1" significa que a imagem que entra na rede está na escala de cinza, Se fosse RGB seria 3. Em EEG, é o número correspondente de canais.
#         "6" são as 6 versões diferentes da imagem. Cada versão é filtrada para destacar algo diferente (borda, curva, textura).
#         "5" é o tamanho das diferentes versões da imagem. Um kernel que irá analisar a imagem através de pixles de 5x5.
#         """
#         self.conv1 = nn.Conv2d(1, 6, 5)

#         """ Conexão:
#         "6" pois o conv1 recebeu 6 canais e serão usadas as mesmas 6 como entrada. A saída de um tem que ser a entrada da próxima.
#         dos 6, são transformadas em "16" canais (aprende features mais complexas).
#         "5" o kernel usa o filtro 5x5 novamente.
#         """
#         self.conv2 = nn.Conv2d(6, 16, 5)

#         """ 
#         modelo: y = Wx + b. Sempre vai formar uma reta em um gráfico, por isso é chamada de função linear. Nessa função:
#             - W = weight (peso). O valor de W afeta o quanto a reta está inclinada. Girando sempre em um ponto específico se alterar somente ele.
#             - x = entrada. Input.
#             - b = viés. Posição da reta.
#             - y = saída. Output.
#         W e b são os parâmetros importantes para a IA. Quanto mais parâmetros, maior a IA, maior o processamento.
#         Em EEG seria: a conexão onde todos os neurônios da entrada se ligam com os neurônios da saída.
#         """
#         # Camada Linear.
#         # Linear lida com listas de números. (vetores 1D)
#         # Antes era 3D e agora isá ser transformado em 1D.
#         # 16 é o número de canais que saiu da conv2d, 5*5 é o tamanho final da imagem, quer resumir tudo em 120 features. 
#         self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5 * 5 da dimensão da imagem
#         # a fc2 pega os 120 e resume para 84.
#         self.fc2 = nn.Linear(120, 84)
#         # a fc3 pega os 84 e resume para 10.
#         self.fc3 = nn.Linear(84,10)

#     def forward(self, input):
#         # (USAR A IMAGEM DO SITE PYTORCH COMO REFERÊNCIA PARA ENTENDER OS NÚMEROS DOS PARÂMETROS)
#         # A convolução da camada C1: 1 entrada de canal da imagem, 6 canais de saída, kernel de 5x5
#         # Um quadrado de convolução de 5x5, isto é usado na ativação da função RELU e
#         # a saída é um tensor com o tamanho (N, 6, 28, 28), onde N é o número de Batch
#         c1 = F.relu(self.conv1(input))

#         # As subamostras da camada S2: um grid de 2x2, puramente funcional.
#         # esta camada não tem nenhum parâmetro e a saída um tensor (N, 6, 14, 14)
#         s2 = F.max_pool2d(c1, (2,2))

#         # Convolução da camada C3: 6 canais de entrada, 16 canais de saída.
#         # quadrado de convolução 5x5, usado no RELU e
#         # a saída um tensor (N, 16, 10, 10)
#         c3 = F.relu(self.conv2(s2))

#         # Subamostras da camada s4: um grid de 2x2 puramente funcional.
#         # esta camada não tem nenhum parâmetro e a saída um tensor (N, 16, 5, 5)
#         s4 = F.max_pool2d(c3, 2)
#         # Operação Flatten: puramente funcional, a saída um tensor (N, 400)
#         s4 = torch.flatten(s4, 1)

#         # Conexão completa camada f5: tensor (N, 400) como entrada e a saída um tensor (N, 120). É usado uma função de ativação RELU (Zera valores negativos).
#         f5 = F.relu(self.fc1(s4))

#         # Conexão completa camada f6: um tensor (N, 120) como entrada e a saída um tensor (N, 84)
#         f6 = F.relu(self.fc2(f5))

#         # Conexão completa camada OUTPUT: um tensor (N, 84) como entrada e a saída um tensor (N, 10)
#         output = F.relu(self.fc3(f6))
#         return output
    
#     """
#     RESUMO DO QUE FOI FEITO:
#     Foi usado uma imagem de 32x32 como entrada, fez a primeira convolução que dividiu em 6 camadas de 28x28 
#     (TamanhoSaida = TamanhoEntrada - TamanhoKernel + 1) utilizando um kernel de 5x5 pixels, 
#     resultou em uma divisão de subcamadas utilizando um grid de 2x2 pixels que resultou em 6 camadas de 14x14 (28 / 2). 
#     Faz outra convolução dessas 6 camadas das subamostras, resultando em 16 canais de saída de 10x10 (14 - 5 + 1). 
#     Dessas 16 subcamadas, irá ser passado um grid de 2x2 pixels e gera um 16 subamostras 5x5. 
#     Faz a operação flatten para planar as camadas geradas, gerando um tensor (N, 400 (16 * 5 * 5)). 
#     Faz a primeira conexão completa, um tensor de 120 que vai virar 84, outra conexão que de 84 irá ser passado como 10, 
#     e no fim retornando a saída com as features filtradas.
#     """

# net = Net()
# print(net)

# Recebe um sinal EEG complexo e ruidoso e retorna um vetor de 64 números (serão cérebros relaxados e sobrecarregados)
class EEGEmbedding(nn.Module):
    def __init__(self):
        super(EEGEmbedding, self).__init__()
        # Convolução
        # Altura x Largura = 33 x 17 para 14 canais EEG (14, 33, 17)
        # O preprocessing.py transformou o EEG bruto em espectograma para ver as frequências cerebrais
        self.conv1 = nn.Conv2d(14, 64, 3) # 14 canais de EEG, 64 filtros que a rede irá aprender (out_channels), tamanho do kernel 3
        self.relu = nn.ReLU() # deixa passar somente os valores úteis
        self.pool = nn.MaxPool2d(2) # reduz a imagem pela metade para ficar mais leve

        # Linear
        # É preciso saber quantos números restaram depois de passar pelo filtro e pelo pooling
        # Seguindo as contas:
        """
        1. Entrada: 33 x 17 (frequência x tempo)
        2. Depois do self.conv1 (kernel 3):
            2.1 subtrai 2 (3-1): Como o kernel tem tamanho 3, ele precisa de espaço. Para o kernel estar no centro, ele precisa de vizinhos.
                Com o kernel 3, o centro precisa de 1 vizinho à esquerda e à direita. Por isso ele não pode parar na borda da imagem, pois falta vizinho.
                Logo, se olhar pela fórmula: N (tamanho da imagem) - K (tamanho do kernel) + 1, se rearranjar fica:
                    N - (K + 1) -> N - (3 - 1) -> N - 2. A convolução sempre come K - 1 pixles da dimensão.
                2.1.1: 33 - 2 = 31
                2.1.2: 17 - 2 = 15
            2.2 resultado da imagem: 31 x 15
        3. depois do self.pool de tamanho 2:
            3.1 divide por 2 (arredondando para baixo):
                3.1.1: 31 / 2 = 15
                3.1.2: 15 / 2 = 7
            3.2 resultado final é uma imagem de tamanho 15 x 7
        4. O cálculo do Flatten: tendo 64 filtros e cada um gerou uma imagem de 15x7. Para entrar na camada linear, é preciso fazer a conta:
            4.1: 64 * 15 * 7 = 6720
        """
        self.fc1 = nn.Linear(64 * 15 * 7, 64)

    def forward(self, x):
        # Passa pela convolução
        x = self.conv1(x)
        x = self.relu(x)
        # Passa pelo pooling
        x = self.pool(x)

        # Flatten. x.size(0) é o Batch size e o -1 é para o PyTorch calcular o valor do Batch (64 * 15 * 7)
        x = x.view(x.size(0), -1)
        # Passa pela camada linear
        x = self.fc1(x)

        return x