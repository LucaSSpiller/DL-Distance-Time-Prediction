# Introdução Distance-Time Prediction - Deep Learning
Passo a passo do funcionamento deste pequeno projeto desenvolvido na hands on da Pós Tech de IA para Devs (da FIAP). 
O projeto consiste na criação, treinamento e utilização de uma rede neural simples usando a biblioteca PyTorch. 

# Objetivo do Projeto
Criar, treinar e utilizar uma rede neural simples para prever o tempo de conclusão de uma determinada distância, com base em um conjunto de dados fornecido.

# Estrutura do Código

## Importação das Bibliotecas
```
import torch
import torch.nn as nn
import torch.optim as optim
``` 
- `torch:` Biblioteca principal do PyTorch para manipulação de tensores.
- `torch.nn:` Módulo que contém classes e funções para criar redes neurais.
- `torch.optim:` Módulo que fornece algoritmos de otimização para ajustar os parâmetros do modelo.

## Definição dos Dados
```
X = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0], 
                  [30.5], [63.0], [67.0], [29.0], [62.0], 
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)
```
- `X:` Tensor contendo os dados de entrada (distâncias).
- `y:` Tensor contendo os resultados esperados (tempos de conclusão).

## Construção da Rede Neural
### Definição da Estrutura

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
- `class Net(nn.Module):` Definição da classe da rede neural, que herda de nn.Module.
- `__init__:` Método construtor, define as camadas da rede.
- `self.fc1:` Primeira camada totalmente conectada (linear) com 1 neurônio de entrada e 5 de saída.
- `self.fc2:` Segunda camada totalmente conectada (linear) com 5 neurônios de entrada e 1 de saída.
- `forward:` Método que define a passagem de dados pela rede (feedforward).
- `torch.relu:` Função de ativação ReLU (Rectified Linear Unit) aplicada à saída da primeira camada.

## Instanciação do Modelo

```
model = Net()
```
- `model:` Instância da rede neural definida pela classe Net.

## Definição da Função de Perda e Otimizador

```
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
- `criterion:` Função de perda MSE (Mean Squared Error), que mede a diferença entre as previsões do modelo e os valores reais.
- `optimizer:` Otimizador SGD (Stochastic Gradient Descent) usado para ajustar os parâmetros do modelo. O parâmetro lr (learning rate) define a taxa de aprendizado.

## Treinamento da Rede Neural

```
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 99:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
```
- `for epoch in range(1000):` Loop de treinamento que executa 1000 épocas.
- `optimizer.zero_grad():` Zera os gradientes dos parâmetros.
- `outputs = model(X):` Calcula as previsões do modelo para os dados X.
- `loss = criterion(outputs, y):` Calcula a perda entre as previsões e os valores reais.
- `loss.backward():` Calcula os gradientes da perda em relação aos parâmetros.
- `optimizer.step():` Atualiza os parâmetros do modelo com base nos gradientes.
- `if epoch % 100 == 99:` A cada 100 épocas, imprime o número da época e o valor da perda.

## Previsão com o Modelo Treinado

```
with torch.no_grad():
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')
```
- `with torch.no_grad():` Desativa a computação de gradientes, economizando memória e aumentando a eficiência.
- `predicted = model(torch.tensor([[10.0]], dtype=torch.float32)):` Faz uma previsão usando a rede neural treinada para uma entrada de 10 km.
- `print(f'Previsão de tempo de conclusão: {predicted.item()} minutos'):` Imprime o resultado da previsão.
