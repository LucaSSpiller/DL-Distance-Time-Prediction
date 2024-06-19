import torch
import torch.nn as nn     # Responsável por estruturar a rede
import torch.optim as optim     # Responsável por ajustar e melhorar o modelo


# No X, dados que o modelo vai usar para aprender
X = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)


# No y, os resultados esperados, que mostra como o modelo deve prever
y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0], 
                  [30.5], [63.0], [67.0], [29.0], [62.0], 
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)



# Construção da Rede Neural
# Estrutura com camadas
# Essas camadas permitirão que os dados sejam processados na entrada, e que sejam aplicadas funções de ativação
# Onde essas funções permitirão que a rede aprenda padrões bem complexos
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Atualizando para aceitar apenas 1 valor de entrada, pois agora temos apenas a distância
        self.fc1 = nn.Linear(1, 5)   # De 2 para 1 na entrada
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Treinar a Rede
# Construir um Loop FOR, onde a rede faz as previsões, e calcula o quanto as previsões estão erradas (perda)
# A partir das perdas, ele ajusta os pesos das redes para melhorar as previsões (esse processo todo é repetido varias vezes)


for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 99:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')


# Pedir para ele fazer uma previsão de quanto demoraria para fazer uma distância de 10 Km por ex:
with torch.no_grad():
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')
