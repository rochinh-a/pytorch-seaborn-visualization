# Importando as bibliotecas necessárias
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
from mpl_toolkits.mplot3d import Axes3D

def set_seed(seed: int = 100):
    """Define uma seed para reprodutibilidade."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def create_dataset() -> tuple:
    """Carrega o conjunto de dados Iris para classificação.
    
    Returns:
        Tupla contendo features e labels
    """
    iris = load_iris()
    X = iris.data[:, :2]  # Usando apenas as duas primeiras features para visualização 2D
    y = iris.target
    return X, y

class SimpleNeuralNetwork(nn.Module):
    """Uma rede neural simples para classificação binária."""
    
    def __init__(self, input_size: int = 2):
        """Inicializa a arquitetura da rede.
        
        Args:
            input_size: Dimensão da camada de entrada
        """
        super().__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 3)  # Alterado para 3 classes
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define o passo forward da rede."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100
) -> list:
    """Treina o modelo e retorna o histórico de perdas.
    
    Args:
        model: Modelo PyTorch a ser treinado
        X_train: Dados de treino
        y_train: Labels de treino
        criterion: Função de perda
        optimizer: Otimizador
        epochs: Número de épocas de treinamento
        
    Returns:
        Lista com histórico de perdas
    """
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f'Época {epoch}, Loss: {loss.item():.4f}')
            
    return losses

def plot_training_results(losses: list):
    """Plota o gráfico de perda durante o treinamento e salva a figura."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(losses)), y=losses)
    plt.title('Curva de Aprendizado')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.grid(True)
    plt.savefig('images/curva_aprendizado.png')  # Salva a figura
    plt.show()

def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> float:
    """Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Labels de teste
        
    Returns:
        Acurácia do modelo
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
    return accuracy

def plot_decision_boundary(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor
) -> None:
    """Plota a fronteira de decisão do modelo.
    
    Args:
        model: Modelo treinado
        X: Features
        y: Labels
    """
    model.eval()
    
    # Definindo a grade de pontos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Fazendo previsões para cada ponto da grade
    with torch.no_grad():
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        outputs = model(grid)
        _, predicted = torch.max(outputs, 1)
        Z = predicted.reshape(xx.shape)
    
    # Plotando
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k', alpha=0.8)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.title('Fronteira de Decisão')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_decision_boundary_3d(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor
) -> None:
    """Plota a fronteira de decisão do modelo em 3D e salva a figura."""
    model.eval()
    
    # Definindo a grade de pontos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Fazendo previsões para cada ponto da grade
    with torch.no_grad():
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        outputs = model(grid)
        _, predicted = torch.max(outputs, 1)
        Z = predicted.reshape(xx.shape)
    
    # Plotando
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='coolwarm', edgecolor='k')
    ax.set_title('Fronteira de Decisão 3D')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Classe')
    plt.savefig('images/fronteira_decisao_3d.png')  # Salva a figura
    plt.show()

def make_prediction(
    model: nn.Module,
    features: list
) -> int:
    """Faz previsão para novos dados.
    
    Args:
        model: Modelo treinado
        features: Lista com as features de entrada
        
    Returns:
        Classe prevista
    """
    if len(features) != 2:
        raise ValueError("features deve conter exatamente 2 elementos")
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(features)
        output = model(x)
        _, predicted = torch.max(output, 0)
    return predicted.item()

def main():
    # Definindo a seed
    set_seed(42)
    
    # Carregando e dividindo o dataset
    X, y = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Convertendo para tensores PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Instanciando o modelo
    model = SimpleNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Treinando o modelo
    losses = train_model(model, X_train, y_train, criterion, optimizer)

    # Plotando os resultados do treinamento
    plot_training_results(losses)

    # Avaliando o modelo
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Acurácia no conjunto de teste: {accuracy:.2%}')

    # Plotando a fronteira de decisão
    plot_decision_boundary_3d(model, X_test, y_test)

    # Exemplo de previsão
    novo_dado = [5.0, 3.5]  # Exemplo de novo dado
    predicao = make_prediction(model, novo_dado)
    print(f'Previsão para {novo_dado}: Classe {predicao}')

if __name__ == "__main__":
    main()