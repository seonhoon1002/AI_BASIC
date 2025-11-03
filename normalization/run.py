import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import train, test
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_bn import CNN_BN

if __name__ == "__main__":
    # 하이퍼파라미터
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # 데이터셋
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_BN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 및 평가
    for epoch in range(num_epochs):
        loss = train(model, device, train_loader, optimizer, criterion)
        acc = test(model, device, test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {loss:.4f}, Accuracy: {acc:.2f}%')
