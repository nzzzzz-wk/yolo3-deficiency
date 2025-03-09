import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 生成随机数据
if torch.cuda.is_available():
    device_str = 'cuda'
elif torch.mps.is_available():
    device_str = 'mps'
else:
    device_str = 'cpu'
print("Current Device:", device_str)
device = torch.device(device_str)
X = torch.randn(100, 10).to(device)
y = torch.randn(100, 1).to(device)

# 初始化模型并移动到 GPU
model = SimpleNet().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练 5 轮
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/5], Loss: {loss.item()}")
