import torch

if torch.cuda.is_available():
    device_str = 'cuda'
elif torch.mps.is_available():
    device_str = 'mps'
else:
    device_str = 'cpu'
print("Current Device:", device_str)
device = torch.device(device_str)
# 创建一个随机张量并移动到 GPU
x = torch.rand((3, 3)).to(device)
y = torch.rand((3, 3)).to(device)

# 在 GPU 上进行矩阵加法计算
z = x + y

print("张量 x (GPU):", x)
print("张量 y (GPU):", y)
print("张量 z (x + y) (GPU):", z)
print("计算设备:", z.device)  # 确保输出是 "cuda:0"
