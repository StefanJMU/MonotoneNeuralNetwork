import torch
import matplotlib.pyplot as plt
from MonoNN import MonoNetwork


def create_dataset(n_samples, func):
    x = 2 * torch.rand(n_samples, 1) - 1
    x = 20 * x
    y = func(x)
    return x, y

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = MonoNetwork([10, 10, 10, 1], input_monotonicity=[1]).to(device)
optim = torch.optim.Adam(model.parameters(), 1e-2)

n_instances = 1000
n_epochs = 250
batch_size = 50
x, y = create_dataset(n_instances, lambda x: torch.sin(x))


for epoch in range(0, n_epochs):
    # Shuffle
    idx = torch.randperm(n_instances)
    x = x[idx].to(device)
    y = y[idx].to(device)
    avg_loss = 0.
    for i in range(0, n_instances-batch_size, batch_size):
        batch_x = x[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        y_pred = model(batch_x)
        loss = ((y_pred - batch_y)**2).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()
        avg_loss += loss.item()

    print(epoch)
    print("Average loss of epoch: ", avg_loss / n_instances)

_, order = torch.sort(x, dim=0)
order = order.squeeze(dim=-1)
x = x[order]
y = y[order]

with torch.no_grad():
    pred = model(x).squeeze(dim=1).numpy()

x = x.squeeze(dim=1)
y = y.squeeze(dim=1)
plt.plot(x, y, c='b', label='ground truth')
plt.plot(x, pred, c='r', label='approximation')
plt.title("Approximation")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

plt.show()



