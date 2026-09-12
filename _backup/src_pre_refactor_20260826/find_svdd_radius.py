import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import prepare_data
from deep_svdd import DeepSVDD

# Load Monday–Thursday
X_train, _, y_train, _, _, _, _ = prepare_data(split_by_day=True)

X_benign = X_train[y_train == 0]

loader = DataLoader(
    TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
    batch_size=4096,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepSVDD(69, 16).to(device)
model.load_state_dict(
    torch.load("saved_models/deep_svdd/deep_svdd_model.pth", map_location=device)
)
model.eval()

center = torch.load(
    "saved_models/deep_svdd/center.pt",
    map_location=device
)

distances = []

with torch.no_grad():
    for (x,) in loader:
        z = model(x.to(device))
        d = torch.sum((z - center) ** 2, dim=1)
        distances.extend(d.cpu().numpy())

distances = np.array(distances)

r95 = np.percentile(distances, 95)
r99 = np.percentile(distances, 99)

print(f"95% Radius : {r95:.8f}")
print(f"99% Radius : {r99:.8f}")

torch.save(
    torch.tensor(r99),
    "saved_models/deep_svdd/radius.pt"
)

print("Radius saved.")