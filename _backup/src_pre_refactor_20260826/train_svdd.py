import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import prepare_data
from deep_svdd import DeepSVDD


# =====================================================
# INITIALIZE CENTER (Mean of Benign Embeddings)
# =====================================================

def initialize_center(model, loader, device):

    model.eval()
    embeddings = []

    with torch.no_grad():

        for (x,) in loader:

            x = x.to(device)
            z = model(x)
            embeddings.append(z)

    embeddings = torch.cat(embeddings, dim=0)
    center = embeddings.mean(dim=0)

    return center


# =====================================================
# LOAD DATA
# =====================================================

(
    X_train,
    _,
    y_train,
    _,
    scaler,
    _,
    feature_names
) = prepare_data(split_by_day=True)

# Keep ONLY BENIGN samples
benign_mask = (y_train == 0)
X_benign = X_train[benign_mask]

print("\n========== DEEP SVDD TRAINING ==========")
print(f"Benign Samples : {len(X_benign)}")


# =====================================================
# DATALOADER
# =====================================================

X_tensor = torch.tensor(
    X_benign,
    dtype=torch.float32
)

dataset = TensorDataset(X_tensor)

loader = DataLoader(
    dataset,
    batch_size=4096,
    shuffle=True
)


# =====================================================
# MODEL
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device : {device}")

model = DeepSVDD(
    input_dim=69,
    latent_dim=16
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-6
)


# =====================================================
# INITIALIZE CENTER
# =====================================================

center = initialize_center(
    model,
    loader,
    device
).detach()

print("Center initialized.")


# =====================================================
# TRAINING LOOP
# =====================================================

EPOCHS = 25

for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0.0

    for (x,) in loader:

        x = x.to(device)

        optimizer.zero_grad()

        z = model(x)

        dist = torch.sum((z - center) ** 2, dim=1)
        loss = dist.mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)

    print(f"Epoch [{epoch+1:02d}/25] Loss : {epoch_loss:.10f}")


# =====================================================
# SAVE MODEL
# =====================================================

save_dir = "saved_models/deep_svdd"
os.makedirs(save_dir, exist_ok=True)

torch.save(
    model.state_dict(),
    os.path.join(save_dir, "deep_svdd_model.pth")
)

torch.save(
    center,
    os.path.join(save_dir, "center.pt")
)

joblib.dump(
    scaler,
    os.path.join(save_dir, "scaler.pkl")
)

joblib.dump(
    feature_names,
    os.path.join(save_dir, "feature_names.pkl")
)

print("\n========== TRAINING COMPLETE ==========")
print("Files saved to:", save_dir)