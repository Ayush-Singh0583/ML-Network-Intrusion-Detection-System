import os
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

from preprocessing import prepare_data
from deep_svdd import DeepSVDD


# ==========================================
# LOAD FRIDAY DATA
# ==========================================

(
    _,
    X_test,
    _,
    y_test,
    _,
    _,
    _
) = prepare_data(split_by_day=True)

# Binary labels for visualization
y_binary = np.where(y_test == "BENIGN", "BENIGN", "ATTACK")


# ==========================================
# LOAD MODEL
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepSVDD(input_dim=69, latent_dim=16).to(device)

model.load_state_dict(
    torch.load(
        "saved_models/deep_svdd/deep_svdd_model.pth",
        map_location=device
    )
)

model.eval()

center = torch.load(
    "saved_models/deep_svdd/center.pt",
    map_location=device
)


# ==========================================
# COMPUTE DISTANCES
# ==========================================

X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():

    embeddings = model(X_tensor)

    distances = torch.sum(
        (embeddings - center) ** 2,
        dim=1
    ).cpu().numpy()


# ==========================================
# STATISTICS
# ==========================================

benign_dist = distances[y_binary == "BENIGN"]
attack_dist = distances[y_binary == "ATTACK"]

print("\n========== SVDD DISTANCE STATS ==========\n")

print(f"Benign Mean : {benign_dist.mean():.6f}")
print(f"Attack Mean : {attack_dist.mean():.6f}")

print(f"Benign Max  : {benign_dist.max():.6f}")
print(f"Attack Max  : {attack_dist.max():.6f}")


# ==========================================
# PLOT
# ==========================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10,6))

plt.hist(
    benign_dist,
    bins=100,
    alpha=0.6,
    density=True,
    label="BENIGN"
)

plt.hist(
    attack_dist,
    bins=100,
    alpha=0.6,
    density=True,
    label="ATTACK"
)

plt.xlabel("Distance from SVDD Center")
plt.ylabel("Density")
plt.title("Deep SVDD Distance Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/svdd_distance_distribution.png",
    dpi=300
)

plt.close()

print("\nGraph saved to results/svdd_distance_distribution.png")