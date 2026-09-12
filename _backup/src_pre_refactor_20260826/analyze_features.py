import joblib

features = joblib.load("models/feature_names.pkl")

print("=" * 60)
print("TOTAL FEATURES :", len(features))
print("=" * 60)

for i, feature in enumerate(features, start=1):
    print(f"{i:02d}. {feature}")