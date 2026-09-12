import pandas as pd
import json

# Load one of your datasets
df = pd.read_csv("data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()
# Remove unnecessary columns
drop_columns = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

df.drop(columns=drop_columns, inplace=True, errors="ignore")

# Remove label
sample = df.drop(columns="Label").iloc[0]

# Convert to dictionary
sample_dict = sample.to_dict()
print(df.columns.tolist())
# Save JSON
with open("sample_request.json", "w") as f:
    json.dump(sample_dict, f, indent=4)

print("sample_request.json created successfully.")