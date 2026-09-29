# Dataset

This project uses the **CIC-IDS2017** dataset.

Due to GitHub file size limits, the dataset is not included.

Download the CSVs (the `MachineLearningCSV` set) from the Canadian Institute
for Cybersecurity (CIC) and place all eight files in this directory:

```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

These names (including the original spellings `workingHours` and
`Infilteration`) are the manifest in `src/config.py`; `python src/run.py cache`
stops with a clear error if any file is missing.
