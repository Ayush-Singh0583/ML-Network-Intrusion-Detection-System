# Dataset

This project uses the **CIC-IDS2017** dataset.

Due to GitHub file size limits, the dataset is not included.

> **⚠️ CORRECTION (2026-10-04)** — the next paragraph says to download the
> `MachineLearningCSV` set. Download **`GeneratedLabelledFlows.zip`** instead
> (same CIC page, the folder inside is called `TrafficLabelling`, and the eight
> files are expected under the same names as below).
>
> Why: the `MachineLearningCSV` files have no source address and no timestamp.
> The per-flow models train on either set, but the study in `src/study.py` needs
> both columns for the behaviour layer (ports per source per minute), for
> ordering flows in time, and for its confidence intervals. With the
> `MachineLearningCSV` files experiment E4 refuses to run and E5 runs with two
> layers instead of three.
>
> After copying the files in, check them before anything else:
>
> ```bash
> python src/study.py doctor
> ```
>
> Unpack the zip so that the eight CSVs sit directly in this folder, not in a
> sub-folder.
>
> `doctor` prints, per file: the row count, the column count, whether source
> address and timestamp are there, the hours the timestamps parse onto, and the
> share of them that parse. Under `[problem]` it lists any file that is
> missing, whose timestamps do not land on its own date, or whose timestamps
> parse outside 08:00-18:00 (the capture ran 09:00-17:00, and the files print a
> 12-hour clock with no AM/PM), and it says so if the folder holds a mixture of
> the two downloads. If it reports a file as missing, the name does not match
> the list below: rename the file, do not edit the list.
>
> The cache notices when a CSV has been replaced (by file size) and rebuilds
> itself on the next command. `python src/run.py cache --rebuild` forces it.
>
> The original text is retained below.

> **⚠️ ADDED (2026-10-05) — what the first `doctor` run found in this folder,
> and three rules that follow.**
>
> On 2026-10-04 this folder did not hold one download. Seven files were
> `MachineLearningCSV` files (79 columns). The eighth, the Friday DDoS capture,
> was a `GeneratedLabelledFlows` file (85 columns) that had been cleaned by
> hand: two rows fewer than the repo's own notebook first counted (225,743
> against 225,745), and timestamps written as `07-07-2017 03:30` where the
> notebook shows `7/7/2017 3:30`. `doctor` could read none of its timestamps.
> The full record is in `wiki/Dataset-CICIDS2017.md`. All eight files were
> replaced from `GeneratedLabelledFlows.zip` on 2026-10-05, and `doctor` then
> listed no problem (`runs/96e8769-20261005-011910-study-doctor`).
>
> 1. **All eight files come from `GeneratedLabelledFlows.zip`.** They have the
>    same names as the ones already here, so they replace them. Take all
>    eight, not only the ones that look wrong.
> 2. **Never clean a capture by hand, and never open one in a spreadsheet
>    program and save it.** Saving rewrites the date column (and can shorten
>    decimals). Copy the files straight out of the zip. The cleaning is done
>    in code (`src/preprocessing.py`), where it is written down and counted:
>    a negative or unreadable `Flow Duration` and a blank label drop the row;
>    an empty or infinite value is filled with the training split's median
>    and the row stays; a zero duration is kept (a one-packet flow is still
>    traffic).
> 3. **`doctor` must list no `[problem]` before anything else is run.** A
>    timestamp column it cannot read is printed as `time=UNREADABLE`, with an
>    example of the value. `python src/study.py all` runs `doctor` first and
>    stops if it lists a problem.
>
> Other files in this folder (`custom_sample.csv`, say) are ignored: the eight
> names below are the whole manifest.

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
