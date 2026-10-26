# Explainable Intrusion Detection — Random Forest Edition

This version preserves the original statistical dashboard and adds a Random Forest binary classifier, holdout evaluation, confusion matrix, classification report, feature importance, and a single-record prediction tool.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Use CIC-IDS2017

1. Download the CIC-IDS2017 machine-learning CSV files from the official Canadian Institute for Cybersecurity page:
   https://www.unb.ca/cic/datasets/ids-2017.html
2. Extract the CSV files locally.
3. In the app, choose **Upload CSV** and upload one CSV file containing a label column such as `Label`.
4. The app maps `BENIGN`/`Benign`/`Normal` to Normal and all other label values to Attack.
5. Use the maximum-row control to limit training time. CIC-IDS2017 files are large; start with 5,000–25,000 rows on a laptop.

## Important notes

- The real CIC-IDS2017 dataset is not bundled because it is large and should be downloaded from its source.
- The built-in demo data is synthetic and must not be used to claim real-world performance.
- Metrics use a stratified random holdout split. CIC-IDS2017 can contain duplicates and time-related dependence; this split may overestimate generalization. For a research-quality result, use a deduplicated, time-aware or day-based train/test split.
- Global Random Forest feature importance is not a causal explanation for an individual prediction.
- This educational app is not a production security monitoring system.
- If a CSV uses different label-column names, rename the label column to `Label` before uploading.
