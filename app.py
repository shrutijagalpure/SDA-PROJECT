import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Explainable Intrusion Detection", page_icon="🛡️", layout="wide")

FEATURES = ["Packet_Length", "Flow_Duration", "Packets_Per_Second", "Source_Bytes", "Destination_Bytes"]
LABEL_CANDIDATES = ["Label", "label", "class", "Class", "Label ", " Attack", "attack", "BinaryLabel", "binary_label"]

@st.cache_data
def make_demo_data(n_normal=180, n_attack=70, seed=42):
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame({
        "Packet_Length": np.clip(rng.normal(520, 110, n_normal), 80, None),
        "Flow_Duration": np.clip(rng.normal(2.8, 0.8, n_normal), 0.3, None),
        "Packets_Per_Second": np.clip(rng.normal(38, 10, n_normal), 5, None),
        "Source_Bytes": np.clip(rng.normal(4200, 1100, n_normal), 500, None),
        "Destination_Bytes": np.clip(rng.normal(5100, 1300, n_normal), 500, None),
        "Label": "BENIGN"
    })
    attack = pd.DataFrame({
        "Packet_Length": np.clip(rng.normal(900, 220, n_attack), 100, None),
        "Flow_Duration": np.clip(rng.normal(0.9, 0.35, n_attack), 0.1, None),
        "Packets_Per_Second": np.clip(rng.normal(115, 28, n_attack), 20, None),
        "Source_Bytes": np.clip(rng.normal(12000, 3200, n_attack), 1000, None),
        "Destination_Bytes": np.clip(rng.normal(15500, 4000, n_attack), 1000, None),
        "Label": "ATTACK"
    })
    return pd.concat([normal, attack], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def find_label_column(df):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate.strip().lower() in normalized:
            return normalized[candidate.strip().lower()]
    for c in df.columns:
        if str(c).strip().lower() in {"label", "class", "attack", "binary_label", "target"}:
            return c
    return None

def binary_label(value):
    text = str(value).strip().lower()
    benign_values = {"benign", "normal", "0", "false", "normal traffic"}
    return "Normal" if text in benign_values else "Attack"

def prepare_ml_data(df, label_col, max_rows, seed=42):
    work = clean_column_names(df)
    label_col = next((c for c in work.columns if c == label_col.strip()), label_col)
    work = work.loc[work[label_col].notna()].copy()
    if len(work) > max_rows:
        # Stratified sample where possible, to keep browser training responsive.
        try:
            work, _ = train_test_split(work, train_size=max_rows, random_state=seed, stratify=work[label_col])
        except Exception:
            work = work.sample(n=max_rows, random_state=seed)
    y = work[label_col].map(binary_label)
    X = work.drop(columns=[label_col]).copy()
    # Use numeric flow features only. CIC-IDS2017 includes some duplicate/blank columns.
    X.columns = [str(c).strip() for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Remove columns that are entirely missing; remaining missing values are imputed inside the training pipeline.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # Drop columns that can directly identify a record or are commonly non-feature metadata.
    remove = [c for c in X.columns if c.lower() in {"flow id", "source ip", "destination ip", "timestamp", "unnamed: 0"}]
    X = X.drop(columns=remove, errors="ignore")
    return X, y, work

st.title("🛡️ Explainable Intrusion Detection")
st.caption("Statistics + Random Forest machine learning for network traffic classification")

with st.sidebar:
    st.header("⚙️ Controls")
    source = st.radio("Data source", ["Built-in Demo Dataset", "Upload CSV"])
    z_threshold = st.slider("Z-score threshold", 1.5, 4.0, 2.0, 0.1)
    st.markdown("---")
    st.info("ML uses a labeled CSV when uploaded. The demo data is synthetic and only demonstrates the workflow.")

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload labeled network traffic CSV (CIC-IDS2017 supported)", type=["csv"])
    if uploaded is None:
        st.warning("Upload a labeled CSV, or select Built-in Demo Dataset.")
        st.stop()
    try:
        df = clean_column_names(pd.read_csv(uploaded, low_memory=False))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    df = make_demo_data()

label_col = find_label_column(df)
if label_col is None:
    st.error("A label column is required to train the ML model. Expected a column such as Label, Class, attack, or binary_label.")
    st.stop()

# ----- Statistical analysis (retains original five-feature dashboard where available) -----
alias_map = {
    "Packet_Length": ["Packet_Length", "Packet Length", "PacketLength", "Pkt Len Mean", "Average Packet Size"],
    "Flow_Duration": ["Flow_Duration", "Flow Duration", "FlowDuration", "Flow Duration (Microseconds)"],
    "Packets_Per_Second": ["Packets_Per_Second", "Packets/s", "Flow Packets/s", "Flow Packets Per Second"],
    "Source_Bytes": ["Source_Bytes", "Source Bytes", "Total Length of Fwd Packets"],
    "Destination_Bytes": ["Destination_Bytes", "Destination Bytes", "Total Length of Bwd Packets"]
}
rename = {}
for target, choices in alias_map.items():
    for c in choices:
        if c in df.columns:
            rename[c] = target
            break
df = df.rename(columns=rename)
available_stats = [c for c in FEATURES if c in df.columns]
for c in available_stats:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if available_stats:
    stat_X = df[available_stats].replace([np.inf, -np.inf], np.nan)
    means, medians = stat_X.mean(), stat_X.median()
    stds = stat_X.std(ddof=1).replace(0, np.nan).fillna(1)
    z_scores = ((stat_X - means) / stds).abs().fillna(0)
    df["Max_Z_Score"] = z_scores.max(axis=1)
    df["Detection"] = np.where(df["Max_Z_Score"] >= z_threshold, "Suspicious", "Normal")
    df["Main_Reason"] = z_scores.idxmax(axis=1)
    df["Reason_Z"] = z_scores.max(axis=1)
else:
    stat_X = pd.DataFrame(index=df.index)
    means = medians = stds = z_scores = pd.DataFrame()
    df["Max_Z_Score"] = 0.0
    df["Detection"] = "Not available"
    df["Main_Reason"] = "N/A"
    df["Reason_Z"] = 0.0

# ----- Random Forest model training -----
st.sidebar.subheader("🤖 ML settings")
max_rows = st.sidebar.select_slider("Maximum rows for ML training", options=[5000, 10000, 25000, 50000, 100000], value=25000)
test_size = st.sidebar.slider("Test-set size", 0.1, 0.4, 0.2, 0.05)
n_trees = st.sidebar.select_slider("Random Forest trees", options=[50, 100, 150, 200, 300], value=150)

X, y, model_source = prepare_ml_data(df, label_col, int(max_rows))
if X.shape[1] == 0 or y.nunique() < 2:
    st.error("The uploaded data needs at least one usable numeric feature and both Normal/Benign and Attack labels.")
    st.stop()

# Evaluation split: keep identical numeric feature rows in the same partition.
# This reduces exact-feature-row leakage, but does not prevent near-duplicate or
# time/capture leakage. A separate capture/file is stronger for final research claims.
with st.sidebar:
    split_method = st.selectbox(
        "Evaluation split",
        ["Group split (keep identical feature rows together)", "Random stratified split (baseline)"],
        index=0,
        help="Group split prevents identical numeric feature vectors from appearing in both train and test. It is not a substitute for testing on a separate capture/file."
    )

try:
    if split_method.startswith("Group split"):
        # Hash each numeric feature row; identical feature vectors share a group.
        # pandas hashing is deterministic for this in-memory feature table.
        row_groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=row_groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                "The grouped split produced only one class in train or test. "
                "Try a larger maximum-row setting or use the random stratified baseline, "
                "and validate with a separate capture/file for the paper."
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
except (ValueError, StopIteration) as exc:
    st.error(f"Unable to split this dataset for evaluation: {exc}")
    st.stop()


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_random_forest(X_train, y_train, n_estimators):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=1
        ))
    ])
    model.fit(X_train, y_train)
    return model

model = train_random_forest(X_train, y_train, n_trees)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Attack", zero_division=0)
recall = recall_score(y_test, y_pred, pos_label="Attack", zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label="Attack", zero_division=0)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Statistical Analysis", "🔎 Explainability",
    "🤖 ML Evaluation", "🧪 Test One Record"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Statistically Suspicious", int((df["Detection"] == "Suspicious").sum()))
    c3.metric("ML Test Accuracy", f"{accuracy:.1%}")
    c4.metric("Numeric ML Features", X.shape[1])
    st.subheader("Statistical Detection Summary")
    summary = df["Detection"].value_counts().rename_axis("Class").reset_index(name="Count")
    summary["Percentage"] = (summary["Count"] / max(len(df), 1) * 100).round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots()
    ax.bar(summary["Class"], summary["Count"])
    ax.set_title("Normal vs Statistically Suspicious Traffic")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)
    st.subheader("ML label distribution")
    st.dataframe(y.value_counts().rename_axis("Class").reset_index(name="Count"), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Descriptive Statistics")
    if available_stats:
        stats = pd.DataFrame({
            "Mean": stat_X.mean(), "Median": stat_X.median(), "Std. Deviation": stat_X.std(),
            "Minimum": stat_X.min(), "Maximum": stat_X.max(),
            "Q1": stat_X.quantile(0.25), "Q3": stat_X.quantile(0.75)
        }).round(2)
        st.dataframe(stats, use_container_width=True)
        feature = st.selectbox("Choose feature", available_stats)
        fig, ax = plt.subplots()
        ax.hist(stat_X[feature].dropna(), bins=25)
        ax.axvline(means[feature], linestyle="--", label="Mean")
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        st.subheader("Correlation Matrix")
        corr = stat_X.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(available_stats))); ax.set_yticks(range(len(available_stats)))
        ax.set_xticklabels(available_stats, rotation=45, ha="right"); ax.set_yticklabels(available_stats)
        for i in range(len(available_stats)):
            for j in range(len(available_stats)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Feature Correlation Heatmap")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        st.info("The uploaded CSV does not contain the five original summary features. ML still uses its numeric features.")

with tab3:
    st.subheader("Why was a record flagged?")
    st.write("Statistical explanation uses absolute Z-scores. ML explanation uses Random Forest feature importance.")
    if available_stats:
        suspicious = df[df["Detection"] == "Suspicious"]
        if suspicious.empty:
            st.success("No statistically suspicious records at the selected threshold.")
        else:
            idx = st.selectbox("Select a suspicious record", suspicious.index.tolist(), format_func=lambda i: f"Record {i}")
            row = df.loc[idx]
            explain = pd.DataFrame({
                "Feature": available_stats,
                "Observed Value": [row[f] for f in available_stats],
                "Mean": [means[f] for f in available_stats],
                "Absolute Z-score": [z_scores.loc[idx, f] for f in available_stats]
            }).sort_values("Absolute Z-score", ascending=False).round(3)
            explain["Contribution"] = np.where(explain["Absolute Z-score"] >= z_threshold, "Strong anomaly", "Within threshold")
            st.error(f"Statistical flag: {row['Main_Reason']} has |Z| = {row['Reason_Z']:.2f}.")
            st.dataframe(explain, use_container_width=True, hide_index=True)
            fig, ax = plt.subplots()
            ax.bar(explain["Feature"], explain["Absolute Z-score"])
            ax.axhline(z_threshold, linestyle="--", label="Threshold")
            ax.set_ylabel("|Z-score|"); ax.set_title("Feature-Level Statistical Explanation")
            ax.tick_params(axis="x", rotation=35); ax.legend()
            st.pyplot(fig)
    st.markdown("#### Global ML feature importance")
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.named_steps["classifier"].feature_importances_
    }).sort_values("Importance", ascending=False).head(20)
    st.dataframe(importances.round(5), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots()
    plot_imp = importances.sort_values("Importance")
    ax.barh(plot_imp["Feature"], plot_imp["Importance"])
    ax.set_title("Top 20 Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    st.caption("Feature importance describes the model's overall reliance on features; it is not a causal explanation for an individual prediction.")

with tab4:
    st.subheader("Random Forest Model Evaluation")
    if split_method.startswith("Group split"):
        st.caption(f"Evaluation uses a grouped holdout split ({len(X_test):,} test rows) from {len(X):,} selected rows. Identical numeric feature rows are kept in one partition.")
        st.warning("This check reduces exact-feature-row leakage only. Similar/near-duplicate flows and capture/time leakage may remain; final research validation should test on a separate capture/file.")
    else:
        st.caption(f"Evaluation uses a stratified random holdout split ({len(X_test):,} test rows) from {len(X):,} selected rows.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.3f}")
    m2.metric("Attack Precision", f"{precision:.3f}")
    m3.metric("Attack Recall", f"{recall:.3f}")
    m4.metric("Attack F1-score", f"{f1:.3f}")
    labels = ["Normal", "Attack"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Attack"], rotation=15)
    ax.set_yticklabels(["Actual Normal", "Actual Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.download_button("Download test predictions CSV",
        data=pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": y_pred}).to_csv(index=False).encode("utf-8"),
        file_name="intrusion_detection_test_predictions.csv", mime="text/csv")
    st.warning("Metrics describe only this uploaded file and selected split. Do not generalize these results to unseen network captures without a separate-file or time-based evaluation.")

with tab5:
    st.subheader("Test a Single Network Record")
    st.write("Enter numeric feature values used by the trained model. For CIC-IDS2017, select the features to enter below.")
    chosen_features = st.multiselect("Features for manual test", list(X.columns), default=list(X.columns[:min(5, len(X.columns))]))
    if not chosen_features:
        st.info("Select at least one feature.")
    else:
        values = {}
        med = X[chosen_features].median()
        cols = st.columns(min(4, len(chosen_features)))
        for i, f in enumerate(chosen_features):
            values[f] = cols[i % len(cols)].number_input(f, value=float(med[f]) if pd.notna(med[f]) else 0.0, key=f"manual_{f}")
        if st.button("🔍 Detect Intrusion", type="primary"):
            record = pd.DataFrame([{f: float(values.get(f, X[f].median())) for f in X.columns}])
            # Unselected features default to dataset medians.
            prediction = model.predict(record)[0]
            probabilities = model.predict_proba(record)[0]
            st.write(f"Prediction: **{prediction.upper()}**")
            st.write(f"Model confidence (maximum class probability): **{probabilities.max():.1%}**")
            prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probabilities}).sort_values("Probability", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            row_imp = pd.DataFrame({"Feature": X.columns, "Global importance": model.named_steps["classifier"].feature_importances_})
            row_imp = row_imp.sort_values("Global importance", ascending=False).head(10)
            st.caption("The table below shows the top global features used by the model, not per-record causal attribution.")
            st.dataframe(row_imp.round(5), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Statistics Project | Explainable Intrusion Detection | Educational demonstration, not a production cybersecurity system.")
