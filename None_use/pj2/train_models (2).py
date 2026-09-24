import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================
# 0. ตั้งค่าพื้นฐาน
# ============================================================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
IMAGE_DIR = "image"
RANDOM_STATE = 42
TARGET = "category_grouped"

os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# 1. โหลดและเตรียมข้อมูล
# ============================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def clean_authors(value):
    """แปลง "['A', 'B']" ให้เป็น "A B" """
    return re.sub(r"[\[\]']", " ", str(value)).replace('"', " ").strip()


def prepare(df):
    df = df.copy()
    df["description"] = df["description"].fillna("")
    df["authors"] = df["authors"].fillna("").apply(clean_authors)
    df["publisher"] = df["publisher"].fillna("").astype(str)
    df["Title"] = df["Title"].fillna("").astype(str)
    df["text"] = df["Title"] + " " + df["description"]
    df["ratingsCount"] = np.log1p(df["ratingsCount"].fillna(0))
    return df


train_df = prepare(train_df)
test_df = prepare(test_df)

FEATURES = ["text", "authors", "publisher",
            "ratingsCount", "year", "has_rating", "has_description"]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

class_names = y_train.value_counts().index.tolist()  # เรียงตามจำนวนมากไปน้อย

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"จำนวนคลาส: {len(class_names)}")
print(y_train.value_counts(), "\n")


# ============================================================
# 2. Feature Engineering (ทำใน Pipeline เพื่อไม่ให้ข้อมูลรั่ว)
#    - ข้อความ: TF-IDF
#    - ตัวเลข: MinMaxScaler (ค่า 0-1 ไม่ติดลบ ใช้กับ Naive Bayes ได้)
# ============================================================
def build_preprocessor():
    return ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=20000, sublinear_tf=True), "text"),
        ("authors", TfidfVectorizer(min_df=1, max_features=3000), "authors"),
        ("publisher", TfidfVectorizer(min_df=1, max_features=2000), "publisher"),
        ("num", MinMaxScaler(), ["ratingsCount", "year",
                                 "has_rating", "has_description"]),
    ])


# ============================================================
# 3. กำหนด 3 โมเดล + Grid Search (ตามแนวทางบทที่ 5, 6, 7)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Naive Bayes": (
        MultinomialNB(),
        {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},  # Laplace smoothing
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                               max_features="sqrt", class_weight="balanced"),
        {"clf__n_estimators": [100, 200],
         "clf__max_depth": [None, 30],
         "clf__min_samples_split": [2, 5]},
    ),
    "SVM": (
        SVC(random_state=RANDOM_STATE, class_weight="balanced"),
        {"clf__kernel": ["linear", "rbf"],
         "clf__C": [0.1, 1, 10],
         "clf__gamma": ["scale"]},
    ),
}


# ============================================================
# 4. ฟังก์ชันวาด Heatmap (Confusion Matrix)
# ============================================================
def plot_heatmap(cm, labels, title, save_path, fmt="d", cmap="Blues"):
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="white",
                annot_kws={"size": 8}, cbar=True)
    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


# ============================================================
# 5. เทรน ประเมินผล และบันทึกภาพ
# ============================================================
results = []

for name, (estimator, param_grid) in models.items():
    print("=" * 70)
    print(f"กำลังเทรน: {name}")
    print("=" * 70)

    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro",
                        n_jobs=-1, verbose=1)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"Best Params      : {grid.best_params_}")
    print(f"Best CV F1-macro : {grid.best_score_:.2%}")
    print(f"Test Accuracy    : {acc:.2%}")
    print(f"Test F1 (macro)  : {f1_macro:.2%}")
    print(f"Test F1 (weighted): {f1_weighted:.2%}")
    print(f"เวลาที่ใช้        : {elapsed:.1f} วินาที\n")
    print(classification_report(y_test, y_pred, labels=class_names,
                                zero_division=0))

    safe_name = name.lower().replace(" ", "_")

    # Heatmap 1: จำนวนจริง
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plot_heatmap(cm, class_names,
                 f"Confusion Matrix - {name} (Count)\nAccuracy = {acc:.2%}",
                 os.path.join(IMAGE_DIR, f"heatmap_{safe_name}_count.png"),
                 fmt="d", cmap="Blues")

    # Heatmap 2: เปอร์เซ็นต์ต่อแถว (Recall ของแต่ละคลาส)
    cm_norm = confusion_matrix(y_test, y_pred, labels=class_names,
                               normalize="true")
    plot_heatmap(cm_norm, class_names,
                 f"Confusion Matrix - {name} (Normalized by Actual Class)\n"
                 f"F1-macro = {f1_macro:.2%}",
                 os.path.join(IMAGE_DIR, f"heatmap_{safe_name}_normalized.png"),
                 fmt=".2f", cmap="Greens")

    results.append({"Model": name, "Accuracy": acc,
                    "F1 (macro)": f1_macro, "F1 (weighted)": f1_weighted})

# ============================================================
# 6. สรุปเปรียบเทียบ 3 โมเดล
# ============================================================
summary = pd.DataFrame(results).set_index("Model")
print("=" * 70)
print("สรุปผลเปรียบเทียบ")
print("=" * 70)
print((summary * 100).round(2).astype(str) + "%")

ax = summary.plot(kind="bar", figsize=(9, 6), rot=0, colormap="viridis")
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v:.1%}" for v in container.datavalues],
                 fontsize=9, padding=2)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "model_comparison.png"),
            dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print(f"\nบันทึกภาพทั้งหมดในโฟลเดอร์ '{IMAGE_DIR}' เรียบร้อยแล้ว")
