import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ==========================================
# 1. Experimental Data Design (10 Trials, 5 Patients)
# ==========================================

trials_data = [
    # --- Diabetes Group (Trap: Age) ---
    {"id": "T1", "text": "Type 2 Diabetes Study for Adults", "cond": "Diabetes", "minAge": 18, "maxAge": 65, "sex": "ALL"},
    {"id": "T2", "text": "Pediatric Diabetes Insulin Study", "cond": "Diabetes", "minAge": 0,  "maxAge": 17, "sex": "ALL"},
    
    # --- Hypertension Group (Trap: Sex & Age) ---
    {"id": "T3", "text": "Hypertension Study for Senior Males", "cond": "Hypertension", "minAge": 65, "maxAge": 100, "sex": "MALE"},
    {"id": "T4", "text": "General Hypertension Medication Study", "cond": "Hypertension", "minAge": 18, "maxAge": 80, "sex": "ALL"},
    
    # --- Breast Cancer Group (Trap: Sex) ---
    {"id": "T5", "text": "Breast Cancer Treatment for Women", "cond": "Breast Cancer", "minAge": 18, "maxAge": 99, "sex": "FEMALE"},
    {"id": "T6", "text": "Rare Male Breast Cancer Study", "cond": "Breast Cancer", "minAge": 18, "maxAge": 99, "sex": "MALE"},
    
    # --- Asthma Group (Trap: None, General) ---
    {"id": "T7", "text": "Asthma Relief for All Ages", "cond": "Asthma", "minAge": 0, "maxAge": 99, "sex": "ALL"},
    
    # --- Distractors (Irrelevant) ---
    {"id": "T8", "text": "Healthy Volunteer Flu Vaccine", "cond": "Flu", "minAge": 18, "maxAge": 60, "sex": "ALL"},
    {"id": "T9", "text": "Kidney Failure Dialysis Study", "cond": "Kidney Disease", "minAge": 40, "maxAge": 80, "sex": "ALL"},
    {"id": "T10", "text": "Migraine Headache Study", "cond": "Migraine", "minAge": 18, "maxAge": 50, "sex": "ALL"},
]

df_trials = pd.DataFrame(trials_data)
# Concatenate text for TF-IDF
df_trials["fullText"] = df_trials["text"] + " " + df_trials["cond"]

# 5 Synthetic Patients
patients_data = [
    {"id": "P1", "desc": "Child (10yo) with Diabetes", "age": 10, "sex": "MALE", "cond": "Diabetes"},
    {"id": "P2", "desc": "Adult (40yo) Male with Hypertension", "age": 40, "sex": "MALE", "cond": "Hypertension"},
    {"id": "P3", "desc": "Male with Breast Cancer", "age": 55, "sex": "MALE", "cond": "Breast Cancer"},
    {"id": "P4", "desc": "Senior (70yo) Female with Hypertension", "age": 70, "sex": "FEMALE", "cond": "Hypertension"},
    {"id": "P5", "desc": "Adult (30yo) with Asthma", "age": 30, "sex": "FEMALE", "cond": "Asthma"},
]

# ==========================================
# 2. Ground Truth Identification
# ==========================================

def get_ground_truth(patient, trial):
    # 1. Condition must match
    if patient["cond"] != trial["cond"]: return 0
    # 2. Age must be within range
    if patient["age"] < trial["minAge"] or patient["age"] > trial["maxAge"]: return 0
    # 3. Sex must match
    if trial["sex"] != "ALL" and trial["sex"] != patient["sex"]: return 0
    
    return 1 # Match

# Build Ground Truth Matrix
ground_truth_matrix = np.zeros((len(patients_data), len(trials_data)))

print("=== 1. Pre-defined Ground Truth ===")
print("1 = Should Recommend, 0 = Should Not Recommend\n")
header = [t["id"] for t in trials_data]
print(f"{'Patient':<35} | " + "  ".join(header))
print("-" * 80)

truth_flat = [] 
for i, p in enumerate(patients_data):
    row_vals = []
    for j, t in enumerate(trials_data):
        is_match = get_ground_truth(p, t)
        ground_truth_matrix[i, j] = is_match
        row_vals.append(str(int(is_match)))
        truth_flat.append(is_match)
    print(f"{p['desc']:<35} | " + "   ".join(row_vals))

# ==========================================
# 3. Model Definitions
# ==========================================

class SimpleTFIDF:
    def __init__(self, df):
        self.df = df
        self.vec = TfidfVectorizer(stop_words='english')
        self.mx = self.vec.fit_transform(df["fullText"])
        
    def predict(self, query_text, top_k=3):
        q = self.vec.transform([query_text])
        sim = cosine_similarity(q, self.mx).flatten()
        top_idx = np.argsort(sim)[::-1][:top_k]
        return top_idx

class SimpleHybrid:
    def __init__(self, df, tfidf_model):
        self.df = df
        self.tfidf = tfidf_model
        
    def predict(self, patient, top_k=3):
        # 1. Broad Recall
        candidates_idx = self.tfidf.predict(patient["cond"], top_k=5)
        
        scores = []
        for idx in candidates_idx:
            row = self.df.iloc[idx]
            score = 1.0 
            
            # Rule-based Filtering
            if patient["age"] < row["minAge"] or patient["age"] > row["maxAge"]:
                score = -1.0 # Discard
            if row["sex"] != "ALL" and row["sex"] != patient["sex"]:
                score = -1.0 # Discard
                
            scores.append((idx, score))
            
        # 2. Re-ranking
        scores.sort(key=lambda x: x[1], reverse=True)
        final_idx = [x[0] for x in scores if x[1] > 0][:top_k]
        return final_idx

# ==========================================
# 4. Run Experiment
# ==========================================

tfidf_model = SimpleTFIDF(df_trials)
hybrid_model = SimpleHybrid(df_trials, tfidf_model)
K = 3

pred_flat_tfidf = []
pred_flat_hybrid = []

for i, p in enumerate(patients_data):
    recs_tfidf = tfidf_model.predict(p["cond"] + " " + p["desc"], top_k=K)
    recs_hybrid = hybrid_model.predict(p, top_k=K)
    
    for j in range(len(trials_data)):
        # Check TFIDF
        if j in recs_tfidf: pred_flat_tfidf.append(1)
        else: pred_flat_tfidf.append(0)
            
        # Check Hybrid
        if j in recs_hybrid: pred_flat_hybrid.append(1)
        else: pred_flat_hybrid.append(0)

# ==========================================
# 5. Visualization & Results
# ==========================================

def plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Not Rec', 'Rec'], 
                yticklabels=['Not Rel', 'Rel'])
    ax.set_title(title)
    ax.set_ylabel('Ground Truth (Actual)')
    ax.set_xlabel('Model Prediction')
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    text = f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}"
    ax.text(0.5, -0.25, text, ha='center', transform=ax.transAxes, fontweight='bold')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_confusion_matrix(truth_flat, pred_flat_tfidf, "Baseline: TF-IDF (Text Only)", axes[0])
plot_confusion_matrix(truth_flat, pred_flat_hybrid, "Proposed: Hybrid Model (Text+Rules)", axes[1])

plt.tight_layout()
plt.show()

# Print Case Analysis
print("\n=== Case Analysis Details (5 Patients) ===")
for i, p in enumerate(patients_data):
    gt_ids = [t['id'] for t in trials_data if get_ground_truth(p, t) == 1]
    
    # Get prediction IDs
    recs_tfidf_idx = tfidf_model.predict(p["cond"] + " " + p["desc"], top_k=3)
    recs_tfidf_ids = [trials_data[idx]['id'] for idx in recs_tfidf_idx]
    
    recs_hybrid_idx = hybrid_model.predict(p, top_k=3)
    recs_hybrid_ids = [trials_data[idx]['id'] for idx in recs_hybrid_idx]
    
    print(f"\n[Case {p['id']}] {p['desc']}")
    print(f"  ✅ Ground Truth:   {gt_ids}")
    print(f"  📊 TF-IDF Recs:    {recs_tfidf_ids}")
    print(f"  🚀 Hybrid Recs:    {recs_hybrid_ids}")