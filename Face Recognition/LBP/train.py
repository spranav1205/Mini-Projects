import json
import os
import urllib.request
import zipfile
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage import data
from PIL import Image

from lbp import LBP

# =========================================================
# CONFIGURATION
# =========================================================
NUM_STAGES = 5
STAGE_ESTIMATORS = [5, 10, 15, 20, 25]
PATCH_SIZE = 24

# Target percentage of true faces that must survive each stage
TARGET_TPR = 0.995 

DATASET_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"

# Initialize with 59 bins matching Uniform LBP mapping
extractor = LBP(window_size=PATCH_SIZE, pieces=2, num_bins=59)

data_dir = "att_faces"
zip_path = "att_faces.zip"

# =====================================================
# Download and Extract Dataset
# =====================================================
if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

if not os.path.isdir(data_dir):
    print("Extracting dataset...")
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

# =====================================================
# Positive Samples Base Extraction
# =====================================================
print("Extracting initial positive samples...")
X_pos_list = []

for root, _, files in os.walk(data_dir):
    for fname in files:
        if not fname.lower().endswith(".pgm"):
            continue

        path = os.path.join(root, fname)
        im = Image.open(path).convert("L").resize((PATCH_SIZE, PATCH_SIZE))
        img = np.array(im, dtype=np.uint8)
        feats = extractor.extract_feature(img, 0, 0)
        X_pos_list.append(feats)

X_pos = np.array(X_pos_list, dtype=np.float32)
print("Positive samples shape:", X_pos.shape)

# =====================================================
# Negative Samples Diverse Background Pool
# =====================================================
print("Extracting initial negative samples...")
background_pool = [
    data.camera(), data.coins(), data.moon(), data.page(),
    data.clock(), data.chelsea(), data.coffee()
]

gray_scenes = []
for scene in background_pool:
    if scene.ndim == 3:
        gray = np.dot(scene[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = scene.astype(np.uint8)
    gray_scenes.append(gray)

X_neg_list = []
np.random.seed(42)
sampled_coordinates = set()

while len(X_neg_list) < len(X_pos):
    scene_idx = np.random.randint(0, len(gray_scenes))
    scene = gray_scenes[scene_idx]
    h, w = scene.shape
    
    y = np.random.randint(0, h - PATCH_SIZE)
    x = np.random.randint(0, w - PATCH_SIZE)
    
    coord_id = (scene_idx, y, x)
    if coord_id in sampled_coordinates:
        continue
    sampled_coordinates.add(coord_id)
    
    crop = scene[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
    feats = extractor.extract_feature(crop, 0, 0)
    X_neg_list.append(feats)

surviving_negatives = np.array(X_neg_list, dtype=np.float32)
print("Negative samples shape:", surviving_negatives.shape)

# =========================================================
# MULTI-STAGE CASCADE TRAINING WITH MINING
# =========================================================
cascade_stages = []

for stage_idx in range(NUM_STAGES):
    print(f"\n==============================")
    print(f"Training Cascade Stage {stage_idx + 1} / {NUM_STAGES}")
    print(f"==============================")
    print("Positives available :", len(X_pos))
    print("Negatives available :", len(surviving_negatives))

    X_stage = np.vstack([X_pos, surviving_negatives])
    y_stage = np.hstack([np.ones(len(X_pos)), np.zeros(len(surviving_negatives))])

    # No 'algorithm' parameter needed here anymore; defaults to pure discrete SAMME
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=STAGE_ESTIMATORS[stage_idx],
        random_state=42
    )
    clf.fit(X_stage, y_stage)

    # Export stage stumps cleanly using public API predictions
    stage_stumps = []
    for i, estimator in enumerate(clf.estimators_):
        tree = estimator.tree_

        if tree.node_count == 1:
            # If the tree didn't split, predict using a mock zero array
            dummy_feat = np.zeros((1, X_stage.shape[1]))
            pred_class = int(estimator.predict(dummy_feat)[0])
            stage_stumps.append({
                "weight": float(clf.estimator_weights_[i]),
                "feature_idx": -1,
                "threshold": -1.0,
                "left_pred": pred_class,
                "right_pred": pred_class
            })
            continue

        # Get threshold details
        f_idx = int(tree.feature[0])
        thresh = float(tree.threshold[0])

        # Safely determine left vs right class labels using direct tree predictions
        dummy_left = np.zeros((1, X_stage.shape[1]))
        dummy_right = np.zeros((1, X_stage.shape[1]))
        dummy_left[0, f_idx] = thresh - 0.01  # Force left branch path
        dummy_right[0, f_idx] = thresh + 0.01 # Force right branch path

        left_pred = int(estimator.predict(dummy_left)[0])
        right_pred = int(estimator.predict(dummy_right)[0])

        stage_stumps.append({
            "weight": float(clf.estimator_weights_[i]),
            "feature_idx": f_idx,
            "threshold": thresh,
            "left_pred": left_pred,
            "right_pred": right_pred
        })

    # Evaluator block for scoring thresholds
    def _evaluate_discrete_score(features):
        score = 0.0
        for stump in stage_stumps:
            f_idx = stump["feature_idx"]
            if f_idx == -1:
                pred_class = stump["left_pred"]
            else:
                pred_class = stump["left_pred"] if features[f_idx] <= stump["threshold"] else stump["right_pred"]
            
            direction = 1.0 if pred_class == 1 else -1.0
            score += stump["weight"] * direction
        return score

    # Dynamic Thresholding
    pos_scores = np.array([_evaluate_discrete_score(p) for p in X_pos])
    percentile_idx = (1.0 - TARGET_TPR) * 100
    stage_threshold = float(np.percentile(pos_scores, percentile_idx))
    
    print(f"Calculated Stage Passing Threshold: {stage_threshold:.4f}")

    cascade_stages.append({
        "stage_id": stage_idx,
        "stage_threshold": stage_threshold,
        "stumps": stage_stumps
    })

    # Hard Negative Mining Loop
    next_negatives = []
    for neg_feat in surviving_negatives:
        stage_score = _evaluate_discrete_score(neg_feat)
        if stage_score >= stage_threshold:
            next_negatives.append(neg_feat)

    surviving_negatives = np.array(next_negatives, dtype=np.float32)
    print("False Positive (Hard Negatives) remaining:", len(surviving_negatives))

    if len(surviving_negatives) == 0:
        print("No background patterns leaked through. Ending cascade early.")
        break

# =====================================================
# Structured Model Export
# =====================================================
cascade_model = {
    "patch_size": PATCH_SIZE,
    "stages": cascade_stages
}

with open("cascade_model.json", "w") as f:
    json.dump(cascade_model, f, indent=4)

print("\nSuccessfully compiled true attentional cascade into cascade_model.json")