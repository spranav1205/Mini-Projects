import os
import json
import numpy as np
import matplotlib.pyplot as plt  # Replaced ImageDraw with matplotlib

from PIL import Image
from lbp import LBP

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_PATH = "cascade_model.json"
INPUT_IMAGE_PATH = "crowd.jpg"
STEP_SIZE = 4
NMS_IOU_THRESHOLD = 0.3


# =========================================================
# IOU
# =========================================================
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


# =========================================================
# NON-MAX SUPPRESSION
# =========================================================
def non_max_suppression(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return []

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        remaining = []

        for idx in idxs[1:]:
            iou = compute_iou(boxes[current], boxes[idx])
            if iou < iou_thresh:
                remaining.append(idx)

        idxs = np.array(remaining)

    return [boxes[i] for i in keep]


# =========================================================
# CASCADE INFERENCE
# =========================================================
def predict_lbp_real_cascade(
    bw_image,
    extractor,
    model_path="cascade_model.json",
    step_size=4,
    iou_thresh=0.3
):
    """
    Sliding-window attentional cascade inference with NMS.
    """
    with open(model_path, "r") as f:
        cascade_data = json.load(f)

    stages = cascade_data["stages"]
    patch_size = cascade_data["patch_size"]
    img_h, img_w = bw_image.shape

    detections = []
    detection_scores = []

    print(f"Scanning image ({img_w}x{img_h})...")

    # Sliding Window
    for y in range(0, img_h - patch_size + 1, step_size):
        for x in range(0, img_w - patch_size + 1, step_size):

            feats = extractor.extract_feature(bw_image, y, x)
            passed = True
            total_score = 0.0

            # Cascade Stages
            for stage in stages:
                stumps = stage["stumps"]
                stage_threshold = stage["stage_threshold"]
                agg_score = 0.0

                for stump in stumps:
                    f_idx = stump["feature_idx"]

                    if f_idx == -1:
                        pred_class = stump["left_pred"]
                    else:
                        if feats[f_idx] <= stump["threshold"]:
                            pred_class = stump["left_pred"]
                        else:
                            pred_class = stump["right_pred"]

                    direction = 1.0 if pred_class == 1 else -1.0
                    agg_score += stump["weight"] * direction

                # Early rejection
                if agg_score < stage_threshold:
                    passed = False
                    break

                total_score += agg_score

            # Save Detection
            if passed:
                detections.append((x, y, patch_size, patch_size))
                detection_scores.append(total_score)

    # Apply NMS filtering
    final_boxes = non_max_suppression(
        detections,
        detection_scores,
        iou_thresh=iou_thresh
    )

    return final_boxes


# =========================================================
# MAIN EXECUTIVE PIPELINE
# =========================================================
if __name__ == "__main__":

    # File validation checks
    if not os.path.exists(MODEL_PATH):
        print(f"Missing model: {MODEL_PATH}")
        exit(1)

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Missing image: {INPUT_IMAGE_PATH}")
        exit(1)

    # Load model configuration metadata
    with open(MODEL_PATH, "r") as f:
        meta = json.load(f)

    PATCH_SIZE = meta["patch_size"]

    # Rebuild extractor matching training metrics
    extractor = LBP(
        window_size=PATCH_SIZE,
        pieces=2,
        num_bins=59
    )

    print(f"Loading image: {INPUT_IMAGE_PATH}")
    color_im = Image.open(INPUT_IMAGE_PATH)
    bw_im = color_im.convert("L")
    bw_image_matrix = np.array(bw_im, dtype=np.uint8)

    # Run attentional detector
    found_faces = predict_lbp_real_cascade(
        bw_image=bw_image_matrix,
        extractor=extractor,
        model_path=MODEL_PATH,
        step_size=STEP_SIZE,
        iou_thresh=NMS_IOU_THRESHOLD
    )

    print(f"\nFinal detections after NMS: {len(found_faces)}")

    # -----------------------------------------------------
    # Matplotlib Plotting Engine (Replaced file saving)
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the background source image safely
    ax.imshow(color_im)
    
    # Draw boxes as patch outlines over the Matplotlib axis canvas
    for (x, y, w, h) in found_faces:
        rect = plt.Rectangle(
            (x, y), w, h, 
            fill=False, 
            edgecolor='#00FF00',  # Clean lime green hex
            linewidth=2
        )
        ax.add_patch(rect)
        
    # Formatting adjustments to clean up image view boundaries
    ax.set_title(f"Cascade Detections (Count: {len(found_faces)})", fontsize=14, pad=10)
    ax.axis('off')  # Strip out grid axis coordinate lines for clear presentation
    
    plt.tight_layout()
    plt.show()  # Launches interactive render view window