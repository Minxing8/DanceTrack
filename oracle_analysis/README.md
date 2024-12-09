## Oracle Analysis

This repository contains scripts that analyze different tracking algorithms, focusing on their core components such as IoU matching, motion models, and appearance features. Below is a detailed explanation of the provided scripts and their connection to the respective tracking algorithms.

---

## 1. `analysis_iou.py`: Only Using IoU Matching

### Code Analysis
- **Tracker Class**: `from iou import Tracker`
- **Initialization**: `tracker = Tracker(det_thresh=0.4)`
- **Update Method**: `tracker.update(det)`  
  The script passes only detection results (`det`) without incorporating appearance features or image information.

### Key Points
- This tracker relies solely on IoU (Intersection over Union) to match bounding boxes.
- **No Motion Model**: Does not use a Kalman filter or any other motion prediction mechanism.
- **No Appearance Features**: Ignores appearance-based matching.

### Conclusion
This script implements a basic IoU-based tracker that associates bounding boxes based only on their spatial overlap.

---

## 2. `analysis_sort.py`: IoU Matching + Motion Model

### Code Analysis
- **Tracker Class**: `from sort import Sort`
- **Initialization**: `tracker = Sort(det_thresh=0.4)`
- **Update Method**: `tracker.update(det)`  
  The script passes only detection results (`det`).

### Key Points
- **Kalman Filter**: Although not explicitly visible in the code, the `Sort` algorithm inherently uses a Kalman filter for motion prediction.
- **Association Step**: Combines predicted states from the Kalman filter with IoU for bounding box matching.

### Explanation of `Sort`
- A classic multi-object tracking algorithm.
- Uses a Kalman filter for motion prediction and IoU for object association.

### Conclusion
This script implements a tracker that combines IoU-based matching with a motion model (Kalman filter).

---

## 3. `analysis_deepsort.py`: IoU Matching + Motion Model + Appearance Matching

### Code Analysis
- **Tracker Class**: `from deepsort_tracker.deepsort import DeepSort`
- **Initialization**: `tracker = DeepSort(model_path='ckpt.t7', min_confidence=0.4, n_init=0)`
- **Update Method**: `tracker.update(det, image_path)`  
  The script passes both detection results (`det`) and the image path (`image_path`), enabling the use of appearance features.

### Key Points
- **Kalman Filter**: Used for motion prediction.
- **Appearance Features**: Extracted from the image using a ReID model.
- **Association Step**: Combines motion (Kalman filter) and appearance features for matching.

### Explanation of `DeepSORT`
- Extends the `Sort` algorithm by adding appearance-based feature matching.
- Improves tracking robustness in cases of occlusion or similar motion patterns.

### Conclusion
This script implements a tracker that uses IoU, a motion model (Kalman filter), and appearance features for object association.

---

## 4. `analysis_appearance.py`: Only Using Appearance Matching

### Code Analysis
- **Tracker Class**: `from deepsort_tracker.appearance_tracker import ATracker`
- **Initialization**: `tracker = ATracker(model_path='ckpt.t7', min_confidence=0.4, n_init=0)`
- **Update Method**: `tracker.update(det, image_path)`  
  The script passes both detection results (`det`) and the image path (`image_path`), focusing entirely on appearance features.

### Key Points
- **No Motion Model**: Does not use a Kalman filter or IoU matching.
- **Appearance-Based Matching**: Relies solely on appearance features for object association.

### Explanation of `ATracker`
- Primarily depends on appearance features extracted from images.
- Suitable for scenarios where motion models and IoU are not reliable.

### Conclusion
This script implements a tracker that uses only appearance features for object association.

---

## Summary of Tracking Algorithms

| Script                  | IoU Matching | Motion Model (Kalman Filter) | Appearance Matching |
|-------------------------|--------------|------------------------------|----------------------|
| `analysis_iou.py`       | ✅           | ❌                           | ❌                   |
| `analysis_sort.py`      | ✅           | ✅                           | ❌                   |
| `analysis_deepsort.py`  | ✅           | ✅                           | ✅                   |
| `analysis_appearance.py`| ❌           | ❌                           | ✅                   |

---

## Frequently Asked Questions

### Why isn’t the Kalman filter visible in the code?
The Kalman filter is encapsulated within the `Sort` and `DeepSort` classes. Users interact with these classes without directly managing the Kalman filter.

### How to confirm the algorithm's implementation?
Refer to the source code of `sort.py` and `deepsort.py` or consult the original research papers for detailed algorithmic explanations.

--- 

Note: 

1. All detection boxes are ground-truth boxes, therefore, we remove hot launch in association.

2. For analysis of appearance and deepsort, re-id model is borrowed from [ckpt.t7](https://drive.google.com/drive/u/2/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).
