from pathlib import Path
from collections import defaultdict
import csv
import math
import time

import cv2
import yaml
import numpy as np
from ultralytics import YOLO

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# YOLO POSE AUSWERTUNG V2.1
# ============================================================
# Ziel:
#   - Detection-Qualitaet bewerten
#   - rohe Keypoint-Genauigkeit P0...P7 bewerten
#   - daraus abgeleitete Mittelpunkt- und Winkelgenauigkeit bewerten
#   - Heatspreader: gerichteter 360-Grad-Winkel
#   - Slot: symmetrischer 180-Grad-Winkel
#   - CSVs + Legende + Debugbilder erzeugen
#
# WICHTIG:
#   Dieses Skript bewertet absichtlich die ROHE YOLO-Pose.
#   CAD-/Template-Fit und anderes Postprocessing kommen spaeter in
#   eine zweite Stufe, damit YOLO raw gegen YOLO + Postprocessing
#   sauber verglichen werden kann.
#
# Erwartetes Label-Format bei kpt_shape: [8, 3]:
#   class xc yc w h  x0 y0 v0 ... x7 y7 v7
#   alle x/y/w/h normalisiert auf [0, 1]
#
# Visibility Ground Truth:
#   0 = nicht gelabelt / nicht auswertbar
#   1 = gelabelt, aber verdeckt
#   2 = sichtbar
# ============================================================


# ============================================================
# HIER ANPASSEN
# ============================================================

MODEL_PATH = r"C:\FHWS\nanoplus\YOLO VS Code\runs\pose\train\weights\best.pt"
DATA_YAML = r"C:\FHWS\nanoplus\YOLO VS Code\dataset_pose\data.yaml"

SPLIT = "val"                  # "val" oder "test"
IMGSZ = 960                    # gleiche Bildgroesse wie beim Training/Vergleich
CONF = 0.25                    # Mindest-Confidence fuer Objekt-Predictions
BATCH = 2
DEVICE = "cpu"                 # GPU: 0, CPU: "cpu"

NUM_KEYPOINTS = 8

# Klassenbezeichnungen muessen zu data.yaml passen.
HEATSPREADER_CLASS_NAME = "heatspreader"
SLOT_CLASS_NAME = "slot"

# Matching von GT-Objekt und Prediction erfolgt NUR ueber Klasse + Bounding-Box-IoU.
# Dadurch werden Keypointfehler nicht zirkulaer fuer das Matching benutzt.
MATCH_IOU_FOR_GEOMETRY = 0.50

# Detection AP / mAP
AP_IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

# PCK = Percentage of Correct Keypoints bei diesen Pixelgrenzen
PCK_THRESHOLDS_PX = (2.0, 5.0, 10.0)

# GT-Keypoints mit visibility >= diesem Wert gehen in die geometrische Auswertung ein.
# 1 bedeutet: verdeckte, aber gelabelte Punkte werden mit ausgewertet.
MIN_GT_VISIBILITY_FOR_EVAL = 1

# Pose-Auswertung:
# True (empfohlen): Mittelpunkt/Winkel werden nur berechnet, wenn ALLE fuer die
# jeweilige Groesse benoetigten GT-Keypoints auswertbar sind. Dadurch entspricht
# die Messgroesse immer exakt derselben geometrischen Definition.
# False: Teilmengen sind erlaubt; GT und Prediction verwenden dann trotzdem immer
# exakt dieselbe GT-Visibility-Maske.
REQUIRE_COMPLETE_POSE_KEYPOINTS = True

# Debugbilder
SAVE_DEBUG_IMAGES = True
DEBUG_MAX_IMAGES = 30           # None = alle Bilder mit mindestens einem Match
DEBUG_SORT_METRIC = "center"    # "center", "kp_mean", "kp_max", "angle"
DEBUG_MIN_SCORE = None           # z.B. 3.0; None = keine Untergrenze

# Ausgabe
# Alle Ergebnisse werden gesammelt unter:
#   evaluation_keypoint/<NAME>/
# Dadurch kann der Name eines Auswertungslaufs oben schnell angepasst werden.
NAME = "pose_evaluation_v2"
OUTPUT_DIR = Path("evaluation_keypoint") / NAME

OBJECT_CSV = "pose_eval_objects.csv"
KEYPOINT_CSV = "pose_eval_keypoints.csv"
SUMMARY_CSV = "pose_eval_summary.csv"
KEYPOINT_SUMMARY_CSV = "pose_eval_keypoint_summary.csv"
LEGEND_TXT = "pose_eval_legend.txt"
DEBUG_DIR_NAME = "debug_pose"


# ============================================================
# GEOMETRISCHE DEFINITIONEN DER ROHEN POSE
# ============================================================
# Heatspreader:
#   Mittelpunkt: Mittelwert der vier Aussenpunkte P0...P3.
#   360-Grad-Richtung: Mittelpunkt(P4,P5) -> Mittelpunkt(P6,P7).
#   Diese gerichtete Achse nutzt die semantische Zuordnung der inneren
#   asymmetrischen Merkmale und kann dadurch 0...360 Grad unterscheiden.
#
# Slot:
#   Mittelpunkt: Mittelwert aller auswertbaren P0...P7.
#   Winkel: Hauptachse (PCA) aller auswertbaren Punkte, modulo 180 Grad.
#   Da der Slot symmetrisch ist, sind theta und theta+180 Grad gleichwertig.
#
# Winkelkonvention im BILD-Koordinatensystem:
#   0 Grad   = nach rechts
#   90 Grad  = nach unten
#   180 Grad = nach links
#   270 Grad = nach oben
# (Bild-y zeigt nach unten. Fuer Roboterkoordinaten wird spaeter transformiert.)


# ============================================================
# BASISFUNKTIONEN
# ============================================================

def norm_path(p):
    return str(Path(p).resolve()).lower()


def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, names.get(str(cls_id), str(cls_id)))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def class_name_to_id(names, class_name):
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v) == class_name:
                return int(k)
    elif isinstance(names, list):
        for i, v in enumerate(names):
            if str(v) == class_name:
                return i
    return None


def resolve_image_paths(data_yaml_path, split):
    data = load_data_yaml(data_yaml_path)

    root = Path(data.get("path", Path(data_yaml_path).parent))
    if not root.is_absolute():
        root = Path(data_yaml_path).parent / root

    if split not in data or data[split] in (None, ""):
        raise KeyError(f"Split '{split}' ist in der data.yaml nicht definiert.")

    split_entry = data[split]
    if not isinstance(split_entry, list):
        split_entry = [split_entry]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = []

    for entry in split_entry:
        p = Path(entry)
        if not p.is_absolute():
            p = root / p

        if p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue

                img_p = Path(line)
                if not img_p.is_absolute():
                    img_p = root / img_p

                images.append(img_p)

        elif p.is_dir():
            for candidate in p.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in exts:
                    images.append(candidate)

        else:
            raise FileNotFoundError(f"Split-Pfad nicht gefunden: {p}")

    return sorted(set(images))


def image_to_label_path(image_path):
    """
    Erwartete Standardstruktur:
        .../images/val/bild.jpg
        .../labels/val/bild.txt
    oder:
        .../images/bild.jpg
        .../labels/bild.txt
    """
    p = Path(image_path)
    parts = list(p.parts)

    image_indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise ValueError(f"Bildpfad enthaelt keinen 'images'-Ordner: {image_path}")

    idx = image_indices[-1]
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def get_image_size(image_path):
    """Rueckgabe: width, height."""
    if Image is not None:
        with Image.open(image_path) as im:
            return im.size

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht gelesen werden: {image_path}")

    h, w = img.shape[:2]
    return w, h


def xywhn_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    xc *= img_w
    yc *= img_h
    bw *= img_w
    bh *= img_h

    return np.array([
        xc - bw / 2.0,
        yc - bh / 2.0,
        xc + bw / 2.0,
        yc + bh / 2.0,
    ], dtype=np.float32)


def bbox_iou_xyxy(box1, box2):
    a = np.asarray(box1, dtype=np.float32)
    b = np.asarray(box2, dtype=np.float32)

    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def bbox_diagonal(box):
    box = np.asarray(box, dtype=np.float32)
    return float(math.hypot(float(box[2] - box[0]), float(box[3] - box[1])))


def euclidean_error(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def angle_from_vector_360(vec):
    """
    Winkel im Bildkoordinatensystem [0, 360).
    x nach rechts, y nach unten.
    """
    vx, vy = float(vec[0]), float(vec[1])
    if math.hypot(vx, vy) < 1e-12:
        return None
    return math.degrees(math.atan2(vy, vx)) % 360.0


def angle_error_360_deg(pred_angle, gt_angle):
    if pred_angle is None or gt_angle is None:
        return None
    diff = abs(float(pred_angle) - float(gt_angle)) % 360.0
    return min(diff, 360.0 - diff)


def angle_error_180_deg(pred_angle, gt_angle):
    if pred_angle is None or gt_angle is None:
        return None
    diff = abs(float(pred_angle) - float(gt_angle)) % 180.0
    return min(diff, 180.0 - diff)


def safe_stats(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "mean": "",
            "median": "",
            "p95": "",
            "max": "",
        }

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def mean_or_blank(values):
    vals = [float(v) for v in values if v not in ("", None) and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else ""


# ============================================================
# GROUND TRUTH EINLESEN
# ============================================================

def read_gt_labels(image_path):
    img_w, img_h = get_image_size(image_path)
    label_path = image_to_label_path(image_path)

    gts = []

    if not label_path.exists():
        return gts

    for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        expected_len = 5 + NUM_KEYPOINTS * 3

        if len(parts) != expected_len:
            print(
                f"WARNUNG: Ueberspringe ungueltige Labelzeile in {label_path.name}, "
                f"Zeile {line_idx + 1}: erwartet {expected_len} Werte, gefunden {len(parts)}."
            )
            continue

        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        box = xywhn_to_xyxy(xc, yc, bw, bh, img_w, img_h)

        raw = np.asarray([float(x) for x in parts[5:]], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        kpts_xy = raw[:, :2].copy()
        kpts_xy[:, 0] *= img_w
        kpts_xy[:, 1] *= img_h
        visibility = raw[:, 2].astype(int)

        gts.append({
            "cls": cls_id,
            "box": box,
            "kpts": kpts_xy,
            "visibility": visibility,
            "gt_object_id": line_idx,
        })

    return gts


# ============================================================
# YOLO PREDICTIONS EXTRAHIEREN
# ============================================================

def extract_preds_from_result(result):
    preds = []

    if result.boxes is None or len(result.boxes) == 0:
        return preds

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy().astype(float)

    if result.keypoints is None:
        raise RuntimeError(
            "Das Modell liefert keine Keypoints. Bitte pruefen, ob wirklich ein Pose-Modell geladen wurde."
        )

    kpts_xy = result.keypoints.xy.cpu().numpy().astype(np.float32)

    if kpts_xy.shape[1] != NUM_KEYPOINTS:
        raise RuntimeError(
            f"Modell liefert {kpts_xy.shape[1]} Keypoints, erwartet werden {NUM_KEYPOINTS}."
        )

    kpt_conf_obj = result.keypoints.conf
    if kpt_conf_obj is not None:
        kpt_conf = kpt_conf_obj.cpu().numpy().astype(float)
    else:
        kpt_conf = np.full((len(boxes), NUM_KEYPOINTS), np.nan, dtype=float)

    if not (len(boxes) == len(classes) == len(confs) == len(kpts_xy)):
        raise RuntimeError("Inkonsistente Anzahl Boxen/Klassen/Confidences/Keypoints in Result.")

    for pred_idx, (box, cls_id, obj_conf, kp_xy, kp_conf) in enumerate(
        zip(boxes, classes, confs, kpts_xy, kpt_conf)
    ):
        preds.append({
            "cls": int(cls_id),
            "conf": float(obj_conf),
            "box": box,
            "kpts": kp_xy,
            "kpt_conf": np.asarray(kp_conf, dtype=float),
            "pred_index": pred_idx,
        })

    return preds


# ============================================================
# POSE AUS KEYPOINTS ABLEITEN
# ============================================================

def valid_gt_indices(visibility):
    return np.where(np.asarray(visibility) >= MIN_GT_VISIBILITY_FOR_EVAL)[0]


def heatspreader_center(kpts, visibility=None):
    """
    Geometrischer Mittelpunkt aus den vier Aussenpunkten P0...P3.

    WICHTIG fuer eine faire Fehlerberechnung:
    Wenn eine Visibility-Maske uebergeben wird, muss fuer GT und Prediction
    dieselbe Maske verwendet werden. Bei REQUIRE_COMPLETE_POSE_KEYPOINTS=True
    wird der Mittelpunkt nur ausgewertet, wenn P0...P3 vollstaendig auswertbar sind.
    """
    required = np.array([0, 1, 2, 3], dtype=int)
    indices = required.copy()

    if visibility is not None:
        vis = np.asarray(visibility)
        valid = vis[required] >= MIN_GT_VISIBILITY_FOR_EVAL
        if REQUIRE_COMPLETE_POSE_KEYPOINTS and not np.all(valid):
            return None
        indices = required[valid]

    if len(indices) < 2:
        return None

    return np.asarray(kpts, dtype=float)[indices].mean(axis=0)


def heatspreader_angle_360(kpts, visibility=None):
    """
    Gerichtete Achse:
        midpoint(P4,P5) -> midpoint(P6,P7)

    Dadurch wird die 180-Grad-Ambiguitaet durch die semantisch unterschiedlichen
    inneren Heatspreader-Merkmale aufgeloest.
    """
    left_idx = np.array([4, 5], dtype=int)
    right_idx = np.array([6, 7], dtype=int)

    if visibility is not None:
        vis = np.asarray(visibility)
        left_valid = vis[left_idx] >= MIN_GT_VISIBILITY_FOR_EVAL
        right_valid = vis[right_idx] >= MIN_GT_VISIBILITY_FOR_EVAL

        if REQUIRE_COMPLETE_POSE_KEYPOINTS and (not np.all(left_valid) or not np.all(right_valid)):
            return None

        left_idx = left_idx[left_valid]
        right_idx = right_idx[right_valid]

    if len(left_idx) == 0 or len(right_idx) == 0:
        return None

    pts = np.asarray(kpts, dtype=float)
    left_center = pts[left_idx].mean(axis=0)
    right_center = pts[right_idx].mean(axis=0)

    return angle_from_vector_360(right_center - left_center)


def slot_center(kpts, visibility=None):
    """
    Mittelpunkt aus P0...P7. Bei strikter Pose-Auswertung nur, wenn alle
    acht benoetigten GT-Keypoints auswertbar sind.
    """
    pts = np.asarray(kpts, dtype=float)
    required = np.arange(len(pts), dtype=int)

    if visibility is None:
        indices = required
    else:
        vis = np.asarray(visibility)
        valid = vis[required] >= MIN_GT_VISIBILITY_FOR_EVAL
        if REQUIRE_COMPLETE_POSE_KEYPOINTS and not np.all(valid):
            return None
        indices = required[valid]

    if len(indices) < 2:
        return None

    return pts[indices].mean(axis=0)


def slot_angle_180(kpts, visibility=None):
    """
    Slot-Hauptachse via PCA, modulo 180 Grad.
    Das Vorzeichen eines PCA-Eigenvektors ist beliebig; genau deshalb wird hier
    nur die symmetrische Achsenorientierung [0, 180) ausgewertet.
    """
    pts = np.asarray(kpts, dtype=float)

    required = np.arange(len(pts), dtype=int)

    if visibility is None:
        indices = required
    else:
        vis = np.asarray(visibility)
        valid = vis[required] >= MIN_GT_VISIBILITY_FOR_EVAL
        if REQUIRE_COMPLETE_POSE_KEYPOINTS and not np.all(valid):
            return None
        indices = required[valid]

    if len(indices) < 2:
        return None

    selected = pts[indices]
    centered = selected - selected.mean(axis=0)

    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]

    angle = angle_from_vector_360(principal)
    if angle is None:
        return None

    return angle % 180.0


def derive_pose(cls_id, names, kpts, visibility=None):
    class_name = get_class_name(names, cls_id)

    if class_name == HEATSPREADER_CLASS_NAME:
        return {
            "center": heatspreader_center(kpts, visibility),
            "angle": heatspreader_angle_360(kpts, visibility),
            "angle_period": 360,
        }

    if class_name == SLOT_CLASS_NAME:
        return {
            "center": slot_center(kpts, visibility),
            "angle": slot_angle_180(kpts, visibility),
            "angle_period": 180,
        }

    # Fallback fuer spaetere weitere Klassen:
    return {
        "center": slot_center(kpts, visibility),
        "angle": None,
        "angle_period": None,
    }


# ============================================================
# AP / mAP DETECTION
# ============================================================

def voc_ap(recalls, precisions):
    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_for_class(preds_cls, gt_boxes_by_image, iou_thr):
    n_gt = sum(len(v) for v in gt_boxes_by_image.values())

    if n_gt == 0:
        return None

    if len(preds_cls) == 0:
        return 0.0

    preds_sorted = sorted(preds_cls, key=lambda x: x["conf"], reverse=True)
    used = {
        img_key: np.zeros(len(boxes), dtype=bool)
        for img_key, boxes in gt_boxes_by_image.items()
    }

    tp = np.zeros(len(preds_sorted), dtype=np.float32)
    fp = np.zeros(len(preds_sorted), dtype=np.float32)

    for i, pred in enumerate(preds_sorted):
        img_key = pred["image_key"]
        gt_boxes = gt_boxes_by_image.get(img_key, [])

        best_iou = 0.0
        best_j = -1

        for j, gt_box in enumerate(gt_boxes):
            if used[img_key][j]:
                continue

            iou = bbox_iou_xyxy(pred["box"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1.0
            used[img_key][best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / max(n_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    return voc_ap(recalls, precisions)


# ============================================================
# GREEDY MATCHING FUER GEOMETRIE
# ============================================================

def greedy_match_image(gt_list, pred_list, iou_thr):
    """
    Globales greedy Matching innerhalb eines Bildes:
    1. alle klassenkompatiblen GT/Pred-Paare bilden
    2. nach IoU absteigend sortieren
    3. jedes GT und jede Prediction maximal einmal verwenden

    Rueckgabe:
        matches: list[(gt_idx, pred_idx, iou)]
        unmatched_gt: set[int]
        unmatched_pred: set[int]
    """
    candidates = []

    for gi, gt in enumerate(gt_list):
        for pi, pred in enumerate(pred_list):
            if gt["cls"] != pred["cls"]:
                continue
            iou = bbox_iou_xyxy(gt["box"], pred["box"])
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, float(iou)))

    unmatched_gt = set(range(len(gt_list))) - used_gt
    unmatched_pred = set(range(len(pred_list))) - used_pred

    return matches, unmatched_gt, unmatched_pred


# ============================================================
# DETAILAUSWERTUNG MATCHES
# ============================================================

def evaluate_matches(gt_by_image, pred_by_image, names):
    object_rows = []
    keypoint_rows = []
    debug_objects = []

    total_gt = 0
    total_preds = 0
    total_matches = 0
    total_unmatched_preds = 0

    image_keys = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))

    for image_key in image_keys:
        gt_list = gt_by_image.get(image_key, [])
        pred_list = pred_by_image.get(image_key, [])

        total_gt += len(gt_list)
        total_preds += len(pred_list)

        matches, unmatched_gt, unmatched_pred = greedy_match_image(
            gt_list, pred_list, MATCH_IOU_FOR_GEOMETRY
        )

        total_matches += len(matches)
        total_unmatched_preds += len(unmatched_pred)

        # Gematchte Objekte
        for gi, pi, iou in matches:
            gt = gt_list[gi]
            pred = pred_list[pi]
            class_name = get_class_name(names, gt["cls"])

            diag = bbox_diagonal(gt["box"])
            vis = gt["visibility"]
            eval_indices = valid_gt_indices(vis)

            kp_errors = []
            kp_norm_errors = []

            for kp_idx in range(NUM_KEYPOINTS):
                gt_xy = gt["kpts"][kp_idx]
                pred_xy = pred["kpts"][kp_idx]
                gt_vis = int(vis[kp_idx])
                kp_conf = float(pred["kpt_conf"][kp_idx]) if np.isfinite(pred["kpt_conf"][kp_idx]) else ""

                evaluable = gt_vis >= MIN_GT_VISIBILITY_FOR_EVAL

                if evaluable:
                    error_px = euclidean_error(gt_xy, pred_xy)
                    error_norm = error_px / diag if diag > 0 else None
                    kp_errors.append(error_px)
                    if error_norm is not None:
                        kp_norm_errors.append(error_norm)
                else:
                    error_px = ""
                    error_norm = ""

                keypoint_rows.append({
                    "image": gt["image_path"],
                    "class_id": gt["cls"],
                    "class": class_name,
                    "gt_object_id": gt["gt_object_id"],
                    "pred_index": pred["pred_index"],
                    "keypoint_index": kp_idx,
                    "keypoint_name": f"P{kp_idx}",
                    "gt_x_px": float(gt_xy[0]),
                    "gt_y_px": float(gt_xy[1]),
                    "gt_visibility": gt_vis,
                    "evaluable": evaluable,
                    "pred_x_px": float(pred_xy[0]),
                    "pred_y_px": float(pred_xy[1]),
                    "keypoint_confidence": kp_conf,
                    "error_px": error_px,
                    "error_normalized_bbox_diagonal": error_norm,
                })

            kp_stats = safe_stats(kp_errors)
            kp_norm_stats = safe_stats(kp_norm_errors)

            # KRITISCH: GT und Prediction muessen fuer die Pose aus EXAKT denselben
            # semantischen Keypoints berechnet werden. Die GT-Visibility-Maske wird
            # deshalb auch auf die Prediction angewendet. Andernfalls wuerde z.B.
            # GT-center(P0,P1) mit Prediction-center(P0,P1,P2,P3) verglichen.
            pose_visibility = gt["visibility"]
            gt_pose = derive_pose(gt["cls"], names, gt["kpts"], pose_visibility)
            pred_pose = derive_pose(pred["cls"], names, pred["kpts"], pose_visibility)

            if gt_pose["center"] is not None and pred_pose["center"] is not None:
                center_error = euclidean_error(gt_pose["center"], pred_pose["center"])
            else:
                center_error = ""

            angle_error = ""
            if gt_pose["angle"] is not None and pred_pose["angle"] is not None:
                if gt_pose["angle_period"] == 360:
                    angle_error = angle_error_360_deg(pred_pose["angle"], gt_pose["angle"])
                elif gt_pose["angle_period"] == 180:
                    angle_error = angle_error_180_deg(pred_pose["angle"], gt_pose["angle"])

            row = {
                "image": gt["image_path"],
                "class_id": gt["cls"],
                "class": class_name,
                "gt_object_id": gt["gt_object_id"],
                "pred_index": pred["pred_index"],
                "matched": True,
                "bbox_iou": iou,
                "object_confidence": pred["conf"],
                "evaluable_keypoints": int(len(eval_indices)),
                "center_evaluable": gt_pose["center"] is not None and pred_pose["center"] is not None,
                "angle_evaluable": gt_pose["angle"] is not None and pred_pose["angle"] is not None,
                "kp_mean_error_px": kp_stats["mean"],
                "kp_median_error_px": kp_stats["median"],
                "kp_p95_error_px": kp_stats["p95"],
                "kp_max_error_px": kp_stats["max"],
                "kp_mean_error_normalized": kp_norm_stats["mean"],
                "center_gt_x_px": float(gt_pose["center"][0]) if gt_pose["center"] is not None else "",
                "center_gt_y_px": float(gt_pose["center"][1]) if gt_pose["center"] is not None else "",
                "center_pred_x_px": float(pred_pose["center"][0]) if pred_pose["center"] is not None else "",
                "center_pred_y_px": float(pred_pose["center"][1]) if pred_pose["center"] is not None else "",
                "center_error_px": center_error,
                "angle_period_deg": gt_pose["angle_period"] if gt_pose["angle_period"] is not None else "",
                "angle_gt_deg": gt_pose["angle"] if gt_pose["angle"] is not None else "",
                "angle_pred_deg": pred_pose["angle"] if pred_pose["angle"] is not None else "",
                "angle_error_deg": angle_error,
            }
            object_rows.append(row)

            debug_objects.append({
                "image_key": image_key,
                "image_path": gt["image_path"],
                "class": class_name,
                "gt_box": gt["box"],
                "pred_box": pred["box"],
                "gt_kpts": gt["kpts"],
                "pred_kpts": pred["kpts"],
                "visibility": gt["visibility"],
                "kpt_conf": pred["kpt_conf"],
                "gt_center": gt_pose["center"],
                "pred_center": pred_pose["center"],
                "gt_angle": gt_pose["angle"],
                "pred_angle": pred_pose["angle"],
                "angle_period": gt_pose["angle_period"],
                "bbox_iou": iou,
                "object_conf": pred["conf"],
                "kp_mean_error_px": kp_stats["mean"],
                "kp_max_error_px": kp_stats["max"],
                "center_error_px": center_error,
                "angle_error_deg": angle_error,
            })

        # Nicht gematchte Ground Truths als Objektzeilen dokumentieren
        for gi in sorted(unmatched_gt):
            gt = gt_list[gi]
            class_name = get_class_name(names, gt["cls"])
            object_rows.append({
                "image": gt["image_path"],
                "class_id": gt["cls"],
                "class": class_name,
                "gt_object_id": gt["gt_object_id"],
                "pred_index": "",
                "matched": False,
                "bbox_iou": "",
                "object_confidence": "",
                "evaluable_keypoints": int(len(valid_gt_indices(gt["visibility"]))),
                "center_evaluable": False,
                "angle_evaluable": False,
                "kp_mean_error_px": "",
                "kp_median_error_px": "",
                "kp_p95_error_px": "",
                "kp_max_error_px": "",
                "kp_mean_error_normalized": "",
                "center_gt_x_px": "",
                "center_gt_y_px": "",
                "center_pred_x_px": "",
                "center_pred_y_px": "",
                "center_error_px": "",
                "angle_period_deg": "",
                "angle_gt_deg": "",
                "angle_pred_deg": "",
                "angle_error_deg": "",
            })

    return {
        "object_rows": object_rows,
        "keypoint_rows": keypoint_rows,
        "debug_objects": debug_objects,
        "total_gt": total_gt,
        "total_preds": total_preds,
        "matched_count": total_matches,
        "unmatched_preds": total_unmatched_preds,
    }


# ============================================================
# DETECTION PRECISION / RECALL BEI FESTER CONF
# ============================================================

def detection_counts_per_class(gt_by_image, pred_by_image, class_ids, iou_thr=0.50):
    counts = {
        cls_id: {"tp": 0, "fp": 0, "fn": 0}
        for cls_id in class_ids
    }

    image_keys = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))

    for image_key in image_keys:
        gt_list = gt_by_image.get(image_key, [])
        pred_list = pred_by_image.get(image_key, [])

        for cls_id in class_ids:
            gt_cls = [g for g in gt_list if g["cls"] == cls_id]
            pred_cls = [p for p in pred_list if p["cls"] == cls_id]
            matches, unmatched_gt, unmatched_pred = greedy_match_image(gt_cls, pred_cls, iou_thr)

            counts[cls_id]["tp"] += len(matches)
            counts[cls_id]["fn"] += len(unmatched_gt)
            counts[cls_id]["fp"] += len(unmatched_pred)

    return counts


# ============================================================
# SUMMARY
# ============================================================

def build_keypoint_summary(keypoint_rows, names):
    class_ids = sorted({int(r["class_id"]) for r in keypoint_rows})
    rows = []

    for cls_id in class_ids:
        cls_name = get_class_name(names, cls_id)

        for kp_idx in range(NUM_KEYPOINTS):
            selected = [
                r for r in keypoint_rows
                if int(r["class_id"]) == cls_id
                and int(r["keypoint_index"]) == kp_idx
                and bool(r["evaluable"])
                and r["error_px"] != ""
            ]

            errors = [float(r["error_px"]) for r in selected]
            norms = [float(r["error_normalized_bbox_diagonal"]) for r in selected if r["error_normalized_bbox_diagonal"] != ""]
            confs = [float(r["keypoint_confidence"]) for r in selected if r["keypoint_confidence"] != ""]

            stats = safe_stats(errors)
            norm_stats = safe_stats(norms)

            row = {
                "class": cls_name,
                "keypoint": f"P{kp_idx}",
                "n": len(errors),
                "error_mean_px": stats["mean"],
                "error_median_px": stats["median"],
                "error_p95_px": stats["p95"],
                "error_max_px": stats["max"],
                "error_mean_normalized_bbox_diagonal": norm_stats["mean"],
                "keypoint_confidence_mean": float(np.mean(confs)) if confs else "",
            }

            for thr in PCK_THRESHOLDS_PX:
                key = f"PCK_at_{thr:g}px"
                row[key] = float(np.mean(np.asarray(errors) <= thr)) if errors else ""

            rows.append(row)

    return rows


def build_summary(
    object_rows,
    keypoint_rows,
    preds_by_class,
    gt_boxes_by_class_image,
    names,
    detection_counts,
):
    class_ids = sorted(set(gt_boxes_by_class_image.keys()) | set(preds_by_class.keys()))
    summary = []

    for cls_id in class_ids:
        cls_name = get_class_name(names, cls_id)

        ap_values = {}
        for thr in AP_IOU_THRESHOLDS:
            ap_values[round(float(thr), 2)] = compute_ap_for_class(
                preds_by_class.get(cls_id, []),
                gt_boxes_by_class_image.get(cls_id, {}),
                float(thr),
            )

        ap50 = ap_values.get(0.50)
        ap75 = ap_values.get(0.75)
        valid_aps = [v for v in ap_values.values() if v is not None]
        map50_95 = float(np.mean(valid_aps)) if valid_aps else None

        counts = detection_counts.get(cls_id, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        matched_objects = [
            r for r in object_rows
            if int(r["class_id"]) == cls_id and bool(r["matched"])
        ]

        kp_rows = [
            r for r in keypoint_rows
            if int(r["class_id"]) == cls_id
            and bool(r["evaluable"])
            and r["error_px"] != ""
        ]

        kp_errors = [float(r["error_px"]) for r in kp_rows]
        kp_norm = [
            float(r["error_normalized_bbox_diagonal"])
            for r in kp_rows
            if r["error_normalized_bbox_diagonal"] != ""
        ]
        bbox_ious = [float(r["bbox_iou"]) for r in matched_objects if r["bbox_iou"] != ""]
        center_errors = [float(r["center_error_px"]) for r in matched_objects if r["center_error_px"] != ""]
        angle_errors = [float(r["angle_error_deg"]) for r in matched_objects if r["angle_error_deg"] != ""]
        kp_confs = [
            float(r["keypoint_confidence"])
            for r in kp_rows
            if r["keypoint_confidence"] != ""
        ]

        kp_stats = safe_stats(kp_errors)
        kp_norm_stats = safe_stats(kp_norm)
        iou_stats = safe_stats(bbox_ious)
        center_stats = safe_stats(center_errors)
        angle_stats = safe_stats(angle_errors)

        row = {
            "class": cls_name,
            "gt_objects": sum(len(v) for v in gt_boxes_by_class_image.get(cls_id, {}).values()),
            "predictions": len(preds_by_class.get(cls_id, [])),
            "tp_at_iou50": tp,
            "fp_at_iou50": fp,
            "fn_at_iou50": fn,
            "precision_at_conf_and_iou50": precision,
            "recall_at_conf_and_iou50": recall,
            "mAP50_detection": ap50 if ap50 is not None else "",
            "mAP75_detection": ap75 if ap75 is not None else "",
            "mAP50_95_detection": map50_95 if map50_95 is not None else "",
            "bbox_iou_mean_matched": iou_stats["mean"],
            "kp_error_mean_px": kp_stats["mean"],
            "kp_error_median_px": kp_stats["median"],
            "kp_error_p95_px": kp_stats["p95"],
            "kp_error_max_px": kp_stats["max"],
            "kp_error_mean_normalized_bbox_diagonal": kp_norm_stats["mean"],
            "center_n": len(center_errors),
            "center_error_mean_px": center_stats["mean"],
            "center_error_median_px": center_stats["median"],
            "center_error_p95_px": center_stats["p95"],
            "center_error_max_px": center_stats["max"],
            "angle_n": len(angle_errors),
            "angle_error_mean_deg": angle_stats["mean"],
            "angle_error_median_deg": angle_stats["median"],
            "angle_error_p95_deg": angle_stats["p95"],
            "angle_error_max_deg": angle_stats["max"],
            "keypoint_confidence_mean": float(np.mean(kp_confs)) if kp_confs else "",
        }

        for thr in PCK_THRESHOLDS_PX:
            row[f"PCK_at_{thr:g}px"] = (
                float(np.mean(np.asarray(kp_errors) <= thr)) if kp_errors else ""
            )

        summary.append(row)

    # all-Zeile: Detection mAP wird als Klassenmittel gebildet,
    # Geometrie als Pool aller passenden Messungen.
    if summary:
        all_kp_rows = [r for r in keypoint_rows if bool(r["evaluable"]) and r["error_px"] != ""]
        all_matched = [r for r in object_rows if bool(r["matched"])]

        all_kp_errors = [float(r["error_px"]) for r in all_kp_rows]
        all_kp_norm = [
            float(r["error_normalized_bbox_diagonal"])
            for r in all_kp_rows
            if r["error_normalized_bbox_diagonal"] != ""
        ]
        all_bbox_iou = [float(r["bbox_iou"]) for r in all_matched if r["bbox_iou"] != ""]
        all_center = [float(r["center_error_px"]) for r in all_matched if r["center_error_px"] != ""]
        all_angle = [float(r["angle_error_deg"]) for r in all_matched if r["angle_error_deg"] != ""]
        all_kp_conf = [
            float(r["keypoint_confidence"])
            for r in all_kp_rows
            if r["keypoint_confidence"] != ""
        ]

        kp_stats = safe_stats(all_kp_errors)
        kp_norm_stats = safe_stats(all_kp_norm)
        iou_stats = safe_stats(all_bbox_iou)
        center_stats = safe_stats(all_center)
        angle_stats = safe_stats(all_angle)

        total_tp = sum(detection_counts[c]["tp"] for c in detection_counts)
        total_fp = sum(detection_counts[c]["fp"] for c in detection_counts)
        total_fn = sum(detection_counts[c]["fn"] for c in detection_counts)

        all_row = {
            "class": "all",
            "gt_objects": sum(r["gt_objects"] for r in summary),
            "predictions": sum(r["predictions"] for r in summary),
            "tp_at_iou50": total_tp,
            "fp_at_iou50": total_fp,
            "fn_at_iou50": total_fn,
            "precision_at_conf_and_iou50": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0,
            "recall_at_conf_and_iou50": total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0,
            "mAP50_detection": mean_or_blank([r["mAP50_detection"] for r in summary]),
            "mAP75_detection": mean_or_blank([r["mAP75_detection"] for r in summary]),
            "mAP50_95_detection": mean_or_blank([r["mAP50_95_detection"] for r in summary]),
            "bbox_iou_mean_matched": iou_stats["mean"],
            "kp_error_mean_px": kp_stats["mean"],
            "kp_error_median_px": kp_stats["median"],
            "kp_error_p95_px": kp_stats["p95"],
            "kp_error_max_px": kp_stats["max"],
            "kp_error_mean_normalized_bbox_diagonal": kp_norm_stats["mean"],
            "center_n": len(all_center),
            "center_error_mean_px": center_stats["mean"],
            "center_error_median_px": center_stats["median"],
            "center_error_p95_px": center_stats["p95"],
            "center_error_max_px": center_stats["max"],
            "angle_n": len(all_angle),
            # ACHTUNG: all-Winkel mischt 360-Heatspreader- und 180-Slotfehler.
            # Fuer technische Aussagen besser klassenweise betrachten.
            "angle_error_mean_deg": angle_stats["mean"],
            "angle_error_median_deg": angle_stats["median"],
            "angle_error_p95_deg": angle_stats["p95"],
            "angle_error_max_deg": angle_stats["max"],
            "keypoint_confidence_mean": float(np.mean(all_kp_conf)) if all_kp_conf else "",
        }

        for thr in PCK_THRESHOLDS_PX:
            all_row[f"PCK_at_{thr:g}px"] = (
                float(np.mean(np.asarray(all_kp_errors) <= thr)) if all_kp_errors else ""
            )

        summary.insert(0, all_row)

    return summary


# ============================================================
# CSV / TEXT AUSGABE
# ============================================================

def save_csv(path, rows):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def write_legend(path):
    pck_text = ", ".join(f"PCK@{t:g}px" for t in PCK_THRESHOLDS_PX)

    text = f"""YOLO POSE AUSWERTUNG V2.1 - LEGENDE
=================================

1) ALLGEMEINES
--------------
Dieses Skript bewertet die ROHE YOLO-Pose vor CAD-/Template-Postprocessing.
Objektmatching fuer geometrische Fehler erfolgt ausschliesslich ueber gleiche Klasse
und Bounding-Box-IoU >= {MATCH_IOU_FOR_GEOMETRY:.2f}.

CONF = {CONF:.3f}
IMGSZ = {IMGSZ}
GT-Keypoints werden ausgewertet bei visibility >= {MIN_GT_VISIBILITY_FOR_EVAL}.
Strikte Pose-Auswertung (vollstaendige benoetigte Keypoints) = {REQUIRE_COMPLETE_POSE_KEYPOINTS}.

2) WINKELKONVENTION
-------------------
Bildkoordinaten: x nach rechts, y nach unten.
0 Grad   = rechts
90 Grad  = unten
180 Grad = links
270 Grad = oben

Heatspreader:
- center = Mittelwert P0..P3 (Aussenpunkte)
- angle = gerichtete Achse midpoint(P4,P5) -> midpoint(P6,P7)
- Winkelbereich / Fehlerperiodizitaet = 360 Grad

Slot:
- center = Mittelwert aller auswertbaren P0..P7
- angle = Hauptachse der 8 Punkte via PCA
- Winkelbereich / Fehlerperiodizitaet = 180 Grad
- theta und theta + 180 Grad gelten als identisch, weil der Slot symmetrisch ist.

3) GROUND-TRUTH VISIBILITY
--------------------------
0 = nicht gelabelt / nicht auswertbar
1 = gelabelt, aber verdeckt
2 = sichtbar

4) DETECTION-METRIKEN
---------------------
bbox_iou:
Intersection over Union zwischen axis-aligned GT- und Prediction-Bounding-Box.

mAP50_detection:
Mittlere Average Precision bei IoU = 0.50.

mAP75_detection:
Mittlere Average Precision bei IoU = 0.75.

mAP50_95_detection:
Mittel der AP-Werte fuer IoU 0.50, 0.55, ..., 0.95.

precision_at_conf_and_iou50:
TP / (TP + FP) fuer Predictions nach CONF-Filter und IoU >= 0.50.

recall_at_conf_and_iou50:
TP / (TP + FN) fuer Predictions nach CONF-Filter und IoU >= 0.50.

5) KEYPOINT-METRIKEN
--------------------
error_px:
Euklidischer Abstand zwischen GT-Keypoint und vorhergesagtem Keypoint in Originalbild-Pixeln.

error_normalized_bbox_diagonal:
error_px geteilt durch die Diagonale der GT-Bounding-Box.
Beispiel: 0.01 = Fehler entspricht 1 % der GT-Boxdiagonale.

PCK:
Percentage of Correct Keypoints = Anteil der auswertbaren Keypoints, deren Pixelabstand
kleiner/gleich einer Schwelle ist. Ausgegeben werden: {pck_text}.

keypoint_confidence:
Von YOLO ausgegebene Confidence des einzelnen Keypoints.
Nicht mit object_confidence verwechseln.

6) POSE-METRIKEN
----------------
center_error_px:
Euklidischer Abstand zwischen GT- und Prediction-Mittelpunkt. Fuer beide Seiten wird
EXAKT dieselbe GT-Visibility-Maske verwendet. Damit werden niemals unterschiedliche
Keypoint-Teilmengen miteinander verglichen.
Bei strikter Pose-Auswertung wird ein Mittelpunkt nur berechnet, wenn alle fuer seine
Definition benoetigten Keypoints auswertbar sind.

center_n / angle_n:
Anzahl der Objekte, die tatsaechlich in die jeweilige Mittelpunkt-/Winkelstatistik
eingegangen sind.

angle_error_deg Heatspreader:
Kleinster zyklischer Fehler bei 360-Grad-Periodizitaet.

angle_error_deg Slot:
Kleinster Achsenfehler bei 180-Grad-Periodizitaet.

Mean   = arithmetischer Mittelwert
Median = 50-%-Quantil
P95    = 95-%-Quantil; 95 % der Werte liegen darunter oder gleich
Max    = groesster beobachteter Fehler

7) CSV-DATEIEN
--------------
{OBJECT_CSV}
Eine Zeile pro GT-Objekt. Enthaelt Detection-Match, Keypoint-Gesamtfehler,
Mittelpunkt- und Winkelfehler. Nicht erkannte GT-Objekte bleiben mit matched=False enthalten.

{KEYPOINT_CSV}
Eine Zeile pro Keypoint eines gematchten Objekts. Wichtig fuer P0...P7-Einzelanalyse,
Visibility, Confidence und spaetere Confidence-vs.-Fehler-Untersuchungen.

{SUMMARY_CSV}
Kompakte Gesamtstatistik pro Klasse plus 'all'. Fuer Winkel technische Aussagen
vorzugsweise klassenweise treffen, weil 'all' 360-Grad-Heatspreader- und
180-Grad-Slot-Winkelfehler gemeinsam poolt.

{KEYPOINT_SUMMARY_CSV}
Statistik fuer jeden einzelnen P0...P7 getrennt nach Klasse.

8) DEBUGBILDER - FARBEN / SYMBOLE
---------------------------------
GRUEN:
Ground Truth (Box, Keypoints, Mittelpunkt, Richtungspfeil)

ROT:
Prediction (Box, Keypoints, Mittelpunkt, Richtungspfeil)

CYAN:
Verbindung vom GT-Keypoint zum zugehoerigen Prediction-Keypoint.
Die Linienlaenge visualisiert direkt den Keypointfehler.

P0...P7:
Index des Keypoints. GT und Prediction tragen denselben Index.

GT C / PR C:
Ground-Truth-Mittelpunkt / Prediction-Mittelpunkt.

GT theta / PR theta:
Aus den Keypoints berechnete Orientierung.
Heatspreader 360 Grad, Slot 180 Grad.

9) WICHTIGE INTERPRETATION
--------------------------
Die Detection-mAP bewertet, ob das Objekt gefunden wird.
Die Keypoint-, Mittelpunkt- und Winkelmetriken bewerten die fuer Pick-and-Place
entscheidende geometrische Genauigkeit.
Diese V1 ist die Baseline fuer den spaeteren Vergleich mit CAD-/Geometrie-Fit.
"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ============================================================
# DEBUGBILDER
# ============================================================

# OpenCV BGR
COLOR_GT = (0, 255, 0)
COLOR_PRED = (0, 0, 255)
COLOR_LINK = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (20, 20, 20)


def draw_cross(img, point, color, size=8, thickness=2):
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def draw_angle_arrow(img, center, angle_deg, color, length=80, thickness=2):
    if center is None or angle_deg is None:
        return
    rad = math.radians(float(angle_deg))
    start = (int(round(center[0])), int(round(center[1])))
    end = (
        int(round(center[0] + length * math.cos(rad))),
        int(round(center[1] + length * math.sin(rad))),
    )
    cv2.arrowedLine(img, start, end, color, thickness, cv2.LINE_AA, tipLength=0.18)


def draw_text_with_bg(img, text, org, color=COLOR_TEXT, scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + baseline + 2), COLOR_PANEL, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_legend_panel(img):
    lines = [
        ("LEGENDE", COLOR_TEXT),
        ("GRUEN = Ground Truth", COLOR_GT),
        ("ROT   = Prediction", COLOR_PRED),
        ("CYAN  = GT -> Prediction Fehler", COLOR_LINK),
        ("P0...P7 = Keypoint-Index", COLOR_TEXT),
        ("Kreuz = Mittelpunkt", COLOR_TEXT),
        ("Pfeil = Orientierung", COLOR_TEXT),
        ("HS: 360 Grad | Slot: 180 Grad", COLOR_TEXT),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    line_h = 20
    panel_w = 330
    panel_h = 12 + len(lines) * line_h

    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    for i, (text, color) in enumerate(lines):
        y = 28 + i * line_h
        cv2.putText(img, text, (18, y), font, scale, color, thickness, cv2.LINE_AA)


def debug_score(obj):
    key = {
        "center": "center_error_px",
        "kp_mean": "kp_mean_error_px",
        "kp_max": "kp_max_error_px",
        "angle": "angle_error_deg",
    }.get(DEBUG_SORT_METRIC, "center_error_px")

    value = obj.get(key, "")
    if value in ("", None):
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def save_debug_images(debug_objects):
    if not SAVE_DEBUG_IMAGES or not debug_objects:
        return 0

    out_dir = OUTPUT_DIR / DEBUG_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for obj in debug_objects:
        grouped[obj["image_key"]].append(obj)

    image_keys = sorted(
        grouped.keys(),
        key=lambda k: max(debug_score(o) for o in grouped[k]),
        reverse=True,
    )

    if DEBUG_MIN_SCORE is not None:
        image_keys = [
            k for k in image_keys
            if max(debug_score(o) for o in grouped[k]) >= DEBUG_MIN_SCORE
        ]

    if DEBUG_MAX_IMAGES is not None:
        image_keys = image_keys[:DEBUG_MAX_IMAGES]

    saved = 0

    for image_key in image_keys:
        objects = grouped[image_key]
        image_path = objects[0]["image_path"]
        img = cv2.imread(str(image_path))

        if img is None:
            continue

        for obj_nr, obj in enumerate(objects):
            gt_box = np.round(obj["gt_box"]).astype(int)
            pr_box = np.round(obj["pred_box"]).astype(int)

            cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), COLOR_GT, 2)
            cv2.rectangle(img, (pr_box[0], pr_box[1]), (pr_box[2], pr_box[3]), COLOR_PRED, 2)

            for i in range(NUM_KEYPOINTS):
                gt_pt = obj["gt_kpts"][i]
                pr_pt = obj["pred_kpts"][i]
                vis = int(obj["visibility"][i])

                gt_xy = tuple(np.round(gt_pt).astype(int))
                pr_xy = tuple(np.round(pr_pt).astype(int))

                if vis >= MIN_GT_VISIBILITY_FOR_EVAL:
                    cv2.line(img, gt_xy, pr_xy, COLOR_LINK, 1, cv2.LINE_AA)
                    cv2.circle(img, gt_xy, 5, COLOR_GT, -1, cv2.LINE_AA)
                    cv2.circle(img, pr_xy, 5, COLOR_PRED, -1, cv2.LINE_AA)

                    cv2.putText(
                        img, f"P{i}", (gt_xy[0] + 6, gt_xy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GT, 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        img, f"P{i}", (pr_xy[0] + 6, pr_xy[1] + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_PRED, 1, cv2.LINE_AA
                    )

            if obj["gt_center"] is not None:
                draw_cross(img, obj["gt_center"], COLOR_GT)
                draw_text_with_bg(
                    img, "GT C", (int(obj["gt_center"][0]) + 10, int(obj["gt_center"][1]) - 10), COLOR_GT
                )

            if obj["pred_center"] is not None:
                draw_cross(img, obj["pred_center"], COLOR_PRED)
                draw_text_with_bg(
                    img, "PR C", (int(obj["pred_center"][0]) + 10, int(obj["pred_center"][1]) + 18), COLOR_PRED
                )

            draw_angle_arrow(img, obj["gt_center"], obj["gt_angle"], COLOR_GT)
            draw_angle_arrow(img, obj["pred_center"], obj["pred_angle"], COLOR_PRED)

            metrics = [
                f'{obj["class"]} | IoU {obj["bbox_iou"]:.3f} | ObjConf {obj["object_conf"]:.3f}',
                f'KP mean {fmt(obj["kp_mean_error_px"])} px | KP max {fmt(obj["kp_max_error_px"])} px',
                f'Center {fmt(obj["center_error_px"])} px | Angle {fmt(obj["angle_error_deg"])} deg',
            ]

            text_x = max(10, int(pr_box[0]))
            text_y = max(30, int(pr_box[1]) - 48 - obj_nr * 4)
            for line_idx, line in enumerate(metrics):
                draw_text_with_bg(img, line, (text_x, text_y + 18 * line_idx), COLOR_TEXT, scale=0.46)

        draw_legend_panel(img)

        out_path = out_dir / f"debug_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(out_path), img)
        saved += 1

    return saved


# ============================================================
# KONSOLE
# ============================================================

def fmt(x, digits=3):
    if x == "" or x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "-"
        return f"{x:.{digits}f}"
    return str(x)


def print_summary_table(summary):
    headers = [
        "Klasse", "mAP50", "mAP75", "mAP50-95", "Prec", "Recall",
        "KP Mean px", "KP P95 px", "Center Mean px", "Center P95 px",
        "Angle Mean", "Angle P95", "PCK@5px",
    ]

    rows = []
    for r in summary:
        rows.append([
            str(r["class"]),
            fmt(r["mAP50_detection"]),
            fmt(r["mAP75_detection"]),
            fmt(r["mAP50_95_detection"]),
            fmt(r["precision_at_conf_and_iou50"]),
            fmt(r["recall_at_conf_and_iou50"]),
            fmt(r["kp_error_mean_px"]),
            fmt(r["kp_error_p95_px"]),
            fmt(r["center_error_mean_px"]),
            fmt(r["center_error_p95_px"]),
            fmt(r["angle_error_mean_deg"]),
            fmt(r["angle_error_p95_deg"]),
            fmt(r.get("PCK_at_5px", "")),
        ])

    if not rows:
        print("Keine Summary-Daten vorhanden.")
        return

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


# ============================================================
# VALIDIERUNG DER KONFIGURATION
# ============================================================

def validate_configuration(data, names):
    kpt_shape = data.get("kpt_shape")
    if kpt_shape is not None:
        if int(kpt_shape[0]) != NUM_KEYPOINTS:
            raise ValueError(
                f"data.yaml kpt_shape={kpt_shape}, aber NUM_KEYPOINTS={NUM_KEYPOINTS}."
            )
        if len(kpt_shape) > 1 and int(kpt_shape[1]) != 3:
            print(
                "WARNUNG: data.yaml verwendet keine 3D-Keypoints [x,y,visibility]. "
                "GT-Visibility-Auswertung ist dann nicht wie vorgesehen moeglich."
            )

    hs_id = class_name_to_id(names, HEATSPREADER_CLASS_NAME)
    slot_id = class_name_to_id(names, SLOT_CLASS_NAME)

    if hs_id is None:
        print(f"WARNUNG: Klasse '{HEATSPREADER_CLASS_NAME}' nicht in data.yaml gefunden.")
    if slot_id is None:
        print(f"WARNUNG: Klasse '{SLOT_CLASS_NAME}' nicht in data.yaml gefunden.")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data_yaml(DATA_YAML)
    names = data.get("names", {})
    validate_configuration(data, names)

    image_paths = resolve_image_paths(DATA_YAML, SPLIT)
    print(f"Bilder gefunden: {len(image_paths)}")
    print("Lese Ground-Truth-Pose-Labels ...")

    gt_by_image = defaultdict(list)
    gt_boxes_by_class_image = defaultdict(lambda: defaultdict(list))

    for image_path in image_paths:
        key = norm_path(image_path)
        gts = read_gt_labels(image_path)

        for gt in gts:
            gt["image_key"] = key
            gt["image_path"] = str(image_path)
            gt_by_image[key].append(gt)
            gt_boxes_by_class_image[gt["cls"]][key].append(gt["box"])

    print("Starte YOLO-Pose-Inferenz ...")
    model = YOLO(MODEL_PATH)

    pred_by_image = defaultdict(list)
    preds_by_class = defaultdict(list)

    source_list = [str(p) for p in image_paths]
    yolo_inference_ms = []
    yolo_preprocess_ms = []
    yolo_postprocess_ms = []

    results = model.predict(
        source=source_list,
        imgsz=IMGSZ,
        conf=CONF,
        batch=BATCH,
        device=DEVICE,
        stream=True,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        key = norm_path(image_path)
        preds = extract_preds_from_result(result)

        for pred in preds:
            pred["image_key"] = key
            pred_by_image[key].append(pred)
            preds_by_class[pred["cls"]].append(pred)

        speed = getattr(result, "speed", None) or {}
        if speed.get("preprocess") is not None:
            yolo_preprocess_ms.append(float(speed["preprocess"]))
        if speed.get("inference") is not None:
            yolo_inference_ms.append(float(speed["inference"]))
        if speed.get("postprocess") is not None:
            yolo_postprocess_ms.append(float(speed["postprocess"]))

    print("Matche Objekte und berechne Keypoint-/Posefehler ...")

    evaluation = evaluate_matches(gt_by_image, pred_by_image, names)

    class_ids = sorted(set(gt_boxes_by_class_image.keys()) | set(preds_by_class.keys()))
    det_counts = detection_counts_per_class(
        gt_by_image, pred_by_image, class_ids, iou_thr=0.50
    )

    summary = build_summary(
        evaluation["object_rows"],
        evaluation["keypoint_rows"],
        preds_by_class,
        gt_boxes_by_class_image,
        names,
        det_counts,
    )

    keypoint_summary = build_keypoint_summary(evaluation["keypoint_rows"], names)

    object_csv_path = OUTPUT_DIR / OBJECT_CSV
    keypoint_csv_path = OUTPUT_DIR / KEYPOINT_CSV
    summary_csv_path = OUTPUT_DIR / SUMMARY_CSV
    kp_summary_csv_path = OUTPUT_DIR / KEYPOINT_SUMMARY_CSV
    legend_path = OUTPUT_DIR / LEGEND_TXT

    save_csv(object_csv_path, evaluation["object_rows"])
    save_csv(keypoint_csv_path, evaluation["keypoint_rows"])
    save_csv(summary_csv_path, summary)
    save_csv(kp_summary_csv_path, keypoint_summary)
    write_legend(legend_path)

    saved_debug = save_debug_images(evaluation["debug_objects"])

    runtime = time.perf_counter() - t0
    missed = evaluation["total_gt"] - evaluation["matched_count"]

    print("\n===== GESAMT =====")
    print(f"Bilder: {len(image_paths)}")
    print(f"GT-Objekte: {evaluation['total_gt']}")
    print(f"Gematcht (IoU >= {MATCH_IOU_FOR_GEOMETRY:.2f}): {evaluation['matched_count']}")
    print(f"Nicht erkannt / nicht gematcht: {missed}")
    print(f"Predictions gesamt: {evaluation['total_preds']}")
    print(f"Unmatched Predictions: {evaluation['unmatched_preds']}")

    if yolo_preprocess_ms:
        print(f"YOLO Preprocess Mittel: {np.mean(yolo_preprocess_ms):.2f} ms/Bild")
    if yolo_inference_ms:
        print(f"YOLO Inferenz Mittel: {np.mean(yolo_inference_ms):.2f} ms/Bild")
        print(f"YOLO Inferenz FPS rechnerisch: {1000.0 / np.mean(yolo_inference_ms):.2f}")
    if yolo_postprocess_ms:
        print(f"YOLO Postprocess Mittel: {np.mean(yolo_postprocess_ms):.2f} ms/Bild")

    print(f"Gesamtlaufzeit Skript: {runtime:.2f} s")
    print(f"\nObjekt-CSV: {object_csv_path}")
    print(f"Keypoint-CSV: {keypoint_csv_path}")
    print(f"Summary-CSV: {summary_csv_path}")
    print(f"Keypoint-Summary-CSV: {kp_summary_csv_path}")
    print(f"Legende: {legend_path}")
    print(f"Debug-Bilder: {saved_debug} in '{OUTPUT_DIR / DEBUG_DIR_NAME}'")

    print("\n===== SUMMARY =====")
    print_summary_table(summary)

    print("\nHinweis:")
    print("- Heatspreader-Winkel wird mit 360-Grad-Periodizitaet bewertet.")
    print("- Slot-Winkel wird mit 180-Grad-Periodizitaet bewertet.")
    print("- Die Datei pose_eval_legend.txt erklaert alle Metriken und Debug-Symbole.")


if __name__ == "__main__":
    main()
