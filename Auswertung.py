
from pathlib import Path
from collections import defaultdict
import csv
import math

import cv2
import yaml
import numpy as np
from ultralytics import YOLO


# =========================
# HIER ANPASSEN
# =========================

MODEL_PATH = r"C:\FHWS\nanoplus\YOLO VS Code\runs\obb\train-2\weights\best.pt"
DATA_YAML = r"C:\FHWS\nanoplus\YOLO VS Code\dataset_yolo8_obb\data.yaml"

SPLIT = "val"          # val oder test
IMGSZ = 960            # gleiche Größe wie beim Training / Vergleich
CONF = 0.50            # 0.25   Mindest-Confidence
MATCH_IOU = 0.75       # 0.50   ab wann Prediction und Label als dasselbe Objekt gelten

MM_PER_PIXEL = None    # später z.B. 0.015 eintragen, wenn du Kamera-Kalibrierung hast
OUT_CSV = "angle_eval.csv"


# =========================
# HILFSFUNKTIONEN
# =========================

def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, names.get(str(cls_id), str(cls_id)))
    if isinstance(names, list) and cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def resolve_image_paths(data_yaml_path, split):
    data = load_data_yaml(data_yaml_path)

    root = Path(data.get("path", Path(data_yaml_path).parent))
    if not root.is_absolute():
        root = Path(data_yaml_path).parent / root

    split_entry = data[split]
    split_path = Path(split_entry)

    if not split_path.is_absolute():
        split_path = root / split_path

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            p = Path(line.strip())
            if not p.is_absolute():
                p = root / p
            images.append(p)
        return images

    if split_path.is_dir():
        images = []
        for ext in image_extensions:
            images.extend(split_path.rglob(f"*{ext}"))
        return sorted(images)

    raise FileNotFoundError(f"Split-Pfad nicht gefunden: {split_path}")


def image_to_label_path(image_path):
    s = image_path.as_posix()
    if "/images/" not in s:
        raise ValueError(f"Bildpfad enthält keinen /images/-Ordner: {image_path}")
    return Path(s.replace("/images/", "/labels/")).with_suffix(".txt")


def order_poly(poly):
    center = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
    return poly[np.argsort(angles)]


def polygon_iou(poly1, poly2):
    poly1 = order_poly(poly1).astype(np.float32)
    poly2 = order_poly(poly2).astype(np.float32)

    area1 = abs(cv2.contourArea(poly1))
    area2 = abs(cv2.contourArea(poly2))

    if area1 <= 0 or area2 <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(poly1, poly2)
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def long_side_angle_deg(poly):
    """
    Gibt den Winkel der langen Rechteckseite zurück.
    Ergebnis: 0 bis 180 Grad.
    0° und 180° gelten als gleich.
    """
    poly = order_poly(poly)
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)

    v = edges[np.argmax(lengths)]
    angle = math.degrees(math.atan2(v[1], v[0]))

    return angle % 180


def angle_error_180_deg(pred_angle, gt_angle):
    """
    Periodischer Winkelfehler.
    Beispiel: 179° und 1° ergeben 2° Fehler, nicht 178°.
    """
    diff = abs(pred_angle - gt_angle) % 180
    return min(diff, 180 - diff)


def center(poly):
    return poly.mean(axis=0)


def read_gt_labels(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht gelesen werden: {image_path}")

    h, w = img.shape[:2]
    label_path = image_to_label_path(image_path)

    gts = []

    if not label_path.exists():
        return gts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue

        cls_id = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)

        coords[:, 0] *= w
        coords[:, 1] *= h

        gts.append({
            "cls": cls_id,
            "poly": coords,
            "angle": long_side_angle_deg(coords),
            "center": center(coords),
        })

    return gts


# =========================
# AUSWERTUNG
# =========================

def main():
    data = load_data_yaml(DATA_YAML)
    names = data.get("names", {})

    model = YOLO(MODEL_PATH)
    image_paths = resolve_image_paths(DATA_YAML, SPLIT)

    rows = []
    total_predictions = 0
    matched_predictions = 0

    for image_path in image_paths:
        gts = read_gt_labels(image_path)

        result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )[0]

        preds = []

        if result.obb is not None and result.obb.data.shape[0] > 0:
            pred_polys = result.obb.xyxyxyxy.cpu().numpy()
            pred_cls = result.obb.cls.cpu().numpy().astype(int)
            pred_conf = result.obb.conf.cpu().numpy()

            for poly, cls_id, conf in zip(pred_polys, pred_cls, pred_conf):
                preds.append({
                    "cls": int(cls_id),
                    "conf": float(conf),
                    "poly": poly.astype(np.float32),
                    "angle": long_side_angle_deg(poly),
                    "center": center(poly),
                })

        total_predictions += len(preds)
        used_pred_indices = set()

        for gt in gts:
            best_iou = 0.0
            best_idx = None

            for i, pred in enumerate(preds):
                if i in used_pred_indices:
                    continue

                if pred["cls"] != gt["cls"]:
                    continue

                iou = polygon_iou(gt["poly"], pred["poly"])

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            class_name = get_class_name(names, gt["cls"])

            if best_idx is None or best_iou < MATCH_IOU:
                rows.append({
                    "image": str(image_path),
                    "class": class_name,
                    "matched": False,
                    "iou": best_iou,
                    "conf": "",
                    "gt_angle_deg": gt["angle"],
                    "pred_angle_deg": "",
                    "angle_error_deg": "",
                    "center_error_px": "",
                    "center_error_mm": "",
                })
                continue

            pred = preds[best_idx]
            used_pred_indices.add(best_idx)
            matched_predictions += 1

            angle_error = angle_error_180_deg(pred["angle"], gt["angle"])
            center_error_px = float(np.linalg.norm(pred["center"] - gt["center"]))
            center_error_mm = center_error_px * MM_PER_PIXEL if MM_PER_PIXEL is not None else ""

            rows.append({
                "image": str(image_path),
                "class": class_name,
                "matched": True,
                "iou": best_iou,
                "conf": pred["conf"],
                "gt_angle_deg": gt["angle"],
                "pred_angle_deg": pred["angle"],
                "angle_error_deg": angle_error,
                "center_error_px": center_error_px,
                "center_error_mm": center_error_mm,
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matched_rows = [r for r in rows if r["matched"]]
    missed_rows = [r for r in rows if not r["matched"]]

    print("\n===== GESAMT =====")
    print(f"Bilder: {len(image_paths)}")
    print(f"GT-Objekte: {len(rows)}")
    print(f"Gematcht: {len(matched_rows)}")
    print(f"Nicht erkannt / falsch gematcht: {len(missed_rows)}")
    print(f"Predictions gesamt: {total_predictions}")
    print(f"Unmatched Predictions grob: {total_predictions - matched_predictions}")
    print(f"CSV gespeichert unter: {OUT_CSV}")

    if matched_rows:
        angle_errors = np.array([r["angle_error_deg"] for r in matched_rows], dtype=float)
        center_errors = np.array([r["center_error_px"] for r in matched_rows], dtype=float)

        print("\n===== WINKELFEHLER =====")
        print(f"Mean:   {angle_errors.mean():.3f}°")
        print(f"Median: {np.median(angle_errors):.3f}°")
        print(f"P95:    {np.percentile(angle_errors, 95):.3f}°")
        print(f"Max:    {angle_errors.max():.3f}°")

        print("\n===== MITTELPUNKTFEHLER =====")
        print(f"Mean:   {center_errors.mean():.3f} px")
        print(f"Median: {np.median(center_errors):.3f} px")
        print(f"P95:    {np.percentile(center_errors, 95):.3f} px")
        print(f"Max:    {center_errors.max():.3f} px")

        by_class = defaultdict(list)
        for r in matched_rows:
            by_class[r["class"]].append(r)

        print("\n===== PRO KLASSE =====")
        for cls_name, cls_rows in by_class.items():
            cls_angle = np.array([r["angle_error_deg"] for r in cls_rows], dtype=float)
            cls_center = np.array([r["center_error_px"] for r in cls_rows], dtype=float)

            print(
                f"{cls_name}: "
                f"n={len(cls_rows)}, "
                f"angle_mean={cls_angle.mean():.3f}°, "
                f"angle_median={np.median(cls_angle):.3f}°, "
                f"center_mean={cls_center.mean():.3f}px"
            )


if __name__ == "__main__":
    main()
