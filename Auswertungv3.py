
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
# HIER ANPASSEN
# ============================================================

MODEL_PATH = r"C:\FHWS\nanoplus\YOLO VS Code\runs\obb\train-2\weights\best.pt"
DATA_YAML = r"C:\FHWS\nanoplus\YOLO VS Code\dataset_yolo8_obb\data.yaml"

SPLIT = "val"              # "val" oder "test"
IMGSZ = 960                # gleiche Bildgröße wie beim Training/Vergleich
CONF = 0.50                # Mindest-Confidence für Predictions
BATCH = 2                  # höher = schneller, aber mehr VRAM/RAM
DEVICE = "cpu"             # GPU: 0, CPU: "cpu"

MATCH_IOU_FOR_ERROR = 0.50 # Matching-Schwelle für Winkel-/Mittelpunktfehler

DETAIL_CSV = "angle_eval_details.csv"
SUMMARY_CSV = "angle_eval_summary.csv"

SAVE_DEBUG_IMAGES = True
DEBUG_DIR = "debug_obb"
DEBUG_MAX_IMAGES = 30              # None = alle Debug-Bilder speichern
DEBUG_MIN_CENTER_ERROR_PX = None   # z.B. 20.0; None = keine Untergrenze


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


def resolve_image_paths(data_yaml_path, split):
    data = load_data_yaml(data_yaml_path)

    root = Path(data.get("path", Path(data_yaml_path).parent))
    if not root.is_absolute():
        root = Path(data_yaml_path).parent / root

    split_entry = data[split]
    if not isinstance(split_entry, list):
        split_entry = [split_entry]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
            for ext in exts:
                images.extend(p.rglob(f"*{ext}"))

        else:
            raise FileNotFoundError(f"Split-Pfad nicht gefunden: {p}")

    return sorted(set(images))


def image_to_label_path(image_path):
    s = Path(image_path).as_posix()

    if "/images/" not in s:
        raise ValueError(f"Bildpfad enthält keinen /images/-Ordner: {image_path}")

    return Path(s.replace("/images/", "/labels/")).with_suffix(".txt")


def get_image_size(image_path):
    """
    Schneller als cv2.imread, weil nur der Bild-Header gelesen wird.
    Rückgabe: width, height
    """
    if Image is not None:
        with Image.open(image_path) as im:
            return im.size

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht gelesen werden: {image_path}")

    h, w = img.shape[:2]
    return w, h


def order_poly(poly):
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    c = poly.mean(axis=0)

    angles = np.arctan2(
        poly[:, 1] - c[1],
        poly[:, 0] - c[0]
    )

    return poly[np.argsort(angles)]


def polygon_iou(poly1, poly2):
    p1 = order_poly(poly1)
    p2 = order_poly(poly2)

    area1 = abs(cv2.contourArea(p1))
    area2 = abs(cv2.contourArea(p2))

    if area1 <= 0 or area2 <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def center(poly):
    return np.asarray(poly, dtype=np.float32).reshape(4, 2).mean(axis=0)


def long_side_angle_deg(poly):
    """
    Winkel der langen Rechteckseite.
    Ergebnisbereich: [0, 180)
    Dadurch sind 0° und 180° automatisch gleichwertig.
    """
    p = order_poly(poly)

    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)

    v = edges[int(np.argmax(lengths))]
    angle = math.degrees(math.atan2(v[1], v[0]))

    return angle % 180


def angle_error_180_deg(pred_angle, gt_angle):
    """
    Periodischer Winkelfehler für OBBs.
    Beispiel: 179° und 1° ergeben 2° Fehler, nicht 178°.
    """
    diff = abs(pred_angle - gt_angle) % 180
    return min(diff, 180 - diff)


def read_gt_labels(image_path):
    w, h = get_image_size(image_path)
    label_path = image_to_label_path(image_path)

    gts = []

    if not label_path.exists():
        return gts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) != 9:
            continue

        cls_id = int(float(parts[0]))

        coords = np.array(
            [float(x) for x in parts[1:]],
            dtype=np.float32
        ).reshape(4, 2)

        coords[:, 0] *= w
        coords[:, 1] *= h

        gts.append({
            "cls": cls_id,
            "poly": coords,
            "angle": long_side_angle_deg(coords),
            "center": center(coords),
        })

    return gts


def extract_preds_from_result(result):
    preds = []

    if result.obb is None or len(result.obb) == 0:
        return preds

    pred_polys = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    pred_cls = result.obb.cls.cpu().numpy().astype(int)
    pred_conf = result.obb.conf.cpu().numpy()

    for poly, cls_id, conf in zip(pred_polys, pred_cls, pred_conf):
        poly = poly.astype(np.float32)

        preds.append({
            "cls": int(cls_id),
            "conf": float(conf),
            "poly": poly,
            "angle": long_side_angle_deg(poly),
            "center": center(poly),
        })

    return preds


# ============================================================
# AP / mAP BERECHNUNG
# ============================================================

def voc_ap(recalls, precisions):
    """
    Kontinuierliche AP-Berechnung mit Precision Envelope.
    """
    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]

    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_for_class(preds_cls, gt_polys_by_image, iou_thr):
    """
    preds_cls:
        Liste von Predictions einer Klasse über alle Bilder.

    gt_polys_by_image:
        dict image_key -> list[gt_poly]
    """
    n_gt = sum(len(v) for v in gt_polys_by_image.values())

    if n_gt == 0:
        return None

    if len(preds_cls) == 0:
        return 0.0

    preds_sorted = sorted(preds_cls, key=lambda x: x["conf"], reverse=True)

    used = {
        img_key: np.zeros(len(polys), dtype=bool)
        for img_key, polys in gt_polys_by_image.items()
    }

    tp = np.zeros(len(preds_sorted), dtype=np.float32)
    fp = np.zeros(len(preds_sorted), dtype=np.float32)

    for i, pred in enumerate(preds_sorted):
        img_key = pred["image_key"]
        gt_polys = gt_polys_by_image.get(img_key, [])

        best_iou = 0.0
        best_j = -1

        for j, gt_poly in enumerate(gt_polys):
            if used[img_key][j]:
                continue

            iou = polygon_iou(pred["poly"], gt_poly)

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
# MATCHING FÜR WINKEL-/MITTELPUNKTFEHLER
# ============================================================

def match_for_geometry_errors(gt_by_image, pred_by_image, names):
    detail_rows = []
    debug_rows = []

    total_gt = 0
    matched_count = 0
    total_preds = sum(len(v) for v in pred_by_image.values())

    used_pred_indices = defaultdict(set)

    for image_key, gt_list in gt_by_image.items():
        pred_list = pred_by_image.get(image_key, [])
        total_gt += len(gt_list)

        for gt in gt_list:
            best_iou = 0.0
            best_pred_idx = None

            for pred_idx, pred in enumerate(pred_list):
                if pred_idx in used_pred_indices[image_key]:
                    continue

                if pred["cls"] != gt["cls"]:
                    continue

                iou = polygon_iou(gt["poly"], pred["poly"])

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            class_name = get_class_name(names, gt["cls"])

            if best_pred_idx is None or best_iou < MATCH_IOU_FOR_ERROR:
                detail_rows.append({
                    "image": gt["image_path"],
                    "class_id": gt["cls"],
                    "class": class_name,
                    "matched": False,
                    "iou": best_iou,
                    "conf": "",
                    "gt_angle_deg": gt["angle"],
                    "pred_angle_deg": "",
                    "angle_error_deg": "",
                    "center_error_px": "",
                })
                continue

            pred = pred_list[best_pred_idx]
            used_pred_indices[image_key].add(best_pred_idx)
            matched_count += 1

            angle_error = angle_error_180_deg(pred["angle"], gt["angle"])
            center_error_px = float(np.linalg.norm(pred["center"] - gt["center"]))

            detail_rows.append({
                "image": gt["image_path"],
                "class_id": gt["cls"],
                "class": class_name,
                "matched": True,
                "iou": best_iou,
                "conf": pred["conf"],
                "gt_angle_deg": gt["angle"],
                "pred_angle_deg": pred["angle"],
                "angle_error_deg": angle_error,
                "center_error_px": center_error_px,
            })

            debug_rows.append({
                "image_key": image_key,
                "image_path": gt["image_path"],
                "class": class_name,
                "iou": best_iou,
                "angle_error_deg": angle_error,
                "center_error_px": center_error_px,
                "gt_poly": gt["poly"],
                "pred_poly": pred["poly"],
                "gt_center": gt["center"],
                "pred_center": pred["center"],
            })

    used_total = sum(len(s) for s in used_pred_indices.values())
    unmatched_preds = total_preds - used_total

    return detail_rows, debug_rows, total_gt, matched_count, total_preds, unmatched_preds


# ============================================================
# SUMMARY
# ============================================================

def build_summary(detail_rows, preds_by_class, gt_by_class_image, names):
    class_ids = sorted(set(gt_by_class_image.keys()) | set(preds_by_class.keys()))
    summary = []

    ap50_values = []
    ap75_values = []

    for cls_id in class_ids:
        cls_name = get_class_name(names, cls_id)

        ap50 = compute_ap_for_class(
            preds_by_class.get(cls_id, []),
            gt_by_class_image.get(cls_id, {}),
            0.50
        )

        ap75 = compute_ap_for_class(
            preds_by_class.get(cls_id, []),
            gt_by_class_image.get(cls_id, {}),
            0.75
        )

        matched = [
            r for r in detail_rows
            if r["class_id"] == cls_id and r["matched"]
        ]

        if matched:
            ious = np.array([r["iou"] for r in matched], dtype=float)
            angle_errors = np.array([r["angle_error_deg"] for r in matched], dtype=float)
            center_errors = np.array([r["center_error_px"] for r in matched], dtype=float)

            mean_iou = float(ious.mean())
            mean_angle = float(angle_errors.mean())
            median_angle = float(np.median(angle_errors))
            max_angle = float(angle_errors.max())
            mean_center_px = float(center_errors.mean())
            max_center_px = float(center_errors.max())
        else:
            mean_iou = ""
            mean_angle = ""
            median_angle = ""
            max_angle = ""
            mean_center_px = ""
            max_center_px = ""

        if ap50 is not None:
            ap50_values.append(ap50)

        if ap75 is not None:
            ap75_values.append(ap75)

        summary.append({
            "Klasse": cls_name,
            "mAP50": ap50 if ap50 is not None else "",
            "mAP75": ap75 if ap75 is not None else "",
            "Intersection over Union (IoU)": mean_iou,
            "mittlerer Winkel-Fehler": mean_angle,
            "Median Winkel-Fehler": median_angle,
            "max Fehler Winkel-Fehler": max_angle,
            "Mittelpunktfehler px": mean_center_px,
            "max Mittelpunktfehler": max_center_px,
        })

    all_matched = [r for r in detail_rows if r["matched"]]

    if all_matched:
        all_ious = np.array([r["iou"] for r in all_matched], dtype=float)
        all_angles = np.array([r["angle_error_deg"] for r in all_matched], dtype=float)
        all_centers = np.array([r["center_error_px"] for r in all_matched], dtype=float)

        all_mean_iou = float(all_ious.mean())
        all_mean_angle = float(all_angles.mean())
        all_median_angle = float(np.median(all_angles))
        all_max_angle = float(all_angles.max())
        all_mean_center_px = float(all_centers.mean())
        all_max_center_px = float(all_centers.max())
    else:
        all_mean_iou = ""
        all_mean_angle = ""
        all_median_angle = ""
        all_max_angle = ""
        all_mean_center_px = ""
        all_max_center_px = ""

    summary.insert(0, {
        "Klasse": "all",
        "mAP50": float(np.mean(ap50_values)) if ap50_values else "",
        "mAP75": float(np.mean(ap75_values)) if ap75_values else "",
        "Intersection over Union (IoU)": all_mean_iou,
        "mittlerer Winkel-Fehler": all_mean_angle,
        "Median Winkel-Fehler": all_median_angle,
        "max Fehler Winkel-Fehler": all_max_angle,
        "Mittelpunktfehler px": all_mean_center_px,
        "max Mittelpunktfehler": all_max_center_px,
    })

    return summary


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=3):
    if x == "" or x is None:
        return "-"

    if isinstance(x, bool):
        return str(x)

    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"

    return str(x)


def print_summary_table(summary):
    headers = [
        "Klasse",
        "mAP50",
        "mAP75",
        "IoU",
        "Winkel Mean",
        "Winkel Median",
        "Winkel Max",
        "Center px",
        "Center Max",
    ]

    keys = [
        "Klasse",
        "mAP50",
        "mAP75",
        "Intersection over Union (IoU)",
        "mittlerer Winkel-Fehler",
        "Median Winkel-Fehler",
        "max Fehler Winkel-Fehler",
        "Mittelpunktfehler px",
        "max Mittelpunktfehler",
    ]

    optimal_row = [
        "Optimal",
        "1.000",
        "1.000",
        "1.000",
        "0.000°",
        "0.000°",
        "0.000°",
        "0.000 px",
        "0.000 px",
    ]

    table = [optimal_row]

    for row in summary:
        table.append([fmt(row[k]) for k in keys])

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in table))
        for i in range(len(headers))
    ]

    line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    optimal = " | ".join(optimal_row[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(line)
    print(optimal)
    print(sep)

    for r in table[1:]:
        print(" | ".join(r[i].ljust(widths[i]) for i in range(len(headers))))


# ============================================================
# DEBUG-BILDER
# ============================================================

def save_debug_images(debug_rows):
    if not SAVE_DEBUG_IMAGES or not debug_rows:
        return 0

    out_dir = Path(DEBUG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    for r in debug_rows:
        if DEBUG_MIN_CENTER_ERROR_PX is not None:
            if r["center_error_px"] < DEBUG_MIN_CENTER_ERROR_PX:
                continue

        grouped[r["image_key"]].append(r)

    if not grouped:
        return 0

    image_keys = sorted(
        grouped.keys(),
        key=lambda k: max(r["center_error_px"] for r in grouped[k]),
        reverse=True
    )

    if DEBUG_MAX_IMAGES is not None:
        image_keys = image_keys[:DEBUG_MAX_IMAGES]

    saved = 0

    for image_key in image_keys:
        rows = grouped[image_key]
        image_path = rows[0]["image_path"]

        debug_img = cv2.imread(str(image_path))

        if debug_img is None:
            continue

        for r in rows:
            gt_poly = order_poly(r["gt_poly"]).astype(int)
            pred_poly = order_poly(r["pred_poly"]).astype(int)

            cv2.polylines(
                debug_img,
                [gt_poly],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.polylines(
                debug_img,
                [pred_poly],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2
            )

            gt_c = r["gt_center"].astype(int)
            pred_c = r["pred_center"].astype(int)

            cv2.circle(debug_img, tuple(gt_c), 5, (0, 255, 0), -1)
            cv2.circle(debug_img, tuple(pred_c), 5, (0, 0, 255), -1)

            cv2.line(debug_img, tuple(gt_c), tuple(pred_c), (255, 0, 0), 1)

            text = (
                f'{r["class"]} | IoU {r["iou"]:.2f} | '
                f'A {r["angle_error_deg"]:.2f} deg | '
                f'C {r["center_error_px"]:.1f}px'
            )

            text_pos = (int(pred_c[0]) + 8, int(pred_c[1]) - 8)

            cv2.putText(
                debug_img,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        out_path = out_dir / f"debug_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(out_path), debug_img)
        saved += 1

    return saved


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.perf_counter()

    data = load_data_yaml(DATA_YAML)
    names = data.get("names", {})

    image_paths = resolve_image_paths(DATA_YAML, SPLIT)

    print(f"Bilder gefunden: {len(image_paths)}")
    print("Lese Ground-Truth-Labels ...")

    gt_by_image = defaultdict(list)
    gt_by_class_image = defaultdict(lambda: defaultdict(list))

    for image_path in image_paths:
        key = norm_path(image_path)
        gts = read_gt_labels(image_path)

        for gt in gts:
            gt["image_key"] = key
            gt["image_path"] = str(image_path)

            gt_by_image[key].append(gt)
            gt_by_class_image[gt["cls"]][key].append(gt["poly"])

    print("Starte YOLO-Inferenz ...")

    model = YOLO(MODEL_PATH)

    pred_by_image = defaultdict(list)
    preds_by_class = defaultdict(list)

    source_list = [str(p) for p in image_paths]

    results = model.predict(
        source=source_list,
        imgsz=IMGSZ,
        conf=CONF,
        batch=BATCH,
        device=DEVICE,
        stream=True,
        verbose=False
    )

    for image_path, result in zip(image_paths, results):
        # Wichtig:
        # Wir verwenden hier bewusst den originalen image_path aus image_paths
        # und NICHT result.path, weil result.path bei Batch-Inferenz anders sein kann.
        key = norm_path(image_path)

        preds = extract_preds_from_result(result)

        for pred in preds:
            pred["image_key"] = key

            pred_by_image[key].append(pred)
            preds_by_class[pred["cls"]].append(pred)

    print("Berechne Fehler und mAP ...")

    detail_rows, debug_rows, total_gt, matched_count, total_preds, unmatched_preds = match_for_geometry_errors(
        gt_by_image,
        pred_by_image,
        names
    )

    summary = build_summary(
        detail_rows,
        preds_by_class,
        gt_by_class_image,
        names
    )

    save_csv(DETAIL_CSV, detail_rows)
    save_csv(SUMMARY_CSV, summary)

    saved_debug = save_debug_images(debug_rows)

    runtime = time.perf_counter() - t0
    missed = total_gt - matched_count

    print("\n===== GESAMT =====")
    print(f"Bilder: {len(image_paths)}")
    print(f"GT-Objekte: {total_gt}")
    print(f"Gematcht: {matched_count}")
    print(f"Nicht erkannt / falsch gematcht: {missed}")
    print(f"Predictions gesamt: {total_preds}")
    print(f"Unmatched Predictions grob: {unmatched_preds}")
    print(f"Detail-CSV gespeichert unter: {DETAIL_CSV}")
    print(f"Summary-CSV gespeichert unter: {SUMMARY_CSV}")
    print(f"Debug-Bilder gespeichert: {saved_debug} in '{DEBUG_DIR}'")
    print(f"Laufzeit: {runtime:.2f} s")

    print("\n===== SUMMARY =====")
    print_summary_table(summary)


if __name__ == "__main__":
    main()
