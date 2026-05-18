#!/usr/bin/env python3
"""
COCO evaluation script with YOLO/COCO-aligned settings.

Features:
- Multi-IoU evaluation (default: 0.50–0.95 @ 0.05)
- Per-class Precision / Recall / F1
- Per-IoU KPI breakdown
- Production-ready JSON export with runtime metrics
- Optimized vectorized numpy operations
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------

DEFAULT_IOU_THR = np.arange(0.50, 0.96, 0.05)
DEFAULT_MAX_DETS = 300
DEFAULT_CONFIDENCE_THR = 0.0
DEFAULT_COUNTS_IOU_REF = 0.50

# ------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------


class CocoEvaluator:
    """COCO evaluator with per-class and per-IoU KPIs (optimized with numpy)."""

    def __init__(self, json_log: str) -> None:
        self.json_log = json_log

        self.coco_gt = None
        self.coco_dt = None
        self.evaluator = None
        self.iou_thrs = DEFAULT_IOU_THR
        self.max_dets = DEFAULT_MAX_DETS

        self.predictions = None
        self.annotation_file = None
        self.evaluation_file = None
        self.cat_ids = None
        self.cat_map = None

    # ========================================================
    # Core Loading & Evaluation
    # ========================================================

    def prepare_data(self) -> None:
        """Load COCO annotations and predictions."""
        with open(self.json_log) as f:
            raw_dt = json.load(f)

        metadata = raw_dt.get("metadata", {})
        self.annotation_file = metadata.get("annotation_file", "<unknown>")
        self.evaluation_file = metadata.get("evaluation_file", "<unknown>")

        self.coco_gt = COCO()
        with open(self.annotation_file) as f:
            self.coco_gt.dataset = json.load(f)
            self.coco_gt.createIndex()

        self.cat_ids = self.coco_gt.getCatIds()
        self.cat_map = {c["id"]: c["name"] for c in self.coco_gt.loadCats(self.cat_ids)}

        # Extract predictions from JSON structure (support both "inference" and "predictions" keys)
        self.predictions = raw_dt.get("inference", raw_dt.get("predictions", []))
        if not isinstance(self.predictions, list):
            raise ValueError(
                f"Predictions not found or not a list. Got: {type(self.predictions)}"
            )

        self.coco_dt = self.coco_gt.loadRes(self.predictions)

    def evaluate(self) -> None:
        """Run COCO evaluation."""
        self.evaluator = COCOeval(self.coco_gt, self.coco_dt, "bbox")
        self.evaluator.params.iouThrs = self.iou_thrs
        self.evaluator.params.maxDets = [self.max_dets]
        self.evaluator.params.useCats = True

        self.evaluator.evaluate()
        self.evaluator.accumulate()

    # ========================================================
    # Vectorized KPI Computation (Numpy-optimized)
    # ========================================================

    def _compute_metrics_from_arrays(
        self, prec: np.ndarray, rec: np.ndarray
    ) -> Tuple[float, float, float]:
        """Efficiently compute precision, recall, and F1 from arrays."""
        prec = prec[prec >= 0]
        rec = rec[rec >= 0]

        p = float(np.mean(prec)) if len(prec) > 0 else 0.0
        r = float(np.mean(rec)) if len(rec) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return p, r, f1

    def _compute_tp_fp_fn_cube(self, num_classes: int, num_ious: int) -> np.ndarray:
        """Compute TP/FP/FN per [class, iou] from COCOeval evalImgs internals."""
        counts = np.zeros((num_classes, num_ious, 3), dtype=np.int64)  # tp, fp, fn
        cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        area_all = tuple(self.evaluator.params.areaRng[0])

        for eval_img in self.evaluator.evalImgs:
            if not eval_img:
                continue

            # Use canonical COCO slice only: area=all and configured maxDet.
            # Otherwise, entries are duplicated across area buckets.
            if tuple(eval_img.get("aRng", ())) != area_all:
                continue
            if int(eval_img.get("maxDet", -1)) != int(self.max_dets):
                continue

            cat_id = eval_img.get("category_id")
            if cat_id not in cat_to_idx:
                continue

            class_idx = cat_to_idx[cat_id]

            dt_matches = np.asarray(eval_img.get("dtMatches", []))
            dt_ignore = np.asarray(eval_img.get("dtIgnore", []), dtype=bool)
            gt_matches = np.asarray(eval_img.get("gtMatches", []))
            gt_ignore = np.asarray(eval_img.get("gtIgnore", []), dtype=bool)

            if dt_matches.size == 0 or gt_matches.size == 0:
                continue

            for iou_idx in range(num_ious):
                dt_m = dt_matches[iou_idx]
                dt_i = dt_ignore[iou_idx] if dt_ignore.ndim > 1 else dt_ignore
                gt_m = gt_matches[iou_idx]

                tp = int(np.sum((dt_m > 0) & (~dt_i)))
                fp = int(np.sum((dt_m == 0) & (~dt_i)))
                fn = int(np.sum((gt_m == 0) & (~gt_ignore)))

                counts[class_idx, iou_idx, 0] += tp
                counts[class_idx, iou_idx, 1] += fp
                counts[class_idx, iou_idx, 2] += fn

        return counts

    def compute_detection_kpis(self) -> Dict:
        """
        Compute detection metrics with vectorized numpy operations.
        Returns comprehensive per-class, per-IoU, and aggregated metrics.
        """
        precision = self.evaluator.eval["precision"]  # [T,R,K,A,M]
        recall = self.evaluator.eval["recall"]  # [T,K,A,M]

        num_ious = len(self.iou_thrs)
        num_classes = len(self.cat_ids)

        # Pre-allocate arrays for vectorized operations
        class_metrics = np.zeros((num_classes, num_ious, 3))  # prec, rec, f1
        class_counts = self._compute_tp_fp_fn_cube(num_classes, num_ious)  # tp, fp, fn

        # Vectorized per-class, per-IoU computation
        for cat_idx in range(num_classes):
            for iou_idx in range(num_ious):
                p = precision[iou_idx, :, cat_idx, 0, -1]
                r = recall[iou_idx, cat_idx, 0, -1]

                p_clean = p[p >= 0]
                r_clean = (
                    r[r >= 0]
                    if hasattr(r, "__len__")
                    else np.array([r]) if r >= 0 else np.array([])
                )

                p_val = float(np.mean(p_clean)) if len(p_clean) > 0 else 0.0
                r_val = float(np.mean(r_clean)) if len(r_clean) > 0 else 0.0
                f1_val = (
                    (2 * p_val * r_val / (p_val + r_val))
                    if (p_val + r_val) > 0
                    else 0.0
                )

                class_metrics[cat_idx, iou_idx] = [p_val, r_val, f1_val]

        # Compute aggregates efficiently
        class_avg = np.mean(class_metrics, axis=1)  # [num_classes, 3]
        iou_avg = np.mean(class_metrics, axis=0)  # [num_ious, 3]
        overall_avg = np.mean(class_metrics, axis=(0, 1))  # [3]

        # Build output structure
        output = {
            "per_class": {},
            "avg_class": {},
        }

        # ========== per_class_per_iou: Full breakdown [class][iou] ==========
        for cat_idx, cat_id in enumerate(self.cat_ids):
            class_name = self.cat_map[cat_id]
            output["per_class"][class_name] = {
                "per_iou": {},
                "avg_iou": {},
            }

            for iou_idx, iou_thr in enumerate(self.iou_thrs):
                iou_key = f"{iou_thr:.2f}"
                tp = int(class_counts[cat_idx, iou_idx, 0])
                fp = int(class_counts[cat_idx, iou_idx, 1])
                fn = int(class_counts[cat_idx, iou_idx, 2])

                output["per_class"][class_name]["per_iou"][iou_key] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "metrics": {
                        "precision": float(class_metrics[cat_idx, iou_idx, 0]),
                        "recall": float(class_metrics[cat_idx, iou_idx, 1]),
                        "f1_score": float(class_metrics[cat_idx, iou_idx, 2]),
                    },
                }

            class_tp = int(np.sum(class_counts[cat_idx, :, 0]))
            class_fp = int(np.sum(class_counts[cat_idx, :, 1]))
            class_fn = int(np.sum(class_counts[cat_idx, :, 2]))

            output["per_class"][class_name]["avg_iou"] = {
                "tp": class_tp,
                "fp": class_fp,
                "fn": class_fn,
                "metrics": {
                    "precision": float(class_avg[cat_idx, 0]),
                    "recall": float(class_avg[cat_idx, 1]),
                    "f1_score": float(class_avg[cat_idx, 2]),
                },
            }

        # ========== avg_class: Average class performance for each IoU ==========
        for iou_idx, iou_thr in enumerate(self.iou_thrs):
            iou_key = f"{iou_thr:.2f}"
            iou_tp = int(np.sum(class_counts[:, iou_idx, 0]))
            iou_fp = int(np.sum(class_counts[:, iou_idx, 1]))
            iou_fn = int(np.sum(class_counts[:, iou_idx, 2]))

            output["avg_class"][iou_key] = {
                "tp": iou_tp,
                "fp": iou_fp,
                "fn": iou_fn,
                "metrics": {
                    "precision": float(iou_avg[iou_idx, 0]),
                    "recall": float(iou_avg[iou_idx, 1]),
                    "f1_score": float(iou_avg[iou_idx, 2]),
                },
            }

        # Keep overall aggregate as explicit entry for convenience.
        overall_tp = int(np.sum(class_counts[:, :, 0]))
        overall_fp = int(np.sum(class_counts[:, :, 1]))
        overall_fn = int(np.sum(class_counts[:, :, 2]))
        output["avg_class"]["avg"] = {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "metrics": {
                "precision": float(overall_avg[0]),
                "recall": float(overall_avg[1]),
                "f1_score": float(overall_avg[2]),
            },
        }

        return output

    def compute_runtime_kpis(self) -> Dict:
        """Compute runtime KPIs from predictions (vectorized)."""
        runtimes = np.array([elem.get("runtime_us", 0.0) for elem in self.predictions])
        valid_runtimes = runtimes[runtimes > 0]

        if len(valid_runtimes) == 0:
            return {"available": False}

        return {
            "available": True,
            "inference_time_us": {
                "mean": float(np.mean(valid_runtimes)),
                "median": float(np.median(valid_runtimes)),
                "std": float(np.std(valid_runtimes)),
                "min": float(np.min(valid_runtimes)),
                "max": float(np.max(valid_runtimes)),
                "p95": float(np.percentile(valid_runtimes, 95)),
                "p99": float(np.percentile(valid_runtimes, 99)),
            },
            "throughput_fps": {
                "mean": (
                    float(1e6 / np.mean(valid_runtimes))
                    if len(valid_runtimes) > 0
                    else 0.0
                ),
            },
        }

    # ========================================================
    # Production JSON Export
    # ========================================================

    def build_production_report(self, detection_kpis: Dict, runtime_kpis: Dict) -> Dict:
        """Build comprehensive production-ready evaluation report."""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "evaluation_config": {
                    "iou_thresholds": [float(f"{i:.2f}") for i in self.iou_thrs],
                    "max_detections": self.max_dets,
                    "annotation_file": str(self.annotation_file),
                    "num_categories": len(self.cat_ids),
                    "num_predictions": len(self.predictions),
                },
            },
            "performance": {
                "detection": detection_kpis,
                "runtime": runtime_kpis,
            },
        }
        return report

    def export_json(self, report: Dict) -> str:
        """Export report to JSON file."""
        os.makedirs(os.path.dirname(self.evaluation_file), exist_ok=True)
        with open(self.evaluation_file, "w") as f:
            json.dump(report, f, indent=2, sort_keys=False)

        file_size_kb = os.path.getsize(self.evaluation_file) / 1024
        print(f"\n✓ Report exported: {self.evaluation_file} ({file_size_kb:.1f} KB)")
        return self.evaluation_file

    # ========================================================
    # Console Output (Pretty Print)
    # ========================================================

    @staticmethod
    def print_summary(report: Dict) -> None:
        """Print concise summary table from production report."""
        detection = report["performance"]["detection"]
        per_class = detection["per_class"]

        table = []
        for class_name in sorted(per_class.keys()):
            m = per_class[class_name]["avg_iou"]["metrics"]
            table.append(
                [
                    class_name,
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1_score']:.4f}",
                ]
            )

        # Overall average row
        overall = detection["avg_class"]["avg"]["metrics"]
        table.append(
            [
                "━━ OVERALL ━━",
                f"{overall['precision']:.4f}",
                f"{overall['recall']:.4f}",
                f"{overall['f1_score']:.4f}",
            ]
        )

        print("\n" + "=" * 70)
        print("DETECTION METRICS (Averaged over IoU thresholds)")
        print("=" * 70)
        print(
            tabulate(
                table,
                headers=["Class", "Precision", "Recall", "F1-Score"],
                tablefmt="simple",
                floatfmt=".4f",
            )
        )

        # IoU threshold summary
        per_iou = {k: v for k, v in detection["avg_class"].items() if k != "avg"}
        iou_table = [
            [
                iou,
                f"{m['metrics']['precision']:.4f}",
                f"{m['metrics']['recall']:.4f}",
                f"{m['metrics']['f1_score']:.4f}",
            ]
            for iou, m in sorted(per_iou.items())
        ]
        iou_table.append(
            [
                "━━ OVERALL ━━",
                f"{overall['precision']:.4f}",
                f"{overall['recall']:.4f}",
                f"{overall['f1_score']:.4f}",
            ]
        )

        print("\n" + "=" * 70)
        print("METRICS BY IoU THRESHOLD")
        print("=" * 70)
        print(
            tabulate(
                iou_table,
                headers=["IoU Threshold", "Precision", "Recall", "F1-Score"],
                tablefmt="simple",
            )
        )

        # Runtime metrics
        runtime = report["performance"]["runtime"]
        if runtime.get("available"):
            rt = runtime["inference_time_us"]
            print("\n" + "=" * 70)
            print("RUNTIME PERFORMANCE")
            print("=" * 70)
            runtime_table = [
                ["Mean", f"{rt['mean']:.2f}", "us"],
                ["Median", f"{rt['median']:.2f}", "us"],
                ["Max", f"{rt['max']:.2f}", "us"],
                ["P95", f"{rt['p95']:.2f}", "us"],
                ["Throughput", f"{runtime['throughput_fps']['mean']:.1f}", "FPS"],
            ]
            print(
                tabulate(
                    runtime_table,
                    headers=["Metric", "Value", "Unit"],
                    tablefmt="simple",
                )
            )
            print("=" * 70 + "\n")


# ============================================================
# CLI & Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COCO evaluation with production-ready KPI export"
    )
    parser.add_argument("--boxlog", required=True, help="Predictions JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        # Initialize and run evaluation
        evaluator = CocoEvaluator(json_log=args.boxlog)
        evaluator.prepare_data()
        evaluator.evaluate()

        # Compute KPIs
        detection_kpis = evaluator.compute_detection_kpis()
        runtime_kpis = evaluator.compute_runtime_kpis()

        # Build and export report
        report = evaluator.build_production_report(detection_kpis, runtime_kpis)
        evaluator.export_json(report)
        evaluator.print_summary(report)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
