#!/usr/bin/env python3
"""
COCO evaluation script with YOLO/COCO-aligned settings.

Features:
- Multi-IoU evaluation (default: 0.50–0.95 @ 0.05)
- Per-class Precision / Recall / F1
- Per-IoU KPI breakdown
- JSON export only
"""

import argparse
import json
import os
from typing import Dict, List
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
DEFAULT_WORKDIR = "./tmp_eval"


# ------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------


class CocoEvaluator:
    """COCO evaluator with per-class and per-IoU KPIs."""

    def __init__(self, boxlog_path: str) -> None:
        self.boxlog_path = boxlog_path

        parts = Path(boxlog_path).parts
        if "predictions" not in parts:
            raise ValueError(f"'predictions' not found in json file path", boxlog_path)

        split = parts.index("predictions")
        self.ds_root = Path(*parts[:split])
        self.ds_pred = Path(*parts[split + 1 :])

        os.makedirs(self.workdir, exist_ok=True)

        self.coco_gt = None
        self.coco_dt = None
        self.evaluator = None

    # --------------------------------------------------------

    def prepare_data(self) -> None:
        """Normalize prediction format and load COCO files."""
        with open(self.gt_path) as f:
            gt_data = json.load(f)

        with open(self.dt_path) as f:
            raw_dt = json.load(f)

        dt_data = raw_dt.get("inference", raw_dt)

        self.coco_gt = COCO()
        self.coco_gt.dataset = gt_data
        self.coco_gt.createIndex()

        self.coco_dt = self.coco_gt.loadRes(dt_data)

    # --------------------------------------------------------

    def evaluate(self) -> None:
        """Run COCO evaluation."""
        self.evaluator = COCOeval(self.coco_gt, self.coco_dt, "bbox")
        self.evaluator.params.iouThrs = self.iou_thrs
        self.evaluator.params.maxDets = [self.max_dets]
        self.evaluator.params.useCats = True

        self.evaluator.evaluate()
        self.evaluator.accumulate()

    # --------------------------------------------------------
    # Per-class averaged KPIs
    # --------------------------------------------------------

    def compute_per_class_kpis(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class KPIs averaged over IoUs."""

        cat_ids = self.coco_gt.getCatIds()
        cat_map = {c["id"]: c["name"] for c in self.coco_gt.loadCats(cat_ids)}

        precision = self.evaluator.eval["precision"]  # [T,R,K,A,M]
        recall = self.evaluator.eval["recall"]  # [T,K,A,M]

        results = {}

        for idx, cat_id in enumerate(cat_ids):
            p = precision[:, :, idx, 0, -1]
            p = p[p >= 0]
            prec = float(p.mean()) if len(p) else 0.0

            r = recall[:, idx, 0, -1]
            rec = float(np.mean(r[r >= 0])) if np.any(r >= 0) else 0.0

            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

            results[cat_map[cat_id]] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }

        return results

    # --------------------------------------------------------
    # Per-class per-IoU breakdown
    # --------------------------------------------------------

    def compute_per_class_per_iou_kpis(self) -> Dict:
        """Compute per-class KPIs for each IoU threshold."""

        cat_ids = self.coco_gt.getCatIds()
        cat_map = {c["id"]: c["name"] for c in self.coco_gt.loadCats(cat_ids)}

        precision = self.evaluator.eval["precision"]
        recall = self.evaluator.eval["recall"]

        output = {
            "summary": {"iou_thrs": [float(f"{i:.2f}") for i in self.iou_thrs]},
            "per_class": {},
        }

        for cat_idx, cat_id in enumerate(cat_ids):
            class_name = cat_map[cat_id]

            per_iou = {}
            prec_vals = []
            rec_vals = []

            for iou_idx, iou in enumerate(self.iou_thrs):
                p = precision[iou_idx, :, cat_idx, 0, -1]
                p = p[p >= 0]

                r = recall[iou_idx, cat_idx, 0, -1]

                prec = float(p.mean()) if len(p) else 0.0
                rec = float(r) if r >= 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

                per_iou[f"{iou:.2f}"] = {
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }

                prec_vals.append(prec)
                rec_vals.append(rec)

            avg_prec = float(np.mean(prec_vals)) if prec_vals else 0.0
            avg_rec = float(np.mean(rec_vals)) if rec_vals else 0.0
            avg_f1 = (
                2 * avg_prec * avg_rec / (avg_prec + avg_rec)
                if (avg_prec + avg_rec)
                else 0.0
            )

            output["per_class"][class_name] = {
                "avg": {
                    "precision": avg_prec,
                    "recall": avg_rec,
                    "f1": avg_f1,
                },
                "per_iou": per_iou,
            }

        return output

    # --------------------------------------------------------

    def export_json(self, data: Dict) -> None:
        path = f"{self.export_prefix}_per_iou.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✔ Per‑IoU KPIs saved to {path}")

    # --------------------------------------------------------

    @staticmethod
    def print_table(kpis: Dict[str, Dict[str, float]]) -> None:
        table = [
            [
                cls,
                f"{vals['precision']:.3f}",
                f"{vals['recall']:.3f}",
                f"{vals['f1']:.3f}",
            ]
            for cls, vals in kpis.items()
        ]

        print("\nPer‑class KPIs (averaged over IoUs):")
        print(
            tabulate(
                table,
                headers=["Class", "Precision", "Recall", "F1‑Score"],
                tablefmt="github",
            )
        )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COCO evaluation with per‑IoU KPI export"
    )

    parser.add_argument("--boxlog", required=True, help="Predictions JSON File")
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main() -> None:
    args = parse_args()

    evaluator = CocoEvaluator(boxlog_path=args.boxlog)
    evaluator.prepare_data()
    evaluator.evaluate()

    avg_kpis = evaluator.compute_per_class_kpis()
    per_iou_kpis = evaluator.compute_per_class_per_iou_kpis()

    evaluator.print_table(avg_kpis)
    evaluator.export_json(per_iou_kpis)


if __name__ == "__main__":
    main()
