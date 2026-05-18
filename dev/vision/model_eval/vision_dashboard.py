from datetime import datetime
import json
from pathlib import Path
import os
import torch
from uuid import uuid4
from typing import Optional
from elasticsearch import Elasticsearch, helpers

# elasticsearch Version 8.7 explicitly required

"""
# search for all documents without timestap
POST /confusion-matrix/_delete_by_query
{
  "query": {
    "bool": {
      "must_not": {
        "exists": {
          "field": "timestamp"
          }
        }
      }
    }
}
"""

SERVER = "https://kibana.hco000122.mars.cloud.zf-world.com:5601/app/dashboards"

CREDENTIALS = dict(
    ES_WEBURL=os.getenv(
        "ES_WEBURL",
        "https://elasticsearch-ocp010033.apps.titan.hco000139.mars.cloud.zf-world.com:443",
    ),
    ES_APIKEY=os.getenv(
        "ES_APIKEY",
        "TmFGNklKNEI0elRfcDllVjFacHo6bFdKRklZT3BRWUtWdzFFbXVOejJQZw==",
    ),
    ES_INDEX=os.getenv("ES_INDEX", "confusion-matrix-2"),
    ES_VERIFY_CERTS=os.getenv("ES_VERIFY_CERTS", "true").lower()
    in ("1", "true", "yes", "on"),
    ES_CA_CERTS=os.getenv("ES_CA_CERTS"),
    ES_COMPATIBILITY_VERSION=os.getenv("ES_COMPATIBILITY_VERSION", "8"),
)

DEFAULT_BRANCH = os.getenv("BRANCH_NAME", "<unknown branch>")
DEFAULT_COMMIT = os.getenv("GIT_COMMIT", "<unknown commit>")
DEFAULT_BUILD = os.getenv("BUILD_NUMBER", "<unknown build>")
DEFAULT_TAG = os.getenv("BUILD_TAG", "<unknown tag>")


class Dashboard:
    def __init__(
        self,
        ES_WEBURL: str,
        ES_APIKEY: str,
        ES_INDEX: str,
        ES_VERIFY_CERTS: bool = False,
        ES_CA_CERTS: Optional[str] = None,
        ES_COMPATIBILITY_VERSION: Optional[str] = "8",
    ):
        client_kwargs = {
            "api_key": ES_APIKEY,
            "verify_certs": ES_VERIFY_CERTS,
        }

        if ES_COMPATIBILITY_VERSION:
            client_kwargs["headers"] = {
                "Accept": f"application/vnd.elasticsearch+json; compatible-with={ES_COMPATIBILITY_VERSION}",
                "Content-Type": f"application/vnd.elasticsearch+json; compatible-with={ES_COMPATIBILITY_VERSION}",
            }

        if ES_VERIFY_CERTS:
            # Prefer explicit CA path; otherwise use system trust store if available.
            ca_bundle = ES_CA_CERTS or "/etc/ssl/certs/ca-certificates.crt"
            if os.path.exists(ca_bundle):
                client_kwargs["ca_certs"] = ca_bundle

        self.el_client = Elasticsearch(ES_WEBURL, **client_kwargs)
        self.es_index = ES_INDEX

    def load_kpis(self, path: Path) -> dict:
        scores = {}
        with path.open("r") as file:
            scores = json.load(file)
        return scores

    def parse_kpis(self, eval_result: dict) -> dict:
        parsed = list()

        classes = eval_result["per_class"]

        for cls, cls_data in classes.items():
            ious = cls_data["per_iou"]
            for iou_thresh, iou_data in ious.items():
                for metric, score in iou_data.items():
                    parsed.append(
                        {
                            "kpi_metric": metric,
                            "kpi_score": score,
                            "det_class": cls,
                            "iou_thrs": float(iou_thresh),
                        }
                    )

        for cls, cls_data in classes.items():
            ious = cls_data["per_iou"]
            for iou_thresh, iou_data in ious.items():
                for metric, score in iou_data.items():
                    parsed.append(
                        {
                            "kpi_metric": metric,
                            "kpi_score": score,
                            "det_class": cls,
                            "iou_thrs": float(iou_thresh),
                        }
                    )

        return parsed

    def scores_to_docs(self, scores: dict) -> list:
        header = dict(
            {
                "timestamp": datetime.now().isoformat(),
                "identifier": str(uuid4()),
                "build": {
                    "Number": os.getenv("BUILD_NUMBER", DEFAULT_BUILD),
                    "Commit": os.getenv("GIT_COMMIT", DEFAULT_COMMIT),
                    "Branch": os.getenv("BRANCH_NAME", DEFAULT_BRANCH),
                    "Tag": os.getenv("BUILD_TAG", DEFAULT_TAG),
                },
            }
        )

        documents = self.parse_kpis(scores)

        return [
            {
                "_index": self.es_index,
                "_source": {
                    **header,
                    **content,
                },
            }
            for content in documents
        ]

    def upload_docs(self, documents: list):
        print(f"Uploading {len(documents)} document(s) to Elasticsearch...")

        if not documents:
            print("No documents to upload.")
            return False
        try:
            helpers.bulk(self.el_client, documents)
            print(f"{len(documents)} document(s) uploaded successfully.")
            return True
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return False


if __name__ == "__main__":
    dest = Path(
        "/workspaces/yolo26-hailo/res/datasets/collection/coco/evaluations/instances_val2017-2026-05-13-11-50-45.json"
    )

    board = Dashboard(**CREDENTIALS)

    kpi_result = board.load_kpis(dest)
    kpi_parsed = board.scores_to_docs(kpi_result)

    board.upload_docs(kpi_parsed)
