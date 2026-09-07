"""Single-command FaceChain pipeline entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from blockchain import BlockchainError, LocalEthereumVerifier
from face_pipeline import FaceEncoder, FacePipelineError
from reverse_search import ReverseSearchError, search_by_image


def canonical_json(record: dict[str, Any]) -> str:
    """Deterministic serialization used for every fingerprint calculation."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_record(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def build_record(image_path: Path, face_metadata: dict[str, Any], search_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "face_search_result",
        "input_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "face": face_metadata,
        "search_provider": "Google Lens via SerpApi",
        "matched_result": {
            "url": search_result["url"],
            "platform": search_result["platform"],
            "source": search_result["source"],
            "title": search_result["title"],
            "snippet": search_result["snippet"],
            "match_type": search_result["match_type"],
            "result_type": search_result["result_type"],
            "result_section": search_result["result_section"],
            "post_url_validation": search_result["post_url_validation"],
            "thumbnail_url": search_result["thumbnail_url"],
            "matched_image_url": search_result["matched_image_url"],
            "image_similarity": search_result["image_similarity"],
        },
    }


def save_artifact(record: dict[str, Any], fingerprint: str, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = artifacts_dir / f"record_{timestamp}_{fingerprint[:12]}.json"
    payload = {"record": record, "canonical_sha256": fingerprint}
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face scan -> live reverse-image search -> Ethereum verification")
    parser.add_argument("image", type=Path, help="Path to an image containing a face you are authorized to process")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Directory for the canonical JSON proof")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"ERROR: Input image does not exist: {image_path}", file=sys.stderr)
        return 2

    try:
        print("[1/4] Detecting and encoding face...")
        face = FaceEncoder().encode(image_path)
        print(f"      Detected {face.face_count} face(s); selected {face.embedding_dimension}-D embedding.")

        print("[2/4] Running genuine reverse-image search...")
        match = search_by_image(image_path)
        print(f"      Live {match['match_type']} result selected from {match['source']}.")

        record = build_record(image_path, face.metadata(), match)
        local_hash = fingerprint_record(record)
        artifact_path = save_artifact(record, local_hash, args.artifacts_dir)

        print("[3/4] Writing fingerprint to blockchain...")
        verifier = LocalEthereumVerifier()
        receipt = verifier.write_fingerprint(local_hash)

        print("[4/4] Re-verifying on-chain record...")
        saved = json.loads(artifact_path.read_text(encoding="utf-8"))
        recomputed_hash = fingerprint_record(saved["record"])
        verified, on_chain_hash = verifier.verify(receipt.transaction_hash, recomputed_hash)

    except (FacePipelineError, ReverseSearchError, BlockchainError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n================ VERIFICATION ================")
    print(f"Search result: {match['url']}")
    print(f"Platform: {match['platform']}")
    print(f"Match type: {match['match_type']}")
    print(f"Result type: {match['result_type']}")
    print(f"Transaction hash: {receipt.transaction_hash}")
    print(f"Block number: {receipt.block_number}")
    print(f"Local SHA-256: {recomputed_hash}")
    print(f"On-chain SHA-256: {on_chain_hash}")
    print(f"Proof artifact: {artifact_path.resolve()}")
    print(f"VERIFIED: {verified}")
    print("================================================")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(run())
