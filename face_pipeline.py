"""Face detection and embedding for the FaceChain CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
from PIL import Image


class FacePipelineError(RuntimeError):
    """Raised when an input image cannot be encoded as a face."""


@dataclass(frozen=True)
class FaceEncoding:
    """The embedding plus non-sensitive metadata needed by the pipeline."""

    embedding: list[float]
    face_count: int
    confidence: float
    bounding_box: list[float]

    @property
    def embedding_dimension(self) -> int:
        return len(self.embedding)

    def metadata(self) -> dict[str, Any]:
        return {
            "face_count": self.face_count,
            "selected_face_confidence": round(self.confidence, 6),
            "selected_face_bounding_box": [round(value, 2) for value in self.bounding_box],
            "embedding_dimension": self.embedding_dimension,
        }


class FaceEncoder:
    """MTCNN detector and FaceNet (InceptionResnetV1) embedding model."""

    def __init__(self) -> None:
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.detector = MTCNN(keep_all=True, device=self.device)
            # The weights are downloaded by facenet-pytorch on the first run if absent.
            self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        except Exception as exc:
            raise FacePipelineError(
                "Could not initialize MTCNN/FaceNet. Check the package installation "
                "and network access for the first model-weights download."
            ) from exc

    def encode(self, image_path: str | Path) -> FaceEncoding:
        path = Path(image_path)
        if not path.is_file():
            raise FacePipelineError(f"Input image does not exist: {path}")

        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except (OSError, ValueError) as exc:
            raise FacePipelineError(f"Could not read image '{path}': {exc}") from exc

        boxes, probabilities = self.detector.detect(image)
        if boxes is None or probabilities is None:
            raise FacePipelineError("No face was detected in the input image.")

        valid_faces = [
            (box, float(probability))
            for box, probability in zip(boxes, probabilities)
            if box is not None and probability is not None and np.isfinite(probability)
        ]
        if not valid_faces:
            raise FacePipelineError("No face was detected with a usable confidence score.")

        # Prefer confidence; use face area as a deterministic tie-breaker.
        box, confidence = max(
            valid_faces,
            key=lambda item: (item[1], max(0.0, item[0][2] - item[0][0]) * max(0.0, item[0][3] - item[0][1])),
        )
        crop = self._crop(image, box)
        face_tensor = self._face_tensor(crop)

        with torch.inference_mode():
            embedding = self.embedder(face_tensor).squeeze(0).cpu().tolist()

        return FaceEncoding(
            embedding=embedding,
            face_count=len(valid_faces),
            confidence=confidence,
            bounding_box=[float(value) for value in box],
        )

    @staticmethod
    def _crop(image: Image.Image, box: np.ndarray) -> Image.Image:
        left, top, right, bottom = box.tolist()
        left = max(0, int(np.floor(left)))
        top = max(0, int(np.floor(top)))
        right = min(image.width, int(np.ceil(right)))
        bottom = min(image.height, int(np.ceil(bottom)))
        if right <= left or bottom <= top:
            raise FacePipelineError("The detected face bounding box is invalid.")
        return image.crop((left, top, right, bottom))

    def _face_tensor(self, crop: Image.Image) -> torch.Tensor:
        # PIL-backed arrays may be read-only; copy ensures torch receives writable memory.
        pixels = np.array(crop.resize((160, 160), Image.Resampling.BILINEAR), copy=True)
        return fixed_image_standardization(
            torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        ).unsqueeze(0).to(self.device)
