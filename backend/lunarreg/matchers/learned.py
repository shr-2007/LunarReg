from __future__ import annotations

import numpy as np

from ..config import RegistrationConfig
from ..types import MatchSet


def _empty_matches() -> MatchSet:
    return MatchSet(
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=object),
    )


class LightGlueMatcher:
    """Optional SuperPoint + LightGlue adapter.

    Install requirements-learned.txt before selecting this matcher. Model
    weights are resolved by the upstream package and are not bundled here.
    """

    name = "lightglue"

    def __init__(self, config: RegistrationConfig):
        self.config = config

    def match(self, source_views: dict[str, np.ndarray], reference_views: dict[str, np.ndarray]) -> MatchSet:
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import rbd
        except ImportError as exc:
            raise RuntimeError(
                "LightGlue is optional. Install requirements-learned.txt or use matcher: sift."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = SuperPoint(max_num_keypoints=self.config.sift_features).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)

        def tensor(image):
            return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)

        image0 = tensor(source_views["intensity"])
        image1 = tensor(reference_views["intensity"])
        with torch.inference_mode():
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches01 = [rbd(value) for value in (feats0, feats1, matches01)]

        matches = matches01["matches"].detach().cpu().numpy()
        if matches.size == 0:
            return _empty_matches()
        key0 = feats0["keypoints"].detach().cpu().numpy()[matches[:, 0]]
        key1 = feats1["keypoints"].detach().cpu().numpy()[matches[:, 1]]
        scores_t = matches01.get("scores")
        scores = (
            scores_t.detach().cpu().numpy().astype(np.float32)
            if scores_t is not None
            else np.ones(len(matches), dtype=np.float32)
        )
        labels = np.full(len(matches), "lightglue", dtype=object)
        return MatchSet(key0.astype(np.float32), key1.astype(np.float32), scores, labels)


class LoFTRMatcher:
    """Optional Kornia LoFTR adapter."""

    name = "loftr"

    def __init__(self, config: RegistrationConfig):
        self.config = config

    def match(self, source_views: dict[str, np.ndarray], reference_views: dict[str, np.ndarray]) -> MatchSet:
        try:
            import kornia.feature as KF
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LoFTR is optional. Install requirements-learned.txt or use matcher: sift."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KF.LoFTR(pretrained="outdoor").eval().to(device)

        def tensor(image):
            return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)

        with torch.inference_mode():
            output = model(
                {
                    "image0": tensor(source_views["intensity"]),
                    "image1": tensor(reference_views["intensity"]),
                }
            )
        key0 = output["keypoints0"].detach().cpu().numpy().astype(np.float32)
        key1 = output["keypoints1"].detach().cpu().numpy().astype(np.float32)
        confidence = output["confidence"].detach().cpu().numpy().astype(np.float32)
        labels = np.full(len(confidence), "loftr", dtype=object)
        return MatchSet(key0, key1, confidence, labels)
