# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse

import cv2
from models.ediffiqa_onnx import eDifFIQAOnnx

from uniface.detection import SCRFD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="eDifFIQA ONNX face quality inference")
    parser.add_argument("images", type=str, nargs="+", help="One or more image paths")
    parser.add_argument("--model", type=str, required=True, help="Path to eDifFIQA ONNX model")
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="Skip detection; treat inputs as already-aligned 112x112 face crops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = eDifFIQAOnnx(args.model)
    detector = None if args.aligned else SCRFD()

    print(f"{'image':<40} {'face':>5} {'score':>10}")
    print("-" * 60)

    for image_path in args.images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"{image_path:<40} {'-':>5} {'READ_ERR':>10}")
            continue

        if args.aligned:
            score = quality.score_aligned(image)
            print(f"{image_path:<40} {1:>5} {score:>10.4f}")
            continue

        faces = detector.detect(image)
        if len(faces) == 0:
            print(f"{image_path:<40} {0:>5} {'NO_FACE':>10}")
            continue

        for idx, face in enumerate(faces, start=1):
            score = quality.get_quality(image, face.landmarks)
            print(f"{image_path:<40} {idx:>5} {score:>10.4f}")


if __name__ == "__main__":
    main()
