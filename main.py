# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse

import cv2
from models import get_model
import numpy as np
import torch
from torchvision import transforms

from uniface import face_alignment
from uniface.detection import SCRFD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='eDifFIQA PyTorch Face Quality Inference - Score one or more images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('images', type=str, nargs='+', help='One or more image paths')
    parser.add_argument(
        '--variant',
        type=str,
        default='t',
        choices=['t', 's', 'm', 'l'],
        help='Model variant (tiny/small/medium/large)',
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to PyTorch state dict (.pth/.bin). Defaults to weights/ediffiqa_<variant>.pth',
    )
    parser.add_argument(
        '--aligned',
        action='store_true',
        help='Skip detection; treat inputs as already-aligned 112x112 face crops.',
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    """Matches the upstream eDifFIQA inference transform."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.variant == 't' and args.weights is None:
        raise SystemExit(
            'eDifFIQA-T PyTorch weights are not redistributed by the upstream authors '
            '(only an ONNX is published). Either pass --weights with a locally obtained '
            'ediffiqaT.pth, or use onnx_inference.py with weights/ediffiqa_t.onnx.'
        )

    weights_path = args.weights or f'weights/ediffiqa_{args.variant}.pth'
    print(f'Loading eDifFIQA-{args.variant.upper()} from {weights_path} on {args.device}')

    model = get_model(args.variant, return_feat=False)
    state_dict = torch.load(weights_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(args.device).eval()

    transform = build_transform()
    detector = None if args.aligned else SCRFD()

    print(f'{"image":<40} {"face":>5} {"score":>10}')
    print('-' * 60)

    for image_path in args.images:
        image = cv2.imread(image_path)
        if image is None:
            print(f'{image_path:<40} {"-":>5} {"READ_ERR":>10}')
            continue

        if args.aligned:
            crops = [image]
        else:
            faces = detector.detect(image)
            if len(faces) == 0:
                print(f'{image_path:<40} {0:>5} {"NO_FACE":>10}')
                continue
            crops = [face_alignment(image, f.landmarks.astype(np.float32))[0] for f in faces]

        for idx, crop in enumerate(crops, start=1):
            # eDifFIQA expects RGB; ToPILImage assumes the array layout, so swap channels first.
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(args.device)
            score = model(tensor).squeeze().detach().cpu().item()
            print(f'{image_path:<40} {idx:>5} {score:>10.4f}')


if __name__ == '__main__':
    main()
