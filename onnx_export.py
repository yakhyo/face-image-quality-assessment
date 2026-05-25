# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse
import os

from models import get_model
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an eDifFIQA PyTorch checkpoint to ONNX")
    parser.add_argument(
        "-w",
        "--weights",
        required=True,
        type=str,
        help="Path to PyTorch state dict (.pth/.bin)",
    )
    parser.add_argument(
        "-v",
        "--variant",
        required=True,
        type=str,
        choices=["t", "s", "m", "l"],
        help="Model variant (tiny/small/medium/large)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to <weights-basename>.onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size for ONNX export",
    )
    return parser.parse_args()


@torch.no_grad()
def export(args: argparse.Namespace) -> None:
    device = "cpu"  # ONNX export is simpler on CPU and avoids autocast surprises.

    model = get_model(args.variant, return_feat=False)
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded eDifFIQA-{args.variant.upper()} from {args.weights}")

    output_path = args.output or f"{os.path.splitext(os.path.basename(args.weights))[0]}.onnx"
    print(f"==> Exporting to '{output_path}' (opset={args.opset})")

    dummy_input = torch.randn(1, 3, 112, 112)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch_size"}, "quality": {0: "batch_size"}}
        print("Exporting with dynamic batch size.")
    else:
        print("Exporting with fixed input size: (1, 3, 112, 112)")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["quality"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    export(parse_args())
