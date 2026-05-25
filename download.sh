#!/bin/bash

# Download eDifFIQA model weights from GitHub releases
# Usage: ./download.sh

BASE_URL="https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights"

WEIGHTS_DIR="weights"
mkdir -p ${WEIGHTS_DIR}

echo "Downloading model weights..."

# eDifFIQA-T (MobileFaceNet)
echo "Downloading eDifFIQA-T (ediffiqa_t.pth)..."
curl -L "${BASE_URL}/ediffiqa_t.pth" -o "${WEIGHTS_DIR}/ediffiqa_t.pth"

echo "Downloading eDifFIQA-T (ediffiqa_t.onnx)..."
curl -L "${BASE_URL}/ediffiqa_t.onnx" -o "${WEIGHTS_DIR}/ediffiqa_t.onnx"

# eDifFIQA-S (IResNet-18)
echo "Downloading eDifFIQA-S (ediffiqa_s.pth)..."
curl -L "${BASE_URL}/ediffiqa_s.pth" -o "${WEIGHTS_DIR}/ediffiqa_s.pth"

echo "Downloading eDifFIQA-S (ediffiqa_s.onnx)..."
curl -L "${BASE_URL}/ediffiqa_s.onnx" -o "${WEIGHTS_DIR}/ediffiqa_s.onnx"

# eDifFIQA-M (IResNet-50)
echo "Downloading eDifFIQA-M (ediffiqa_m.pth)..."
curl -L "${BASE_URL}/ediffiqa_m.pth" -o "${WEIGHTS_DIR}/ediffiqa_m.pth"

echo "Downloading eDifFIQA-M (ediffiqa_m.onnx)..."
curl -L "${BASE_URL}/ediffiqa_m.onnx" -o "${WEIGHTS_DIR}/ediffiqa_m.onnx"

# eDifFIQA-L (IResNet-100)
echo "Downloading eDifFIQA-L (ediffiqa_l.pth)..."
curl -L "${BASE_URL}/ediffiqa_l.pth" -o "${WEIGHTS_DIR}/ediffiqa_l.pth"

echo "Downloading eDifFIQA-L (ediffiqa_l.onnx)..."
curl -L "${BASE_URL}/ediffiqa_l.onnx" -o "${WEIGHTS_DIR}/ediffiqa_l.onnx"

echo "Done! Weights saved to ${WEIGHTS_DIR}/"
ls -lh ${WEIGHTS_DIR}/
