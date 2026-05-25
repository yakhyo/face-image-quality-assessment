import cv2
import numpy as np
import onnxruntime as ort

from uniface import face_alignment


class eDifFIQAOnnx:
    """eDifFIQA ONNX runtime quality estimator.

    Paper:  https://ieeexplore.ieee.org/document/10468647
    Source: https://github.com/LSIbabnikz/eDifFIQA
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        self.model_path = model_path

        if providers is None:
            available = ort.get_available_providers()
            preferred = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in preferred if p in available]

        self.session = ort.InferenceSession(model_path, providers=providers)

        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        self.input_size = tuple(input_config.shape[2:4][::-1])
        self.output_name = self.session.get_outputs()[0].name

        # ((image / 255) - 0.5) / 0.5  ==  (image - 127.5) / 127.5
        self.normalization_mean = 127.5
        self.normalization_std = 127.5

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(
            face_image,
            scalefactor=1.0 / self.normalization_std,
            size=self.input_size,
            mean=(self.normalization_mean,) * 3,
            swapRB=True,
        )

    def score_aligned(self, aligned_face: np.ndarray) -> float:
        blob = self.preprocess(aligned_face)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        return float(np.squeeze(output))

    def get_quality(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        if image is None or landmarks is None:
            raise ValueError('Image and landmarks must not be None')
        aligned, _ = face_alignment(image, landmarks.astype(np.float32))
        return self.score_aligned(aligned)
