from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def load_images(base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    original_path = base_dir / "original.jpg"
    defect_path = base_dir / "defect.jpg"

    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    defect = cv2.imread(str(defect_path), cv2.IMREAD_COLOR)

    if original is None:
        raise FileNotFoundError(f"Failed to read reference image: {original_path}")
    if defect is None:
        raise FileNotFoundError(f"Failed to read defect image: {defect_path}")
    if original.shape[:2] != defect.shape[:2]:
        raise ValueError(
            "Input images must have the same size, "
            f"got {original.shape[:2]} and {defect.shape[:2]}"
        )

    return original, defect


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    height, width = mask.shape

    for label in range(1, num_labels):
        x, y, w, h, _ = stats[label]
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        cleaned[labels == label] = 255

    return cleaned


def component_score(
    area: int,
    aspect_ratio: float,
    mean_response: float,
    max_response: float,
) -> float:
    elongated_bonus = max(aspect_ratio - 1.2, 0.0) * 20.0
    return mean_response * 4.0 + area * 0.03 + max_response * 0.2 + elongated_bonus


def extract_best_component(
    mask: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    min_area: int = 120,
) -> Tuple[np.ndarray, Optional[BBox], float]:
    binary = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_mask = np.zeros_like(binary)
    best_bbox: Optional[BBox] = None
    best_score = 0.0

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        size_a, size_b = rect[1]
        short_side = max(min(size_a, size_b), 1.0)
        long_side = max(size_a, size_b)
        aspect_ratio = long_side / short_side

        if score_map is None:
            mean_response = float(area)
            max_response = float(area)
        else:
            values = score_map[labels == label]
            mean_response = float(values.mean())
            max_response = float(values.max())

        score = component_score(area, aspect_ratio, mean_response, max_response)
        if score <= best_score:
            continue

        best_score = score
        best_bbox = (int(x), int(y), int(w), int(h))
        best_mask = component

    return best_mask, best_bbox, best_score


def detect_by_reference_tophat(
    original: np.ndarray,
    defect: np.ndarray,
) -> Tuple[np.ndarray, Optional[BBox], float]:
    original_gray = preprocess_gray(original)
    defect_gray = preprocess_gray(defect)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    original_tophat = cv2.morphologyEx(original_gray, cv2.MORPH_TOPHAT, kernel)
    defect_tophat = cv2.morphologyEx(defect_gray, cv2.MORPH_TOPHAT, kernel)
    score_map = cv2.subtract(defect_tophat, original_tophat)
    score_map = cv2.GaussianBlur(score_map, (5, 5), 0)

    threshold_value, binary = cv2.threshold(
        score_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if threshold_value < 15:
        _, binary = cv2.threshold(score_map, 15, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.dilate(binary, open_kernel, iterations=1)
    binary = remove_border_components(binary)

    return extract_best_component(binary, score_map=score_map, min_area=150)


def detect_by_adaptive(defect: np.ndarray) -> Tuple[np.ndarray, Optional[BBox], float]:
    gray = preprocess_gray(defect)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    score_map = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )

    binary = cv2.adaptiveThreshold(
        score_map,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -4,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    binary = remove_border_components(binary)

    return extract_best_component(binary, score_map=score_map, min_area=120)


def detect_by_canny(defect: np.ndarray) -> Tuple[np.ndarray, Optional[BBox], float]:
    gray = preprocess_gray(defect)
    score_map = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )

    edges = cv2.Canny(score_map, 40, 120)
    link_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, link_kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, link_kernel, iterations=2)
    binary = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    binary = remove_border_components(binary)

    return extract_best_component(binary, score_map=score_map, min_area=120)


def draw_detection(
    defect_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[BBox],
    method_name: str,
) -> np.ndarray:
    result = defect_bgr.copy()
    overlay = result.copy()
    overlay[mask > 0] = (0, 255, 0)
    result = cv2.addWeighted(overlay, 0.25, result, 0.75, 0)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        result,
        f"Method: {method_name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return result


def select_detection(original: np.ndarray, defect: np.ndarray) -> Tuple[np.ndarray, Optional[BBox], str]:
    methods = [
        ("reference_tophat", lambda: detect_by_reference_tophat(original, defect)),
        ("adaptive_threshold", lambda: detect_by_adaptive(defect)),
        ("canny_merge", lambda: detect_by_canny(defect)),
    ]

    for method_name, method in methods:
        mask, bbox, score = method()
        if bbox is not None and score > 0:
            return mask, bbox, method_name

    empty_mask = np.zeros(defect.shape[:2], dtype=np.uint8)
    return empty_mask, None, "none"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "result.jpg"

    original, defect = load_images(base_dir)
    mask, bbox, method_name = select_detection(original, defect)

    result = draw_detection(defect, mask, bbox, method_name)
    if bbox is None:
        cv2.putText(
            result,
            "No confident defect found",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), result):
        raise IOError(f"Failed to save result image: {output_path}")

    if bbox is None:
        print(f"Saved result to: {output_path}")
        print("No confident defect was detected.")
        return

    x, y, w, h = bbox
    print(f"Saved result to: {output_path}")
    print(f"Selected method: {method_name}")
    print(f"Defect bbox: x={x}, y={y}, w={w}, h={h}")


if __name__ == "__main__":
    main()
