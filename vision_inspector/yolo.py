from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloAnnotation:
    """One YOLO-format object detection annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class YoloFormatError(ValueError):
    """Raised when a YOLO annotation line has an invalid format."""


def parse_yolo_line(line: str) -> YoloAnnotation:
    """
    Parse one YOLO detection annotation line.

    Expected format:
        <class_id> <x_center> <y_center> <width> <height>

    Example:
        0 0.512 0.483 0.214 0.376
    """

    stripped = line.strip()

    if not stripped:
        raise YoloFormatError("YOLO annotation line is empty.")

    parts = stripped.split()

    if len(parts) != 5:
        raise YoloFormatError(
            f"Expected 5 values, got {len(parts)}: {stripped}"
        )

    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise YoloFormatError(
            f"Invalid class id: {parts[0]}"
        ) from exc

    try:
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError as exc:
        raise YoloFormatError(
            f"Invalid coordinate value: {stripped}"
        ) from exc

    if class_id < 0:
        raise YoloFormatError(
            f"Class id must be non-negative: {class_id}"
        )

    values = {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }

    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise YoloFormatError(
                f"{name} must be between 0 and 1: {value}"
            )

    if width <= 0.0:
        raise YoloFormatError(
            f"Width must be greater than 0: {width}"
        )

    if height <= 0.0:
        raise YoloFormatError(
            f"Height must be greater than 0: {height}"
        )

    return YoloAnnotation(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def load_yolo_file(
    path: str | Path,
) -> list[YoloAnnotation]:
    """Load all annotations from a YOLO label file."""

    label_path = Path(path)

    if not label_path.is_file():
        raise FileNotFoundError(
            f"YOLO label file not found: {label_path}"
        )

    annotations: list[YoloAnnotation] = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                annotation = parse_yolo_line(line)
            except YoloFormatError as exc:
                raise YoloFormatError(
                    f"{label_path}:{line_number}: {exc}"
                ) from exc

            annotations.append(annotation)

    return annotations
