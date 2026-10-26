from dataclasses import dataclass
from pathlib import Path

from vision_inspector.dataset import (
    DatasetScanResult,
    scan_yolo_dataset,
)
from vision_inspector.yolo import (
    YoloFormatError,
    load_yolo_file,
)


@dataclass(frozen=True)
class LabelValidationIssue:
    """A validation problem found in one YOLO label file."""

    label_file: Path
    message: str


@dataclass(frozen=True)
class DatasetValidationResult:
    """Combined scan and label-validation results."""

    scan_result: DatasetScanResult
    valid_label_files: tuple[Path, ...]
    empty_label_files: tuple[Path, ...]
    invalid_label_files: tuple[LabelValidationIssue, ...]
    total_annotations: int

    @property
    def valid_label_count(self) -> int:
        return len(self.valid_label_files)

    @property
    def empty_label_count(self) -> int:
        return len(self.empty_label_files)

    @property
    def invalid_label_count(self) -> int:
        return len(self.invalid_label_files)

    @property
    def has_errors(self) -> bool:
        return (
            self.scan_result.missing_label_count > 0
            or self.scan_result.orphan_label_count > 0
            or self.invalid_label_count > 0
        )


def validate_scan_result(
    scan_result: DatasetScanResult,
) -> DatasetValidationResult:
    """
    Validate all YOLO label files found in a scan result.

    Empty label files are tracked separately because they can represent
    valid background images containing no annotated objects.
    """

    valid_label_files: list[Path] = []
    empty_label_files: list[Path] = []
    invalid_label_files: list[LabelValidationIssue] = []

    total_annotations = 0

    for label_file in scan_result.label_files:
        try:
            annotations = load_yolo_file(label_file)
        except YoloFormatError as exc:
            invalid_label_files.append(
                LabelValidationIssue(
                    label_file=label_file,
                    message=str(exc),
                )
            )
            continue

        if not annotations:
            empty_label_files.append(label_file)
            continue

        valid_label_files.append(label_file)
        total_annotations += len(annotations)

    return DatasetValidationResult(
        scan_result=scan_result,
        valid_label_files=tuple(valid_label_files),
        empty_label_files=tuple(empty_label_files),
        invalid_label_files=tuple(invalid_label_files),
        total_annotations=total_annotations,
    )


def validate_yolo_dataset(
    dataset_root: str | Path,
) -> DatasetValidationResult:
    """Scan and validate a YOLO-format dataset."""

    scan_result = scan_yolo_dataset(dataset_root)

    return validate_scan_result(scan_result)
