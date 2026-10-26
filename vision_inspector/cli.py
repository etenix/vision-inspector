import argparse
from pathlib import Path
from typing import Sequence

from vision_inspector.dataset import DatasetStructureError
from vision_inspector.validator import (
    DatasetValidationResult,
    validate_yolo_dataset,
)


VERSION = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="vision-inspector",
        description=(
            "Inspect and validate computer vision datasets "
            "and annotations."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan and validate a YOLO-format dataset.",
        description=(
            "Scan a YOLO dataset containing images/ and "
            "labels/ directories."
        ),
    )

    scan_parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the YOLO dataset root directory.",
    )

    scan_parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed file-level problems.",
    )

    return parser


def _print_summary(
    result: DatasetValidationResult,
) -> None:
    """Print the validation summary."""

    scan = result.scan_result

    print(f"Vision Inspector v{VERSION}")
    print()
    print(f"Dataset: {scan.dataset_root}")
    print()
    print(f"Images:             {scan.image_count}")
    print(f"Labels:             {scan.label_count}")
    print(f"Matched pairs:      {scan.matched_pair_count}")
    print(f"Missing labels:     {scan.missing_label_count}")
    print(f"Orphan labels:      {scan.orphan_label_count}")
    print()
    print(f"Valid label files:  {result.valid_label_count}")
    print(f"Empty label files:  {result.empty_label_count}")
    print(f"Invalid labels:     {result.invalid_label_count}")
    print(f"Total annotations:  {result.total_annotations}")
    print()

    if result.has_errors:
        print("Status: Issues found")
    else:
        print("Status: OK")


def _print_details(
    result: DatasetValidationResult,
) -> None:
    """Print detailed validation problems."""

    scan = result.scan_result

    if scan.missing_label_images:
        print()
        print("Images without label files:")

        for image_file in scan.missing_label_images:
            relative_path = image_file.relative_to(
                scan.images_dir
            )
            print(f"  - {relative_path.as_posix()}")

    if scan.orphan_label_files:
        print()
        print("Label files without images:")

        for label_file in scan.orphan_label_files:
            relative_path = label_file.relative_to(
                scan.labels_dir
            )
            print(f"  - {relative_path.as_posix()}")

    if result.invalid_label_files:
        print()
        print("Invalid label files:")

        for issue in result.invalid_label_files:
            relative_path = issue.label_file.relative_to(
                scan.labels_dir
            )

            print(f"  - {relative_path.as_posix()}")
            print(f"    {issue.message}")

    if result.empty_label_files:
        print()
        print("Empty label files:")

        for label_file in result.empty_label_files:
            relative_path = label_file.relative_to(
                scan.labels_dir
            )
            print(f"  - {relative_path.as_posix()}")


def run_scan(
    dataset_path: Path,
    show_details: bool = False,
) -> int:
    """
    Scan and validate one YOLO dataset.

    Returns:
        0 when no errors are found.
        1 when validation issues are found.
        2 when the dataset structure is invalid.
    """

    try:
        result = validate_yolo_dataset(dataset_path)
    except DatasetStructureError as exc:
        print(f"Error: {exc}")
        return 2

    _print_summary(result)

    if show_details:
        _print_details(result)

    if result.has_errors:
        return 1

    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the Vision Inspector command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        return run_scan(
            dataset_path=args.dataset,
            show_details=args.details,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2
