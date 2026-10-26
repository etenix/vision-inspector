from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }
)

LABEL_EXTENSIONS = frozenset({".txt"})


class DatasetStructureError(ValueError):
    """Raised when a dataset directory has an invalid structure."""


@dataclass(frozen=True)
class DatasetScanResult:
    """Summary produced by scanning a YOLO dataset."""

    dataset_root: Path
    images_dir: Path
    labels_dir: Path
    image_files: tuple[Path, ...]
    label_files: tuple[Path, ...]
    matched_pairs: tuple[tuple[Path, Path], ...]
    missing_label_images: tuple[Path, ...]
    orphan_label_files: tuple[Path, ...]

    @property
    def image_count(self) -> int:
        return len(self.image_files)

    @property
    def label_count(self) -> int:
        return len(self.label_files)

    @property
    def matched_pair_count(self) -> int:
        return len(self.matched_pairs)

    @property
    def missing_label_count(self) -> int:
        return len(self.missing_label_images)

    @property
    def orphan_label_count(self) -> int:
        return len(self.orphan_label_files)


def _collect_files(
    directory: Path,
    allowed_extensions: frozenset[str],
) -> tuple[Path, ...]:
    """Collect supported files recursively in a stable order."""

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed_extensions
    ]

    return tuple(
        sorted(
            files,
            key=lambda path: path.relative_to(directory).as_posix(),
        )
    )


def _build_file_index(
    files: tuple[Path, ...],
    base_directory: Path,
    file_type: str,
) -> dict[Path, Path]:
    """
    Build an index using each file's relative path without its suffix.

    Example:
        images/train/cat.jpg -> train/cat
        labels/train/cat.txt -> train/cat
    """

    index: dict[Path, Path] = {}

    for path in files:
        relative_key = path.relative_to(base_directory).with_suffix("")

        if relative_key in index:
            existing = index[relative_key]

            raise DatasetStructureError(
                f"Duplicate {file_type} stem detected: "
                f"{existing} and {path}"
            )

        index[relative_key] = path

    return index


def scan_yolo_dataset(
    dataset_root: str | Path,
) -> DatasetScanResult:
    """
    Scan a YOLO dataset containing images/ and labels/ directories.

    Images and labels are matched by their relative path and stem.

    Example:
        images/train/sample.jpg
        labels/train/sample.txt
    """

    root = Path(dataset_root).expanduser().resolve()

    if not root.is_dir():
        raise DatasetStructureError(
            f"Dataset directory not found: {root}"
        )

    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir():
        raise DatasetStructureError(
            f"Images directory not found: {images_dir}"
        )

    if not labels_dir.is_dir():
        raise DatasetStructureError(
            f"Labels directory not found: {labels_dir}"
        )

    image_files = _collect_files(
        images_dir,
        IMAGE_EXTENSIONS,
    )

    label_files = _collect_files(
        labels_dir,
        LABEL_EXTENSIONS,
    )

    image_index = _build_file_index(
        image_files,
        images_dir,
        "image",
    )

    label_index = _build_file_index(
        label_files,
        labels_dir,
        "label",
    )

    image_keys = set(image_index)
    label_keys = set(label_index)

    matched_keys = sorted(
        image_keys & label_keys,
        key=lambda path: path.as_posix(),
    )

    missing_label_keys = sorted(
        image_keys - label_keys,
        key=lambda path: path.as_posix(),
    )

    orphan_label_keys = sorted(
        label_keys - image_keys,
        key=lambda path: path.as_posix(),
    )

    matched_pairs = tuple(
        (
            image_index[key],
            label_index[key],
        )
        for key in matched_keys
    )

    missing_label_images = tuple(
        image_index[key]
        for key in missing_label_keys
    )

    orphan_label_files = tuple(
        label_index[key]
        for key in orphan_label_keys
    )

    return DatasetScanResult(
        dataset_root=root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_files=image_files,
        label_files=label_files,
        matched_pairs=matched_pairs,
        missing_label_images=missing_label_images,
        orphan_label_files=orphan_label_files,
    )
