import pytest

from vision_inspector.dataset import (
    DatasetStructureError,
    scan_yolo_dataset,
)


def create_dataset_directories(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    return images_dir, labels_dir


def test_scan_dataset_with_matched_pairs(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "image_001.jpg").write_bytes(b"fake image")
    (images_dir / "image_002.png").write_bytes(b"fake image")

    (labels_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (labels_dir / "image_002.txt").write_text(
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    result = scan_yolo_dataset(tmp_path)

    assert result.image_count == 2
    assert result.label_count == 2
    assert result.matched_pair_count == 2
    assert result.missing_label_count == 0
    assert result.orphan_label_count == 0


def test_scan_detects_missing_label(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "image_001.jpg").write_bytes(b"fake image")
    (images_dir / "image_002.jpg").write_bytes(b"fake image")

    (labels_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    result = scan_yolo_dataset(tmp_path)

    assert result.image_count == 2
    assert result.label_count == 1
    assert result.matched_pair_count == 1
    assert result.missing_label_count == 1

    assert (
        result.missing_label_images[0].name
        == "image_002.jpg"
    )


def test_scan_detects_orphan_label(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "image_001.jpg").write_bytes(b"fake image")

    (labels_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (labels_dir / "image_002.txt").write_text(
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    result = scan_yolo_dataset(tmp_path)

    assert result.matched_pair_count == 1
    assert result.orphan_label_count == 1

    assert (
        result.orphan_label_files[0].name
        == "image_002.txt"
    )


def test_scan_supports_nested_directories(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    nested_images = images_dir / "train"
    nested_labels = labels_dir / "train"

    nested_images.mkdir()
    nested_labels.mkdir()

    (nested_images / "sample.jpg").write_bytes(b"fake image")
    (nested_labels / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    result = scan_yolo_dataset(tmp_path)

    assert result.image_count == 1
    assert result.label_count == 1
    assert result.matched_pair_count == 1

    image_path, label_path = result.matched_pairs[0]

    assert image_path.relative_to(images_dir).as_posix() == (
        "train/sample.jpg"
    )
    assert label_path.relative_to(labels_dir).as_posix() == (
        "train/sample.txt"
    )


def test_scan_ignores_unsupported_files(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "image_001.jpg").write_bytes(b"fake image")
    (images_dir / "notes.md").write_text(
        "not an image",
        encoding="utf-8",
    )

    (labels_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (labels_dir / "metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )

    result = scan_yolo_dataset(tmp_path)

    assert result.image_count == 1
    assert result.label_count == 1
    assert result.matched_pair_count == 1


def test_missing_dataset_directory_is_rejected(tmp_path):
    missing_directory = tmp_path / "missing"

    with pytest.raises(
        DatasetStructureError,
        match="Dataset directory not found",
    ):
        scan_yolo_dataset(missing_directory)


def test_missing_images_directory_is_rejected(tmp_path):
    (tmp_path / "labels").mkdir()

    with pytest.raises(
        DatasetStructureError,
        match="Images directory not found",
    ):
        scan_yolo_dataset(tmp_path)


def test_missing_labels_directory_is_rejected(tmp_path):
    (tmp_path / "images").mkdir()

    with pytest.raises(
        DatasetStructureError,
        match="Labels directory not found",
    ):
        scan_yolo_dataset(tmp_path)


def test_duplicate_image_stem_is_rejected(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "sample.jpg").write_bytes(b"fake image")
    (images_dir / "sample.png").write_bytes(b"fake image")

    (labels_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetStructureError,
        match="Duplicate image stem",
    ):
        scan_yolo_dataset(tmp_path)
