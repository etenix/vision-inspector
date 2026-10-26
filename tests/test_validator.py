from vision_inspector.validator import (
    validate_yolo_dataset,
)


def create_dataset_directories(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    return images_dir, labels_dir


def test_validate_dataset_with_valid_labels(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "image_001.jpg").write_bytes(b"fake image")
    (images_dir / "image_002.jpg").write_bytes(b"fake image")

    (labels_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    (labels_dir / "image_002.txt").write_text(
        "2 0.3 0.3 0.15 0.15\n",
        encoding="utf-8",
    )

    result = validate_yolo_dataset(tmp_path)

    assert result.valid_label_count == 2
    assert result.empty_label_count == 0
    assert result.invalid_label_count == 0
    assert result.total_annotations == 3
    assert result.has_errors is False


def test_validate_detects_empty_label_file(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "background.jpg").write_bytes(b"fake image")
    (labels_dir / "background.txt").write_text(
        "",
        encoding="utf-8",
    )

    result = validate_yolo_dataset(tmp_path)

    assert result.valid_label_count == 0
    assert result.empty_label_count == 1
    assert result.invalid_label_count == 0
    assert result.total_annotations == 0
    assert result.has_errors is False

    assert result.empty_label_files[0].name == "background.txt"


def test_validate_detects_invalid_label_file(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "broken.jpg").write_bytes(b"fake image")

    (labels_dir / "broken.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 invalid 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    result = validate_yolo_dataset(tmp_path)

    assert result.valid_label_count == 0
    assert result.empty_label_count == 0
    assert result.invalid_label_count == 1
    assert result.total_annotations == 0
    assert result.has_errors is True

    issue = result.invalid_label_files[0]

    assert issue.label_file.name == "broken.txt"
    assert ":2:" in issue.message


def test_validate_counts_only_valid_annotations(tmp_path):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "valid.jpg").write_bytes(b"fake image")
    (images_dir / "broken.jpg").write_bytes(b"fake image")

    (labels_dir / "valid.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    (labels_dir / "broken.txt").write_text(
        "2 1.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    result = validate_yolo_dataset(tmp_path)

    assert result.valid_label_count == 1
    assert result.invalid_label_count == 1
    assert result.total_annotations == 2


def test_missing_label_marks_dataset_as_having_errors(tmp_path):
    images_dir, _ = create_dataset_directories(tmp_path)

    (images_dir / "missing.jpg").write_bytes(b"fake image")

    result = validate_yolo_dataset(tmp_path)

    assert result.scan_result.missing_label_count == 1
    assert result.has_errors is True


def test_orphan_label_marks_dataset_as_having_errors(tmp_path):
    _, labels_dir = create_dataset_directories(tmp_path)

    (labels_dir / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    result = validate_yolo_dataset(tmp_path)

    assert result.scan_result.orphan_label_count == 1
    assert result.valid_label_count == 1
    assert result.total_annotations == 1
    assert result.has_errors is True
