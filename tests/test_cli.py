from vision_inspector.cli import main


def create_dataset_directories(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    return images_dir, labels_dir


def test_cli_scan_valid_dataset(tmp_path, capsys):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "sample.jpg").write_bytes(b"fake image")

    (labels_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "scan",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Vision Inspector v0.1.0" in output
    assert "Images:             1" in output
    assert "Labels:             1" in output
    assert "Matched pairs:      1" in output
    assert "Total annotations:  1" in output
    assert "Status: OK" in output


def test_cli_scan_reports_missing_label(tmp_path, capsys):
    images_dir, _ = create_dataset_directories(tmp_path)

    (images_dir / "missing.jpg").write_bytes(
        b"fake image"
    )

    exit_code = main(
        [
            "scan",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Missing labels:     1" in output
    assert "Status: Issues found" in output


def test_cli_details_show_missing_label_file(
    tmp_path,
    capsys,
):
    images_dir, _ = create_dataset_directories(tmp_path)

    (images_dir / "missing.jpg").write_bytes(
        b"fake image"
    )

    exit_code = main(
        [
            "scan",
            str(tmp_path),
            "--details",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Images without label files:" in output
    assert "missing.jpg" in output


def test_cli_details_show_invalid_label(
    tmp_path,
    capsys,
):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "broken.jpg").write_bytes(
        b"fake image"
    )

    (labels_dir / "broken.txt").write_text(
        "0 invalid 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "scan",
            str(tmp_path),
            "--details",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Invalid label files:" in output
    assert "broken.txt" in output
    assert ":1:" in output


def test_cli_details_show_empty_label(
    tmp_path,
    capsys,
):
    images_dir, labels_dir = create_dataset_directories(
        tmp_path
    )

    (images_dir / "background.jpg").write_bytes(
        b"fake image"
    )

    (labels_dir / "background.txt").write_text(
        "",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "scan",
            str(tmp_path),
            "--details",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Empty label files:  1" in output
    assert "Empty label files:" in output
    assert "background.txt" in output
    assert "Status: OK" in output


def test_cli_invalid_dataset_structure(
    tmp_path,
    capsys,
):
    missing_dataset = tmp_path / "missing"

    exit_code = main(
        [
            "scan",
            str(missing_dataset),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 2
    assert "Error:" in output
    assert "Dataset directory not found" in output
