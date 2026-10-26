import pytest

from vision_inspector.yolo import (
    YoloAnnotation,
    YoloFormatError,
    load_yolo_file,
    parse_yolo_line,
)


def test_parse_valid_yolo_line():
    annotation = parse_yolo_line(
        "0 0.512 0.483 0.214 0.376"
    )

    assert annotation == YoloAnnotation(
        class_id=0,
        x_center=0.512,
        y_center=0.483,
        width=0.214,
        height=0.376,
    )


def test_parse_line_with_extra_spaces():
    annotation = parse_yolo_line(
        "   3   0.721   0.532   0.108   0.224   "
    )

    assert annotation.class_id == 3
    assert annotation.x_center == pytest.approx(0.721)
    assert annotation.y_center == pytest.approx(0.532)
    assert annotation.width == pytest.approx(0.108)
    assert annotation.height == pytest.approx(0.224)


def test_empty_line_is_invalid():
    with pytest.raises(YoloFormatError):
        parse_yolo_line("")


def test_missing_value_is_invalid():
    with pytest.raises(YoloFormatError):
        parse_yolo_line(
            "0 0.5 0.5 0.2"
        )


def test_invalid_class_id_is_rejected():
    with pytest.raises(YoloFormatError):
        parse_yolo_line(
            "person 0.5 0.5 0.2 0.2"
        )


def test_negative_class_id_is_rejected():
    with pytest.raises(YoloFormatError):
        parse_yolo_line(
            "-1 0.5 0.5 0.2 0.2"
        )


def test_coordinate_outside_normalized_range_is_rejected():
    with pytest.raises(YoloFormatError):
        parse_yolo_line(
            "0 1.2 0.5 0.2 0.2"
        )


def test_zero_width_is_rejected():
    with pytest.raises(YoloFormatError):
        parse_yolo_line(
            "0 0.5 0.5 0.0 0.2"
        )


def test_load_yolo_file(tmp_path):
    label_file = tmp_path / "sample.txt"

    label_file.write_text(
        "0 0.5 0.5 0.2 0.3\n"
        "2 0.3 0.4 0.1 0.15\n",
        encoding="utf-8",
    )

    annotations = load_yolo_file(label_file)

    assert len(annotations) == 2
    assert annotations[0].class_id == 0
    assert annotations[1].class_id == 2


def test_load_yolo_file_skips_blank_lines(tmp_path):
    label_file = tmp_path / "sample.txt"

    label_file.write_text(
        "0 0.5 0.5 0.2 0.3\n"
        "\n"
        "   \n"
        "1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    annotations = load_yolo_file(label_file)

    assert len(annotations) == 2


def test_load_invalid_file_reports_line_number(tmp_path):
    label_file = tmp_path / "broken.txt"

    label_file.write_text(
        "0 0.5 0.5 0.2 0.3\n"
        "1 invalid 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(YoloFormatError) as exc_info:
        load_yolo_file(label_file)

    assert ":2:" in str(exc_info.value)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_yolo_file(
            "this_file_does_not_exist.txt"
        )
