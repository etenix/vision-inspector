Vision Inspector

Vision Inspector is a lightweight Python toolkit for inspecting and validating computer vision datasets and annotations.
The current version focuses on YOLO object-detection datasets and provides dataset structure inspection, image/label matching, annotation parsing, validation, and a command-line interface.

Features

Vision Inspector currently supports:
Recursive scanning of YOLO-style images/ and labels/ directories
Image and label matching by relative path and filename stem
Detection of images without label files
Detection of label files without corresponding images
Detection of ambiguous duplicate image stems
YOLO annotation parsing
Validation of normalized bounding-box values
Detection of malformed annotation lines
Invalid file and line-number reporting
Empty label file detection
Annotation counting
Detailed validation output
Command-line interface
Automated test coverage with pytest
Supported image extensions:
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff

Dataset Structure

Vision Inspector expects a YOLO-style dataset with separate images and labels directories.
Example:
dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── train/
│       └── image_003.png
│
└── labels/
    ├── image_001.txt
    ├── image_002.txt
    └── train/
        └── image_003.txt
Images and labels are matched using their relative paths and filename stems.
For example:
images/train/sample.jpg
labels/train/sample.txt
are treated as a matching pair.
However:
images/train/sample.jpg
labels/val/sample.txt
are not treated as a matching pair because their relative paths differ.

YOLO Annotation Format

Vision Inspector currently validates standard YOLO object-detection annotations.
Each annotation line must contain:
<class_id> <x_center> <y_center> <width> <height>
Example:
0 0.512 0.483 0.214 0.376
3 0.721 0.532 0.108 0.224
Requirements:
class_id must be a non-negative integer.
x_center must be between 0 and 1.
y_center must be between 0 and 1.
width must be greater than 0 and no greater than 1.
height must be greater than 0 and no greater than 1.
Blank lines are ignored.
An empty label file is tracked separately and is not automatically treated as an error because it may represent a valid background image with no annotated objects.

Installation

Requirements
Python 3.10 or later
Git
pytest for development and testing
Clone the repository
git clone git@github.com:etenix/vision-inspector.git
cd vision-inspector
Install
Install the project in editable mode with development dependencies:
python -m pip install -e ".[dev]"
Verify the installation:
vision-inspector --version
Expected output:
vision-inspector 0.1.0

Usage

Scan a dataset
vision-inspector scan /path/to/dataset
Example output:
Vision Inspector v0.1.0
Dataset: /path/to/dataset
Images:             100
Labels:              98
Matched pairs:       97
Missing labels:       3
Orphan labels:        1
Valid label files:   95
Empty label files:    2
Invalid labels:       1
Total annotations:  328
Status: Issues found
Show detailed problems
Use the --details option:
vision-inspector scan /path/to/dataset --details
Example:
Images without label files:
  - train/image_004.jpg
Label files without images:
  - train/image_105.txt
Invalid label files:
  - train/image_021.txt
    /path/to/dataset/labels/train/image_021.txt:2: Invalid coordinate value
Empty label files:
  - train/background_001.txt

Exit Codes

The command-line interface uses exit codes that can also be consumed by scripts or automated workflows.
Exit code	Meaning
0	Dataset validation completed without errors
1	Dataset validation completed and issues were found
2	Dataset structure is invalid
Empty label files alone do not cause exit code 1.

Validation Rules

Vision Inspector currently checks for:
Dataset structure
Dataset root exists
images/ directory exists
labels/ directory exists
Image/label matching
Matching image and label pairs
Missing label files
Orphan label files
Duplicate image stems with different supported extensions
YOLO annotations
Correct number of fields
Valid integer class IDs
Non-negative class IDs
Numeric bounding-box values
Normalized coordinate ranges
Non-zero width and height
Invalid file and line-number reporting

Running Tests

Run the complete test suite:
pytest -v
The current test suite covers:
YOLO annotation parsing
Invalid annotation handling
YOLO label file loading
Dataset scanning
Nested dataset directories
Missing labels
Orphan labels
Duplicate image stems
Empty label files
Dataset validation
CLI output
CLI exit codes

Project Structure

vision-inspector/
├── README.md
├── LICENSE
├── pyproject.toml
│
├── vision_inspector/
│   ├── __init__.py
│   ├── cli.py
│   ├── dataset.py
│   ├── validator.py
│   └── yolo.py
│
└── tests/
    ├── test_cli.py
    ├── test_dataset.py
    ├── test_validator.py
    └── test_yolo.py

Roadmap

Planned future improvements include:
Image integrity and decoding validation
Bounding-box visualization
Dataset and class statistics
Dataset summary reports
Annotation filtering and search
Additional annotation validation rules
Desktop graphical interface
COCO annotation support
Pascal VOC annotation support
Export and conversion utilities
The roadmap is intentionally incremental. Vision Inspector aims to remain modular, testable, and useful for practical computer vision dataset workflows.

Development Philosophy

Vision Inspector is developed incrementally with an emphasis on:
Small, focused modules
Explicit validation behavior
Clear error reporting
Automated testing
Reproducible command-line workflows
Maintainable and extensible code

License

This project is licensed under the MIT License. See LICENSE for details.
