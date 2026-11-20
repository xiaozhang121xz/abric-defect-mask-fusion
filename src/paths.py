"""Resolve dataset roots from a user-provided environment variable."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def find_root():
    """Return a project data root; set FABRIC_PROJECT_ROOT for custom layouts."""
    configured = os.environ.get("FABRIC_PROJECT_ROOT")
    candidates = [configured] if configured else []
    candidates.extend((os.path.join(os.path.dirname(HERE), "data"), HERE))
    for base in candidates:
        if base and os.path.isdir(os.path.join(base, "data", "szbm")):
            return base
    raise RuntimeError(
        "Cannot find data/szbm. Set FABRIC_PROJECT_ROOT to the project data root."
    )


ROOT = find_root()
DATA_DIR = os.path.join(ROOT, "data")
SZBM_DIR = os.path.join(DATA_DIR, "szbm")
CLASS_DIR = os.path.join(DATA_DIR, "ai-zzdn-class")
CLASS_BINARY_DIR = os.path.join(DATA_DIR, "ai-zzdn-class-binary")
CLASS_COARSE_DIR = os.path.join(DATA_DIR, "ai-zzdn-class-coarse")
CLASS_FIVE_DIR = os.path.join(DATA_DIR, "ai-zzdn-class-five")
CODE_DIR = HERE

_PARENT = os.path.dirname(CODE_DIR)
EXCEL_PATH = os.path.join(_PARENT, "class_mapping.xlsx")
COARSE_XLSX = os.path.join(_PARENT, "coarse_class_mapping.xlsx")
DETECT_CLASSES_JSON = os.path.join(DATA_DIR, "ai-zzdn", "classes.json")
