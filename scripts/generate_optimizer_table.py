from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, List, Optional, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Always import the local repository instead of an already installed Mealpy.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mealpy
from mealpy.optimizer import Optimizer

DEFAULT_OUTPUT_PATH = (PROJECT_ROOT / "docs" / "source" / "_generated" / "optimizer_table.csv")

GROUP_LABELS = {"bio_based": "Bio-inspired", "evolutionary_based": "Evolutionary", "game_based": "Game",
    "human_based": "Human", "math_based": "Mathematics", "music_based": "Music", "physics_based": "Physics",
    "sota_based": "SOTA", "swarm_based": "Swarm", "system_based": "System", }

GROUP_ORDER = {group_name: index for index, group_name in enumerate(GROUP_LABELS)}
KIND_LABELS = {"original": "Original", "variant": "Variant", "hybrid": "Hybrid", "sota": "SOTA", "developed": "Developed", }
KIND_ORDER = {"original": 0, "variant": 1, "hybrid": 2, "sota": 3, "developed": 4, }

# The CSV is parsed as reStructuredText. Escaping the asterisk avoids it being
# interpreted as incomplete emphasis markup while still displaying "*" in HTML.
REPEAT_MARKER = r"\*"


@dataclass(frozen=True)
class AlgorithmRow:
    """Normalized metadata for one optimizer class."""

    group_key: str
    group: str
    name: Optional[str]
    module: str
    class_name: str
    year: Optional[int]
    parameters: int
    difficulty: str
    kind: str
    source_line: int


def normalize_value(value: Any) -> str:
    """Convert strings and string-based Enum members to plain strings."""
    return str(getattr(value, "value", value))


def get_group_key(module_name: str) -> Optional[str]:
    """Extract a known Mealpy group from a fully qualified module name."""
    for component in module_name.split("."):
        if component in GROUP_LABELS:
            return component
    return None


def get_source_line(cls: Type[Any]) -> int:
    """Return the first source-code line of a class."""
    try:
        _, line_number = inspect.getsourcelines(cls)
        return line_number
    except (OSError, TypeError):
        return sys.maxsize


def count_constructor_parameters(cls: Type[Optimizer]) -> int:
    """
    Count explicitly declared constructor parameters.

    The following parameters are excluded:
    - self
    - *args
    - **kwargs
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return 0
    excluded_kinds = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, }

    return sum(1 for parameter in signature.parameters.values() if
               parameter.name != "self" and parameter.kind not in excluded_kinds)


def discover_algorithm_modules() -> Iterable[ModuleType]:
    """Discover and import all Mealpy algorithm modules."""
    package_prefix = "{}.".format(mealpy.__name__)
    for module_info in pkgutil.walk_packages(mealpy.__path__, prefix=package_prefix, ):
        if module_info.ispkg:
            continue
        if get_group_key(module_info.name) is None:
            continue
        try:
            yield importlib.import_module(module_info.name)
        except Exception as error:
            raise RuntimeError("Failed to import algorithm module: {!r}".format(module_info.name)) from error


def get_optimizer_classes(module: ModuleType, ) -> List[Type[Optimizer]]:
    """Find public Optimizer subclasses defined directly in a module."""
    classes = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls is Optimizer:
            continue
        if not issubclass(cls, Optimizer):
            continue
        if cls.__dict__.get("DEPRECATED", False):
            continue

        # Ignore classes imported into this module from somewhere else.
        if cls.__module__ != module.__name__:
            continue
        if cls.__name__.startswith("_"):
            continue
        if inspect.isabstract(cls):
            continue

        classes.append(cls)

    classes.sort(key=get_source_line)
    return classes


def validate_opt_info(module: ModuleType, cls: Type[Optimizer], info: Any, ) -> None:
    """Validate the metadata required by the generated table."""
    qualified_name = "{}.{}".format(module.__name__, cls.__name__, )
    if info is None:
        raise ValueError("{} defines OPT_INFO=None.".format(qualified_name))

    required_fields = ("difficulty", "kind",)
    missing_fields = [field_name for field_name in required_fields if not hasattr(info, field_name)]
    if missing_fields:
        raise ValueError(
            "{} has invalid OPT_INFO. Missing fields: {}.".format(qualified_name, ", ".join(missing_fields), ))

    difficulty = normalize_value(info.difficulty)
    kind = normalize_value(info.kind)
    valid_difficulties = {"easy", "medium", "hard", "nightmare", }
    if difficulty not in valid_difficulties:
        raise ValueError("{} has invalid difficulty {!r}.".format(qualified_name, difficulty, ))
    if kind not in KIND_LABELS:
        raise ValueError("{} has invalid kind {!r}.".format(qualified_name, kind, ))

    name = getattr(info, "name", None)
    year = getattr(info, "year", None)

    if name is not None:
        if not isinstance(name, str):
            raise TypeError("{} OPT_INFO.name must be a string or None.".format(qualified_name))
        if not name.strip():
            raise ValueError("{} OPT_INFO.name cannot be empty.".format(qualified_name))

    if year is not None:
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("{} OPT_INFO.year must be an integer or None.".format(qualified_name))
        if not 1800 <= year <= 2100:
            raise ValueError("{} has invalid publication year {}.".format(qualified_name, year, ))


def create_algorithm_row(module: ModuleType, cls: Type[Optimizer], ) -> AlgorithmRow:
    """Convert one Optimizer class into a normalized table row."""
    # Use cls.__dict__ so inherited OPT_INFO values are not accepted.
    info = cls.__dict__.get("OPT_INFO")
    validate_opt_info(module=module, cls=cls, info=info, )
    group_key = get_group_key(module.__name__)

    if group_key is None:
        raise ValueError("Cannot determine group for module {!r}.".format(module.__name__))
    module_name = module.__name__.rsplit(".", 1)[-1]

    return AlgorithmRow(group_key=group_key, group=GROUP_LABELS[group_key], name=getattr(info, "name", None),
        module=module_name, class_name=cls.__name__, year=getattr(info, "year", None),
        parameters=count_constructor_parameters(cls), difficulty=normalize_value(info.difficulty),
        kind=normalize_value(info.kind), source_line=get_source_line(cls), )


def collect_algorithm_rows(strict: bool = False, ) -> Tuple[List[AlgorithmRow], List[str]]:
    """Collect metadata from all Mealpy Optimizer subclasses."""
    rows = []
    missing_metadata = []

    for module in discover_algorithm_modules():
        for cls in get_optimizer_classes(module):
            qualified_name = "{}.{}".format(module.__name__, cls.__name__, )
            if "OPT_INFO" not in cls.__dict__:
                missing_metadata.append(qualified_name)
                continue
            rows.append(create_algorithm_row(module=module, cls=cls, ))

    if strict and missing_metadata:
        formatted_classes = "\n".join("  - {}".format(class_name) for class_name in missing_metadata)

        raise RuntimeError("The following Optimizer classes do not define OPT_INFO:\n{}".format(formatted_classes))

    rows.sort(key=lambda row: (GROUP_ORDER.get(row.group_key, sys.maxsize),
                               row.module.casefold(), KIND_ORDER.get(row.kind, sys.maxsize), row.source_line, row.class_name.casefold(),))

    return rows, missing_metadata


def render_csv(rows: List[AlgorithmRow], include_kind: bool = True, ) -> str:
    """Render optimizer metadata as CSV text."""
    headers = ["Group", "Name", "Module", "Class", "Year", "Paras", "Difficulty", ]
    if include_kind:
        headers.append("Kind")

    output = StringIO(newline="")
    writer = csv.writer(output, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n", )
    writer.writerow(headers)
    previous_module = None

    for row in rows:
        displayed_module = (REPEAT_MARKER if previous_module == row.module else row.module)
        displayed_name = (row.name if row.name is not None else REPEAT_MARKER)
        displayed_year = (row.year if row.year is not None else REPEAT_MARKER)
        csv_row = [row.group, displayed_name, displayed_module, row.class_name, displayed_year, row.parameters,
            row.difficulty, ]
        if include_kind:
            csv_row.append(KIND_LABELS.get(row.kind, row.kind.replace("_", " ").title(), ))

        writer.writerow(csv_row)
        previous_module = row.module

    return output.getvalue()


def write_text_file(output_path: Path, content: str, ) -> None:
    """Write generated text using UTF-8 and stable Unix line endings."""
    output_path.parent.mkdir(parents=True, exist_ok=True, )
    with output_path.open(mode="w", encoding="utf-8", newline="", ) as file:
        file.write(content)


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file without changing line endings."""
    with path.open(mode="r", encoding="utf-8", newline="", ) as file:
        return file.read()


def write_or_check_output(output_path: Path, content: str, check: bool = False, ) -> int:
    """Write the CSV or verify that an existing CSV is up to date."""
    if check:
        if not output_path.exists():
            print("Generated CSV does not exist: {}".format(output_path), file=sys.stderr, )
            return 1
        existing_content = read_text_file(output_path)
        if existing_content != content:
            print("Generated CSV is outdated: {}".format(output_path), file=sys.stderr, )
            return 1
        print("Generated CSV is up to date: {}".format(output_path))
        return 0

    write_text_file(output_path=output_path, content=content, )
    print("Generated optimizer table: {}".format(output_path))
    return 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=("Generate the Mealpy optimizer classification table as CSV."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=("Output CSV path. Default: {}".format(DEFAULT_OUTPUT_PATH)), )
    parser.add_argument("--strict", action="store_true", help=("Fail if any public Optimizer subclass does not define OPT_INFO."), )
    parser.add_argument("--without-kind", action="store_true", help="Generate the seven-column table without Kind.", )
    parser.add_argument("--check", action="store_true", help=("Check whether the existing CSV is up to date without modifying it."), )

    return parser.parse_args()


def main() -> int:
    """Generate the optimizer classification CSV."""
    args = parse_arguments()
    rows, missing_metadata = collect_algorithm_rows(strict=args.strict, )
    csv_content = render_csv(rows=rows, include_kind=not args.without_kind, )
    print("Discovered {} optimizer classes with OPT_INFO.".format(len(rows)))

    if missing_metadata and not args.strict:
        print("Skipped {} classes without OPT_INFO:".format(len(missing_metadata)))
        for class_name in missing_metadata:
            print("  - {}".format(class_name))

    return write_or_check_output(output_path=args.output, content=csv_content, check=args.check, )


if __name__ == "__main__":
    raise SystemExit(main())
