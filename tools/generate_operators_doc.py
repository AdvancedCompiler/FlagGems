#!/usr/bin/env python3
"""
Tool to generate FlagGems operators documentation.

This script extracts all operators from flag_gems.ops and flag_gems.fused
modules and generates a markdown document in the format of docs/operators.md.

Usage:
    python -m tools.generate_operators_doc [--output OUTPUT_PATH]

Output:
    By default, generates docs/operators.md in the project root.
"""

import argparse
import sys
from pathlib import Path
from typing import List


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is at flaggems-312/tools/generate_operators_doc.py
    # Project root is two levels up
    return Path(__file__).parent.parent


def extract_all_from_init(init_file_path: Path) -> List[str]:
    """
    Extract the __all__ list from an __init__.py file.

    Args:
        init_file_path: Path to the __init__.py file

    Returns:
        List of operator names
    """
    if not init_file_path.exists():
        print(f"Warning: {init_file_path} not found", file=sys.stderr)
        return []

    content = init_file_path.read_text()

    # Find __all__ = [...] block
    start = content.find("__all__")
    if start == -1:
        return []

    bracket_start = content.find("[", start)
    if bracket_start == -1:
        return []

    depth = 0
    bracket_end = bracket_start
    for i, char in enumerate(content[bracket_start:], start=bracket_start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                bracket_end = i
                break

    all_content = content[bracket_start + 1 : bracket_end]

    operators = []
    for line in all_content.split("\n"):
        line = line.strip().strip(",").strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            op = line[1:-1]
        elif line.startswith("'") and line.endswith("'"):
            op = line[1:-1]
        else:
            continue
        if op:
            operators.append(op)

    return operators


def format_operator_name(op: str) -> str:
    """
    Format operator name for markdown.

    In-place operators (ending with _) need to escape the underscore
    in markdown (e.g., "abs_ -> abs\\_").
    Args:
        op: Operator name

    Returns:
        Formatted operator name for markdown
    """
    if op.endswith("_"):
        return op[:-1] + "\\_"
    return op


def extract_base_operator(op: str) -> tuple:
    """
    Extract the base operator name from a variant.

    For example:
    - "bitwise_and_scalar" -> ("bitwise_and", False)
    - "bitwise_and_scalar_" -> ("bitwise_and", True)
    - "add_out" -> ("add", False)
    - "add" -> ("add", False)
    - "add_" -> ("add", True)

    Args:
        op: Operator name

    Returns:
        Tuple of (base_name, is_inplace)
    """
    # Suffixes that indicate overloaded variants (to be filtered out)
    # Order matters: longer suffixes first for proper matching
    overload_suffixes = [
        "_tensor_float",
        "_float_tensor",
        "_tensor_tensor",
        "_tensor_scalar",
        "_scalar_other",
        "_scalar_self",
        "_self_int",
        "_self_tensor",
        "_backward",
        "_forward",
        "_start",
        "_m",
        "_out",
        "_tensor",
        "_scalar",
        "_dim",
        "_dims",
        "_mode",
        "_int",
    ]

    is_inplace = False
    base = op

    # Check if it's an in-place operator (ends with _ but not __)
    if base.endswith("_") and not base.endswith("__"):
        is_inplace = True
        base = base[:-1]

    # Try to strip overload suffixes
    for suffix in overload_suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    return (base, is_inplace)


def filter_variants(operators: List[str]) -> List[str]:
    """
    Filter out variant operators.

    Filtering rules:
    1. For operator x, only keep x and x_ (in-place version)
    2. Filter out overloaded variants:
       - _out (e.g., "add_out")
       - _tensor (e.g., "bitwise_and_tensor")
       - _scalar (e.g., "bitwise_and_scalar")
       - _tensor_float, _float_tensor, _tensor_tensor (e.g., "normal_tensor_float")
       - _self_int, _self_tensor (e.g., "repeat_interleave_self_int")
       - _dim, _dims (e.g., "all_dim")
       - _backward, _forward (e.g., "softmax_backward")
    3. If base operator doesn't exist but variants do, generate the base

    Args:
        operators: List of operator names

    Returns:
        Filtered list of operator names
    """
    op_set = set(operators)
    base_ops = set()  # Non-inplace base operators
    inplace_ops = set()  # In-place base operators

    # First pass: identify operators that are already base forms
    for op in operators:
        base, is_inplace = extract_base_operator(op)
        # If the operator IS the base form (no suffix was stripped)
        if op == base or op == base + "_":
            if is_inplace:
                inplace_ops.add(op)
            else:
                base_ops.add(op)

    # Second pass: for variants, extract base and add if not exists
    for op in operators:
        base, is_inplace = extract_base_operator(op)

        # Skip if this is already a base form
        if op == base or op == base + "_":
            continue

        # This is a variant - we need to add the base if it doesn't exist
        if is_inplace:
            # e.g., bitwise_and_scalar_ -> need bitwise_and_
            inplace_name = base + "_"
            if inplace_name not in op_set and inplace_name not in inplace_ops:
                inplace_ops.add(inplace_name)
        else:
            # e.g., bitwise_and_scalar -> need bitwise_and
            if base not in op_set and base not in base_ops:
                base_ops.add(base)

    # Combine results
    result = base_ops | inplace_ops

    return list(result)


def generate_operators_doc(output_path: Path = None) -> List[str]:
    """
    Generate the operators documentation.

    Args:
        output_path: Optional path to write the output file

    Returns:
        List of all operator names (sorted)
    """
    project_root = get_project_root()

    ops_init = project_root / "src" / "flag_gems" / "ops" / "__init__.py"
    fused_init = project_root / "src" / "flag_gems" / "fused" / "__init__.py"

    ops_list = extract_all_from_init(ops_init)
    fused_list = extract_all_from_init(fused_init)

    all_operators = list(set(ops_list + fused_list))

    # Filter out variant operators (_backward, _forward)
    all_operators = filter_variants(all_operators)

    all_operators.sort(key=lambda x: (x.lstrip("_").lower(), x))

    print(f"Found {len(ops_list)} operators in flag_gems.ops")
    print(f"Found {len(fused_list)} operators in flag_gems.fused")
    print(f"Total unique operators: {len(all_operators)}")

    lines = ["## Operator List", ""]

    for op in all_operators:
        lines.append(f"- {format_operator_name(op)}")

    markdown_content = "\n".join(lines) + "\n"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content)
        print(f"Generated operators documentation: {output_path}")

    return all_operators


def main():
    parser = argparse.ArgumentParser(
        description="Generate FlagGems operators documentation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for the generated markdown file",
        default=None,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if operators.md is up to date, exit with 0 if yes, 1 if not",
    )

    args = parser.parse_args()

    project_root = get_project_root()

    if args.output is None:
        args.output = project_root / "docs" / "operators.md"

    operators = generate_operators_doc(args.output)

    if args.check:
        if args.output.exists():
            existing_content = args.output.read_text()
            new_content = "\n".join(["## Operator List", ""]) + "\n"
            new_content += "\n".join(
                f"- {format_operator_name(op)}" for op in operators
            )
            new_content += "\n"

            if existing_content.strip() == new_content.strip():
                print("operators.md is up to date")
                sys.exit(0)
            else:
                print("operators.md is out of date")
                sys.exit(1)
        else:
            print(f"{args.output} does not exist")
            sys.exit(1)


if __name__ == "__main__":
    main()
