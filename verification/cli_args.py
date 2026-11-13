"""Command line arguments shared by the verification test scripts."""

import argparse


def positive_int(value):
    """argparse type: an integer >= 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return number


def add_particles_per_cell_argument(parser):
    """--particles-per-cell: overrides ``released_particles_per_cell`` of the
    test's SETTINGS (default None keeps the test's own value)."""
    parser.add_argument(
        "--particles-per-cell", type=positive_int, default=None, metavar="N",
        help="released particles per release cell (default: the test's SETTINGS)")
