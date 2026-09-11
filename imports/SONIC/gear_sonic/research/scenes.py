"""Deterministic primitive scenes shared by physics spawning and field queries."""
from .geometry import Solid


def stair_side_clutter():
    # Outside the pinned stair reference's swept path, not an avoidance benchmark.
    return (
        Solid("box", "box", (1.1, 0.0, 0.3), (0.4, 0.5, 0.6), yaw=0.25),
        Solid("pole", "cylinder", (-1.0, 0.4, 0.9), (0.12, 0.12, 1.8)),
        Solid("rail", "box", (0.85, 0.35, 1.5), (0.08, 2.4, 0.08)),
        Solid("beam", "box", (0., 0.3, 3.1), (1.9, 0.15, 0.15)),
        Solid("opening_left", "box", (-0.85, 1.8, 1.0), (0.25, 0.2, 2.0)),
        Solid("opening_right", "box", (0.85, 1.8, 1.0), (0.25, 0.2, 2.0)),
    )


def contact_fixture_solids():
    # Deliberately separated test stations: no cross-fixture contacts.
    return (
        Solid("tread", "box", (0., 0., 0.25), (1., 1., 0.5), support_top=True),
        Solid("rail", "box", (3., 0., 0.7), (0.1, 1., 0.1)),
        Solid("beam", "box", (6., 0., 1.0), (1., 1., 0.2)),
        Solid("pole", "cylinder", (9., 0., 0.5), (0.2, 0.2, 1.0)),
    )
