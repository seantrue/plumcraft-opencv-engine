"""The engine as PlumCraft sees it. The golden catalogue catches a renamed port, which would break
saved workflows: when a change is meant, `UPDATE_GOLDEN=1 uv run pytest` rewrites it."""

import json
import os
from pathlib import Path

from plumcraft_engine.cli import Local
from plumcraft_engine.conformance import Run

from opencv_engine import OpencvEngine

GOLDEN = Path(__file__).parent / "golden" / "schema.json"


def test_the_catalogue_matches_the_golden(engine):
    schema = engine.schema()
    got = {"wire_types": schema["wire_types"], "nodes": schema["nodes"]}
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
    assert got == json.loads(GOLDEN.read_text("utf-8"))


def test_the_demo_runs_every_node(demo):
    conn = Local(OpencvEngine)
    try:
        states = Run(conn, demo, "demo").states()
    finally:
        conn.close()
    assert states == {n: "done" for n in demo["nodes"]}


def test_the_cartoon_group_is_the_cartoon_node_in_steps(engine):
    """`examples/cartoon.json`: every node runs, and its steps, followed by hand with the values in
    the file, look like the Cartoon node: close, not equal, as it works in floats and the node in 8 bits."""
    graph = json.loads((Path(__file__).parents[1] / "examples" / "cartoon.json").read_text("utf-8"))
    conn = Local(OpencvEngine)
    try:
        assert Run(conn, graph, "cartoon").states() == {n: "done" for n in graph["nodes"]}
    finally:
        conn.close()
    v = {n: d.get("values", {}) for n, d in graph["nodes"].items()}
    source = engine.test_picture(v["source"]["width"], v["source"]["height"])
    flat = engine.pyramid_down(engine.pyramid_down(source))
    for n in ("smooth1", "smooth2", "smooth3"):
        flat = engine.bilateral(flat, v[n]["diameter"], v[n]["sigma_colour"], v[n]["sigma_space"])
    flat = engine.pyramid_up(engine.pyramid_up(flat))
    lines = engine.adaptive_threshold(engine.median_blur(engine.to_grey(source), v["median"]["size"]), v["lines"]["size"], v["lines"]["offset"], False)
    steps = engine.bitwise(flat, flat, "and", lines)
    node = engine.cartoon(source, 2, 3, v["lines"]["size"])
    assert steps.shape == node.shape
    assert abs(steps - node).mean() < 0.05, abs(steps - node).mean()
