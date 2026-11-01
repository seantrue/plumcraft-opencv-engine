"""PlumCraft's conformance checks: what every engine must do, in process and over HTTP."""

from plumcraft_engine.conformance import assert_conformant

from opencv_engine import OpencvEngine


def test_the_engine_is_conformant(demo):
    assert_conformant(OpencvEngine, demo=demo, slow="opencv.slow")
