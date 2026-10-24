"""Regla de semillas (núcleo §4.1)."""

import numpy as np
import pytest

from src.common.seeds import fijar_semillas, make_rng, seeds_for_runs


def test_make_rng_reproducible_y_solo_importa_la_suma():
    assert make_rng(0, 5).random() == make_rng(5, 0).random() == make_rng(3, 2).random()


def test_seeds_for_runs():
    assert seeds_for_runs(10, 3) == [10, 11, 12]


@pytest.mark.parametrize("malo", [-1, 1.5, True, "3"])
def test_semilla_invalida(malo):
    with pytest.raises((TypeError, ValueError)):
        make_rng(malo, 0)


def test_fijar_semillas_reproduce_torch_y_numpy():
    torch = pytest.importorskip("torch")
    a_np = fijar_semillas(7).random(3)
    a_t = torch.rand(3)
    b_np = fijar_semillas(7).random(3)
    b_t = torch.rand(3)
    assert np.array_equal(a_np, b_np) and torch.equal(a_t, b_t)
