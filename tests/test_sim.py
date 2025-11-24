# tests/test_sim.py
# -*- coding: utf-8 -*-

"""
Suite de pruebas unitaria para sim.py

Cubre:
- ALU (operaciones y flags).
- Pipeline demo (correctitud básica y métricas).
- Caché directa (API y experimentos).
- Barrido B/C (consistencia de parámetros).
- Tabla de verdad resumida de ALU (generación).

Ejecución:
  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List

import sim


# ============================================================================
# Utilidades de apoyo
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class _GlobalSnapshot:
    """Guarda/restaura globals relevantes de sim.py para aislar pruebas."""

    def __init__(self) -> None:
        self.ALU_ASSERTIONS_ENABLED = sim.ALU_ASSERTIONS_ENABLED
        self.ALU_WIDTH_BITS = sim.ALU_WIDTH_BITS
        self.REG_WIDTH_BITS = sim.REG_WIDTH_BITS
        self.CACHE_CAPACITY_BYTES = sim.CACHE_CAPACITY_BYTES
        self.CACHE_LINE_BYTES = sim.CACHE_LINE_BYTES
        self.CACHE_HIT_TIME = sim.CACHE_HIT_TIME
        self.CACHE_MISS_PENALTY = sim.CACHE_MISS_PENALTY
        self.PSEUDO_RANDOM_SEED = sim.PSEUDO_RANDOM_SEED
        self.EXPERIMENT_STEPS = sim.EXPERIMENT_STEPS
        self.EXPERIMENT_RANGE_BYTES = sim.EXPERIMENT_RANGE_BYTES
        self.EXPERIMENT_STRIDE_BYTES = sim.EXPERIMENT_STRIDE_BYTES

    def restore(self) -> None:
        sim.ALU_ASSERTIONS_ENABLED = self.ALU_ASSERTIONS_ENABLED
        sim.ALU_WIDTH_BITS = self.ALU_WIDTH_BITS
        sim.REG_WIDTH_BITS = self.REG_WIDTH_BITS
        sim.CACHE_CAPACITY_BYTES = self.CACHE_CAPACITY_BYTES
        sim.CACHE_LINE_BYTES = self.CACHE_LINE_BYTES
        sim.CACHE_HIT_TIME = self.CACHE_HIT_TIME
        sim.CACHE_MISS_PENALTY = self.CACHE_MISS_PENALTY
        sim.PSEUDO_RANDOM_SEED = self.PSEUDO_RANDOM_SEED
        sim.EXPERIMENT_STEPS = self.EXPERIMENT_STEPS
        sim.EXPERIMENT_RANGE_BYTES = self.EXPERIMENT_RANGE_BYTES
        sim.EXPERIMENT_STRIDE_BYTES = self.EXPERIMENT_STRIDE_BYTES


# ============================================================================
# ALU — operaciones y flags
# ============================================================================

class TestALUCore(unittest.TestCase):
    """Pruebas de núcleo ALU: operaciones A1 y flags A2."""

    def setUp(self) -> None:
        self._snap = _GlobalSnapshot()
        # Acelera pruebas
        sim.ALU_ASSERTIONS_ENABLED = True
        sim.ALU_WIDTH_BITS = 16

    def tearDown(self) -> None:
        self._snap.restore()

    def test_add_overflow_signed(self) -> None:
        n = sim.ALU_WIDTH_BITS
        alu = sim.ALU(n)
        max_pos = (1 << (n - 1)) - 1
        r = alu.exec(sim.AluOp.ADD, max_pos, 1)
        self.assertEqual(r.v, 1, "ADD debería marcar overflow con signo en max_pos+1")

    def test_sub_borrow_flag(self) -> None:
        alu = sim.ALU(sim.ALU_WIDTH_BITS)
        r = alu.exec(sim.AluOp.SUB, 0, 1)
        self.assertEqual(r.c, 0, "SUB 0-1 debe indicar borrow (C=0 por convención no-borrow)")

    def test_logic_identities(self) -> None:
        n = sim.ALU_WIDTH_BITS
        alu = sim.ALU(n)
        x = 0x1234
        self.assertEqual(alu.exec(sim.AluOp.AND, x, 0).value, 0)
        self.assertEqual(alu.exec(sim.AluOp.ORR, x, 0).value, sim.mask_n_bits(x, n))
        self.assertEqual(alu.exec(sim.AluOp.EOR, x, x).value, 0)
        self.assertEqual(sim.mask_n_bits(x ^ alu.exec(sim.AluOp.NOT, x).value, n), (1 << n) - 1)

    def test_shift_and_slt(self) -> None:
        n = sim.ALU_WIDTH_BITS
        alu = sim.ALU(n)
        # SLL/SRL
        a = 0b11
        self.assertEqual(alu.exec(sim.AluOp.SLL, a, 1).value, sim.mask_n_bits(a << 1, n))
        self.assertEqual(alu.exec(sim.AluOp.SRL, a, 1).value, 1)
        # SLT signed vs unsigned
        min_neg = 1 << (n - 1)
        max_pos = (1 << (n - 1)) - 1
        self.assertEqual(alu.exec(sim.AluOp.SLT_SIGNED, min_neg, max_pos).value, 1)
        self.assertEqual(alu.exec(sim.AluOp.SLT_UNSIGNED, min_neg, max_pos).value, 0)


# ============================================================================
# Pipeline — programa de demostración y métricas
# ============================================================================

class TestPipelineDemo(unittest.TestCase):
    """Correctitud del programa demo y métricas consistentes."""

    def setUp(self) -> None:
        self._snap = _GlobalSnapshot()
        # Acelera y estabiliza ejecución
        sim.ALU_ASSERTIONS_ENABLED = True
        sim.PSEUDO_RANDOM_SEED = 1337

    def tearDown(self) -> None:
        self._snap.restore()

    def test_run_program_metrics_small(self) -> None:
        program = sim.build_demo_program()
        m = sim.run_program_and_metrics(program, data_len=8)
        # Estructura mínima
        for k in ("cycles", "retired", "stalls", "cpi_milli", "stored_max", "expected_max"):
            self.assertIn(k, m)
        # Consistencias básicas
        self.assertGreater(m["cycles"], 0)
        self.assertGreater(m["retired"], 0)
        self.assertGreaterEqual(m["cycles"], m["retired"])
        self.assertEqual(m["stored_max"], m["expected_max"], "El valor almacenado debe coincidir con el máximo esperado")


# ============================================================================
# Caché directa — API básica y experimentos
# ============================================================================

class TestCacheDirectMapped(unittest.TestCase):
    """API de caché y propiedades en patrones simples."""

    def setUp(self) -> None:
        self._snap = _GlobalSnapshot()
        # Parametrización reducida para rapidez
        sim.CACHE_CAPACITY_BYTES = 1024
        sim.CACHE_LINE_BYTES = 32
        sim.CACHE_HIT_TIME = 1
        sim.CACHE_MISS_PENALTY = 50
        sim.EXPERIMENT_STEPS = 2000
        sim.EXPERIMENT_RANGE_BYTES = 16 * 1024
        sim.EXPERIMENT_STRIDE_BYTES = sim.CACHE_CAPACITY_BYTES

    def tearDown(self) -> None:
        self._snap.restore()

    def test_cache_line_and_hit(self) -> None:
        c = sim.DirectMappedCache(sim.CACHE_CAPACITY_BYTES, sim.CACHE_LINE_BYTES)
        addr = 0x1000
        miss1 = c.access(addr)
        hit2 = c.access(addr)
        self.assertFalse(miss1)
        self.assertTrue(hit2)
        rep = c.report()
        self.assertAlmostEqual(rep["hit_rate"], 0.5, delta=0.5)

    def test_experiments_present_and_fields(self) -> None:
        out = sim.run_cache_experiments()
        self.assertCountEqual(list(out.keys()), ["linear", "stride_conflict", "random"])
        for name, rep in out.items():
            for k in ("hit_rate", "miss_rate", "AMAT", "capacity_bytes", "line_bytes", "sets"):
                self.assertIn(k, rep, f"Falta campo {k} en experimento {name}")
            self.assertGreaterEqual(rep["hit_rate"], 0.0)
            self.assertLessEqual(rep["hit_rate"], 1.0)
            self.assertGreater(rep["AMAT"], 0.0)


# ============================================================================
# Barrido B/C — consistencia de combinaciones y métricas
# ============================================================================

class TestSweepBC(unittest.TestCase):
    """Valida combinaciones (C,B), potencias de 2 y AMAT finito."""

    def setUp(self) -> None:
        self._snap = _GlobalSnapshot()
        # Paso pequeño para que sea rápido pero no trivial
        sim.CACHE_HIT_TIME = 1
        sim.CACHE_MISS_PENALTY = 50
        sim.EXPERIMENT_STEPS = 1000
        sim.EXPERIMENT_RANGE_BYTES = 8 * 1024

    def tearDown(self) -> None:
        self._snap.restore()

    def test_sweep_presets_valid(self) -> None:
        rows: List[Dict[str, Any]] = sim.sweep_presets()
        self.assertGreater(len(rows), 0, "El barrido no debe estar vacío")
        valid_scenarios = {"linear", "stride_conflict", "random"}

        for r in rows:
            self.assertIn(r.get("scenario"), valid_scenarios)
            C = int(r.get("C"))
            B = int(r.get("B"))
            self.assertGreater(C, 0)
            self.assertGreater(B, 0)
            self.assertTrue(_is_power_of_two(B))
            self.assertEqual(C % B, 0, "C debe ser múltiplo de B")
            # AMAT finito y positivo
            amat_val = float(r.get("AMAT"))
            self.assertTrue(math.isfinite(amat_val) and amat_val > 0.0)


# ============================================================================
# Tabla de verdad resumida — generación y forma
# ============================================================================

class TestALUTruthTable(unittest.TestCase):
    """Comprueba que la tabla de verdad resumida se genere y tenga campos esperados."""

    def setUp(self) -> None:
        self._snap = _GlobalSnapshot()
        sim.ALU_ASSERTIONS_ENABLED = True
        sim.ALU_WIDTH_BITS = 16

    def tearDown(self) -> None:
        self._snap.restore()

    def test_truth_table_generation(self) -> None:
        table = sim.run_all_alu_tests_and_table()
        self.assertIsInstance(table, list)
        self.assertGreater(len(table), 0, "La tabla no debería estar vacía")
        # Campos esperados en las filas
        row0 = table[0]
        for k in ("op", "a", "b", "value", "Z", "N", "C", "V"):
            self.assertIn(k, row0)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
