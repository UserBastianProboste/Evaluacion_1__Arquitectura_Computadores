#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Interfaz TUI (curses) integral para sim.py

Funciones principales:
  1) Ejecutar: Pipeline + Caché (JSON) [siempre guarda en ./outputs con timestamp]
  2) Ejecutar: Tests ALU completos (JSON)
  3) Ejecutar: Barrido B/C (JSON)
  4) Ejecutar: Pruebas unitarias (unittest) con fallback a tests ALU
  5) Explorar resultados (TUI): listar JSONs de ./outputs, ver detalles “human-friendly”
  6) Exportar CSVs agregados (si hay pandas)
  7) Editor de parámetros
  8) Robustez: contención de errores y fallback a modo CLI si curses no está disponible

Requisitos:
- sim.py debe estar en el mismo directorio.
- Python 3.10+.
"""

from __future__ import annotations

import curses
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sim  # motor/simulador


# ============================================================================
#                          PARÁMETROS EDITABLES (UI)
# ============================================================================

class Params:
    """
    Contenedor de parámetros editables con defaults seguros.
    """

    def __init__(self) -> None:
        # Pipeline / datos
        self.data_len: int = 64

        # Caché (aplica a los 3 experimentos estándar)
        self.cache_capacity: int = sim.CACHE_CAPACITY_BYTES
        self.line_bytes: int = sim.CACHE_LINE_BYTES

        # ALU / reproducibilidad
        self.no_alu_asserts: bool = False
        self.seed: int = sim.PSEUDO_RANDOM_SEED

        # --- Pipeline
        self.forwarding_enabled: bool = sim.FORWARDING_ENABLED

        # Tiempos para AMAT (supuestos explícitos)
        self.hit_time: int = sim.CACHE_HIT_TIME
        self.miss_penalty: int = sim.CACHE_MISS_PENALTY

        # Opciones de salida
        self.outputs_dir: Path = Path("outputs")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_len": self.data_len,
            "cache_capacity": self.cache_capacity,
            "line_bytes": self.line_bytes,
            "no_alu_asserts": self.no_alu_asserts,
            "seed": self.seed,
            "hit_time": self.hit_time,
            "miss_penalty": self.miss_penalty,
            "forwarding_enabled": self.forwarding_enabled,
            "outputs_dir": str(self.outputs_dir.resolve()),
        }


# ============================================================================
#                           UTILIDADES DE SALIDA
# ============================================================================

def ensure_outputs_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(data: Dict[str, Any], outputs_dir: Path, prefix: str) -> Path:
    ensure_outputs_dir(outputs_dir)
    fname = f"{prefix}_{timestamp()}.json"
    out = outputs_dir / fname
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


# ============================================================================
#                             LÓGICA DE EJECUCIÓN
# ============================================================================

def apply_params_to_sim(p: Params) -> None:
    """
    Copia parámetros seleccionados a sim.* antes de ejecutar.
    """
    sim.ALU_ASSERTIONS_ENABLED = not p.no_alu_asserts
    sim.CACHE_CAPACITY_BYTES = int(p.cache_capacity)
    sim.CACHE_LINE_BYTES = int(p.line_bytes)
    sim.CACHE_HIT_TIME = int(p.hit_time)
    sim.CACHE_MISS_PENALTY = int(p.miss_penalty)
    sim.PSEUDO_RANDOM_SEED = int(p.seed)
    sim.FORWARDING_ENABLED = bool(p.forwarding_enabled)

def run_pipeline_and_cache(p: Params) -> Dict[str, Any]:
    """
    Ejecuta:
      - Sanidad ALU (si aplica)
      - Programa demo en pipeline (métricas)
      - Experimentos de caché (AMAT)
    Devuelve estructura lista para serializar.
    """
    apply_params_to_sim(p)

    # 1) Sanidad ALU
    try:
        sim.alu_sanity_checks()
        alu_sanity_ok = True
        alu_sanity_error = None
    except Exception as exc:  # robustez
        alu_sanity_ok = False
        alu_sanity_error = f"{type(exc).__name__}: {exc}"

    # 2) Programa demo
    try:
        program = sim.build_demo_program()
        metrics = sim.run_program_and_metrics(program, data_len=int(p.data_len))
        pipe_error = None
    except Exception as exc:
        metrics = {}
        pipe_error = f"{type(exc).__name__}: {exc}"

    # 3) Experimentos de caché
    try:
        cache_results = sim.run_cache_experiments()
        cache_error = None
    except Exception as exc:
        cache_results = {}
        cache_error = f"{type(exc).__name__}: {exc}"

    return {
        "params": p.as_dict(),
        "alu_sanity": {"ok": alu_sanity_ok, "error": alu_sanity_error},
        "pipeline_metrics": metrics,
        "pipeline_error": pipe_error,
        "cache_experiments": cache_results,
        "cache_error": cache_error,
        "generated_at": time.time(),
    }


def run_alu_full_tests(p: Params) -> Dict[str, Any]:
    """
    Ejecuta batería completa de tests de ALU y tabla de verdad resumida.
    """
    apply_params_to_sim(p)
    try:
        table = sim.run_all_alu_tests_and_table()
        ok = True
        error = None
    except Exception as exc:
        table = []
        ok = False
        error = f"{type(exc).__name__}: {exc}"

    return {
        "params": p.as_dict(),
        "full_tests_ok": ok,
        "error": error,
        "truth_table_summary": table,
        "generated_at": time.time(),
    }


def run_sweep_bc(p: Params) -> Dict[str, Any]:
    """
    Ejecuta el barrido predefinido de B/C (sensibilidad de AMAT).
    """
    apply_params_to_sim(p)
    try:
        sweep = sim.sweep_presets()
        ok = True
        error = None
    except Exception as exc:
        sweep = []
        ok = False
        error = f"{type(exc).__name__}: {exc}"

    return {
        "params": p.as_dict(),
        "sweep_ok": ok,
        "error": error,
        "sweep_results": sweep,
        "generated_at": time.time(),
    }


# ============================================================================
#                  PRUEBAS UNITARIAS (unittest) CON FALLBACK
# ============================================================================

def run_unittest_suite(outputs_dir: Path) -> Dict[str, Any]:
    """
    Intenta descubrir y ejecutar pruebas con 'unittest' en ./tests.
    Falla con resumen estructurado si no hay suite válida.
    """
    import io
    import unittest

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    loader = unittest.TestLoader()
    try:
        suite = loader.discover(start_dir="tests", pattern="test_*.py")
    except Exception as exc:
        return {
            "ok": False,
            "error": f"DiscoveryError: {type(exc).__name__}: {exc}",
            "stdout": "",
            "stderr": "",
            "generated_at": time.time(),
        }

    runner = unittest.TextTestRunner(stream=buf_out, verbosity=2)
    try:
        result = runner.run(suite)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"RunError: {type(exc).__name__}: {exc}",
            "stdout": buf_out.getvalue(),
            "stderr": buf_err.getvalue(),
            "generated_at": time.time(),
        }

    ok = result.wasSuccessful()
    summary = {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
    }

    return {
        "ok": ok,
        "summary": summary,
        "failures": [(str(t), str(tb)) for t, tb in result.failures],
        "errors": [(str(t), str(tb)) for t, tb in result.errors],
        "stdout": buf_out.getvalue(),
        "stderr": buf_err.getvalue(),
        "generated_at": time.time(),
    }


def run_unit_tests_with_fallback(p: Params) -> Dict[str, Any]:
    """
    Ejecuta pruebas unitarias:
      1) 'unittest discover' en ./tests.
      2) Fallback: tests completos de ALU (sim).
    """
    ut_out = run_unittest_suite(p.outputs_dir)
    if "summary" in ut_out or ut_out.get("ok") is True:
        return ut_out
    alu_out = run_alu_full_tests(p)
    return {
        "ok": alu_out.get("full_tests_ok", False),
        "used_fallback": True,
        "fallback_name": "sim.run_all_alu_tests_and_table",
        "fallback_payload": alu_out,
        "generated_at": time.time(),
    }


# ============================================================================
#                         VISUALIZACIÓN (Curses Viewer)
# ============================================================================

OUTPUTS_DIR = Path("outputs")
DERIVED_DIR = OUTPUTS_DIR / "derived"

def list_outputs() -> List[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(OUTPUTS_DIR.glob("*.json"))

def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_pipeline(obj: Dict[str, Any]) -> Dict[str, Any]:
    pm = obj.get("pipeline_metrics") or {}
    return {
        "cycles": pm.get("cycles"),
        "retired": pm.get("retired"),
        "stalls": pm.get("stalls"),
        "cpi_milli": pm.get("cpi_milli"),
        "stored_max": pm.get("stored_max"),
        "expected_max": pm.get("expected_max"),
    }

def normalize_cache_experiments(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    ce = obj.get("cache_experiments") or {}
    rows: List[Dict[str, Any]] = []
    for name, rep in ce.items():
        if not isinstance(rep, dict):
            continue
        rows.append({
            "scenario": name,
            "capacity_bytes": rep.get("capacity_bytes"),
            "line_bytes": rep.get("line_bytes"),
            "sets": rep.get("sets"),
            "accesses": rep.get("accesses"),
            "hits": rep.get("hits"),
            "misses": rep.get("misses"),
            "hit_rate": rep.get("hit_rate"),
            "miss_rate": rep.get("miss_rate"),
            "AMAT": rep.get("AMAT"),
        })
    return rows

def normalize_sweep(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    sweep = obj.get("sweep_results") or []
    rows: List[Dict[str, Any]] = []
    for rep in sweep:
        if not isinstance(rep, dict):
            continue
        rows.append({
            "scenario": rep.get("scenario"),
            "C": rep.get("C"),
            "B": rep.get("B"),
            "hit_rate": rep.get("hit_rate"),
            "miss_rate": rep.get("miss_rate"),
            "AMAT": rep.get("AMAT"),
            "accesses": rep.get("accesses"),
        })
    return rows

# ============================================================================
#   REEMPLAZO: renderizadores “human-friendly” para distintos tipos de JSON
#   Sustituye la función `format_obj_human` anterior por este bloque completo.
#   (Incluye helpers específicos para unittest, alu tests y sweep B/C.)
# ============================================================================

def _format_lines_pipeline_cache(obj: Dict[str, Any], file_name: str) -> List[str]:
    """Vista estándar (pipeline + cache + sweep resumen)."""
    lines: List[str] = []
    lines.append(f"# Archivo: {file_name}")
    lines.append("")
    lines.append("# ==== PIPELINE ====")
    pm = normalize_pipeline(obj)
    if any(v is not None for v in pm.values()):
        for k in ["cycles", "retired", "stalls", "cpi_milli", "stored_max", "expected_max"]:
            lines.append(f"{k:>12}: {pm.get(k)}")
    else:
        lines.append("(sin datos)")
    lines.append("")

    lines.append("# ==== CACHÉ (por escenario) ====")
    cr = normalize_cache_experiments(obj)
    if not cr:
        lines.append("(sin datos)")
    else:
        for r in cr:
            try:
                txt = (f"[{r['scenario']}] "
                       f"C={int(r['capacity_bytes'])}  "
                       f"B={int(r['line_bytes'])}  sets={int(r['sets'])}  "
                       f"hit_rate={r['hit_rate']:.4f}  AMAT={r['AMAT']:.2f}")
            except Exception:
                txt = f"[{r.get('scenario')}] (datos parciales)"
            lines.append(txt)
    lines.append("")

    lines.append("# ==== SWEEP B/C (resumen) ====")
    sr = normalize_sweep(obj)
    if not sr:
        lines.append("(sin datos)")
    else:
        by_scenario: Dict[str, int] = {}
        for r in sr:
            s = str(r.get("scenario"))
            by_scenario[s] = by_scenario.get(s, 0) + 1
        for k, v in by_scenario.items():
            lines.append(f"{k}: {v} combinaciones")
    return lines


def _format_lines_unittest(obj: Dict[str, Any], file_name: str) -> List[str]:
    """Vista específica para resultados de unittest (o fallback)."""
    lines: List[str] = []
    lines.append(f"# Archivo: {file_name}  (unittest)")
    lines.append("")

    ok = obj.get("ok")
    summary = obj.get("summary") or {}
    used_fallback = obj.get("used_fallback", False)

    if used_fallback:
        # Fallback: mostramos resultado de ALU tests integrado aquí
        lines.append("No se encontró suite unittest válida. Se usó fallback de ALU tests.")
        payload = obj.get("fallback_payload") or {}
        lines.extend(_format_lines_alu_tests(payload, file_name + " (fallback)"))
        return lines

    # Suite unittest real
    lines.append("# ==== RESUMEN ====")
    lines.append(f"ok           : {ok}")
    if summary:
        lines.append(f"testsRun     : {summary.get('testsRun')}")
        lines.append(f"failures     : {summary.get('failures')}")
        lines.append(f"errors       : {summary.get('errors')}")
        lines.append(f"skipped      : {summary.get('skipped')}")
    else:
        lines.append("(sin resumen)")

    # Failures y errors (solo nombres y primera línea)
    def _short_pairs(tag: str, pairs: List[List[str]]) -> None:
        if not pairs:
            return
        lines.append(f"\n# ==== {tag.upper()} ====")
        for i, pair in enumerate(pairs, 1):
            test_name, traceback = pair
            first_line = (traceback or "").strip().splitlines()[:1]
            lines.append(f"{i:2d}. {test_name}")
            if first_line:
                lines.append(f"    {first_line[0]}")

    _short_pairs("failures", obj.get("failures") or [])
    _short_pairs("errors", obj.get("errors") or [])

    # Últimas líneas de stdout si existen
    stdout = obj.get("stdout") or ""
    if stdout.strip():
        tail = stdout.strip().splitlines()[-10:]
        lines.append("\n# ==== STDOUT (últimas 10 líneas) ====")
        lines.extend(tail)

    return lines


def _format_lines_alu_tests(obj: Dict[str, Any], file_name: str) -> List[str]:
    """Vista específica para run_alu_tests (tabla verdad resumida)."""
    lines: List[str] = []
    lines.append(f"# Archivo: {file_name}  (ALU tests)")
    lines.append("")
    ok = obj.get("full_tests_ok")
    error = obj.get("error")
    table = obj.get("truth_table_summary") or []
    lines.append("# ==== ESTADO ====")
    lines.append(f"full_tests_ok : {ok}")
    lines.append(f"error         : {error}" if error else "error         : None")

    lines.append("\n# ==== TABLA DE VERDAD (muestra) ====")
    if not table:
        lines.append("(sin filas)")
        return lines

    # Muestra compacta de las primeras N filas por operación
    N = 6
    # Agrupar por op (sin pandas)
    by_op: Dict[str, List[Dict[str, Any]]] = {}
    for row in table:
        op = str(row.get("op"))
        by_op.setdefault(op, []).append(row)

    for op in sorted(by_op.keys()):
        lines.append(f"\n[Op: {op}] (mostrando hasta {N} filas)")
        subset = by_op[op][:N]
        lines.append(" a       b       value   Z N C V")
        for r in subset:
            a = int(r.get("a", 0))
            b = int(r.get("b", 0))
            val = int(r.get("value", 0))
            Z = r.get("Z"); Nf = r.get("N"); C = r.get("C"); V = r.get("V")
            lines.append(f"{a:<7d}{b:<8d}{val:<8d}{Z} {Nf} {C} {V}")

    # Conteo total
    total = sum(len(v) for v in by_op.values())
    lines.append(f"\nTotal de filas en tabla: {total}")
    return lines


def _format_lines_sweep(obj: Dict[str, Any], file_name: str) -> List[str]:
    """Vista específica para run_sweep_bc (sensibilidad B/C)."""
    lines: List[str] = []
    lines.append(f"# Archivo: {file_name}  (sweep B/C)")
    lines.append("")
    ok = obj.get("sweep_ok", None)
    error = obj.get("error", None)
    rows = normalize_sweep(obj)

    lines.append("# ==== ESTADO ====")
    lines.append(f"sweep_ok : {ok}")
    lines.append(f"error    : {error}" if error else "error    : None")

    if not rows:
        lines.append("\n# ==== DATOS ====")
        lines.append("(sin filas)")
        return lines

    # Resumen por escenario y mejores/peores AMAT
    lines.append("\n# ==== RESUMEN ====")
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_scenario.setdefault(str(r.get("scenario")), []).append(r)
    for sc, lst in by_scenario.items():
        count = len(lst)
        # Mejor y peor AMAT
        try:
            best = min(lst, key=lambda x: float(x.get("AMAT", float("inf"))))
            worst = max(lst, key=lambda x: float(x.get("AMAT", float("-inf"))))
            lines.append(f"{sc}: {count} combinaciones | "
                         f"mejor AMAT={best.get('AMAT'):.2f} (C={int(best.get('C'))}, B={int(best.get('B'))}) | "
                         f"peor AMAT={worst.get('AMAT'):.2f} (C={int(worst.get('C'))}, B={int(worst.get('B'))})")
        except Exception:
            lines.append(f"{sc}: {count} combinaciones")

    # Muestra top-5 global por AMAT
    try:
        top = sorted(rows, key=lambda x: float(x.get("AMAT", float("inf"))))[:5]
        lines.append("\n# ==== TOP-5 (AMAT más bajo) ====")
        lines.append(" rank  escenario     C        B     AMAT   hit_rate  miss_rate")
        for i, r in enumerate(top, 1):
            lines.append(f"{i:>4}  {r.get('scenario'):<12}{int(r.get('C')):<8}{int(r.get('B')):<7}"
                         f"{float(r.get('AMAT')):>8.2f}  {float(r.get('hit_rate')):>8.4f}  {float(r.get('miss_rate')):>9.4f}")
    except Exception:
        pass

    return lines


def format_obj_human(obj: Dict[str, Any], file_name: str) -> List[str]:
    """
    Renderizador de detalle sensible al tipo de resultado:
      - unittest -> _format_lines_unittest
      - alu tests -> _format_lines_alu_tests
      - sweep bc  -> _format_lines_sweep
      - por defecto -> pipeline+cache (como antes)
    """
    # Heurísticas de detección por esquema:
    if ("summary" in obj and ("failures" in obj or "errors" in obj)) or obj.get("used_fallback", False):
        return _format_lines_unittest(obj, file_name)

    if "truth_table_summary" in obj or "full_tests_ok" in obj:
        return _format_lines_alu_tests(obj, file_name)

    if "sweep_results" in obj or obj.get("sweep_ok") is True:
        return _format_lines_sweep(obj, file_name)

    # Default (pipeline/cache)
    return _format_lines_pipeline_cache(obj, file_name)


# ---------------------------- curses widgets --------------------------------

def draw_menu(stdscr: "curses._CursesWindow", title: str, items: List[str], idx: int) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(1, 2, title[: w - 4], curses.A_BOLD)
    for i, label in enumerate(items):
        prefix = "➤ " if i == idx else "  "
        text = f"{prefix}{label}"
        if i == idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(3 + i, 4, text[: w - 8])
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(3 + i, 4, text[: w - 8])
    stdscr.addstr(h - 2, 2, "↑/↓ mover  Enter seleccionar   q salir", curses.A_DIM)
    stdscr.refresh()

def scrollable_view(stdscr: "curses._CursesWindow", title: str, lines: List[str]) -> None:
    """
    Muestra una vista de texto scrollable.
    """
    top = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(1, 2, title[: w - 4], curses.A_BOLD)
        usable = h - 5
        for i in range(usable):
            j = top + i
            if j >= len(lines):
                break
            stdscr.addstr(3 + i, 4, lines[j][: w - 8])
        stdscr.addstr(h - 2, 2, "↑/↓ scroll  PgUp/PgDn  q volver", curses.A_DIM)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (27, ord("q")):
            return
        elif key in (curses.KEY_DOWN, ord("j")):
            top = min(max(0, len(lines) - usable), top + 1)
        elif key in (curses.KEY_UP, ord("k")):
            top = max(0, top - 1)
        elif key == curses.KEY_NPAGE:  # PgDn
            top = min(max(0, len(lines) - usable), top + usable)
        elif key == curses.KEY_PPAGE:  # PgUp
            top = max(0, top - usable)

def file_selector(stdscr: "curses._CursesWindow", files: List[Path], title: str) -> Optional[Path]:
    """
    Selector de archivos simple. Devuelve la ruta seleccionada o None.
    """
    if not files:
        h, w = stdscr.getmaxyx()
        stdscr.clear()
        stdscr.addstr(2, 2, "No hay JSONs en ./outputs", curses.A_BOLD)
        stdscr.addstr(h - 2, 2, "q volver", curses.A_DIM)
        stdscr.refresh()
        stdscr.getch()
        return None

    idx = max(0, len(files) - 1)  # por defecto el más reciente
    while True:
        labels = [f.name for f in files]
        draw_menu(stdscr, title, labels + ["Volver"], idx)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % (len(labels) + 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % (len(labels) + 1)
        elif key in (27, ord("q")):
            return None
        elif key in (10, 13):
            if idx == len(labels):
                return None
            return files[idx]

def show_message(stdscr: "curses._CursesWindow", msg: str, pause: bool = True) -> None:
    h, w = stdscr.getmaxyx()
    lines = msg.splitlines()[-3:]
    for i, ln in enumerate(lines):
        stdscr.addstr(h - 6 + i, 2, " " * (w - 4))
        stdscr.addstr(h - 6 + i, 2, ln[: w - 4])
    stdscr.refresh()
    if pause:
        stdscr.addstr(h - 2, 2, "Presiona cualquier tecla para continuar...", curses.A_DIM)
        stdscr.getch()

def prompt_input(stdscr: "curses._CursesWindow", prompt: str, default: str) -> Optional[str]:
    curses.echo()
    h, w = stdscr.getmaxyx()
    line = f"{prompt} [{default}]: "
    stdscr.addstr(h - 4, 2, " " * (w - 4))
    stdscr.addstr(h - 4, 2, line[: w - 4])
    stdscr.refresh()
    try:
        s = stdscr.getstr(h - 4, 2 + len(line), w - len(line) - 4)
    except Exception:
        curses.noecho()
        return None
    curses.noecho()
    if s is None:
        return None
    val = s.decode("utf-8").strip()
    return default if val == "" else val


# ============================================================================
#                             PANDAS / CSV (opcionales)
# ============================================================================

def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None

def export_csv_aggregate() -> Tuple[bool, str]:
    """
    Agrega todos los JSONs en ./outputs y exporta CSVs en ./outputs/derived.
    Retorna (ok, mensaje).
    """
    pd = try_import_pandas()
    if pd is None:
        return False, "pandas no está disponible. Instale con: pip install pandas"

    files = list_outputs()
    if not files:
        return False, "No se encontraron JSONs en ./outputs."

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    pipe_rows: List[Dict[str, Any]] = []
    cache_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    for fp in files:
        try:
            obj = load_json_file(fp)
        except Exception as exc:
            continue

        params = obj.get("params") or {}
        tag = {
            "file": fp.name,
            "seed": params.get("seed"),
            "C": params.get("cache_capacity") or params.get("CACHE_CAPACITY_BYTES"),
            "B": params.get("line_bytes") or params.get("CACHE_LINE_BYTES"),
            "data_len": params.get("DATA_LEN") or params.get("data_len"),
        }

        pm = normalize_pipeline(obj)
        if any(v is not None for v in pm.values()):
            pipe_rows.append({**tag, **pm})

        for row in normalize_cache_experiments(obj):
            cache_rows.append({**tag, **row})

        for row in normalize_sweep(obj):
            sweep_rows.append({**tag, **row})

    pipeline_df = pd.DataFrame(pipe_rows) if pipe_rows else pd.DataFrame()
    cache_df = pd.DataFrame(cache_rows) if cache_rows else pd.DataFrame()
    sweep_df = pd.DataFrame(sweep_rows) if sweep_rows else pd.DataFrame()

    if not pipeline_df.empty:
        pipeline_df.to_csv(DERIVED_DIR / "pipeline_summary.csv", index=False)
    if not cache_df.empty:
        cache_df.to_csv(DERIVED_DIR / "cache_summary.csv", index=False)
    if not sweep_df.empty:
        sweep_df.to_csv(DERIVED_DIR / "sweep_summary.csv", index=False)

    return True, f"CSVs exportados en {DERIVED_DIR.resolve()}"


# ============================================================================
#                                   TUI
# ============================================================================

MENU_ITEMS = [
    "Editar parámetros",
    "Ejecutar: Pipeline + Caché (JSON)",
    "Ejecutar: Tests ALU completos (JSON)",
    "Ejecutar: Barrido B/C (JSON)",
    "Ejecutar: Pruebas unitarias (unittest)",
    "Explorar resultados (TUI)",
    "Exportar CSVs agregados (pandas)",
    "Mostrar parámetros actuales",
    "Salir",
]

PARAM_ITEMS = [
    ("data_len", "Número de elementos 64-bit (pipeline)"),
    ("cache_capacity", "Capacidad C de caché (bytes)"),
    ("line_bytes", "Tamaño de línea B (bytes, potencia de 2)"),
    ("seed", "Semilla PRNG para experimentos"),
    ("hit_time", "Tiempo de hit (ciclos)"),
    ("miss_penalty", "Miss penalty (ciclos)"),
    ("no_alu_asserts", "Desactivar aserciones ALU (True/False)"),
    ("forwarding_enabled", "Habilitar forwarding (True/False)"),

]

def format_params(p: Params) -> str:
    d = p.as_dict()
    lines = ["Parámetros actuales:"]
    for k in ["data_len", "cache_capacity", "line_bytes",
              "seed", "hit_time", "miss_penalty",
              "forwarding_enabled",
              "no_alu_asserts"]:
        lines.append(f"  - {k}: {d[k]}")

    return "\n".join(lines)

def edit_params_screen(stdscr: "curses._CursesWindow", p: Params) -> None:
    idx = 0
    while True:
        items = [f"{name} — {desc}" for name, desc in PARAM_ITEMS] + ["Volver"]
        draw_menu(stdscr, "Editar parámetros", items, idx)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(items)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(items)
        elif key in (10, 13):
            if idx == len(items) - 1:
                return
            name, desc = PARAM_ITEMS[idx]
            current = getattr(p, name)
            default = str(current)
            new_str = prompt_input(stdscr, f"{name} ({desc})", default)
            if new_str is None:
                continue
            try:
                if name in ("no_alu_asserts", "forwarding_enabled"):  # <--- AJUSTE
                    val = new_str.lower() in ("1", "true", "t", "yes", "y")
                else:
                    val = int(new_str)
                    if val < 0:
                        raise ValueError("Debe ser no negativo.")
                setattr(p, name, val)
                show_message(stdscr, f"OK: {name} = {getattr(p, name)}")
            except Exception as exc:
                show_message(stdscr, f"Error al asignar {name}: {exc}")
        elif key in (27, ord("q")):
            return

def explore_results_tui(stdscr: "curses._CursesWindow") -> None:
    files = list_outputs()
    fp = file_selector(stdscr, files, "Selecciona un JSON de ./outputs")
    if fp is None:
        return
    try:
        obj = load_json_file(fp)
        lines = format_obj_human(obj, fp.name)
        scrollable_view(stdscr, "Detalle del resultado", lines)
    except Exception as exc:
        show_message(stdscr, f"Error al cargar/mostrar {fp.name}: {exc}")

def tui(stdscr: "curses._CursesWindow") -> None:
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.timeout(-1)
    p = Params()
    idx = 0

    while True:
        draw_menu(stdscr, "Evaluación — ALU + Pipeline + Caché", MENU_ITEMS, idx)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(MENU_ITEMS)
        elif key in (27, ord("q")):
            break
        elif key in (10, 13):
            choice = MENU_ITEMS[idx]

            if choice == "Editar parámetros":
                edit_params_screen(stdscr, p)

            elif choice == "Mostrar parámetros actuales":
                show_message(stdscr, format_params(p))

            elif choice == "Explorar resultados (TUI)":
                explore_results_tui(stdscr)

            elif choice == "Exportar CSVs agregados (pandas)":
                ok, msg = export_csv_aggregate()
                show_message(stdscr, ("OK: " if ok else "WARN: ") + msg)

            elif choice == "Ejecutar: Pipeline + Caché (JSON)":
                stdscr.clear()
                stdscr.addstr(2, 2, "Ejecutando Pipeline + Caché ...")
                stdscr.refresh()
                try:
                    result = run_pipeline_and_cache(p)
                    out = write_json(result, p.outputs_dir, "run_pipeline_cache")
                    show_message(stdscr, f"Finalizado. JSON: {out}")
                except Exception as exc:
                    show_message(stdscr, f"Error en ejecución: {type(exc).__name__}: {exc}")

            elif choice == "Ejecutar: Tests ALU completos (JSON)":
                stdscr.clear()
                stdscr.addstr(2, 2, "Ejecutando Tests ALU completos ...")
                stdscr.refresh()
                try:
                    result = run_alu_full_tests(p)
                    out = write_json(result, p.outputs_dir, "run_alu_tests")
                    show_message(stdscr, f"Finalizado. JSON: {out}")
                except Exception as exc:
                    show_message(stdscr, f"Error en tests: {type(exc).__name__}: {exc}")

            elif choice == "Ejecutar: Barrido B/C (JSON)":
                stdscr.clear()
                stdscr.addstr(2, 2, "Ejecutando Barrido B/C ...")
                stdscr.refresh()
                try:
                    result = run_sweep_bc(p)
                    out = write_json(result, p.outputs_dir, "run_sweep_bc")
                    show_message(stdscr, f"Finalizado. JSON: {out}")
                except Exception as exc:
                    show_message(stdscr, f"Error en barrido: {type(exc).__name__}: {exc}")

            elif choice == "Ejecutar: Pruebas unitarias (unittest)":
                stdscr.clear()
                stdscr.addstr(2, 2, "Ejecutando pruebas unitarias (unittest) ...")
                stdscr.refresh()
                try:
                    result = run_unit_tests_with_fallback(p)
                    out = write_json(result, p.outputs_dir, "run_unittest")
                    show_message(stdscr, f"Finalizado. JSON: {out}")
                except Exception as exc:
                    show_message(stdscr, f"Error en unittest: {type(exc).__name__}: {exc}")

            elif choice == "Salir":
                break


# ============================================================================
#                      FALLBACK: MODO INTERACTIVO (SIN CURSES)
# ============================================================================

def prompt_cli(prompt: str, current: str) -> str:
    try:
        s = input(f"{prompt} [{current}]: ").strip()
        return current if s == "" else s
    except (EOFError, KeyboardInterrupt):
        return current

def interactive_cli() -> None:
    p = Params()
    print("=== Modo interactivo (fallback) — Evaluación ALU+Pipeline+Caché ===\n")

    while True:
        print(format_params(p))
        print("\nOpciones:")
        print("  1) Editar parámetros")
        print("  2) Ejecutar: Pipeline + Caché (JSON)")
        print("  3) Ejecutar: Tests ALU completos (JSON)")
        print("  4) Ejecutar: Barrido B/C (JSON)")
        print("  5) Ejecutar: Pruebas unitarias (unittest)")
        print("  6) Explorar resultados (último JSON)")
        print("  7) Exportar CSVs agregados (pandas)")
        print("  8) Salir")

        try:
            opt = input("\nSeleccione opción [1-8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSalida segura.")
            return

        if opt == "1":
            try:
                p.data_len = int(prompt_cli("data_len", str(p.data_len)))
                p.cache_capacity = int(prompt_cli("cache_capacity", str(p.cache_capacity)))
                p.line_bytes = int(prompt_cli("line_bytes", str(p.line_bytes)))
                p.seed = int(prompt_cli("seed", str(p.seed)))
                p.hit_time = int(prompt_cli("hit_time", str(p.hit_time)))
                p.miss_penalty = int(prompt_cli("miss_penalty", str(p.miss_penalty)))
                no_alu = prompt_cli("no_alu_asserts (True/False)", str(p.no_alu_asserts))
                p.no_alu_asserts = no_alu.lower() in ("1", "true", "t", "yes", "y")
                print("Parámetros actualizados.\n")
            except Exception as exc:
                print(f"Error al actualizar: {exc}\n")

        elif opt == "2":
            print("Ejecutando Pipeline + Caché ...")
            try:
                result = run_pipeline_and_cache(p)
                out = write_json(result, p.outputs_dir, "run_pipeline_cache")
                print(f"Listo. JSON: {out}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")

        elif opt == "3":
            print("Ejecutando Tests ALU completos ...")
            try:
                result = run_alu_full_tests(p)
                out = write_json(result, p.outputs_dir, "run_alu_tests")
                print(f"Listo. JSON: {out}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")

        elif opt == "4":
            print("Ejecutando Barrido B/C ...")
            try:
                result = run_sweep_bc(p)
                out = write_json(result, p.outputs_dir, "run_sweep_bc")
                print(f"Listo. JSON: {out}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")

        elif opt == "5":
            print("Ejecutando unittest ...")
            try:
                result = run_unit_tests_with_fallback(p)
                out = write_json(result, p.outputs_dir, "run_unittest")
                print(f"Listo. JSON: {out}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")

        elif opt == "6":
            files = list_outputs()
            if not files:
                print("(no hay JSONs en ./outputs)\n")
            else:
                obj = load_json_file(files[-1])
                for ln in format_obj_human(obj, files[-1].name):
                    print(ln)
                print("")

        elif opt == "7":
            ok, msg = export_csv_aggregate()
            print(("OK: " if ok else "WARN: ") + msg + "\n")

        elif opt == "8":
            print("Salida.")
            return
        else:
            print("Opción inválida.\n")


# ============================================================================
#                                PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    """
    Punto de entrada ejecutable.

    Levanta TUI con curses. Si curses falla, usa modo interactivo por consola.
    En todas las ejecuciones se generan JSONs en ./outputs con timestamp.
    """
    try:
        curses.wrapper(tui)
    except Exception as exc:
        sys.stderr.write(f"[WARN] Curses no disponible o falló ({type(exc).__name__}: {exc}).\n")
        sys.stderr.write("Entrando a modo interactivo simple...\n\n")
        interactive_cli()
