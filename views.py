#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
views.py — Explorador/visor para JSONs de ./outputs generados por main.py

Funciones principales:
- Listar outputs disponibles.
- Cargar el último JSON o uno específico.
- Resumir métricas de pipeline, caché y barridos B/C en tablas pandas.
- Exportar CSVs a ./outputs/derived/.
- (Opcional) Graficar resúmenes básicos con matplotlib (sin estilos explícitos).

Uso rápido:
  python views.py --list
  python views.py --latest
  python views.py --file outputs/run_pipeline_cache_20250101_120000.json
  python views.py --aggregate --export-csv
  python views.py --aggregate --export-csv --plot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

OUTPUTS_DIR = Path("outputs")
DERIVED_DIR = OUTPUTS_DIR / "derived"


# --------------------------- helpers sin pandas -----------------------------

def list_outputs() -> List[Path]:
    """Lista JSONs en ./outputs."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(OUTPUTS_DIR.glob("*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    """Carga JSON en dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_output() -> Optional[Path]:
    """Devuelve el JSON más reciente, si existe."""
    files = list_outputs()
    return files[-1] if files else None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------------------------- pandas (opcional) ------------------------------

def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


# --------------------------- normalización ---------------------------------

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
    """Para resultados de run_sweep_bc()."""
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


# --------------------------- impresión simple ------------------------------

def print_pipeline_(obj: Dict[str, Any]) -> None:
    pm = normalize_pipeline(obj)
    print("# ==== PIPELINE ====")
    for k in ["cycles", "retired", "stalls", "cpi_milli", "stored_max", "expected_max"]:
        print(f"{k:>12}: {pm.get(k)}")
    print("")


def print_cache_(obj: Dict[str, Any]) -> None:
    rows = normalize_cache_experiments(obj)
    if not rows:
        print("# ==== CACHÉ ====")
        print("(sin datos)\n")
        return
    print("# ==== CACHÉ (por escenario) ====")
    for r in rows:
        txt = (f"[{r['scenario']}] C={int(r['capacity_bytes'])}  "
               f"B={int(r['line_bytes'])}  sets={int(r['sets'])}  "
               f"hit_rate={r['hit_rate']:.4f}  AMAT={r['AMAT']:.2f}")
        print(txt)
    print("")


def print_sweep_(obj: Dict[str, Any]) -> None:
    rows = normalize_sweep(obj)
    print("# ==== SWEEP B/C ====")
    if not rows:
        print("(sin datos)\n")
        return
    # Un pequeño resumen por escenario:
    by_scenario: Dict[str, int] = {}
    for r in rows:
        s = str(r.get("scenario"))
        by_scenario[s] = by_scenario.get(s, 0) + 1
    for k, v in by_scenario.items():
        print(f"{k}: {v} combinaciones")
    print("")


# --------------------------- agregación con pandas -------------------------

def aggregate_dir(export_csv: bool = False, plot: bool = False) -> None:
    """
    Carga todos los JSONs de ./outputs y genera DataFrames:
    - pipeline_df
    - cache_df
    - sweep_df

    Exporta CSVs a ./outputs/derived/ si export_csv=True.
    Genera gráficas simples si plot=True (requiere matplotlib).
    """
    pd = _try_import_pandas()
    if pd is None:
        print("[WARN] pandas no disponible. Usa --latest/--file para vista simple.")
        return

    files = list_outputs()
    if not files:
        print("[INFO] No se encontraron JSONs en ./outputs")
        return

    ensure_dir(DERIVED_DIR)

    pipe_rows: List[Dict[str, Any]] = []
    cache_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    for fp in files:
        try:
            obj = load_json(fp)
        except Exception as exc:
            print(f"[WARN] no se pudo cargar {fp.name}: {exc}")
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
            pr = {**tag, **pm}
            pipe_rows.append(pr)

        for row in normalize_cache_experiments(obj):
            cache_rows.append({**tag, **row})

        for row in normalize_sweep(obj):
            sweep_rows.append({**tag, **row})

    # Construcción de DataFrames
    pipeline_df = pd.DataFrame(pipe_rows) if pipe_rows else pd.DataFrame()
    cache_df = pd.DataFrame(cache_rows) if cache_rows else pd.DataFrame()
    sweep_df = pd.DataFrame(sweep_rows) if sweep_rows else pd.DataFrame()

    # Exportación CSV (opcional)
    if export_csv:
        if not pipeline_df.empty:
            pipeline_df.to_csv(DERIVED_DIR / "pipeline_summary.csv", index=False)
        if not cache_df.empty:
            cache_df.to_csv(DERIVED_DIR / "cache_summary.csv", index=False)
        if not sweep_df.empty:
            sweep_df.to_csv(DERIVED_DIR / "sweep_summary.csv", index=False)
        print(f"[OK] CSVs generados en {DERIVED_DIR.resolve()}")

    # Gráficos (opcional)
    if plot:
        plt = _try_import_matplotlib()
        if plt is None:
            print("[WARN] matplotlib no disponible. Omite --plot o instálalo.")
        else:
            # Ejemplo: bar plot de AMAT por escenario (últimos resultados)
            if not cache_df.empty:
                last_by_scenario = (cache_df.sort_values("file")
                                    .groupby("scenario", as_index=False).last())
                plt.figure()
                plt.bar(last_by_scenario["scenario"], last_by_scenario["AMAT"])
                plt.title("AMAT por escenario (último por escenario)")
                plt.xlabel("Escenario")
                plt.ylabel("AMAT (ciclos)")
                plt.tight_layout()
                out = DERIVED_DIR / "cache_amat_bar.png"
                plt.savefig(out)
                print(f"[OK] Figura: {out}")

            # Ejemplo: heatmap simple de AMAT (random) vs (C,B) si hay sweep
            if not sweep_df.empty:
                sdf = sweep_df[sweep_df["scenario"] == "random"].copy()
                if not sdf.empty:
                    pivot = sdf.pivot_table(index="C", columns="B", values="AMAT", aggfunc="mean")
                    plt.figure()
                    # Heatmap manual (imshow); sin estilos ni colores específicos
                    plt.imshow(pivot.values, aspect="auto")
                    plt.title("AMAT medio (random) vs (C,B)")
                    plt.xlabel("B (bytes)")
                    plt.ylabel("C (bytes)")
                    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist(), rotation=45)
                    plt.yticks(range(len(pivot.index)), pivot.index.tolist())
                    plt.tight_layout()
                    out = DERIVED_DIR / "sweep_amat_heatmap.png"
                    plt.savefig(out)
                    print(f"[OK] Figura: {out}")

    # Resúmenes rápidos
    if not pipeline_df.empty:
        print("\n# Resumen pipeline (últimos 5):")
        print(pipeline_df.tail(5).to_string(index=False))
    if not cache_df.empty:
        print("\n# Resumen cache (últimos 5):")
        cols = ["file", "scenario", "C", "B", "hit_rate", "AMAT"]
        cols = [c for c in cols if c in cache_df.columns]
        print(cache_df[cols].tail(5).to_string(index=False))
    if not sweep_df.empty:
        print("\n# Resumen sweep (últimos 5):")
        cols = ["file", "scenario", "C", "B", "AMAT"]
        cols = [c for c in cols if c in sweep_df.columns]
        print(sweep_df[cols].tail(5).to_string(index=False))


# --------------------------- CLI -------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="views.py",
        description="Herramienta para navegar/visualizar JSONs generados por main.py"
    )
    parser.add_argument("--list", action="store_true", help="Listar JSONs en ./outputs")
    parser.add_argument("--latest", action="store_true", help="Cargar y mostrar el JSON más reciente")
    parser.add_argument("--file", type=Path, default=None, help="Cargar un JSON específico")
    parser.add_argument("--aggregate", action="store_true", help="Agrega todos los JSONs (pandas)")
    parser.add_argument("--export-csv", action="store_true", help="Exporta CSVs (requiere --aggregate)")
    parser.add_argument("--plot", action="store_true", help="Genera gráficos básicos (requiere --aggregate)")

    args = parser.parse_args()

    if args.list:
        files = list_outputs()
        if not files:
            print("(no hay JSONs en ./outputs)")
            return
        for f in files:
            print(f.name)
        return

    if args.latest:
        fp = latest_output()
        if not fp:
            print("(no hay JSONs en ./outputs)")
            return
        obj = load_json(fp)
        print(f"# Archivo: {fp.name}\n")
        print_pipeline_(obj)
        print_cache_(obj)
        print_sweep_(obj)
        return

    if args.file is not None:
        fp = args.file
        if not fp.exists():
            print(f"[ERROR] No existe: {fp}")
            return
        obj = load_json(fp)
        print(f"# Archivo: {fp.name}\n")
        print_pipeline_(obj)
        print_cache_(obj)
        print_sweep_(obj)
        return

    if args.aggregate:
        aggregate_dir(export_csv=args.export_csv, plot=args.plot)
        return

    # Uso por defecto
    print("Uso:")
    print("  python views.py --list")
    print("  python views.py --latest")
    print("  python views.py --file outputs/<archivo>.json")
    print("  python views.py --aggregate --export-csv [--plot]")


if __name__ == "__main__":
    main()
