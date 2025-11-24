"""
Punto de arranque simplificado para la evaluación de Arquitectura de Computadores.

Ejecuta las pruebas de sanidad de la ALU, corre el programa de demostración en el
pipeline y lanza los experimentos de caché usando los parámetros por defecto
definidos en ``sim.py``. Basta con ejecutar ``python main.py``.
"""

from __future__ import annotations

import sim


def _imprimir_resumen(resultados: dict) -> None:
    """Muestra en consola un resumen legible de la ejecución."""

    parametros = resultados.get("params", {})
    print("# =================== PARÁMETROS UTILIZADOS ===================")
    print(f"ALU_WIDTH_BITS       : {parametros.get('ALU_WIDTH_BITS')}")
    print(f"REG_WIDTH_BITS       : {parametros.get('REG_WIDTH_BITS')}")
    print(f"CACHE_CAPACITY_BYTES : {parametros.get('CACHE_CAPACITY_BYTES')}")
    print(f"CACHE_LINE_BYTES     : {parametros.get('CACHE_LINE_BYTES')}")
    print(f"CACHE_HIT_TIME       : {parametros.get('CACHE_HIT_TIME')}")
    print(f"CACHE_MISS_PENALTY   : {parametros.get('CACHE_MISS_PENALTY')}")
    print(f"PSEUDO_RANDOM_SEED   : {parametros.get('PSEUDO_RANDOM_SEED')}")
    print(f"DATA_LEN             : {parametros.get('DATA_LEN')}")
    print("")

    metrics = resultados.get("pipeline_metrics", {})
    print("# ===================== RESULTADOS PIPELINE =====================")
    print(f"Ciclos totales       : {metrics.get('cycles')}")
    print(f"Instrucciones retir. : {metrics.get('retired')}")
    print(f"Stalls (incl. flush) : {metrics.get('stalls')}")
    print(f"CPI (x1000)          : {metrics.get('cpi_milli')} (milli-CPI)")
    print(f"Máx. almacenado      : {metrics.get('stored_max')}")
    print(f"Máx. esperado        : {metrics.get('expected_max')}")
    print("")

    cache_results = resultados.get("cache_experiments", {})
    print("# ==================== EXPERIMENTOS DE CACHÉ ====================")
    for nombre, rep in cache_results.items():
        print(
            f"[{nombre}] "
            f"hit_rate={rep.get('hit_rate', 0):.4f}  "
            f"miss_rate={rep.get('miss_rate', 0):.4f}  "
            f"AMAT={rep.get('AMAT', 0):.2f} ciclos  "
            f"(C={int(rep.get('capacity_bytes', 0))}B, "
            f"B={int(rep.get('line_bytes', 0))}B, "
            f"sets={int(rep.get('sets', 0))})"
        )


def main() -> None:
    """Ejecuta la simulación completa con los parámetros predeterminados."""

    print("Iniciando pruebas de sanidad de la ALU…")
    sim.alu_sanity_checks()

    print("Ejecutando programa de demostración y experimentos de caché…")
    resultados = sim.collect_all_results()

    _imprimir_resumen(resultados)


if __name__ == "__main__":
    main()
