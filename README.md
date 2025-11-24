# Evaluación 1 — Arquitectura de Computadores

Simulador educativo que implementa una ALU parametrizable, un pipeline ARMv8-A
simplificado y experimentos de caché directa. Los parámetros principales están
expuestos en `sim.py` para facilitar su ajuste.

## Requisitos previos

- Python 3.10 o superior.
- No se necesitan dependencias externas; solo se usa la biblioteca estándar.

## Ejecución rápida

1. **Clona o descarga** este repositorio en tu máquina.
2. **Accede** al directorio del proyecto:
   ```bash
   cd Evaluacion_1__Arquitectura_Computadores
   ```
3. **Inicia la simulación completa** ejecutando el punto de entrada único:
   ```bash
   python main.py
   ```
4. **Observa la salida** en consola. Verás:
   - Parámetros de configuración utilizados.
   - Métricas del pipeline (ciclos, CPI, stalls, etc.).
   - Resultados de los tres experimentos de caché, incluida la AMAT.

## Ajustes opcionales

- Modifica las constantes definidas al inicio de `sim.py` (por ejemplo
  `ALU_WIDTH_BITS`, `CACHE_CAPACITY_BYTES` o `PSEUDO_RANDOM_SEED`) para probar
  escenarios diferentes.
- Si deseas serializar los resultados, aprovecha la función
  `collect_all_results` dentro de `sim.py` y escribe los datos en el formato que
  prefieras.

## Pruebas

Para verificar la integridad del proyecto, ejecuta los tests automatizados:
```bash
python -m pytest
```
