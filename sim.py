#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arquitectura de Computadores — Evaluación 1
Código único organizado por bloques según checklist A–D.


"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ============================================================================
#  PARÁMETROS GLOBALES (Ajustables)  —  Requisito general (p.7 de checklist)
# ============================================================================

# --- ALU ---
ALU_WIDTH_BITS: int = 16  # A1: ancho de palabra (8 o 16 bits).
ALU_USE_CLA_MODEL: bool = False  # A4: si True, reporta latencia tipo CLA.
ALU_ASSERTIONS_ENABLED: bool = True  # A3: habilita aserciones de sanidad.

# --- SIMULADOR ARMv8-A (subconjunto simplificado) ---
REG_COUNT: int = 32  # B: X0..X31.
REG_WIDTH_BITS: int = 64  # B: ancho de registros.
MEM_DEFAULT_WORD_BYTES: int = 8  # B: LDR/STR sobre palabras de 64 bits.
FORWARDING_ENABLED: bool = True  # B2: forwarding activado.
LOAD_USE_STALL_CYCLES: int = 1  # B2: burbuja pos-LDR.
LOAD_USE_STALL_CYCLES_FWD: int = 1     # con forwarding
LOAD_USE_STALL_CYCLES_NOFWD: int = 2   # sin forwarding
BRANCH_PREDICT_NOT_TAKEN: bool = True  # B2: predicción simple.
BRANCH_FLUSH_PENALTY: int = 1  # B2: penalidad de flush.

# --- CACHÉ DIRECTA ---
CACHE_CAPACITY_BYTES: int = 4 * 1024  # C1: capacidad C.
CACHE_LINE_BYTES: int = 32  # C1: tamaño de línea B.
CACHE_E: int = 1  # C1: asociatividad (directa → 1).

# --- Tiempos para AMAT (supuestos explícitos) ---
CACHE_HIT_TIME: int = 1  # C4: tiempo de hit (ciclos).
CACHE_MISS_PENALTY: int = 50  # C4: miss penalty (ciclos).

# --- Experimentos de caché (C3) ---
EXPERIMENT_RANGE_BYTES: int = 256 * 1024
EXPERIMENT_STEPS: int = 20_000
EXPERIMENT_STRIDE_BYTES: int = CACHE_CAPACITY_BYTES

# --- Reproducibilidad ---
PSEUDO_RANDOM_SEED: int = 1337


# ============================================================================
#  UTILIDADES COMUNES
# ============================================================================

def mask_n_bits(value: int, n: int) -> int:
    """
    Aplica máscara de n bits (complemento a dos compatible).

    :param value: Entero a recortar.
    :param n: Número de bits.
    :return: Valor en [0, 2^n - 1].
    """
    if n <= 0:
        return 0
    return value & ((1 << n) - 1)


def to_signed(value: int, n: int) -> int:
    """
    Interpreta 'value' como entero con signo (2's complement) de n bits.

    :param value: Entero sin signo o arbitrario.
    :param n: Ancho de palabra.
    :return: Entero con signo.
    """
    value = mask_n_bits(value, n)
    sign_bit = 1 << (n - 1)
    return value - (1 << n) if (value & sign_bit) else value


def carry_out_add(a: int, b: int, n: int) -> int:
    """
    Acarreo fuera (C) de suma sin signo a+b en n bits.

    :return: 1 si hubo carry-out; 0 en caso contrario.
    """
    full = (a & ((1 << n) - 1)) + (b & ((1 << n) - 1))
    return 1 if full >> n else 0


def overflow_add(a: int, b: int, r: int, n: int) -> int:
    """
    Overflow con signo (V) en suma a+b=r (2's complement).

    :return: 1 si hubo overflow con signo; 0 en caso contrario.
    """
    sa, sb, sr = to_signed(a, n), to_signed(b, n), to_signed(r, n)
    if sa >= 0 and sb >= 0 and sr < 0:
        return 1
    if sa < 0 and sb < 0 and sr >= 0:
        return 1
    return 0


def borrow_in_sub(a: int, b: int, n: int) -> int:
    """
    Indicador 'no borrow' en a - b (convención: C=1 si no hay borrow).

    :return: 1 si NO hubo borrow; 0 si hubo borrow.
    """
    au = a & ((1 << n) - 1)
    bu = b & ((1 << n) - 1)
    return 1 if au >= bu else 0


# ============================================================================
#  # A1. Operaciones (ALU)  —  Mapa mínimo de operaciones
# ============================================================================

class AluOp(Enum):
    """
    Conjunto de operaciones mínimas soportadas por la ALU (Parte A del proyecto).

    Cada miembro representa una operación aritmético-lógica definida en el
    subconjunto obligatorio. Se utiliza como enumeración para mantener claridad,
    evitar el uso de literales mágicos y garantizar consistencia en la ejecución.

    Orden lógico:
      1. Operaciones aritméticas básicas (suma, resta).
      2. Operaciones lógicas binarias (AND, ORR, EOR).
      3. Operación unaria lógica (NOT).
      4. Desplazamientos lógicos (izquierda y derecha).
      5. Comparaciones tipo "set less than" (con y sin signo).
    """

    # 1. Operaciones aritméticas
    ADD = auto()  # ADD: Suma aritmética (a + b).
    SUB = auto()  # SUB: Resta aritmética (a - b).

    # 2. Operaciones lógicas binarias
    AND = auto()  # AND: Conjunción bit a bit (a & b).
    ORR = auto()  # ORR: Disyunción bit a bit (a | b).
    EOR = auto()  # EOR: Disyunción exclusiva bit a bit (a ^ b).

    # 3. Operación unaria lógica
    NOT = auto()  # NOT: Negación bit a bit (~a). Solo requiere operando 'a'.

    # 4. Desplazamientos lógicos
    SLL = auto()  # SLL: Shift lógico a la izquierda (a << b).
    SRL = auto()  # SRL: Shift lógico a la derecha (a >> b).

    # 5. Comparaciones (Set Less Than)
    SLT_SIGNED = auto()  # SLT (signed): Comparación con signo.
    SLT_UNSIGNED = auto()  # SLT (unsigned): Comparación sin signo.



@dataclass
class AluResult:
    """
    Resultado de ALU con flags Z, N, C, V.

    - Z: Zero (resultado == 0)
    - N: Negative (bit de signo del resultado)
    - C: Carry (ADD) / no-borrow (SUB)
    - V: Overflow con signo (2's complement)
    """
    value: int
    z: int
    n: int
    c: int
    v: int


class ALU:
    """
    Unidad Aritmético-Lógica parametrizable (n bits) con flags estándar.
    Cumple A1 (operaciones) y A2 (flags).
    """

    def __init__(self, width_bits: int) -> None:
        """
        Inicializa la ALU.

        :param width_bits: Ancho de palabra en bits.
        """
        self.n = width_bits

    # ----------------------------------------------------------------------
    # # A2. Flags (Z, N, C, V) — Implementación en cada operación
    # ----------------------------------------------------------------------

    def exec(self, op: AluOp, a: int, b: Optional[int] = None) -> AluResult:
        """
        Ejecuta la operación 'op' con operandos 'a' y 'b' (si aplica).

        :param op: Operación de la ALU.
        :param a: Operando A.
        :param b: Operando B (opcional en NOT).
        :return: AluResult con valor y flags (Z,N,C,V).
        """
        n = self.n
        c = 0
        v = 0

        if op == AluOp.ADD:
            r = mask_n_bits(a + (b or 0), n)
            c = carry_out_add(a, (b or 0), n)
            v = overflow_add(a, (b or 0), r, n)

        elif op == AluOp.SUB:
            b_val = (b or 0)
            r = mask_n_bits(a - b_val, n)
            c = borrow_in_sub(a, b_val, n)  # 1=no borrow, 0=borrow
            v = overflow_add(a, mask_n_bits(-b_val, n), r, n)

        elif op == AluOp.AND:
            r = mask_n_bits(a & (b or 0), n)

        elif op == AluOp.ORR:
            r = mask_n_bits(a | (b or 0), n)

        elif op == AluOp.EOR:
            r = mask_n_bits(a ^ (b or 0), n)

        elif op == AluOp.NOT:
            r = mask_n_bits(~a, n)

        elif op == AluOp.SLL:
            shift = (b or 0) & (n - 1)
            r = mask_n_bits(a << shift, n)

        elif op == AluOp.SRL:
            shift = (b or 0) & (n - 1)
            r = (a & ((1 << n) - 1)) >> shift

        elif op == AluOp.SLT_SIGNED:
            r = 1 if to_signed(a, n) < to_signed(b or 0, n) else 0

        elif op == AluOp.SLT_UNSIGNED:
            r = 1 if mask_n_bits(a, n) < mask_n_bits((b or 0), n) else 0

        else:
            raise ValueError(f"Operación ALU no soportada: {op}")

        z = 1 if r == 0 else 0
        n_flag = 1 if (r & (1 << (n - 1))) != 0 else 0
        return AluResult(value=r, z=z, n=n_flag, c=c, v=v)


# ============================================================================
#  # A3. Tests unitarios completos — vectores sistemáticos y tabla resumida
# ============================================================================

def alu_sanity_checks() -> None:
    """
    A3: Sanidad básica de ALU. Casos borde representativos.
    No usa 'unittest' para mantener archivo único.
    """
    if not ALU_ASSERTIONS_ENABLED:
        return

    alu = ALU(ALU_WIDTH_BITS)
    n = ALU_WIDTH_BITS

    # Overflow con signo en ADD: (max_pos + 1) → V=1
    a = (1 << (n - 1)) - 1
    b = 1
    res = alu.exec(AluOp.ADD, a, b)
    assert res.v == 1, "ADD: overflow con signo no detectado (V)."

    # Borrow en SUB: 0 - 1 → C=0 (hubo borrow)
    res = alu.exec(AluOp.SUB, 0, 1)
    assert res.c == 0, "SUB: indicador de borrow incorrecto (C)."

    # Identidades lógicas.
    x = 0x1234
    assert alu.exec(AluOp.AND, x, 0).value == 0, "AND x&0 debe ser 0."
    assert alu.exec(AluOp.ORR, x, 0).value == mask_n_bits(x, n), "OR x|0 debe ser x."
    assert alu.exec(AluOp.EOR, x, x).value == 0, "XOR x^x debe ser 0."
    assert mask_n_bits(x ^ alu.exec(AluOp.NOT, x).value, n) == (1 << n) - 1, "NOT inconsistente."

    # SLT unsigned: (2^n-1) < 1 → False
    assert alu.exec(AluOp.SLT_UNSIGNED, (1 << n) - 1, 1).value == 0, "SLT_UNSIGNED inconsistente."


def _check_flags_zn(value: int, nbits: int) -> Tuple[int, int]:
    """
    Calcula flags Z y N esperados para un valor n-bit.
    """
    v = mask_n_bits(value, nbits)
    z = 1 if v == 0 else 0
    n = 1 if (v & (1 << (nbits - 1))) != 0 else 0
    return z, n


def build_alu_truth_table_summary(nbits: int) -> List[Dict[str, int]]:
    """
    Genera una tabla de verdad bastante resumida por operación con operandos canónicos.
    Esta tabla sirve para documentación y verificación la verdad que bien básica.

    Conjunto de operandos canónicos:
    - 0x0000, 0x0001, 0x00FF/0xFFFF (todo-1), 0x7FFF (max positivo), 0x8000 (min negativo)
    - Para nbits=8 ajusta (0x7F, 0x80, 0xFF), para nbits=16 ajusta (0x7FFF, 0x8000, 0xFFFF).

    :param nbits: Ancho en bits (8 o 16 en este proyecto).
    :return: Lista de filas con campos: op, a, b, value, Z, N, C, V (enteros).
    """
    alu = ALU(nbits)
    ones = (1 << nbits) - 1
    max_pos = (1 << (nbits - 1)) - 1
    min_neg = 1 << (nbits - 1)

    operands = [0x0, 0x1, ones, max_pos, min_neg]
    ops = [
        AluOp.ADD, AluOp.SUB, AluOp.AND, AluOp.ORR, AluOp.EOR,
        AluOp.NOT, AluOp.SLL, AluOp.SRL, AluOp.SLT_SIGNED, AluOp.SLT_UNSIGNED
    ]

    table: List[Dict[str, int]] = []
    for op in ops:
        # Para un resumen compacto, recorremos un subconjunto cruzado A×B.
        # NOT solo usa A; SLL/SRL usan B como shift en {0,1,nbits-1}.
        for a in operands:
            if op == AluOp.NOT:
                r = alu.exec(op, a)
                row = {
                    "op": op.name, "a": a, "b": 0,
                    "value": r.value, "Z": r.z, "N": r.n, "C": r.c, "V": r.v,
                }
                table.append(row)
                continue

            if op in {AluOp.SLL, AluOp.SRL}:
                for shift in [0, 1, max(0, nbits - 1)]:
                    r = alu.exec(op, a, shift)
                    row = {
                        "op": op.name, "a": a, "b": shift,
                        "value": r.value, "Z": r.z, "N": r.n, "C": r.c, "V": r.v,
                    }
                    table.append(row)
                continue

            for b in operands:
                r = alu.exec(op, a, b)
                row = {
                    "op": op.name, "a": a, "b": b,
                    "value": r.value, "Z": r.z, "N": r.n, "C": r.c, "V": r.v,
                }
                table.append(row)

    return table


def alu_full_tests() -> None:
    """
    A3: Batería de pruebas sistemáticas por operación.
    Lanza AssertionError ante cualquier inconsistencia.
    No imprime salida para mantener el módulo silencioso.
    """
    if not ALU_ASSERTIONS_ENABLED:
        return

    n = ALU_WIDTH_BITS
    alu = ALU(n)
    ones = (1 << n) - 1
    max_pos = (1 << (n - 1)) - 1
    min_neg = 1 << (n - 1)

    # ------------------ ADD ------------------
    # Identidades y propiedades básicas
    for a in [0, 1, max_pos, min_neg, ones]:
        r = alu.exec(AluOp.ADD, a, 0)
        assert r.value == mask_n_bits(a, n), "ADD: x + 0 debe ser x."
        z, nf = _check_flags_zn(r.value, n)
        assert (r.z, r.n) == (z, nf), "ADD: flags Z/N incorrectos en identidad."

    # Overflow con signo
    r = alu.exec(AluOp.ADD, max_pos, 1)
    assert r.v == 1, "ADD: overflow esperado (max_pos + 1)."

    # Carry sin signo
    r = alu.exec(AluOp.ADD, ones, 1)
    assert r.c == 1, "ADD: carry-out esperado (all_ones + 1)."

    # ------------------ SUB ------------------
    # Identidades
    for a in [0, 1, max_pos, min_neg, ones]:
        r = alu.exec(AluOp.SUB, a, 0)
        assert r.value == mask_n_bits(a, n), "SUB: x - 0 debe ser x."
        z, nf = _check_flags_zn(r.value, n)
        assert (r.z, r.n) == (z, nf), "SUB: flags Z/N incorrectos en identidad."

    # Borrow: 0 - 1 (C=0 por convención de no-borrow)
    r = alu.exec(AluOp.SUB, 0, 1)
    assert r.c == 0, "SUB: borrow esperado (C=0)."

    # Overflow con signo (min_neg - 1 => overflow)
    r = alu.exec(AluOp.SUB, min_neg, 1)
    assert r.v == 1, "SUB: overflow con signo esperado (min_neg - 1)."

    # ------------------ AND / ORR / EOR ------------------
    Xs = [0, 1, 0x55 & ones, 0xAA & ones, ones]
    for x in Xs:
        assert alu.exec(AluOp.AND, x, 0).value == 0, "AND: x & 0 == 0."
        assert alu.exec(AluOp.ORR, x, 0).value == x, "OR: x | 0 == x."
        assert alu.exec(AluOp.EOR, x, x).value == 0, "XOR: x ^ x == 0."
        # Z/N coherentes
        for op in (AluOp.AND, AluOp.ORR, AluOp.EOR):
            r = alu.exec(op, x, 0 if op != AluOp.ORR else x)
            z, nf = _check_flags_zn(r.value, n)
            assert (r.z, r.n) == (z, nf), f"{op.name}: flags Z/N incorrectos."

    # ------------------ NOT ------------------
    for a in Xs:
        r = alu.exec(AluOp.NOT, a)
        assert mask_n_bits(a ^ r.value, n) == ones, "NOT: x ^ ~x == all_ones."
        z, nf = _check_flags_zn(r.value, n)
        assert (r.z, r.n) == (z, nf), "NOT: flags Z/N incorrectos."

    # ------------------ SLL / SRL ------------------
    for a in [0, 1, 0x3, ones, min_neg, max_pos]:
        for sh in [0, 1, max(0, n - 1)]:
            r = alu.exec(AluOp.SLL, a, sh)
            exp = mask_n_bits(a << sh, n)
            assert r.value == exp, "SLL: desplazamiento incorrecto."
            z, nf = _check_flags_zn(r.value, n)
            assert (r.z, r.n) == (z, nf), "SLL: flags Z/N incorrectos."

            r = alu.exec(AluOp.SRL, a, sh)
            exp = (a & ones) >> sh
            assert r.value == exp, "SRL: desplazamiento incorrecto."
            z, nf = _check_flags_zn(r.value, n)
            assert (r.z, r.n) == (z, nf), "SRL: flags Z/N incorrectos."

    # ------------------ SLT (signed/unsigned) ------------------
    # Escenarios clave: comparación cruzada entre rangos y signos.
    pairs = [
        (0, 1), (1, 0), (max_pos, min_neg), (min_neg, max_pos),
        (ones, 1), (1, ones), (min_neg, min_neg), (max_pos, max_pos),
    ]
    for a, b in pairs:
        ru = alu.exec(AluOp.SLT_UNSIGNED, a, b).value
        rs = alu.exec(AluOp.SLT_SIGNED, a, b).value
        # Reglas de referencia:
        au, bu = mask_n_bits(a, n), mask_n_bits(b, n)
        asg, bsg = to_signed(a, n), to_signed(b, n)
        assert ru == (1 if au < bu else 0), "SLT_UNSIGNED: resultado incorrecto."
        assert rs == (1 if asg < bsg else 0), "SLT_SIGNED: resultado incorrecto."


def run_all_alu_tests_and_table() -> List[Dict[str, int]]:
    """
    Ejecuta 'alu_full_tests' y retorna la tabla de verdad resumida para anexar en informe.
    No imprime ni guarda; el llamador decide el destino.
    """
    alu_full_tests()
    return build_alu_truth_table_summary(ALU_WIDTH_BITS)


# ============================================================================
#  # A4. Análisis de latencia RCA vs. CLA  —  Estimación cualitativa/cuantitativa
# ============================================================================

def estimate_adder_latency(width_bits: int,
                           model: str = "RCA",
                           t_fa: float = 1.0,
                           t_look: float = 0.5) -> float:
    """
    Estima latencia relativa de un sumador n-bit.

    :param width_bits: Ancho n.
    :param model: 'RCA' (ripple-carry) u 'CLA' (carry-lookahead).
    :param t_fa: Retardo relativo por full-adder (RCA).
    :param t_look: Retardo por nivel de lookahead (CLA).
    :return: Latencia relativa (unidades abstractas).
    """
    model = model.upper()
    if model == "RCA":
        return width_bits * t_fa
    if model == "CLA":
        import math
        levels = max(1, math.ceil(math.log2(width_bits)))
        return levels * t_look + 2 * t_fa  # 2*t_fa para sumas locales
    raise ValueError("Modelo no soportado; use 'RCA' o 'CLA'.")


# ============================================================================
#  # B1. Programa en (pseudo) ARMv8-A  —  Suma N y máximo con LDR/STR + CBZ/CBNZ
# ============================================================================

class Op(Enum):
    """
    ISA mínima basada en un subconjunto de ARMv8-A.

    Cada miembro representa una instrucción elemental necesaria para
    modelar el datapath, ejecutar un programa demo y analizar hazards
    en un pipeline de 5 etapas (IF, ID, EX, MEM, WB).

    Orden lógico:
      1. Operaciones aritméticas y lógicas tipo R.
      2. Transferencia de datos (MOV).
      3. Acceso a memoria (LDR, STR).
      4. Control de flujo condicional (CBZ, CBNZ).
      5. No operación (NOP).
    """

    # 1. Operaciones aritméticas y lógicas (formato R)
    ADD = auto()   # rd <- rn + rm   | Suma entre registros.
    SUB = auto()   # rd <- rn - rm   | Resta entre registros.
    AND = auto()   # rd <- rn & rm   | Conjunción bit a bit.
    ORR = auto()   # rd <- rn | rm   | Disyunción bit a bit.
    EOR = auto()   # rd <- rn ^ rm   | Disyunción exclusiva (XOR).

    # 2. Transferencia de datos
    MOV = auto()   # rd <- rn (o imm) | Copia entre registros o carga inmediata.

    # 3. Acceso a memoria (formato D)
    LDR = auto()   # rd <- MEM[rn + imm]  | Carga desde memoria.
    STR = auto()   # MEM[rn + imm] <- rd  | Almacenamiento en memoria.

    # 4. Control de flujo condicional (formato B cond)
    CBZ = auto()   # if rz == 0 -> PC += imm  | Salto condicional en cero.
    CBNZ = auto()  # if rz != 0 -> PC += imm  | Salto condicional distinto de cero.

    # 5. No operación
    NOP = auto()   # No Operation | Instrucción de relleno (pipeline).

@dataclass
class Instr:
    """
    Instrucción simplificada para el simulador de pipeline.

    :param op: Operación.
    :param rd: Registro destino.
    :param rn: Registro fuente 1 / base.
    :param rm: Registro fuente 2.
    :param rz: Registro para prueba en CBZ/CBNZ.
    :param imm: Inmediato (desplazamiento o delta de salto).
    """
    op: Op
    rd: Optional[int] = None
    rn: Optional[int] = None
    rm: Optional[int] = None
    rz: Optional[int] = None
    imm: Optional[int] = None


# ============================================================================
#  # B2. Datapath y Pipeline (5 etapas) — hazards, forwarding, predicción
# ============================================================================

@dataclass
class IfId:
    """Registro IF/ID."""
    instr: Instr = field(default_factory=lambda: Instr(Op.NOP))


@dataclass
class IdEx:
    """Registro ID/EX."""
    instr: Instr = field(default_factory=lambda: Instr(Op.NOP))
    a: int = 0
    b: int = 0
    imm: int = 0


@dataclass
class ExMem:
    """Registro EX/MEM."""
    instr: Instr = field(default_factory=lambda: Instr(Op.NOP))
    alu_res: int = 0
    rd: Optional[int] = None
    mem_addr: Optional[int] = None
    branch_taken: bool = False
    new_pc: Optional[int] = None


@dataclass
class MemWb:
    """Registro MEM/WB."""
    instr: Instr = field(default_factory=lambda: Instr(Op.NOP))
    write_val: int = 0
    rd: Optional[int] = None


class PipelineCPU:
    """
    CPU de 5 etapas: IF, ID, EX, MEM, WB.

    Implementa:
    - Forwarding opcional (EX/MEM, MEM/WB).
    - Burbuja carga-uso.
    - Predicción simple (not-taken) con flush.
    """

    def __init__(self, program: List[Instr]) -> None:
        """
        Inicializa el procesador con un programa dado.

        1. Programa y estado global:
           - PC en 0; contador de ciclos en 0; estado de detención en False.

        2. Arquitectura (estado visible):
           - Banco de registros X0..X31 (X31 actúa como zero-register en lecturas).
           - Memoria principal modelada como mapa addr→valor64 (direcciones en bytes).

        3. Unidades funcionales:
           - ALU con ancho igual a min(ALU_WIDTH_BITS, REG_WIDTH_BITS).

        4. Registros de pipeline:
           - IF/ID, ID/EX, EX/MEM, MEM/WB en estado NOP.

        5. Métricas (para CPI y análisis de hazards):
           - retired_instrs: instrucciones retiradas en WB.
           - stall_count: burbujas por hazards de datos (p. ej., carga-uso).
           - branch_flushes: penalizaciones por control/predicción.
        """
        # 1. Programa y estado global
        self.program: List[Instr] = program
        self.pc: int = 0
        self.cycle: int = 0
        self.halted: bool = False

        # 2. Arquitectura (registros y memoria)
        self.regs: List[int] = [0] * REG_COUNT          # X0..X31; lectura de X31 devuelve 0.
        self.memory: Dict[int, int] = {}                # addr(byte) → palabra de 64 bits.

        # 3. Unidades funcionales
        self.alu: ALU = ALU(width_bits=min(ALU_WIDTH_BITS, REG_WIDTH_BITS))

        # 4. Registros de pipeline (estado inicial NOP)
        self.if_id: IfId = IfId()
        self.id_ex: IdEx = IdEx()
        self.ex_mem: ExMem = ExMem()
        self.mem_wb: MemWb = MemWb()

        # 5. Métricas
        self.retired_instrs: int = 0
        self.stall_count: int = 0
        self.branch_flushes: int = 0
        self.load_use_events = 0
        self.branch_events = 0
        self.branch_taken_events = 0
    # ------------------------------ utilidades ------------------------------

    @staticmethod
    def _sat_reg(value: int) -> int:
        """Satura a REG_WIDTH_BITS."""
        return mask_n_bits(value, REG_WIDTH_BITS)

    def _read_mem64(self, addr: int) -> int:
        """Lee palabra de 64 bits."""
        return self.memory.get(addr, 0)

    def _write_mem64(self, addr: int, val: int) -> None:
        """Escribe palabra de 64 bits."""
        self.memory[addr] = mask_n_bits(val, 64)

    # ------------------------------ etapas ---------------------------------

    def step(self) -> None:
        """
        Un ciclo: WB → MEM → EX → ID → IF. Detiene si halted.
        """
        if self.halted:
            return

        self.cycle += 1

        self._stage_wb()
        self._stage_mem()
        self._stage_ex()
        stall_inserted = self._stage_id()
        if not stall_inserted:
            self._stage_if()

    def _stage_wb(self) -> None:
        """WB: escribe resultados y contabiliza 'retired'."""
        instr = self.mem_wb.instr
        if instr.op in {Op.ADD, Op.SUB, Op.AND, Op.ORR, Op.EOR, Op.MOV, Op.LDR}:
            rd = self.mem_wb.rd
            if rd is not None and rd != 31:
                self.regs[rd] = self._sat_reg(self.mem_wb.write_val)
                self.retired_instrs += 1
        elif instr.op in {Op.STR, Op.CBZ, Op.CBNZ, Op.NOP}:
            if instr.op != Op.NOP:
                self.retired_instrs += 1

        self.mem_wb = MemWb()

    def _stage_mem(self) -> None:
        """MEM: LDR/STR y propagación a WB. Manejo de branches."""
        instr = self.ex_mem.instr
        next_memwb = MemWb(instr=instr)

        if instr.op == Op.LDR:
            addr = self.ex_mem.mem_addr or 0
            next_memwb.write_val = self._read_mem64(addr)
            next_memwb.rd = self.ex_mem.rd

        elif instr.op == Op.STR:
            addr = self.ex_mem.mem_addr or 0
            self._write_mem64(addr, self.ex_mem.alu_res)

        elif instr.op in {Op.ADD, Op.SUB, Op.AND, Op.ORR, Op.EOR, Op.MOV}:
            next_memwb.write_val = self.ex_mem.alu_res
            next_memwb.rd = self.ex_mem.rd

        # Control de salto
        if self.ex_mem.branch_taken and self.ex_mem.new_pc is not None:
            self.pc = self.ex_mem.new_pc
            self.if_id = IfId()
            self.id_ex = IdEx()
            if BRANCH_FLUSH_PENALTY > 0:
                self.branch_flushes += BRANCH_FLUSH_PENALTY

        self.mem_wb = next_memwb
        self.ex_mem = ExMem()

    def _stage_ex(self) -> None:
        """EX: operaciones de ALU y evaluación de branches."""
        instr = self.id_ex.instr
        next_exmem = ExMem(instr=instr)

        a, b, imm = self.id_ex.a, self.id_ex.b, self.id_ex.imm

        if instr.op == Op.ADD:
            r = self.alu.exec(AluOp.ADD, a, b).value
            next_exmem.alu_res = self._sat_reg(r)
            next_exmem.rd = instr.rd

        elif instr.op == Op.SUB:
            r = self.alu.exec(AluOp.SUB, a, b).value
            next_exmem.alu_res = self._sat_reg(r)
            next_exmem.rd = instr.rd

        elif instr.op == Op.AND:
            r = self.alu.exec(AluOp.AND, a, b).value
            next_exmem.alu_res = self._sat_reg(r)
            next_exmem.rd = instr.rd

        elif instr.op == Op.ORR:
            r = self.alu.exec(AluOp.ORR, a, b).value
            next_exmem.alu_res = self._sat_reg(r)
            next_exmem.rd = instr.rd

        elif instr.op == Op.EOR:
            r = self.alu.exec(AluOp.EOR, a, b).value
            next_exmem.alu_res = self._sat_reg(r)
            next_exmem.rd = instr.rd

        elif instr.op == Op.MOV:
            next_exmem.alu_res = self._sat_reg(a if instr.rn is not None else imm)
            next_exmem.rd = instr.rd

        elif instr.op == Op.LDR:
            base = a
            addr = self._sat_reg(base + imm)
            next_exmem.mem_addr = addr
            next_exmem.rd = instr.rd

        elif instr.op == Op.STR:
            base = a
            addr = self._sat_reg(base + imm)
            next_exmem.mem_addr = addr
            next_exmem.alu_res = b if instr.rd is None else self.regs[instr.rd]

        elif instr.op == Op.CBZ:
            taken = (a == 0)
            next_exmem.branch_taken = taken
            if taken:
                next_exmem.new_pc = self.pc + imm

        elif instr.op == Op.CBNZ:
            taken = (a != 0)
            next_exmem.branch_taken = taken
            if taken:
                next_exmem.new_pc = self.pc + imm

        elif instr.op == Op.NOP:
            pass

        self.ex_mem = next_exmem
        self.id_ex = IdEx()

    def _stage_id(self) -> bool:
        """
        ID: lectura de registros y resolución de hazards (burbuja/forwarding).

        Política:
        - Hazard carga-uso: si la instrucción en EX/MEM es LDR y el operando
          actual (en ID) consume su resultado, se inserta una burbuja.
          * Con forwarding  : 1 ciclo de burbuja (dato llega tras MEM).
          * Sin forwarding  : 2 ciclos de burbuja (sin bypass EX/MEM ni MEM/WB).
        - Forwarding: habilitado/inhabilitado según FORWARDING_ENABLED.

        :return: True si se insertó burbuja (stall) y NO se avanza IF en este ciclo.
        """
        instr = self.if_id.instr

        # ------------------------------------------------------------------
        # Penalidad de carga-uso según modo de forwarding (configurable).
        # ------------------------------------------------------------------
        load_use_penalty = (
            LOAD_USE_STALL_CYCLES_FWD if FORWARDING_ENABLED else LOAD_USE_STALL_CYCLES_NOFWD
        )

        # ------------------------------------------------------------------
        # Hazard carga-uso: LDR produce dato en EX/MEM; consumidor en ID.
        # Detecta dependencia RAW inmediata contra el destino de la LDR.
        # ------------------------------------------------------------------
        def is_load_use_hazard() -> bool:
            if self.ex_mem.instr.op != Op.LDR:
                return False
            load_rd = self.ex_mem.rd
            if load_rd is None:
                return False
            # Consumidores potenciales en la instrucción actual (ID):
            consumers = [instr.rn, instr.rm, instr.rz]
            return load_rd in consumers

        # Si hay carga-uso, inserta burbuja acorde al modo seleccionado.
        if is_load_use_hazard():
            # Contabiliza evento de carga-uso (métrica para f_LU).
            self.load_use_events += 1

            # Inserta burbuja: no se propaga ID→EX; IF no avanza (retorna True).
            self.stall_count += load_use_penalty
            self.id_ex = IdEx()  # NOP en ID/EX
            return True

        # ------------------------------------------------------------------
        # Lectura de registros con forwarding condicional.
        # ZR (X31) siempre lee 0. Se permite bypass desde EX/MEM y MEM/WB
        # únicamente si FORWARDING_ENABLED=True.
        # ------------------------------------------------------------------
        def read_reg(idx: Optional[int]) -> int:
            if idx is None or idx == 31:
                return 0

            val = self.regs[idx]

            if FORWARDING_ENABLED:
                # Bypass EX/MEM → ID (resultado ALU aún no escrito).
                if (
                        self.ex_mem.instr.op in {Op.ADD, Op.SUB, Op.AND, Op.ORR, Op.EOR, Op.MOV}
                        and self.ex_mem.rd == idx
                ):
                    val = self.ex_mem.alu_res

                # Bypass MEM/WB → ID (resultado listo para escribir).
                if (
                        self.mem_wb.rd == idx
                        and self.mem_wb.instr.op in {Op.ADD, Op.SUB, Op.AND, Op.ORR, Op.EOR, Op.MOV, Op.LDR}
                ):
                    val = self.mem_wb.write_val

            return val

        # Selección de operandos A/B (en branches se usa rz como fuente A).
        a_src = instr.rz if instr.op in {Op.CBZ, Op.CBNZ} else instr.rn
        a = read_reg(a_src)
        b = read_reg(instr.rm)
        imm = instr.imm or 0

        # Latch de ID→EX con operandos ya resueltos/bypasseados.
        self.id_ex = IdEx(instr=instr, a=a, b=b, imm=imm)

        # Predicción not-taken: IF ya precargó la siguiente; se limpia IF/ID.
        # Si hubiera fallo de predicción, el flush ocurre en MEM (ver _stage_mem).
        self.if_id = IfId()

        # Sin burbuja: se continúa el flujo normal del pipeline.
        return False

    def _stage_if(self) -> None:
        """IF: fetch de instrucción y avance de PC."""
        if self.pc < 0 or self.pc >= len(self.program):
            self.if_id = IfId(Instr(Op.NOP))
            self.halted = True
            return

        instr = self.program[self.pc]
        self.if_id = IfId(instr=instr)
        # Predicción not-taken
        if BRANCH_PREDICT_NOT_TAKEN:
            self.pc += 1
        else:
            if instr.op in {Op.CBZ, Op.CBNZ} and instr.imm is not None:
                self.pc += instr.imm
            else:
                self.pc += 1

        self.branch_events += 1
        if self.ex_mem.branch_taken:
            self.branch_taken_events += 1  # not-taken ⇒ tomado = mispredict

    # ------------------------------ ejecución -------------------------------

    def run(self, max_cycles: int = 1_000_000) -> Tuple[int, int, int]:
        """
        Ejecuta hasta finalizar o alcanzar 'max_cycles'.

        :return: (ciclos_totales, instrucciones_retiradas, stalls_totales)
        """
        while not self.halted and self.cycle < max_cycles:
            self.step()
        return self.cycle, self.retired_instrs, self.stall_count + self.branch_flushes


# ============================================================================
#  Utilidades B: carga de datos, construcción de programa y métricas CPI
# ============================================================================

def patch_branch_imm(program: List[Instr], branch_index: int, target_index: int) -> None:
    """
    Parchea inmediato de CBZ/CBNZ con delta relativo 'target - (branch+1)'.
    """
    delta = target_index - (branch_index + 1)
    instr = program[branch_index]
    if instr.op not in {Op.CBZ, Op.CBNZ}:
        raise ValueError("Solo CBZ/CBNZ soportados para parcheo.")
    instr.imm = delta


def build_demo_program(base_reg: int = 0,
                       len_reg: int = 1,
                       sum_reg: int = 2,
                       max_reg: int = 3,
                       tmp_reg: int = 4,
                       store_dst_reg: int = 5) -> List[Instr]:
    """
    B1: Programa mínimo que suma N elementos (64-bit) y calcula el máximo.
    Usa LDR/STR y CB(N)Z. Subconjunto pseudo-ARM.

    :return: Lista de instrucciones.
    """
    prog: List[Instr] = []

    # sum=0; max=0
    prog += [
        Instr(op=Op.MOV, rd=sum_reg, rn=31),
        Instr(op=Op.MOV, rd=max_reg, rn=31),
    ]

    loop_pos = len(prog)

    # tmp <- 8 (tamaño de palabra)
    prog.append(Instr(op=Op.MOV, rd=tmp_reg, rn=None, imm=MEM_DEFAULT_WORD_BYTES))
    # LDR t <- [base]
    prog.append(Instr(op=Op.LDR, rd=tmp_reg, rn=base_reg, imm=0))
    # sum += t
    # sum += t
    prog.append(Instr(op=Op.ADD, rd=sum_reg, rn=sum_reg, rm=tmp_reg))

    # (t > max) ? max = max + (t - max) = t : max
    # Válido para el patrón de datos lineal estricto (1..N) usado en la demo.
    prog.append(Instr(op=Op.SUB, rd=tmp_reg, rn=tmp_reg, rm=max_reg))  # tmp <- t - max
    prog.append(Instr(op=Op.CBZ, rz=tmp_reg, imm=1))  # si t==max, saltar asignación
    prog.append(Instr(op=Op.ADD, rd=max_reg, rn=max_reg, rm=tmp_reg))  # max <- max + (t - max) = t

    # len--
    prog.append(Instr(op=Op.MOV, rd=tmp_reg, rn=None, imm=1))
    prog.append(Instr(op=Op.SUB, rd=len_reg, rn=len_reg, rm=tmp_reg))

    # base += 8
    prog.append(Instr(op=Op.MOV, rd=tmp_reg, rn=None, imm=MEM_DEFAULT_WORD_BYTES))
    prog.append(Instr(op=Op.ADD, rd=base_reg, rn=base_reg, rm=tmp_reg))

    # CBNZ len -> loop
    back = Instr(op=Op.CBNZ, rz=len_reg, imm=0)
    prog.append(back)
    patch_branch_imm(prog, branch_index=len(prog) - 1, target_index=loop_pos)

    # STR max -> [store_dst]
    prog.append(Instr(op=Op.STR, rd=max_reg, rn=store_dst_reg, imm=0))

    return prog


def load_demo_data(cpu: PipelineCPU,
                   base_addr: int,
                   length: int,
                   pattern: str = "linear") -> None:
    """
    Carga arreglo de N enteros de 64 bits en memoria principal.

    :param cpu: CPU.
    :param base_addr: Dirección base.
    :param length: N elementos.
    :param pattern: 'linear'|'descending'|'xor' (patrones simples).
    """
    for i in range(length):
        if pattern == "linear":
            val = i + 1
        elif pattern == "descending":
            val = (length - i)
        else:
            val = i ^ (length - i)
        cpu._write_mem64(base_addr + i * MEM_DEFAULT_WORD_BYTES, val)


def prime_registers_for_demo(cpu: PipelineCPU,
                             base_addr: int,
                             length: int,
                             store_addr: int) -> None:
    """
    Inicializa X0, X1, X5 y limpia el resto (zero-register implícito en X31).
    """
    for i in range(REG_COUNT):
        cpu.regs[i] = 0
    cpu.regs[0] = base_addr
    cpu.regs[1] = length
    cpu.regs[5] = store_addr


def run_program_and_metrics(program: List[Instr],
                            data_len: int) -> Dict[str, int]:
    """
    Ejecuta el programa en pipeline y reporta métricas (CPI, stalls).
    """
    cpu = PipelineCPU(program=program)
    base_addr = 0x1000
    store_addr = 0x8000

    load_demo_data(cpu, base_addr=base_addr, length=data_len, pattern="linear")
    prime_registers_for_demo(cpu, base_addr=base_addr, length=data_len, store_addr=store_addr)

    cycles, retired, stalls = cpu.run(max_cycles=1_000_000)
    cpi_milli = int((cycles / max(1, retired)) * 1000)
    stored_max = cpu._read_mem64(store_addr)

    return {
        "cycles": cycles,
        "retired": retired,
        "stalls": stalls,
        "cpi_milli": cpi_milli,
        "stored_max": stored_max,
        "expected_max": data_len,
        "events": {
            "load_use": cpu.load_use_events,
            "branch_total": cpu.branch_events,
            "branch_taken": cpu.branch_taken_events,
        },
        "fractions": {
            "f_LU": cpu.load_use_events / max(1, retired),
            "f_br": cpu.branch_taken_events / max(1, retired),
        },
        "cpi": cycles / max(1, retired),
    }


# ============================================================================
#  # C1–C2. Caché Directa (parametrizable) + access/report
# ============================================================================

@dataclass
class CacheLine:
    """Línea de caché directa (válido + tag)."""
    valid: bool = False
    tag: int = 0


class DirectMappedCache:
    """
    Caché directa (E=1) parametrizable por C (capacidad) y B (tamaño de línea).
    Cumple C1 y C2.
    """

    def __init__(self, capacity_bytes: int, line_bytes: int) -> None:
        assert line_bytes > 0 and (line_bytes & (line_bytes - 1)) == 0, "B debe ser potencia de 2."
        assert capacity_bytes > 0 and capacity_bytes % line_bytes == 0, "C debe ser múltiplo de B."

        self.C = capacity_bytes
        self.B = line_bytes
        self.S = self.C // self.B

        self.lines: List[CacheLine] = [CacheLine() for _ in range(self.S)]

        self.hits = 0
        self.misses = 0
        self.accesses = 0

        self.offset_bits = self._log2(self.B)
        self.index_bits = self._log2(self.S)

    @staticmethod
    def _log2(x: int) -> int:
        n = 0
        v = x
        while v > 1:
            v >>= 1
            n += 1
        return n

    def _index_and_tag(self, addr: int) -> Tuple[int, int]:
        index = (addr >> self.offset_bits) & ((1 << self.index_bits) - 1)
        tag = addr >> (self.offset_bits + self.index_bits)
        return index, tag

    def access(self, addr: int) -> bool:
        """
        C2: acceso (lectura/escritura lógica) para simular hit/miss.
        """
        self.accesses += 1
        index, tag = self._index_and_tag(addr)
        line = self.lines[index]

        if line.valid and line.tag == tag:
            self.hits += 1
            return True

        self.misses += 1
        line.valid = True
        line.tag = tag
        return False

    def report(self) -> Dict[str, float]:
        """
        C2: métricas básicas (hits, misses, tasas).
        """
        hit_rate = (self.hits / self.accesses) if self.accesses else 0.0
        return {
            "capacity_bytes": float(self.C),
            "line_bytes": float(self.B),
            "sets": float(self.S),
            "accesses": float(self.accesses),
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": hit_rate,
            "miss_rate": 1.0 - hit_rate,
        }


# ============================================================================
#  # C3. Experimentos mínimos (lineal, stride conflictivo, aleatorio)
# ============================================================================

def experiment_linear(cache: DirectMappedCache,
                      base_addr: int,
                      count: int,
                      step: int) -> Dict[str, float]:
    """
    C3(i): Accesos contiguos con paso 'step' (alta localidad espacial).
    """
    addr = base_addr
    for _ in range(count):
        cache.access(addr)
        addr += step
    return cache.report()


def experiment_stride_conflict(cache: DirectMappedCache,
                               base_a: int,
                               base_b: int,
                               count: int,
                               stride: int) -> Dict[str, float]:
    """
    C3(ii): Dos flujos alternados que colisionan (mismo set) → thrashing.
    """
    addr_a = base_a
    addr_b = base_b
    for _ in range(count):
        cache.access(addr_a)
        cache.access(addr_b)
        addr_a += stride
        addr_b += stride
    return cache.report()


def experiment_random(cache: DirectMappedCache,
                      base_addr: int,
                      span: int,
                      count: int,
                      seed: int) -> Dict[str, float]:
    """
    C3(iii): Accesos uniformes en un rango amplio (baja localidad).
    """
    state = seed & 0xFFFFFFFFFFFFFFFF

    def lcg() -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return state

    for _ in range(count):
        r = lcg() % span
        addr = base_addr + int(r)
        cache.access(addr)
    return cache.report()


# ============================================================================
#  # C4. AMAT = HitTime + MissRate × MissPenalty  —  Sensibilidad B y C
# ============================================================================

def amat(hit_time: int, miss_rate: float, miss_penalty: int) -> float:
    """
    C4: Cálculo canónico de AMAT.
    """
    return hit_time + miss_rate * miss_penalty


def run_cache_experiments() -> Dict[str, Dict[str, float]]:
    """
    Ejecuta los tres experimentos y añade AMAT en cada reporte.
    """
    results: Dict[str, Dict[str, float]] = {}

    # 1) Lineal
    cache1 = DirectMappedCache(CACHE_CAPACITY_BYTES, CACHE_LINE_BYTES)
    rep1 = experiment_linear(cache1, base_addr=0, count=EXPERIMENT_STEPS, step=MEM_DEFAULT_WORD_BYTES)
    rep1["AMAT"] = amat(CACHE_HIT_TIME, rep1["miss_rate"], CACHE_MISS_PENALTY)
    results["linear"] = rep1

    # 2) Stride conflictivo
    cache2 = DirectMappedCache(CACHE_CAPACITY_BYTES, CACHE_LINE_BYTES)
    rep2 = experiment_stride_conflict(cache2,
                                      base_a=0,
                                      base_b=CACHE_CAPACITY_BYTES,
                                      count=EXPERIMENT_STEPS // 2,
                                      stride=EXPERIMENT_STRIDE_BYTES)
    rep2["AMAT"] = amat(CACHE_HIT_TIME, rep2["miss_rate"], CACHE_MISS_PENALTY)
    results["stride_conflict"] = rep2

    # 3) Aleatorio
    cache3 = DirectMappedCache(CACHE_CAPACITY_BYTES, CACHE_LINE_BYTES)
    rep3 = experiment_random(cache3,
                             base_addr=0,
                             span=EXPERIMENT_RANGE_BYTES,
                             count=EXPERIMENT_STEPS,
                             seed=PSEUDO_RANDOM_SEED)
    rep3["AMAT"] = amat(CACHE_HIT_TIME, rep3["miss_rate"], CACHE_MISS_PENALTY)
    results["random"] = rep3

    return results


# ============================================================================
#  # C4. Barrido sistemático de B y C — sensibilidad de AMAT
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    """Retorna True si x es potencia de 2 (>0)."""
    return x > 0 and (x & (x - 1)) == 0


def sweep_cache_bc(capacities: List[int],
                   line_sizes: List[int],
                   steps: int = EXPERIMENT_STEPS,
                   base_addr: int = 0,
                   span: int = EXPERIMENT_RANGE_BYTES,
                   seed: int = PSEUDO_RANDOM_SEED) -> List[Dict[str, float]]:
    """
    Realiza un barrido de parámetros (C, B) y ejecuta los tres experimentos
    (lineal, stride conflictivo, aleatorio) reportando AMAT y tasas.

    :param capacities: Lista de C en bytes (p.ej., [2048, 4096, 8192]).
    :param line_sizes: Lista de B en bytes (potencia de 2) (p.ej., [16, 32, 64]).
    :param steps: Número de accesos por experimento.
    :param base_addr: Dirección base para lineal/aleatorio.
    :param span: Rango para aleatorio.
    :param seed: Semilla PRNG.
    :return: Lista de diccionarios, cada uno con resultados por (C,B,scenario).
    """
    results: List[Dict[str, float]] = []

    for C in capacities:
        for B in line_sizes:
            if C <= 0 or B <= 0 or (C % B) != 0 or not _is_power_of_two(B):
                # Parámetros inválidos; se omite la combinación.
                continue

            # 1) Lineal
            cache = DirectMappedCache(C, B)
            rep = experiment_linear(cache, base_addr=base_addr, count=steps, step=MEM_DEFAULT_WORD_BYTES)
            rep["AMAT"] = amat(CACHE_HIT_TIME, rep["miss_rate"], CACHE_MISS_PENALTY)
            rep.update({"scenario": "linear", "C": float(C), "B": float(B)})
            results.append(rep)

            # 2) Stride conflictivo
            cache = DirectMappedCache(C, B)
            rep = experiment_stride_conflict(cache,
                                             base_a=base_addr,
                                             base_b=base_addr + C,  # fuerza colisión de sets
                                             count=steps // 2,
                                             stride=C)  # stride conflictivo
            rep["AMAT"] = amat(CACHE_HIT_TIME, rep["miss_rate"], CACHE_MISS_PENALTY)
            rep.update({"scenario": "stride_conflict", "C": float(C), "B": float(B)})
            results.append(rep)

            # 3) Aleatorio
            cache = DirectMappedCache(C, B)
            rep = experiment_random(cache,
                                    base_addr=base_addr,
                                    span=span,
                                    count=steps,
                                    seed=seed)
            rep["AMAT"] = amat(CACHE_HIT_TIME, rep["miss_rate"], CACHE_MISS_PENALTY)
            rep.update({"scenario": "random", "C": float(C), "B": float(B)})
            results.append(rep)

    return results


def sweep_presets() -> List[Dict[str, float]]:
    """
    Presets recomendados para el informe:
    - C ∈ {2, 4, 8, 16 KiB}
    - B ∈ {16, 32, 64} B
    """
    capacities = [2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024]
    line_sizes = [16, 32, 64]
    return sweep_cache_bc(capacities, line_sizes)


# ============================================================================
#  # D. (Informe) — No requerido en código; utilidades para exportar resultados
# ============================================================================

def collect_all_results(data_len: int = 64) -> Dict[str, object]:
    """
    Reúne resultados para alimentar tablas/figuras del informe (Parte D).
    No imprime ni guarda; retorna estructura para serialización externa.
    """
    # A: Latencias relativas (ejemplo de estimación)
    lat_rca = estimate_adder_latency(ALU_WIDTH_BITS, "RCA")
    lat_cla = estimate_adder_latency(ALU_WIDTH_BITS, "CLA")

    # B: Métricas de pipeline
    program = build_demo_program()
    metrics = run_program_and_metrics(program, data_len=data_len)

    # C: Experimentos de caché
    cache_results = run_cache_experiments()

    return {
        "params": {
            "ALU_WIDTH_BITS": ALU_WIDTH_BITS,
            "REG_WIDTH_BITS": REG_WIDTH_BITS,
            "CACHE_CAPACITY_BYTES": CACHE_CAPACITY_BYTES,
            "CACHE_LINE_BYTES": CACHE_LINE_BYTES,
            "CACHE_HIT_TIME": CACHE_HIT_TIME,
            "CACHE_MISS_PENALTY": CACHE_MISS_PENALTY,
            "PSEUDO_RANDOM_SEED": PSEUDO_RANDOM_SEED,
            "DATA_LEN": data_len,
        },
        "alu_latency_estimates": {
            "RCA": lat_rca,
            "CLA": lat_cla,
            "CLA_model_enabled": ALU_USE_CLA_MODEL,
        },
        "pipeline_metrics": metrics,
        "cache_experiments": cache_results,
    }


# ============================================================================
#  Punto de entrada referencial Metodo Heredado [Usar Main.py mejor]
# ============================================================================

def main() -> None:
    """
    Punto de entrada de referencia:
    - Ejecuta sanity checks de ALU (A3).
    - Construye y evalúa programa demo (B1/B2).
    - Corre experimentos de caché (C3/C4).
    - Retorna resultados agregados (para serialización externa).
    """
    alu_sanity_checks()
    _ = collect_all_results(data_len=64)
    # No imprime ni guarda por defecto.


if __name__ == "__main__":
    """
    Punto de entrada ejecutable.

    Ejecuta:
    1) Chequeos de sanidad de la ALU (opcional).
    2) Construcción y evaluación del programa demo en el pipeline (métricas CPI).
    3) Tres experimentos de caché directa (lineal, stride conflictivo, aleatorio) y AMAT.

    Soporta parámetros por CLI para ajustar tamaño de datos, caché y salida JSON.
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="eval_arquitectura.py",
        description="Ejecuta la demo de ALU + Pipeline ARMv8-A + Caché directa (CPI y AMAT)."
    )

    parser.add_argument(
        "--no-forwarding",
        action="store_true",
        help="Desactiva el forwarding en el pipeline (duplica la penalidad carga-uso)."
    )

    # Parámetros del programa demo (pipeline).
    parser.add_argument(
        "--data-len",
        type=int,
        default=64,
        help="Número de elementos 64-bit en el arreglo del programa demo (por defecto: 64)."
    )

    # Parámetros de caché (se aplican a TODOS los experimentos).
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=CACHE_CAPACITY_BYTES,
        help=f"Capacidad total C en bytes (por defecto: {CACHE_CAPACITY_BYTES})."
    )
    parser.add_argument(
        "--line-bytes",
        type=int,
        default=CACHE_LINE_BYTES,
        help=f"Tamaño de línea B en bytes (por defecto: {CACHE_LINE_BYTES})."
    )

    # Control de ALU / reproducibilidad.
    parser.add_argument(
        "--no-alu-asserts",
        action="store_true",
        help="Desactiva aserciones de sanidad de la ALU."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PSEUDO_RANDOM_SEED,
        help=f"Semilla para el experimento aleatorio (por defecto: {PSEUDO_RANDOM_SEED})."
    )

    # Salida.
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Ruta para guardar un JSON con resultados (opcional)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso: no imprime resultados en consola, solo guarda JSON si se indica."
    )

    args = parser.parse_args()

    # Actualiza parámetros globales cuando proceda.
    if args.no_alu_asserts:
        ALU_ASSERTIONS_ENABLED = False  # noqa: N806  (ajuste deliberado de global)
    CACHE_CAPACITY_BYTES = int(args.cache_capacity)  # noqa: N806
    CACHE_LINE_BYTES = int(args.line_bytes)  # noqa: N806
    PSEUDO_RANDOM_SEED = int(args.seed)  # noqa: N806
    FORWARDING_ENABLED = not args.no_forwarding  # noqa: N806

    # 1) Chequeos de ALU (opcional).
    alu_sanity_checks()

    # 2) Pipeline: construye y evalúa el programa demo.
    program = build_demo_program()
    metrics = run_program_and_metrics(program, data_len=int(args.data_len))

    # 3) Experimentos de caché (se usan los globals ya actualizados).
    cache_results = run_cache_experiments()

    # Empaqueta resultados.
    result = {
        "params": {
            "ALU_WIDTH_BITS": ALU_WIDTH_BITS,
            "REG_WIDTH_BITS": REG_WIDTH_BITS,
            "CACHE_CAPACITY_BYTES": CACHE_CAPACITY_BYTES,
            "CACHE_LINE_BYTES": CACHE_LINE_BYTES,
            "CACHE_HIT_TIME": CACHE_HIT_TIME,
            "CACHE_MISS_PENALTY": CACHE_MISS_PENALTY,
            "PSEUDO_RANDOM_SEED": PSEUDO_RANDOM_SEED,
            "DATA_LEN": int(args.data_len),
        },
        "pipeline_metrics": metrics,
        "cache_experiments": cache_results,
    }

    # Salida por consola.
    if not args.quiet:
        print("# ===================== RESULTADOS PIPELINE =====================")
        print(f"Ciclos totales       : {metrics['cycles']}")
        print(f"Instrucciones retir. : {metrics['retired']}")
        print(f"Stalls (incl. flush) : {metrics['stalls']}")
        print(f"CPI (x1000)          : {metrics['cpi_milli']} (milli-CPI)")
        print(f"Máx. almacenado      : {metrics['stored_max']}")
        print(f"Máx. esperado        : {metrics['expected_max']}")
        print("")
        print("# ==================== EXPERIMENTOS DE CACHÉ ====================")
        for name, rep in cache_results.items():
            print(f"[{name}] "
                  f"hit_rate={rep['hit_rate']:.4f}  "
                  f"miss_rate={rep['miss_rate']:.4f}  "
                  f"AMAT={rep['AMAT']:.2f} ciclos  "
                  f"(C={int(rep['capacity_bytes'])}B, B={int(rep['line_bytes'])}B, sets={int(rep['sets'])})")

    # Guardado opcional en JSON.
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
