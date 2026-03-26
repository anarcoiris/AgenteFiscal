"""
test_rag_qa.py — Suite de tests tipo test (opción múltiple) para el RAG Tributario.

Evalúa:
  - Si el RAG selecciona la respuesta correcta
  - Si la respuesta está fundamentada en el contexto (grounding check)
  - Tiempo de respuesta y confianza
"""
import os
import sys
import time
import random
import pickle
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from rag_engine import RAGEngine


# ── Preguntas tipo test (opción múltiple) ────────────────────────────────────

PREGUNTAS_TEST = [
    {
        "pregunta": "¿Cuál es el plazo general para presentar la declaración de la Renta en España?",
        "opciones": {
            "A": "Del 1 de enero al 31 de marzo",
            "B": "Del 3 de abril al 30 de junio",
            "C": "Del 1 de julio al 30 de septiembre",
            "D": "Del 1 de octubre al 31 de diciembre",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Qué impuesto grava la renta de las personas físicas en España?",
        "opciones": {
            "A": "IVA",
            "B": "Impuesto de Sociedades",
            "C": "IRPF",
            "D": "ITBIS",
        },
        "correcta": "C",
    },
    {
        "pregunta": "¿A partir de qué importe de rendimientos del trabajo con un solo pagador NO estás obligado a declarar IRPF?",
        "opciones": {
            "A": "15.000€",
            "B": "22.000€",
            "C": "30.000€",
            "D": "12.000€",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Cuál de estos es un rendimiento del capital inmobiliario?",
        "opciones": {
            "A": "Salario de un empleo",
            "B": "Ingresos por alquiler de un piso",
            "C": "Dividendos de acciones",
            "D": "Intereses de una cuenta de ahorro",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Qué tipo impositivo se aplica en el IRPF a las ganancias patrimoniales por venta de un inmueble (primeros 6.000€)?",
        "opciones": {
            "A": "15%",
            "B": "19%",
            "C": "21%",
            "D": "25%",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Qué es el mínimo personal en el IRPF?",
        "opciones": {
            "A": "El salario mínimo interprofesional",
            "B": "La cantidad de renta que no se somete a tributación por satisfacer las necesidades básicas del contribuyente",
            "C": "El importe máximo deducible por vivienda",
            "D": "La base imponible mínima para declarar",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Cuántos pagadores obligan a declarar si los ingresos totales superan 15.000€?",
        "opciones": {
            "A": "1 pagador",
            "B": "2 o más pagadores (si el segundo y sucesivos superan 1.500€)",
            "C": "3 pagadores",
            "D": "No influye el número de pagadores",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Qué se entiende por 'rendimientos del trabajo' en el IRPF?",
        "opciones": {
            "A": "Solo el salario base",
            "B": "Todas las contraprestaciones que deriven del trabajo personal por cuenta ajena",
            "C": "Solo las pensiones de jubilación",
            "D": "Los beneficios empresariales",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Qué es una ganancia patrimonial?",
        "opciones": {
            "A": "El salario anual del contribuyente",
            "B": "Una variación positiva en el valor del patrimonio al transmitir un bien",
            "C": "Los intereses bancarios",
            "D": "Las aportaciones a planes de pensiones",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Qué reducción se puede aplicar por obtención de rendimientos del trabajo en el IRPF?",
        "opciones": {
            "A": "Reducción del 50% para todos los trabajadores",
            "B": "Reducción para rendimientos netos del trabajo inferiores a 19.747,5€ (con cuantías variables)",
            "C": "No existe ninguna reducción por rendimientos del trabajo",
            "D": "Reducción fija de 5.000€ para todos",
        },
        "correcta": "B",
    },
    {
        "pregunta": "¿Cuál de los siguientes gastos no es deducible para un abogado que presta servicios por cuenta propia?",
        "opciones": {
            "A": "Gastos de gasolina de un coche que utiliza principalmente para su actividad, pero también de forma residual para su vida personal.",
            "B": "La prima de un seguro de enfermedad que contrato con una entidad privada.",
            "C": "La cuota de Seguridad Social que paga como autónomo, si también hace aportaciones a la mutualidad de abogacía.",
        },
        "correcta": "A",
    },
]


def format_test_prompt(pregunta_data: dict) -> str:
    """Formatea una pregunta tipo test para el RAG."""
    opciones_text = "\n".join(
        f"  {k}) {v}" for k, v in pregunta_data["opciones"].items()
    )
    return f"""{pregunta_data['pregunta']}

Opciones:
{opciones_text}

Responde indicando la letra de la opción correcta (A, B, C o D) y justifica brevemente tu elección basándote en el contexto."""


def extract_answer_letter(response: str) -> str | None:
    """Extrae la letra de la respuesta del modelo."""
    # Buscar patrones como "La respuesta es B", "Opción B", "B)", etc.
    patterns = [
        r'\b([A-D])\)',
        r'[Oo]pci[oó]n\s+([A-D])',
        r'[Rr]espuesta.*?([A-D])',
        r'[Cc]orrecta.*?([A-D])',
        r'^([A-D])[\.\):\s]',
        r'\*\*([A-D])\*\*',
        r'es la ([A-D])',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()

    # Fallback: buscar la primera letra A-D que aparece aislada
    match = re.search(r'\b([A-D])\b', response[:200])
    if match:
        return match.group(1).upper()
    return None


def check_grounding(response: str, debug_chunks: list) -> dict:
    """Verifica si la respuesta está fundamentada en los chunks recuperados."""
    if not debug_chunks:
        return {"grounded": False, "reason": "No hay chunks para verificar"}

    chunks_text = " ".join(debug_chunks).lower()
    response_lower = response.lower()

    # Buscar señales de hallucination
    hallucination_markers = [
        "modelo 720", "modelo 303", "modelo 130",
        # Si menciona artículos que no están en el contexto
    ]
    hallucinated = []
    for marker in hallucination_markers:
        if marker in response_lower and marker not in chunks_text:
            hallucinated.append(marker)

    if hallucinated:
        return {"grounded": False, "reason": f"Menciona conceptos no presentes en los chunks: {hallucinated}"}

    return {"grounded": True, "reason": "OK"}


def run_test_suite():
    """Ejecuta la batería de tests tipo test."""
    print("=" * 70)
    print("🧪 TEST SUITE — RAG Tributario (Tipo Test)")
    print("=" * 70)

    engine = RAGEngine(on_status=lambda msg: print(f"  [{msg}]"))
    engine.load()
    # Limpiar caché para tests limpios
    engine.clear_cache()

    results = []

    for i, q_data in enumerate(PREGUNTAS_TEST, 1):
        formatted_question = format_test_prompt(q_data)
        print(f"\n{'─'*60}")
        print(f"  Q{i}: {q_data['pregunta']}")
        for k, v in q_data["opciones"].items():
            marker = "→" if k == q_data["correcta"] else " "
            print(f"    {marker} {k}) {v}")

        t0 = time.time()
        result = engine.query(formatted_question, debug=True)
        elapsed = time.time() - t0

        answer = result["answer"]
        confidence = result.get("confidence", "?")
        debug_chunks = result.get("debug_chunks", [])

        # Extraer letra
        selected = extract_answer_letter(answer)
        correct = q_data["correcta"]
        is_correct = selected == correct

        # Grounding check
        grounding = check_grounding(answer, debug_chunks)

        icon = "✅" if is_correct else "❌"
        print(f"  {icon} Seleccionó: {selected or '?'} | Correcta: {correct} | "
              f"Confianza: {confidence} | ⏱️ {elapsed:.1f}s")
        print(f"  📌 Grounding: {'✅' if grounding['grounded'] else '⚠️'} {grounding['reason']}")
        print(f"  Respuesta (primeros 300 chars):")
        print(f"    {answer[:300]}...")

        results.append({
            "pregunta": q_data["pregunta"],
            "seleccionada": selected,
            "correcta": correct,
            "acierto": is_correct,
            "grounded": grounding["grounded"],
            "confianza": confidence,
            "tiempo": elapsed,
        })

    # ── Reporte final ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 REPORTE FINAL")
    print(f"{'='*70}")

    total = len(results)
    aciertos = sum(1 for r in results if r["acierto"])
    grounded = sum(1 for r in results if r["grounded"])
    avg_time = sum(r["tiempo"] for r in results) / total if total else 0

    print(f"  Total preguntas:     {total}")
    print(f"  Aciertos:            {aciertos}/{total} ({100*aciertos/total:.0f}%)")
    print(f"  Grounding OK:        {grounded}/{total} ({100*grounded/total:.0f}%)")
    print(f"  Tiempo promedio:     {avg_time:.1f}s")

    # Detalle por pregunta
    print(f"\n  {'#':<4} {'Acierto':<10} {'Grounded':<10} {'Confianza':<12} {'Tiempo':<8}")
    print(f"  {'─'*44}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<4} {'✅' if r['acierto'] else '❌':<10} "
              f"{'✅' if r['grounded'] else '⚠️':<10} "
              f"{r['confianza']:<12} {r['tiempo']:.1f}s")

    print(f"{'='*70}")


if __name__ == "__main__":
    run_test_suite()
