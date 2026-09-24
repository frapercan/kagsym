#!/usr/bin/env python
"""Panel completo de una corrida: resultado, produccion, salud y alarmas.

No es un volcado de metricas: cada una se muestra con su TENDENCIA -primer
valor, ultimo, y el cambio- porque un numero suelto no dice si vamos bien.

Las alarmas estan derivadas de medidas, no elegidas:
  * verb_signal_noise < 1  -> el ruido tapa lo aprendido; la politica
    determinista se degrada aunque la muestreada parezca sana (medido: L0 con
    0,20 hacia 903 $ desplegada y 15.952 muestreada).
  * kl_per_dim muy por encima de su objetivo -> pasos demasiado grandes.
  * critic_r2 cayendo -> las ventajas se vuelven ruido.
  * x_inaction ~1 -> no batimos a no hacer nada.
  * saturation alta -> PPO recorta la mayoria del lote.
"""
import sqlite3, sys, os
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(RAIZ, "data/mlflow.db")

GRUPOS = [
    ("EL CRITERIO (evaluacion reservada, 100 semillas)", [
        ("1_result/eval_dinero",      "dinero",            "%9.0f"),
        ("1_result/eval_se",          "  su error",        "%9.0f"),
        ("1_result/eval_margen_pct",  "  margen %",        "%9.1f"),
        ("1_result/eval_mejor",       "  mejor hasta hoy", "%9.0f"),
    ]),
    ("EL ANCLA (en entrenamiento, ruidosa)", [
        ("1_result/ancla_dinero",     "nuestro dinero",    "%9.0f"),
        ("1_result/ancla_se",         "  su error",        "%9.0f"),
        ("1_result/ancla_margen_pct", "  margen %",        "%9.1f"),
        ("3_context/rival_money",     "dinero del rival",  "%9.0f"),
    ]),
    ("AGREGADOS (mezclan regimenes: NO son criterio)", [
        ("1_result/money",            "dinero medio",      "%9.0f"),
        ("1_result/margin_pct",       "margen %",          "%9.1f"),
        ("1_result/win_rate",         "win rate",          "%9.3f"),
        ("1_result/x_inaction",       "x inaccion",        "%9.2f"),
    ]),
    ("PRODUCCION (unidades vendidas, rival externo)", [
        ("6_producto/strawberry",     "fresa    (v48: 430)", "%9.1f"),
        ("6_producto/milk",           "leche    (v48: 335)", "%9.1f"),
        ("6_producto/fertilizer",     "fertiliz.(v48: 397)", "%9.1f"),
        ("6_producto/wool",           "lana     (v48: 179)", "%9.1f"),
        ("6_producto/melon",          "melon    (v48:  72)", "%9.1f"),
        ("6_producto/egg",            "huevo",               "%9.1f"),
        ("6_producto/wheat",          "trigo    (v48: 313)", "%9.1f"),
        ("6_producto/TOTAL",          "TOTAL unidades",      "%9.1f"),
    ]),
    ("INGRESO POR PRODUCTO (dolares reales del motor)", [
        ("6_ingreso/TOTAL",              "ingreso total",   "%9.0f"),
        ("6_ingreso/variedad_efectiva",  "variedad efectiva", "%9.2f"),
        ("6_ingreso/cuota_top2",         "cuota top-2 %",   "%9.1f"),
    ]),
    ("SALUD DE LA MAQUINARIA", [
        ("2_health/kl_per_dim",       "kl por dimension",  "%9.4f"),
        ("2_health/kl_target",        "  su objetivo",     "%9.4f"),
        ("2_health/saturation",       "lote recortado",    "%9.3f"),
        ("2_health/critic_r2",        "r2 del critico",    "%9.3f"),
        ("2_health/verb_signal_noise","senal/ruido verbo", "%9.3f"),
        ("2_health/macro_w_norm",     "|W| del macro",     "%9.3f"),
        ("2_health/grad_zero_pct",    "gradiente nulo %",  "%9.1f"),
        ("2_health/manos_saturacion", "diales realizados", "%9.3f"),
        ("4_diag/lr_trunk",           "lr tronco",         "%9.2e"),
        ("4_diag/lr_micro",           "lr micro",          "%9.2e"),
    ]),
]

ALARMAS = [
    ("2_health/verb_signal_noise", lambda v: v < 1.0,
     "el ruido tapa la senal del verbo: la politica DETERMINISTA se degrada"),
    ("1_result/x_inaction", lambda v: v < 1.5,
     "apenas batimos a no hacer nada"),
    ("2_health/saturation", lambda v: v > 0.6,
     "PPO recorta la mayoria del lote: pasos demasiado grandes"),
    ("2_health/critic_r2", lambda v: v < 0.2,
     "el critico no explica el retorno: las ventajas son ruido"),
    ("2_health/grad_zero_pct", lambda v: v > 50,
     "mas de la mitad del gradiente es cero"),
]


def serie(c, run, key):
    q = ("SELECT m.step, m.value FROM metrics m JOIN runs r ON m.run_uuid=r.run_uuid "
         "WHERE r.name=? AND m.key=? ORDER BY m.step")
    return [(int(s), float(v)) for s, v in c.execute(q, (run, key))]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "LARGO"
    c = sqlite3.connect(DB)
    pasos = serie(c, run, "1_result/money")
    if not pasos:
        print(f"sin metricas para '{run}'"); return 1
    print(f"=== {run} ===  pasos {pasos[0][0]} -> {pasos[-1][0]}  "
          f"({len(pasos)} lecturas)\n")
    for titulo, filas in GRUPOS:
        hay = False
        buf = []
        for key, nom, fmt in filas:
            s = serie(c, run, key)
            if not s:
                continue
            hay = True
            ini, fin = s[0][1], s[-1][1]
            # media de las ultimas 5 lecturas, menos ruidosa que la ultima
            ult = float(np.mean([v for _, v in s[-5:]]))
            d = fin - ini
            flecha = "  " if abs(d) < 1e-9 else ("^ " if d > 0 else "v ")
            buf.append(f"  {nom:22s} {fmt % ini} {fmt % ult} {flecha}{fmt % d}")
        if hay:
            print(f"{titulo}")
            print(f"  {'':22s} {'inicio':>9s} {'ult.5':>9s}   {'cambio':>9s}")
            print("\n".join(buf)); print()
    print("ALARMAS")
    saltan = 0
    for key, pred, msg in ALARMAS:
        s = serie(c, run, key)
        if not s:
            continue
        v = float(np.mean([x for _, x in s[-5:]]))
        if pred(v):
            saltan += 1
            print(f"  [!] {key.split('/')[-1]} = {v:.3f}: {msg}")
    if not saltan:
        print("  ninguna")


if __name__ == "__main__":
    sys.exit(main() or 0)
