#!/usr/bin/env python
"""PUERTA de verificacion: cada mecanismo tiene que DEMOSTRAR que disparo.

POR QUE EXISTE. En una sola noche hubo siete mecanismos cableados detras de
una condicion que nadie cumplia -el torneo, el curriculo automatico, la cabeza
espacial, jepa, aux_rival, la metrica de producto y el ancla de MLflow-, mas
cuatro estados que se perdian al reanudar. Ninguno daba error: simplemente no
existian, y el run parecia sano.

Un mecanismo que no imprime prueba de haber disparado NO esta montado, por
mucho que su bandera este puesta. Esto corre un entrenamiento corto con todo
encendido y comprueba la evidencia una por una, en el log y en MLflow.

Se ejecuta ANTES de lanzar una corrida larga. No es un postmortem.
"""
import json, os, re, sqlite3, subprocess, sys, tempfile, time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_ = os.path.join(RAIZ, ".venv312", "bin", "python")

# (nombre, patron en el log, obligatorio)
LOG = [
    ("momentos de Adam por nombre",   r"momentos de Adam: \d+/\d+ tensores emparejados POR NOMBRE", True),
    ("lr del checkpoint restaurado",  r"optimizer restored: trunk", True),
    ("semillas continuan",            r"semillas de entrenamiento desde \d+", True),
    ("liston del mejor heredado",     r"liston heredado de", True),
    ("normalizador de valor",         r"normalizador de valor restaurado", True),
    ("salvavidas sembrado",           r"salvavidas sembrado", True),
    ("dos asientos",                  r"DOS ASIENTOS \(\d+ filas/update\)", True),
    ("rivales de liga asignados",     r"\[liga\]", True),
    ("evaluacion reservada",          r"\[eval\]", True),
    ("cabeza espacial entrena",       r"l_esp=(?!0\.000000)", True),
    ("cabeza aux_rival entrena",      r"l_aux=(?!0\.000000)", True),
    ("objetivo del historico existe", r"fhf=si", True),
    ("rival de autojuego SANO",       r"RIVAL ROTO", False),   # NO debe aparecer
]

MLF = [
    ("criterio: eval reservada",   "1_result/eval_dinero",          lambda v: v > 0),
    ("dispersion del ancla",       "1_result/ancla_se",             lambda v: v >= 0),
    ("fertilizante acreditado",    "6_producto/fertilizer",         lambda v: v > 0),
    ("ingreso real por producto",  "6_ingreso/TOTAL",               lambda v: v > 0),
    ("variedad efectiva",          "6_ingreso/variedad_efectiva",   lambda v: 1 <= v <= 12),
    ("saturacion de diales",       "2_health/manos_saturacion",     lambda v: 0 < v <= 1.5),
    ("pool de la liga",            "3_liga/instantaneas",           lambda v: v >= 1),
    ("salud: kl por dimension",    "2_health/kl_per_dim",           lambda v: v >= 0),
    ("salud: r2 del critico",      "2_health/critic_r2",            lambda v: True),
]


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "runs/liga8/linaje0.pt"
    ck = os.path.join(RAIZ, ck) if not os.path.isabs(ck) else ck
    if not os.path.exists(ck):
        print(f"no existe el checkpoint {ck}"); return 2
    tmp = tempfile.mkdtemp(prefix="montaje_")
    out = os.path.join(tmp, "m.pt")
    # el liston tiene que existir para que se pueda heredar
    json.dump({"mejor": 1.0, "upd": 0}, open(out + ".mejor.json", "w"))
    nombre = f"MONTAJE_{int(time.time())}"
    nombre1 = nombre + "_e1"
    cmd = [PY_, "-m", "kagsym.cli.train", "--resume", ck,
           # 30 DIAS Y 7 TRABAJADORES, no un mundo de juguete. Con episodios
           # de 3 dias no se vende nada y las metricas de producto salen a
           # cero: la puerta daria "FALLA" sobre codigo correcto. Y el reparto
           # necesita sitio para los tres roles sin solaparse (3 + 2 + 2).
           "--days", "30", "--steps", "720", "--envs", "7", "--procs", "7",
           "--updates", "4", "--levels", "6",
           "--dos-asientos", "3", "--rivales-liga", "2",
           "--liga-dir", os.path.join(tmp, "liga"),
           "--rival-flow", "--espacial-weight", "0.3", "--aux-weight", "0.3",
           "--eval-cada", "2", "--eval-n", "6", "--refresh", "2",
           "--out", out, "--run-name", nombre1]
    env = dict(os.environ, KDIAGAUX="1", KAG_LIGA_MAX="4", KAG_LIGA_MINPART="1")
    # DOS ETAPAS. Varios mecanismos solo se pueden verificar AL REANUDAR
    # -el normalizador de valor, el liston, el flujo de semillas- y para eso
    # hace falta un checkpoint escrito por ESTE codigo. La etapa 1 lo produce
    # y la etapa 2 es la que se audita.
    print(f"corriendo la puerta ({nombre}), etapa 1 de 2...", flush=True)
    t0 = time.time()
    r1 = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=RAIZ,
                        timeout=1800)
    if r1.returncode != 0:
        print("  la etapa 1 fallo:\n" + (r1.stdout + r1.stderr)[-1500:])
        return 2
    if not os.path.exists(out + ".ultimo"):
        print("  la etapa 1 no dejo checkpoint"); return 2
    print(f"  etapa 1 lista ({time.time()-t0:.0f}s), reanudando...", flush=True)
    cmd2 = list(cmd)
    cmd2[cmd2.index("--resume") + 1] = out + ".ultimo"
    cmd2[cmd2.index("--run-name") + 1] = nombre
    r = subprocess.run(cmd2, capture_output=True, text=True, env=env, cwd=RAIZ,
                       timeout=1800)
    log = r.stdout + r.stderr
    open(os.path.join(tmp, "log.txt"), "w").write(log)
    print(f"  {time.time()-t0:.0f}s, rc={r.returncode}\n")
    fallos = 0
    print(f"{'mecanismo':34s} {'evidencia en el log':>22s}")
    for nom, pat, debe in LOG:
        hay = re.search(pat, log) is not None
        ok = hay if debe else not hay
        fallos += 0 if ok else 1
        print(f"  {nom:32s} {'OK' if ok else 'FALLA':>22s}")
    print(f"\n{'metrica en MLflow':34s} {'valor':>22s}")
    c = sqlite3.connect(os.path.join(RAIZ, "data/mlflow.db"))
    q = ("SELECT m.value FROM metrics m JOIN runs r ON m.run_uuid=r.run_uuid "
         "WHERE r.name=? AND m.key=? ORDER BY m.step DESC LIMIT 1")
    for nom, key, pred in MLF:
        fila = list(c.execute(q, (nombre, key)))
        if not fila:
            print(f"  {nom:32s} {'AUSENTE':>22s}"); fallos += 1; continue
        v = float(fila[0][0])
        ok = pred(v)
        fallos += 0 if ok else 1
        print(f"  {nom:32s} {v:>15.3f} {'OK' if ok else 'FALLA':>6s}")
    print(f"\n  {'TODO CONECTADO' if fallos == 0 else str(fallos) + ' FALLOS'}"
          f"   log completo en {tmp}/log.txt")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
