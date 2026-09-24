#!/usr/bin/env python
"""Liga de L linajes por TURNOS, con seleccion evolutiva (PBT).

POR QUE POR TURNOS Y NO CONCURRENTE. Concurrente -L redes vivas en el mismo
update- exige envolver las ~520 lineas del cuerpo del update en un bucle por
linaje, con estado por linaje del controlador de KL, del salvavidas y del
normalizador de valor. Por turnos sale lo MISMO en datos por linaje
-660/L filas por update de media- con el lote por update L veces MAYOR, que
para PPO, que es on-policy, es mejor y no peor. El unico coste es que durante
el turno de un linaje los rivales estan congelados: como mucho (L-1)*K updates
de rancidez.

QUE HACE CADA TURNO. El linaje l entrena K updates con:

    la mayoria de trabajadores   DOS ASIENTOS: l contra si mismo, los dos
                                 aportan gradiente
    unos pocos                   contra una instantanea de la LIGA elegida por
                                 PFSP -- ahi es donde se mide el Elo
    uno                          contra v48 sin capar: el ancla ABSOLUTA, la
                                 unica medida que significa algo fuera

PBT. Al cerrar cada ciclo completo, el linaje con peor Elo COPIA los pesos del
mejor y se le perturban los hiperparametros. Copiar, no promediar: dos redes
con inicializacion independiente estan en cuencas distintas y sus unidades
ocultas en orden arbitrario -simetria de permutacion-, asi que su media es
basura, no un hijo.

Y esto convierte los diales que nunca hemos ajustado conjuntamente -lr del
tronco, ratio cabezas/tronco, kl objetivo, peso del shaping- en una BUSQUEDA
que corre mientras entrena, en vez de constantes elegidas a ojo.
"""
import argparse
import json
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_ = os.path.join(RAIZ, ".venv312", "bin", "python")

# Puntos de partida SEPARADOS a proposito: que los linajes empiecen distintos
# tambien es diversidad, y PBT los mueve desde ahi. No son constantes
# elegidas -- son el estado inicial de una busqueda.
# DIVERSIDAD RELATIVA AL ANCESTRO, no absoluta. Estos factores multiplican el
# lr al que el checkpoint de partida YA habia convergido; fijar valores
# absolutos elegidos para entrenar desde cero destroza un checkpoint maduro.
# MEDIDO el 2026-09-24: el ancestro venia con tronco 3,4e-05 y el HIPER0
# absoluto le imponia 3,0e-04 -casi 9x-, con lo que el ancla caia de 41.153 $ a
# 26.251 en diez updates, cada vez que empezaba un turno.
FACTORES = [
    {"lr": 1.0, "lr_heads": 1.0, "kl_target": 0.020, "phi_w": 1.0},
    {"lr": 0.5, "lr_heads": 1.5, "kl_target": 0.030, "phi_w": 0.5},
    {"lr": 2.0, "lr_heads": 2.0, "kl_target": 0.015, "phi_w": 1.5},
]
HIPER0 = None   # se construye desde el ancestro en `hiper_inicial`
LIMITES = {"lr": (1e-5, 3e-3), "lr_heads": (1e-5, 3e-3),
           "kl_target": (5e-3, 1e-1), "phi_w": (0.0, 3.0)}


def lr_ancestro(ck):
    """(lr tronco, lr cabezas) a los que el ancestro habia convergido."""
    if not ck or not os.path.exists(ck):
        return 3.0e-4, 3.0e-4
    import torch
    d = torch.load(ck, map_location="cpu", weights_only=False)
    g = (d.get("opt") or {}).get("param_groups") or []
    if len(g) >= 3:
        return float(g[0]["lr"]), float(g[2]["lr"])
    return 3.0e-4, 3.0e-4


def hiper(d, L, ancestro=None):
    f = os.path.join(d, "hiper.json")
    if os.path.exists(f):
        return {int(k): v for k, v in json.load(open(f)).items()}
    lt, lc = lr_ancestro(ancestro)
    out = {}
    for i in range(L):
        fa = FACTORES[i % len(FACTORES)]
        out[i] = {"lr": lt * fa["lr"], "lr_heads": lc * fa["lr_heads"],
                  "kl_target": fa["kl_target"], "phi_w": fa["phi_w"],
                  # el linaje 0 hereda EXACTAMENTE lo del ancestro, asi que no
                  # hace falta imponerselo: que siga con su optimizador.
                  "_pendiente": (fa["lr"] != 1.0 or fa["lr_heads"] != 1.0)}
    return out


def guarda_hiper(d, h):
    json.dump({str(k): v for k, v in h.items()},
              open(os.path.join(d, "hiper.json"), "w"), indent=1)


def ancla(d, l, n=10):
    """(media, error estandar) de las ultimas n lecturas del ANCLA del linaje.

    EL ANCLA ES DINERO contra el rival externo, y es la unica aptitud valida.
    El Elo del pool NO sirve: MEDIDO el 2026-09-23 en 12 ciclos, subio monotono
    de 988 a 1730 mientras el ancla tocaba techo en el ciclo 5 y se desplomaba
    en el 10. Son instantaneas de tu propio pasado: ganarles sube la puntuacion
    aunque contra el rival de verdad seas cada vez peor.
    """
    import re
    f = os.path.join(d, f"linaje{l}.log")
    if not os.path.exists(f):
        return None, None
    vs = re.findall(r"ANCLA\s+(\d+)\s+vs", open(f, errors="ignore").read())
    if not vs:
        return None, None
    import statistics
    v = [float(x) for x in vs[-n:]]
    se = (statistics.stdev(v) / len(v) ** 0.5) if len(v) > 1 else float("inf")
    return sum(v) / len(v), se


def anclas(d, L, n=10):
    out = {}
    for l in range(L):
        m, se = ancla(d, l, n)
        if m is not None:
            out[l] = (m, se)
    return out


def turno(a, l, h, ciclo):
    ck = os.path.join(a.dir, f"linaje{l}.pt")
    log = os.path.join(a.dir, f"linaje{l}.log")
    cmd = [PY_, "-m", "kagsym.cli.train",
           "--days", str(a.days), "--steps", str(a.days * 24),
           "--envs", str(a.envs), "--procs", str(a.envs),
           "--updates", str(a.k),
           "--levels", "6",
           "--dos-asientos", str(a.envs - 1 - a.rivales),
           "--rivales-liga", str(a.rivales),
           "--liga-dir", a.dir, "--linaje", str(l),
           "--rival-flow", "--refresh", str(a.refresh),
           "--espacial-weight", str(a.espacial),
           "--eval-cada", str(a.eval_cada), "--eval-n", str(a.eval_n),
           "--kl-target", f"{h[l]['kl_target']:.4f}",
           "--out", ck, "--run-name", f"L{l}c{ciclo}"]
    if os.path.exists(ck + ".ultimo"):
        cmd += ["--resume", ck + ".ultimo"]
    # EL LR SOLO SE IMPONE CUANDO CAMBIA. Pasarlo cada turno pisa el valor al
    # que el controlador de KL habia convergido -un `--lr` explicito
    # sobrescribe los cuatro grupos-, asi que cada turno arrancaba con un paso
    # 5x demasiado grande. MEDIDO el 2026-09-24 en el primer turno: el ancla
    # cayo de 39.654 $ a 24.514 en 55 updates y a los 90 aun no habia vuelto.
    # Con 45 turnos eso es una maquina de perder progreso.
    #
    # Se aplica en el PRIMER turno de cada linaje -donde se establece la
    # diversidad de hiperparametros- y despues solo cuando PBT los muta.
    if h[l].get("_pendiente", True):
        cmd += ["--lr", f"{h[l]['lr']:.3e}",
                "--lr-heads", f"{h[l]['lr_heads']:.3e}"]
        h[l]["_pendiente"] = False
    env = dict(os.environ, KAG_PHI_W=f"{h[l]['phi_w']:.3f}")
    t0 = time.time()
    with open(log, "a") as fo:
        fo.write(f"\n===== ciclo {ciclo} linaje {l} :: {' '.join(cmd)}\n")
        fo.flush()
        r = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env,
                           cwd=RAIZ)
    return r.returncode, time.time() - t0


def ultimo_dinero(d, l):
    f = os.path.join(d, f"linaje{l}.log")
    if not os.path.exists(f):
        return None
    for ln in reversed(open(f, errors="ignore").read().splitlines()):
        if ln.startswith("upd ") and "$=" in ln:
            try:
                return float(ln.split("$=")[1].split("vs")[0].strip())
            except Exception:
                return None
    return None


def evalua(ck, n=12, procs=4):
    """Evaluacion DETERMINISTA en semillas reservadas. (dinero, se) o None."""
    if not os.path.exists(ck):
        return None
    r = subprocess.run([PY_, "tools/evalua.py", ck, "--n", str(n),
                        "--procs", str(procs)],
                       capture_output=True, text=True, cwd=RAIZ, timeout=900)
    for ln in r.stdout.splitlines():
        if ln.startswith("EVAL "):
            d = json.loads(ln[5:])
            return d["dinero"], d["se"]
    return None


def sopa(fs, out, pesos=None):
    """Media de varios checkpoints. SOLO con antepasado comun.

    REFUTADA el 2026-09-24. Se dejan las dos medidas y el orden en que se
    hicieron, porque la leccion esta en la diferencia de tamaño de muestra:

      independientes    sopa  9.134 $ vs 22.707 del padre  t -2,69   n=10
      antepasado comun  sopa 48.109 $ vs 43.311 del padre  t +2,24   n=10
      --- y con 200 semillas pareadas, el mismo experimento: ---
      sopa(L0,L2) 45.158 $ vs 48.592 de L2 solo            t -2,62   n=200

    Con la sd REAL de 22.526 $/semilla, diez semillas no distinguen nada. El
    +4.798 que justifico montar una liga entera alrededor de la sopa era
    ruido, y el apoyo de produccion -48.061 contra 45.882 en el ciclo 1- eran
    48 semillas, o sea 0,47 sigmas. Tampoco era evidencia.

    Se conserva el codigo porque el mecanismo puede funcionar en otro regimen,
    pero NO se usa sin volver a medirlo con 200 semillas. Y lo mismo aplica a
    promediar instantaneas de UNA trayectoria (SWA), medido a 200 semillas:
    -4.285 $, t -2,85.

    Con inicializacion independiente las unidades ocultas estan en orden
    arbitrario -simetria de permutacion- y la media es basura. Con antepasado
    comun los checkpoints viven en la misma cuenca y la media BATE A LOS DOS
    PADRES. Por eso todos los linajes de esta liga salen del mismo ancestro.
    """
    import torch
    ds = [torch.load(f, map_location="cpu", weights_only=False) for f in fs]
    w = pesos or [1.0 / len(ds)] * len(ds)
    base = ds[0]
    sd = {}
    for k, v in base["sd"].items():
        if not v.is_floating_point():
            sd[k] = v
            continue
        acc = None
        tot = 0.0
        for wi, d in zip(w, ds):
            t = d["sd"].get(k)
            if t is None or t.shape != v.shape:
                continue
            acc = (wi * t) if acc is None else acc + wi * t
            tot += wi
        sd[k] = (acc / tot) if (acc is not None and tot > 0) else v
    torch.save({**base, "sd": sd}, out)
    return out


def pbt(a, h, anc, rng):
    """El peor COPIA al mejor y muta. La aptitud es el ANCLA, o sea DINERO.

    El umbral no es un numero elegido: se exige que la diferencia supere DOS
    veces la suma de los errores estandar de las dos medias. Por debajo de eso
    la ventaja es ruido de la propia medida y PBT estaria copiando azar.
    """
    if len(anc) < 2:
        return "PBT: aun no hay ancla de al menos dos linajes"
    orden = sorted(anc, key=lambda k: anc[k][0])
    peor, mejor = orden[0], orden[-1]
    dif = anc[mejor][0] - anc[peor][0]
    # CONTRASTE DE DOS SIGMAS sobre la diferencia, no suma de errores. El
    # error de la diferencia de dos medias independientes es la raiz de la
    # suma de cuadrados, no la suma: `2*(se_a+se_b)` era ~1,4x mas estricto de
    # lo necesario y bloqueaba selecciones reales.
    umbral = 2.0 * (anc[mejor][1] ** 2 + anc[peor][1] ** 2) ** 0.5
    if not (dif > umbral):
        return (f"PBT: sin cambios, la ventaja de L{mejor} ({dif:+.0f} $) no "
                f"supera el ruido de la medida ({umbral:.0f} $)")
    import shutil
    # se copia el MEJOR POR ANCLA del ganador -que es lo que `--out` guarda
    # ahora-, no su ultimo estado: el ultimo puede ser posterior a un colapso.
    src = os.path.join(a.dir, f"linaje{mejor}.pt")
    if not os.path.exists(src):
        src = os.path.join(a.dir, f"linaje{mejor}.pt.ultimo")
    dst = os.path.join(a.dir, f"linaje{peor}.pt.ultimo")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    cambios = []
    for k, (lo, hi) in LIMITES.items():
        f = float(rng.choice([0.8, 1.25]))
        nuevo = min(hi, max(lo, h[mejor][k] * f))
        cambios.append(f"{k} {h[peor][k]:.3g}->{nuevo:.3g}")
        h[peor][k] = nuevo
    h[peor]["_pendiente"] = True   # PBT muto: ahora si hay que imponerlo
    return (f"PBT: L{peor} (ancla {anc[peor][0]:.0f} $) COPIA a L{mejor} "
            f"(ancla {anc[mejor][0]:.0f} $, ventaja {dif:+.0f} > ruido "
            f"{umbral:.0f}) y muta :: " + ", ".join(cambios))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="runs/liga")
    p.add_argument("--linajes", type=int, default=3)
    p.add_argument("--ciclos", type=int, default=10)
    p.add_argument("--k", type=int, default=60, help="updates por turno")
    p.add_argument("--envs", type=int, default=11)
    p.add_argument("--rivales", type=int, default=2,
                   help="trabajadores contra instantaneas de la liga")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--refresh", type=int, default=10)
    p.add_argument("--espacial", type=float, default=0.3,
                   help="peso de la cabeza espacial: el unico objetivo con "
                        "etiquetas exactas por casilla")
    p.add_argument("--eval-cada", type=int, default=40,
                   help="cada cuantos updates evalua el entrenador")
    p.add_argument("--eval-n", type=int, default=12,
                   help="semillas reservadas por evaluacion DENTRO del turno, "
                        "que solo elige el mejor checkpoint de ese linaje")
    p.add_argument("--eval-n-ciclo", type=int, default=48,
                   help="semillas para la aptitud de CICLO. Mas altas a "
                        "proposito: una decision que sobrescribe un linaje "
                        "entero -PBT, sopa- no puede tomarse con el ruido que "
                        "basta para ordenar checkpoints de uno solo. MEDIDO: "
                        "se 4.880 $ con 12 semillas, 1.966 $ con 48, por 2,5x "
                        "de tiempo.")
    p.add_argument("--ancestro", default=None,
                   help="checkpoint del que salen TODOS los linajes. Es "
                        "condicion para que la sopa funcione.")
    p.add_argument("--semilla", type=int, default=0)
    a = p.parse_args()
    import numpy as np
    rng = np.random.default_rng(a.semilla)
    os.makedirs(a.dir, exist_ok=True)
    h = hiper(a.dir, a.linajes, a.ancestro)
    guarda_hiper(a.dir, h)
    _mejor_global = [-1e18]
    if a.ancestro:
        import shutil
        for l in range(a.linajes):
            dst = os.path.join(a.dir, f"linaje{l}.pt.ultimo")
            if not os.path.exists(dst):
                shutil.copyfile(a.ancestro, dst)
        print(f"  los {a.linajes} linajes salen de {a.ancestro} "
              f"(ancestro comun: condicion de la sopa)", flush=True)
    print(f"LIGA: {a.linajes} linajes x {a.ciclos} ciclos x {a.k} updates "
          f"= {a.linajes * a.ciclos * a.k} updates totales", flush=True)
    print(f"  reparto por turno: {a.envs - 1 - a.rivales} dos asientos, "
          f"{a.rivales} liga, 1 ancla v48", flush=True)
    for c in range(1, a.ciclos + 1):
        for l in range(a.linajes):
            rc, dt = turno(a, l, h, c)
            din = ultimo_dinero(a.dir, l)
            print(f"  ciclo {c} linaje {l}: rc={rc} {dt/60:.1f} min "
                  f"dinero={din if din is None else f'{din:.0f}'}", flush=True)
            if rc != 0:
                print(f"  ABORTA: el linaje {l} salio con {rc}. Mira "
                      f"{a.dir}/linaje{l}.log", flush=True)
                return 1
        # EVALUACION RESERVADA de cada linaje: es la aptitud, no el ancla
        # -que mide la politica muestreada- ni el Elo -que es una cinta de
        # correr contra tu propio pasado-.
        ev = {}
        for l in range(a.linajes):
            r = evalua(os.path.join(a.dir, f"linaje{l}.pt"), a.eval_n_ciclo)
            if r:
                ev[l] = r
        print(f"  --- fin ciclo {c}: EVAL " + ", ".join(
            f"L{k}={v[0]:.0f}$(se {v[1]:.0f})" for k, v in sorted(ev.items())),
            flush=True)
        # SOPA: media de los DOS mejores. Solo vale porque todos descienden
        # del mismo ancestro; con inicializacion independiente la destroza la
        # simetria de permutacion (medido: -13.573 $, t -2,69).
        if len(ev) >= 2:
            top = sorted(ev, key=lambda k: -ev[k][0])[:2]
            fs = [os.path.join(a.dir, f"linaje{l}.pt") for l in top]
            sp = os.path.join(a.dir, f"sopa_c{c}.pt")
            try:
                sopa(fs, sp)
                r = evalua(sp, a.eval_n_ciclo)
                mejor_l = top[0]
                if r and r[0] > ev[mejor_l][0]:
                    import shutil
                    shutil.copyfile(sp, os.path.join(a.dir, "campeon.pt"))
                    print(f"  SOPA L{top[0]}+L{top[1]} = {r[0]:.0f} $ "
                          f"(se {r[1]:.0f}) GANA al mejor padre "
                          f"({ev[mejor_l][0]:.0f}) -> campeon.pt", flush=True)
                    # el peor hereda la sopa: asi la recombinacion se propaga
                    peor = min(ev, key=lambda k: ev[k][0])
                    # A LOS DOS FICHEROS. `.ultimo` es de donde reanuda el
                    # turno siguiente, pero `.pt` es el que PBT copia como
                    # "el mejor de ese linaje". Escribiendo solo `.ultimo`,
                    # `ev[peor]` decia la puntuacion de la sopa mientras
                    # `linaje{peor}.pt` seguia teniendo el checkpoint viejo:
                    # MEDIDO en el ciclo 1, PBT propago 43.276 $ anunciando
                    # 48.061. El camino principal no se rompia -la sopa si
                    # llegaba al turno siguiente- pero el log mentia.
                    for _dst in (f"linaje{peor}.pt.ultimo", f"linaje{peor}.pt"):
                        shutil.copyfile(sp, os.path.join(a.dir, _dst))
                    ev[peor] = r
                else:
                    if r:
                        print(f"  SOPA L{top[0]}+L{top[1]} = {r[0]:.0f} $ "
                              f"no gana al mejor padre ({ev[mejor_l][0]:.0f})",
                              flush=True)
                    import shutil
                    if ev[mejor_l][0] > _mejor_global[0]:
                        _mejor_global[0] = ev[mejor_l][0]
                        shutil.copyfile(fs[0], os.path.join(a.dir, "campeon.pt"))
                os.remove(sp)
            except Exception as e:
                print(f"  SOPA fallo: {type(e).__name__}: {e}", flush=True)
        print("  " + pbt(a, h, ev, rng), flush=True)
        guarda_hiper(a.dir, h)
    print("LIGA TERMINADA", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
