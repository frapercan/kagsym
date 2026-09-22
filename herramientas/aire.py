"""Que celda sirve para investigar aprendizaje: la que tiene AIRE.

El pre-vuelo anterior filtraba por x_inaccion -si se gana dinero por encima de
no hacer nada-. La celda 12h x 5d lo pasa de sobra (2,7x) y sin embargo esta
MUERTA: CEM sobre las 32 dims no encuentra nada sobre el incumbente (+1,3 %) y
seis entrenamientos de RL con recompensas dispares aterrizan todos en 1085-1095.
Dos metodos que no comparten nada coincidiendo = es el techo, no el metodo.

El criterio correcto es el AIRE: cuanto mejora la busqueda sobre un incumbente
competente. Sin aire no hay nada que aprender y cualquier ablacion mide ruido.

La caja se escala por UNIDAD-TURNO -H * D * (1+tope)- en vez de a ojo. La
anterior escalera se fijo a ojo y el pre-vuelo tumbo 6 de 9 peldanos por
falta de capital.

El ultimo peldano es el campeonato con su configuracion REAL, que rompe el
patron a proposito: 3000 $ para 8640 unidad-turno es 0,35 $/ut frente a los
1,67 de la celda de juguete. El juego de verdad esta estrangulado de capital
comparado con nuestros mundos pequenos, y eso conviene verlo escrito.
"""
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kagsym import spec
from kagsym.symbolic import tasks as T
from kagsym.symbolic.executor import Agent
from kagsym.fastenv import FastEnv
from kagsym.macro import Macro, N_MACRO
import kagsym.macro as _M

BASE = np.load("runs/ligas/init32_l0.npy")
RATIO = 400.0 / (12 * 5 * 4)           # $/unidad-turno de la celda ancla
POB, ELITE, ITERS = 24, 6, 8
SEARCH_SEEDS, SEM_VAL = list(range(101, 107)), list(range(301, 313))

#            H   D  tope  caja (None = por unidad-turno)
CELDAS = [( 12,  5,  3, None),
          ( 24,  5,  3, None),
          ( 12, 10,  3, None),
          ( 24, 10,  5, None),
          ( 24, 30, 11, 3000)]         # campeonato, configuracion REAL

def play(vec, riv, seeds, H, D, CAJA):
    mios = []
    for s in seeds:
        env = FastEnv(configuration={"episodeSteps": H*D, "turnsPerDay": H,
                                     "startingMoney": CAJA}, seed=s)
        o = env.reset()
        a = Agent(episode_steps=H*D, macro=Macro.from_vector(list(vec)))
        r = Agent(episode_steps=H*D, macro=Macro.from_vector(list(riv)))
        while not env.done:
            o, _ = env.step([a(o[0]), r(o[1])])
        mios.append(float(env.rewards()[0]))
    return float(np.mean(mios))

if __name__ == "__main__":
    t0 = time.time()
    print("AIRE por celda: cuanto mejora CEM sobre el incumbente, reevaluado fresco")
    print(f"caja por unidad-turno = {RATIO:.2f} $/ut (ancla: la celda de juguete)\n")
    print(f"  {'celda':>16} {'ut':>6} {'caja':>6} {'incumb':>8} {'CEM':>8} "
          f"{'aire':>7} {'s':>5}")
    for H, D, TOPE, caja in CELDAS:
        t1 = time.time()
        ut = H * D * (1 + TOPE)
        CAJA = int(caja if caja else max(100, round(ut * RATIO / 50) * 50))
        spec.set_turns_per_day(H); spec.set_episode_steps(H*D)
        _M.HAND_CAP = TOPE; T.MICRO_MODE = "residuo"
        inc = play(BASE, BASE, SEM_VAL, H, D, CAJA)
        mu, sg = BASE.astype(float).copy(), np.full(N_MACRO, 0.25)
        best, pts_max = None, -1e9
        for _ in range(ITERS):
            pob = np.clip(mu + sg*np.random.randn(POB, N_MACRO), 0.0, 1.0)
            pts = np.array([play(v, BASE, SEARCH_SEEDS, H, D, CAJA) for v in pob])
            idx = np.argsort(pts)[-ELITE:]
            mu, sg = pob[idx].mean(0), pob[idx].std(0) + 0.02
            if pts.max() > pts_max:
                pts_max, best = float(pts.max()), pob[int(np.argmax(pts))].copy()
        cem = play(best, BASE, SEM_VAL, H, D, CAJA)   # semillas FRESCAS
        aire = 100.0 * (cem - inc) / max(inc, 1.0)
        np.save(f"runs/ligas/aire_{H}h{D}d.npy", best)
        print(f"  {f'{H}h x {D}d t{TOPE}':>16} {ut:>6} {CAJA:>6} {inc:>8.0f} "
              f"{cem:>8.0f} {aire:>+6.1f}% {time.time()-t1:>5.0f}", flush=True)
    print(f"\n  sin aire -> la celda esta agotada: cualquier ablacion mide ruido.")
    print(f"  el autoplay va en la MAS PEQUENA con aire de verdad.")
    print(f"  ({time.time()-t0:.0f}s)")
