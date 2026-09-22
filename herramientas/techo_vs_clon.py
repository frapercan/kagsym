"""Cuanto aire hay por encima del rival EN ESTA CELDA.

Separa las dos hipotesis que explican igual de bien el empate de la rejilla:

  H1  el rival es demasiado fuerte -> hay techo, pero el no nos deja llegar.
      Se arregla con un rival mas flojo para empezar.
  H2  el rival esta pegado al techo de la celda -> no hay sitio arriba y
      ningun rival arregla nada; hay que cambiar de celda.

La medida que las separa: el MEJOR vector fijo de 32 dims contra ESTE mismo
rival. Es la comparacion limpia porque el rival es exactamente eso, un vector
fijo sin condicionar, asi que se busca dentro de su misma clase.

  techo >> 1076   -> H1: hay aire, el rival lo tapa.
  techo ~ 1076    -> H2: no hay aire. Cambiar de celda, no de rival.

Se reevalua en semillas FRESCAS: el maximo de una busqueda esta inflado por
seleccion -medido en esta misma competicion, entre 1,2x y 1,81x-.
"""
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ligas"))
import numpy as np, torch
from kagsym import spec
from kagsym.symbolic import tasks as T
from kagsym.symbolic.executor import Agent
from kagsym.fastenv import FastEnv
from kagsym.macro import Macro, N_MACRO
import kagsym.macro as _M

H, D, TOPE, CAJA = 12, 5, 3, 400
INIT = "runs/ligas/init32_l0.npy"
SEM_BUSCA = list(range(101, 107))      # las que ve la busqueda
SEM_VAL   = list(range(301, 313))      # frescas, las mismas que juzgan la rejilla
POB, ELITE, ITERS = 24, 6, 8

spec.set_turns_per_day(H); spec.set_episode_steps(H*D); _M.TOPE_PEONES = TOPE
T.MODO_MICRO = "residuo"
RIV = list(np.load(INIT))

def play(vec, seeds):
    mios, suyos = [], []
    for s in seeds:
        env = FastEnv(configuration={"episodeSteps": H*D, "turnsPerDay": H,
                                     "startingMoney": CAJA}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H*D, macro=Macro.from_vector(list(vec)))
        rv = Agent(episode_steps=H*D, macro=Macro.from_vector(RIV))
        while not env.done:
            o, _ = env.step([ag(o[0]), rv(o[1])])
        r = env.rewards(); mios.append(float(r[0])); suyos.append(float(r[1]))
    return float(np.mean(mios)), float(np.mean(suyos))

if __name__ == "__main__":
    t0 = time.time()
    print(f"techo de la clase VECTOR FIJO contra el clon, celda {H}h x {D}d "
          f"caja {CAJA} tope {TOPE}")
    b0, r0 = play(RIV, SEM_VAL)
    print(f"  referencias   inaccion {CAJA}   el rival (=nuestro init) {b0:.0f}")
    print(f"  CEM {POB}x{ITERS} sobre {N_MACRO} dims, {len(SEM_BUSCA)} semillas por sorteo\n")
    mu, sg = np.array(RIV, float), np.full(N_MACRO, 0.25)
    mejor_v, mejor_s = None, -1e9
    for it in range(ITERS):
        t1 = time.time()
        pob = np.clip(mu + sg * np.random.randn(POB, N_MACRO), 0.0, 1.0)
        pts = np.array([play(v, SEM_BUSCA)[0] for v in pob])
        idx = np.argsort(pts)[-ELITE:]
        mu, sg = pob[idx].mean(0), pob[idx].std(0) + 0.02
        if pts.max() > mejor_s:
            mejor_s, mejor_v = float(pts.max()), pob[int(np.argmax(pts))].copy()
        if it == 0:
            print(f"  (una iteracion = {time.time()-t1:.0f}s -> ETA "
                  f"{(time.time()-t1)*ITERS:.0f}s)\n", flush=True)
        print(f"   it {it+1:>2}  busqueda {pts.max():>7.0f}  "
              f"sigma medio {sg.mean():.3f}", flush=True)
    m, v = play(mejor_v, SEM_VAL)
    print(f"\n  busqueda (inflada por seleccion)  {mejor_s:>7.0f}")
    print(f"  REEVALUADO en semillas frescas     {m:>7.0f}   rival {v:.0f}   "
          f"margen {m-v:+.0f}")
    print(f"  el rival mismo                     {b0:>7.0f}")
    aire = 100.0 * (m - b0) / b0
    print(f"\n  AIRE sobre el rival: {aire:+.1f} %")
    print(f"  la meseta de la rejilla estaba en ~1090 ({100*(1090-b0)/b0:+.1f} %)")
    print("  -> " + ("H1: hay aire, el rival lo tapa. Bajar el rival."
                     if aire > 8 else
                     "H2: no hay aire en esta celda. Cambiar de celda, no de rival."))
    np.save("runs/ligas/techo_vs_clon.npy", mejor_v)
    print(f"  ({time.time()-t0:.0f}s)")
