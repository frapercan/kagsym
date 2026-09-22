"""Vara fija EMPAREJADA: ¿el modelo completo bate de verdad al estado del arte?

La vara normal compara dos MEDIAS independientes. Con sd de 13.300 $/semilla
eso no distingue un +6,6 % de cero (ver la nota de potencia estadistica). Aqui
se juega la MISMA semilla con los dos checkpoints y se mira la diferencia por
semilla: la varianza del escenario se cancela y hace falta muchisima menos
muestra para el mismo error estandar.

Las semillas se fijan ANTES de mirar nada, y son distintas de las 601-608 de la
vara normal para no premiar el haber ajustado nada a ellas.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H, D, CAJA = 24, 30, 3000
SEM = list(range(9001, 9201))        # 200, fijadas de antemano
PROCS = 6                            # el entrenamiento usa 11; no lo ahogamos


def _una(args):
    ckpt, s = args
    import torch
    # UN HILO POR PROCESO. Torch abre hilos por su cuenta: medido, cada worker
    # consumia 1,6 nucleos, y sumados a los 11 procesos del entrenamiento eran
    # ~21 hilos sobre 12 nucleos. El thrashing hacia la medida indefinida Y
    # frenaba el entrenamiento. Con un hilo, el trabajo son ~800 s de CPU
    # repartidos entre los procesos del pool.
    torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.exacto import tareas as T
    from kagsym.exacto.ejecutor import Agent
    from kagsym.entorno import publico_con_tope
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrar_ckpt import carga_estricta
    from kagsym.redes import mundo as M
    from kagsym.redes.mundo import AgenteE2E, MundoConfig
    import kagsym.macro as _M
    global _CACHE
    try:
        net, ops = _CACHE[ckpt]
    except Exception:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        ops = bool((d.get("cfg") or {}).get("con_ops", False))
        T.MODO_MICRO = "ops" if ops else "residuo"
        net = AgenteE2E(MundoConfig(device="cpu", con_ops=ops))
        carga_estricta(net, d["sd"], ckpt); net.eval()
        try:
            _CACHE[ckpt] = (net, ops)
        except NameError:
            _CACHE = {ckpt: (net, ops)}
    T.MODO_MICRO = "ops" if ops else "residuo"
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.TOPE_PEONES = None
    HIST = torch.zeros(1, M.N_HIST)
    env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                 "startingMoney": CAJA}, seed=s)
    o = env.reset()
    ag = Agent(episode_steps=H * D, macro=Macro.desde_vector([0.5] * N_MACRO))
    rv = publico_con_tope("v48-fast-routes", 10 ** 6)
    dia = None
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            if dia != ob["day"]:
                gr, b = O.encode_obs(ob)
                sal = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0), HIST)
                ag.macro = Macro.desde_vector(torch.sigmoid(sal["macro_mu"])[0].numpy())
                mp = sal["micro"][0].numpy()
                ag.micro = (lambda _o, m=((mp[0], mp[1:]) if ops else mp): m)
                dia = ob["day"]
            try:
                acc_r = rv(o[1])
            except Exception:
                acc_r = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([ag(ob), acc_r])
    r = env.rewards()
    return float(r[0]), float(r[1])


if __name__ == "__main__":
    import multiprocessing as mp
    A, B = sys.argv[1], sys.argv[2]
    t0 = time.time()
    print(f"vara EMPAREJADA: {H}h x {D}d caja {CAJA} contra v48 sin tope, "
          f"{len(SEM)} semillas fijadas de antemano")
    print(f"  A = {A}\n  B = {B}\n")
    # PROGRESO. Sin esto no hay forma de saber si va por el 40 % o el 95 %, y
    # eso ya costo tirar una medida de 15 minutos por no poder estimarla.
    def barre(ck, et):
        out = []
        with mp.Pool(PROCS) as pool:
            for i, r in enumerate(pool.imap(_una, [(ck, s) for s in SEM]), 1):
                out.append(r)
                if i % 20 == 0 or i == len(SEM):
                    print(f"    {et}: {i}/{len(SEM)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        return out
    ra = barre(A, "A")
    rb = barre(B, "B")
    a = np.array([x[0] for x in ra]); va = np.array([x[1] for x in ra])
    b = np.array([x[0] for x in rb]); vb = np.array([x[1] for x in rb])
    d = a - b
    ee = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {'':<10}{'nosotros':>10}{'v48':>10}{'margen%':>10}{'gana':>8}")
    print(f"  {'A':<10}{a.mean():>10.0f}{va.mean():>10.0f}"
          f"{100*(a-va).mean()/va.mean():>9.1f}%{int((a>va).sum()):>5}/{len(SEM)}")
    print(f"  {'B':<10}{b.mean():>10.0f}{vb.mean():>10.0f}"
          f"{100*(b-vb).mean()/vb.mean():>9.1f}%{int((b>vb).sum()):>5}/{len(SEM)}")
    print(f"\n  DIFERENCIA EMPAREJADA A-B: {d.mean():+.0f} $  "
          f"+- {ee:.0f} (ee)   t = {d.mean()/max(1e-9, ee):+.2f}")
    print(f"  sd por semilla: {d.std(ddof=1):.0f} $   "
          f"A gana a B en {int((d>0).sum())}/{len(SEM)}")
    veredicto = ("A ES MEJOR" if d.mean() > 2*ee else
                 "B ES MEJOR" if d.mean() < -2*ee else
                 "INDISTINGUIBLES a 2 ee")
    print(f"\n  -> {veredicto}   ({time.time()-t0:.0f}s)")
