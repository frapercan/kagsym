"""Vara de medir EXTERNA para la run larga de campeonato.

Hace falta porque el entrenador calcula `margen_pct` contra el rival de
ENTRENAMIENTO, y con autoplay ese rival somos nosotros: el margen deja de subir
en cuanto la copia se fortalece, aunque el agente siga mejorando. El propio
codigo lo documenta -en ese mismo tramo el margen contra el 2945 mejoro de
-98,7 % a -82,8 % mientras el retorno no se movia-.

Asi que el progreso se lee contra algo que NO se mueve: v48 en el juego real.

Uso:  python runs/ligas/vara.py runs/largo.pt.ultimo [mas_checkpoints...]
"""
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

H, D, CAJA = 24, 30, 3000
SEM = list(range(601, 609))          # fijas: comparables entre checkpoints

def evalua(ckpt):
    from kagsym import obs as O, spec
    from kagsym.symbolic import tasks as T
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_strict
    from kagsym.nets import world as M
    from kagsym.nets.world import AgenteE2E, MundoConfig
    import kagsym.macro as _M
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    ops = bool((d.get("cfg") or {}).get("con_ops", False))
    T.MODO_MICRO = "ops" if ops else "residuo"
    spec.set_turns_per_day(H); spec.set_episode_steps(H*D); _M.TOPE_PEONES = None
    net = AgenteE2E(MundoConfig(device="cpu", con_ops=ops))
    load_strict(net, d["sd"], ckpt); net.eval()
    HIST = torch.zeros(1, M.N_HIST)
    mios, suyos = [], []
    for s in SEM:
        env = FastEnv(configuration={"episodeSteps": H*D, "turnsPerDay": H,
                                     "startingMoney": CAJA}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H*D, macro=Macro.from_vector([0.5]*N_MACRO))
        rv = public_with_cap("v48-fast-routes", 10**6)
        dia = None
        with torch.no_grad():
            while not env.done:
                ob = o[0]
                if dia != ob["day"]:
                    gr, b = O.encode_obs(ob)
                    sal = net(torch.from_numpy(gr).unsqueeze(0),
                              torch.from_numpy(b).unsqueeze(0), HIST)
                    ag.macro = Macro.from_vector(
                        torch.sigmoid(sal["macro_mu"])[0].numpy())
                    mp = sal["micro"][0].numpy()
                    ag.micro = (lambda _o, m=((mp[0], mp[1:]) if ops else mp): m)
                    dia = ob["day"]
                try:
                    acc_r = rv(o[1])
                except Exception:
                    acc_r = {"farmer": ["PASS"], "hands": [], "market": []}
                o, _ = env.step([ag(ob), acc_r])
        r = env.rewards(); mios.append(float(r[0])); suyos.append(float(r[1]))
    m, v = np.array(mios), np.array(suyos)
    d_ = m - v
    return (m.mean(), v.mean(), d_.mean(),
            d_.std(ddof=1)/np.sqrt(len(d_)), int((m > v).sum()), int(d.get("upd", 0) or 0))

if __name__ == "__main__":
    print(f"vara externa: {H}h x {D}d caja {CAJA} contra v48, "
          f"{len(SEM)} semillas fijas\n")
    print(f"  {'checkpoint':>34} {'upd':>5} {'dinero':>9} {'v48':>9} "
          f"{'margen%':>8} {'gana':>6} {'s':>5}")
    for c in sys.argv[1:]:
        t1 = time.time()
        try:
            m, v, dm, ee, g, upd = evalua(c)
        except Exception as e:
            print(f"  {c[-34:]:>34}  fallo: {str(e)[:50]}", flush=True); continue
        print(f"  {c[-34:]:>34} {upd:>5} {m:>9.0f} {v:>9.0f} "
              f"{100*dm/v:>+7.1f}% {g:>3}/{len(SEM)} {time.time()-t1:>5.0f}",
              flush=True)
