"""¿Los agentes publicos son cintas deterministas o politicas reactivas?

Importa porque decide cuanto autojuego hace falta: una cinta se MEMORIZA, y
memorizar 24 cintas no es aprender a jugar. Una politica reactiva, no.

Dos pruebas independientes:
  1. MISMA configuracion, DOS SEMILLAS distintas. El tablero, el mercado y el
     pueblo cambian; una cinta indexada por paso juega exactamente lo mismo.
     Se mide la fraccion de turnos con accion identica.
  2. OTRA ESCALA (12 h/dia en vez de 24). Una cinta construida para 719 pasos
     se desincroniza y se hunde; una politica reactiva se adapta.

Medido para v48 en su dia: 60.426 $ a 24h x 30d y ~0 a cualquier otra escala.
Esto comprueba si eso vale para todos o solo para el.
"""
import json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def firma(nombre, sem, horas=24, dias=30):
    from kagsym import spec
    from kagsym.entorno import carga_publico
    from kagsym.fastenv import FastEnv
    spec.set_turns_per_day(horas); spec.set_episode_steps(horas * dias)
    ag = carga_publico(nombre)
    env = FastEnv(configuration={"episodeSteps": horas * dias,
                                 "turnsPerDay": horas, "startingMoney": 3000},
                  seed=sem)
    o = env.reset()
    acts = []
    while not env.done:
        a = ag(o[0])
        acts.append(json.dumps([a.get("farmer"), a.get("hands"),
                                a.get("market")], sort_keys=True))
        o, _ = env.step([a, {"farmer": ["PASS"], "hands": [], "market": []}])
    return acts, float(env.rewards()[0])


def analiza(nombre):
    a1, d1 = firma(nombre, 4242)
    a2, d2 = firma(nombre, 9999)
    n = min(len(a1), len(a2))
    igual = sum(1 for i in range(n) if a1[i] == a2[i]) / max(1, n)
    _, d12 = firma(nombre, 4242, horas=12, dias=30)
    return igual, (d1 + d2) / 2, d12


if __name__ == "__main__":
    import multiprocessing as mp
    def uno(nom):
        try:
            return (nom, *analiza(nom))
        except Exception as e:
            return (nom, None, None, None)
    b = json.load(open("data/escalera_valida.json"))
    nombres = [x["nombre"] for x in b]
    if "--pocos" in sys.argv:
        nombres = nombres[::4]
    print(f"{len(nombres)} agentes. 'igual' = % de turnos con la MISMA accion "
          f"en dos semillas distintas\n")
    print(f"  {'agente':<46}{'igual':>7}{'24h $':>9}{'12h $':>9}")
    with mp.Pool(8) as p:
        res = p.map(uno, nombres)
    import numpy as np
    ig = []
    for nom, i, d24, d12 in res:
        if i is None:
            print(f"  {nom[:44]:<46}{'fallo':>7}"); continue
        ig.append(i)
        print(f"  {nom[:44]:<46}{100*i:>6.0f}%{d24:>9.0f}{d12:>9.0f}")
    ig = np.array(ig)
    print(f"\n  CINTAS (>90 % identico): {int((ig>0.9).sum())}/{len(ig)}")
    print(f"  REACTIVOS (<50 %):       {int((ig<0.5).sum())}/{len(ig)}")
    print(f"  mediana de coincidencia: {100*np.median(ig):.0f} %")
