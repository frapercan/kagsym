"""Mide un desplazamiento del macro en las semillas RESERVADAS, y lo hornea.

La busqueda optimiza sobre las semillas que ve. Ese numero no es un resultado,
es una hipotesis: con sd 22.526 $/semilla, el mejor de 32 candidatos sobre 16
semillas esta inflado por seleccion (medido en otra ocasion: +10.708 $).

Aqui se mide EMPAREJADO -mismas semillas, delta y cero- sobre 7101-7300, que
la busqueda no ha visto. La diferencia emparejada cancela el efecto del
tablero, que es el termino que domina.

Si pasa, se hornea: `macro_mu` es un Linear(256,67), asi que sumar el delta a
su BIAS implementa sigmoid(macro_mu(x)+delta) exactamente. Coste cero por
turno y ni una linea de codigo en inferencia.

    .venv312/bin/python tools/valida_delta.py runs/delta_macro.npy [ck] [n]
    .venv312/bin/python tools/valida_delta.py ... --hornear runs/nuevo.pt
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

_ARGV = list(sys.argv)          # `sys.argv` se pisa abajo para importar cem_diales
DELTA = sys.argv[1]
CKP   = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "runs/partida_v2.pt"
N     = int(os.environ.get("VAL_N", 200))
SEM0  = int(os.environ.get("VAL_SEM0", 7101))
os.environ.setdefault("CEM_OUT", "/dev/null")
sys.argv = [sys.argv[0], CKP]
import cem_diales as C
C.CK = CKP


def _t(t):
    s, d = t
    try:
        return s, (0 if d is None else 1), C.episodio(s, d)
    except Exception:
        return s, (0 if d is None else 1), float("nan")


if __name__ == "__main__":
    import multiprocessing as mp
    d = np.load(DELTA)
    sem = [SEM0 + k for k in range(N)]
    print(f"[valida] {DELTA} sobre {CKP}")
    print(f"  {len(d)} diales vivos  |delta| medio {np.abs(d).mean():.3f}  "
          f"max {np.abs(d).max():.3f}")
    print(f"  {N} semillas reservadas {SEM0}-{SEM0+N-1}, EMPAREJADAS", flush=True)
    with mp.Pool(12) as p:
        res = p.map(_t, [(s, x) for s in sem for x in (None, d)], chunksize=1)
    A = {}; B = {}
    for s, w, v in res:
        (B if w == 0 else A)[s] = v
    par = np.array([A[s] - B[s] for s in sem
                    if np.isfinite(A.get(s, np.nan)) and np.isfinite(B.get(s, np.nan))])
    a = np.array([A[s] for s in sem if np.isfinite(A.get(s, np.nan))])
    b = np.array([B[s] for s in sem if np.isfinite(B.get(s, np.nan))])
    se = par.std(ddof=1) / np.sqrt(len(par))
    t = par.mean() / se if se > 0 else 0.0
    print(f"\n  con delta  {a.mean():10,.0f} $")
    print(f"  sin delta  {b.mean():10,.0f} $")
    print(f"  DIFERENCIA EMPAREJADA  {par.mean():+9,.0f} $ +- {se:,.0f}   "
          f"t = {t:+.2f}   n = {len(par)}")
    print(f"  gana en {100*(par>0).mean():.0f}% de los tableros")
    if "--hornear" in _ARGV:
        import torch
        dst = _ARGV[_ARGV.index("--hornear") + 1]
        if t < 2.0:
            print(f"\n  NO se hornea: t={t:+.2f} < 2. No ha superado el ruido.")
            sys.exit(1)
        from kagsym.rampa import es_rampa
        ck = torch.load(CKP, map_location="cpu", weights_only=False)
        if es_rampa(d, C.VIVOS):
            # RAMPA: depende del dia, asi que NO cabe en el bias -que es
            # constante por construccion-. Viaja como campo del checkpoint y
            # la aplica `submit_kagsym/main.py` con la MISMA funcion que la
            # busqueda, `kagsym.rampa.offset`. Antes esto reventaba con
            # "shape mismatch [98] en [49]": ruidoso, pero inservible.
            ck["delta_rampa"] = {"delta": [float(x) for x in d],
                                 "vivos": [int(i) for i in C.VIVOS]}
        else:
            off = torch.zeros(ck["sd"]["macro_mu.bias"].shape)
            off[C.VIVOS] = torch.from_numpy(d.astype(np.float32))
            ck["sd"]["macro_mu.bias"] = ck["sd"]["macro_mu.bias"] + off
        ck["delta_macro"] = {"origen": DELTA, "de": CKP,
                             "dif": float(par.mean()), "se": float(se), "n": len(par)}
        torch.save(ck, dst)
        print(f"\n  horneado -> {dst}  ({par.mean():+,.0f} $ +- {se:,.0f})")
