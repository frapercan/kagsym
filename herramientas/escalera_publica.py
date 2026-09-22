"""Escalera de rivales PUBLICOS por puntuacion, de 50 en 50 puntos.

Por que hace falta: la escalera anterior tenia 11 peldanos y ganabamos 8 de
ellos al 88-100 %. Por el propio criterio del curriculo esos rivales ya no
ensenan nada, y el unico que importaba -v48 sin tope- lo perdiamos al 100 %,
o sea el salto era demasiado grande. Hace falta relleno REAL entre medias.

Como se construye:
  1. leaderboard completo (equipo -> puntuacion). Se pagina con `page_token`;
     `page_size` se queda en 200 pero el token sigue, asi que se llega al final.
  2. cuadernos publicos de la competicion (autor -> ref descargable).
  3. cruce por autor/equipo, y se elige UNO cada `PASO` puntos.

Lo que NO se puede hacer, y conviene tenerlo escrito: el codigo de una
submission ajena no es descargable. Solo lo es el de un CUADERNO publico. Asi
que la escalera solo puede cubrir los tramos donde alguien publico su cuaderno.
"""
import io, json, os, re, sys, time, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMP = "kaggriculture"
ALTO, BAJO, PASO = 2700, 1500, 50
CACHE = "data/escalera_publica.json"


def leaderboard(api):
    tok, rows = None, []
    for _ in range(200):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = api.competition_leaderboard_view(COMP, page_size=200, page_token=tok)
        m = re.search(r"Next Page Token = (\S+)", buf.getvalue())
        nuevo = m.group(1) if m else None
        n0 = len(rows)
        for e in (r or []):
            try:
                rows.append((str(getattr(e, "team_name", "?")), float(e.score)))
            except Exception:
                pass
        if not nuevo or nuevo == tok or len(rows) == n0:
            break
        tok = nuevo
    return rows


def cuadernos(api):
    out, pg = [], 1
    while pg <= 30:
        try:
            ks = api.kernels_list(competition=COMP, page_size=100, page=pg,
                                  sort_by="scoreDescending")
        except Exception:
            break
        if not ks:
            break
        for k in ks:
            out.append({"ref": k.ref, "autor": k.author, "titulo": k.title})
        if len(ks) < 100:
            break
        pg += 1
    return out


if __name__ == "__main__":
    import kaggle
    api = kaggle.KaggleApi(); api.authenticate()
    t0 = time.time()
    if os.path.exists(CACHE) and "--refresca" not in sys.argv:
        d = json.load(open(CACHE))
        lb, ks = d["leaderboard"], d["cuadernos"]
        print(f"cache: {len(lb)} equipos, {len(ks)} cuadernos")
    else:
        print("bajando leaderboard...", flush=True)
        lb = leaderboard(api)
        print(f"  {len(lb)} equipos ({time.time()-t0:.0f}s)", flush=True)
        print("bajando cuadernos...", flush=True)
        ks = cuadernos(api)
        print(f"  {len(ks)} cuadernos ({time.time()-t0:.0f}s)", flush=True)
        os.makedirs("data", exist_ok=True)
        json.dump({"leaderboard": lb, "cuadernos": ks}, open(CACHE, "w"))

    sc = [s for _, s in lb]
    print(f"\nleaderboard: {len(lb)} equipos, {max(sc):.0f} .. {min(sc):.0f}")

    # cruce por nombre, normalizado
    def nrm(x):
        return re.sub(r"[^a-z0-9]", "", str(x).lower())
    por_eq = {}
    for n, s in lb:
        por_eq.setdefault(nrm(n), s)
    casados = []
    for k in ks:
        s = por_eq.get(nrm(k["autor"]))
        if s is None:
            s = por_eq.get(nrm(k["ref"].split("/")[0]))
        if s is not None:
            casados.append({**k, "score": s})
    print(f"cuadernos con puntuacion conocida: {len(casados)} de {len(ks)}"
          f"  ({100*len(casados)/max(1,len(ks)):.0f} %)")
    if casados:
        cs = [c["score"] for c in casados]
        print(f"  su rango: {max(cs):.0f} .. {min(cs):.0f}")

    # un agente cada PASO puntos, descendente
    print(f"\nescalera pedida: de {ALTO} a {BAJO} de {PASO} en {PASO} "
          f"({(ALTO-BAJO)//PASO + 1} peldanos)")
    print(f"  {'tramo':<14}{'elegido':<52}{'score':>7}")
    elegidos, vacios = [], []
    for lo in range(ALTO, BAJO - 1, -PASO):
        cand = [c for c in casados if lo <= c["score"] < lo + PASO]
        if not cand:
            vacios.append(lo); continue
        best = max(cand, key=lambda c: c["score"])
        elegidos.append(best)
        print(f"  {lo}-{lo+PASO-1:<9}{best['ref'][:50]:<52}{best['score']:>7.0f}")
    print(f"\n  cubiertos {len(elegidos)} de {(ALTO-BAJO)//PASO + 1} tramos")
    if vacios:
        print(f"  SIN cuaderno publico: {vacios}")
    json.dump(elegidos, open("data/escalera_elegidos.json", "w"), indent=1)
    print(f"  -> data/escalera_elegidos.json  ({time.time()-t0:.0f}s)")
