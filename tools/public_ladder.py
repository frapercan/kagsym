"""Ladder of PUBLIC opponents by score, every 50 points.

Why it is needed: the previous ladder had 11 rungs and we won 8 of them at
88-100%. By the curriculum's own criterion those opponents teach nothing, and
the only one that mattered -an uncapped v48- we lost 100% of the time, i.e. the
jump was too large. REAL filler is needed in between.

How it is built:
  1. full leaderboard (team -> score). It is paged with `page_token`;
     `page_size` stays at 200 but the token keeps going, so the end is reached.
  2. public notebooks of the competition (author -> downloadable ref).
  3. cross by author/team, and ONE is chosen every `STEP` points.

What CANNOT be done, and is worth writing down: someone else's submission code
is not downloadable. Only a public NOTEBOOK's is. So the ladder can only cover
the bands where somebody published their notebook.
"""
import io, json, os, re, sys, time, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMP = "kaggriculture"
HIGH, LOW, STEP = 2700, 1500, 50
CACHE = "data/public_ladder_cache.json"


def leaderboard(api):
    tok, rows = None, []
    for _ in range(200):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = api.competition_leaderboard_view(COMP, page_size=200, page_token=tok)
        m = re.search(r"Next Page Token = (\S+)", buf.getvalue())
        new = m.group(1) if m else None
        n0 = len(rows)
        for e in (r or []):
            try:
                rows.append((str(getattr(e, "team_name", "?")), float(e.score)))
            except Exception:
                pass
        if not new or new == tok or len(rows) == n0:
            break
        tok = new
    return rows


def notebooks(api):
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
            out.append({"ref": k.ref, "author": k.author, "title": k.title})
        if len(ks) < 100:
            break
        pg += 1
    return out


if __name__ == "__main__":
    import kaggle
    api = kaggle.KaggleApi(); api.authenticate()
    t0 = time.time()
    if os.path.exists(CACHE) and "--refresh" not in sys.argv:
        d = json.load(open(CACHE))
        lb, ks = d["leaderboard"], d["notebooks"]
        print(f"cache: {len(lb)} teams, {len(ks)} notebooks")
    else:
        print("downloading the leaderboard...", flush=True)
        lb = leaderboard(api)
        print(f"  {len(lb)} teams ({time.time()-t0:.0f}s)", flush=True)
        print("downloading the notebooks...", flush=True)
        ks = notebooks(api)
        print(f"  {len(ks)} notebooks ({time.time()-t0:.0f}s)", flush=True)
        os.makedirs("data", exist_ok=True)
        json.dump({"leaderboard": lb, "notebooks": ks}, open(CACHE, "w"))

    sc = [s for _, s in lb]
    print(f"\nleaderboard: {len(lb)} teams, {max(sc):.0f} .. {min(sc):.0f}")

    # cross by name, normalised
    def nrm(x):
        return re.sub(r"[^a-z0-9]", "", str(x).lower())
    by_team = {}
    for n, s in lb:
        by_team.setdefault(nrm(n), s)
    matched = []
    for k in ks:
        s = by_team.get(nrm(k["author"]))
        if s is None:
            s = by_team.get(nrm(k["ref"].split("/")[0]))
        if s is not None:
            matched.append({**k, "score": s})
    print(f"notebooks with a known score: {len(matched)} of {len(ks)}"
          f"  ({100*len(matched)/max(1,len(ks)):.0f}%)")
    if matched:
        cs = [c["score"] for c in matched]
        print(f"  their range: {max(cs):.0f} .. {min(cs):.0f}")

    # one agent every STEP points, descending
    print(f"\nladder requested: from {HIGH} to {LOW} every {STEP} "
          f"({(HIGH-LOW)//STEP + 1} rungs)")
    print(f"  {'band':<14}{'chosen':<52}{'score':>7}")
    chosen, empty = [], []
    for lo in range(HIGH, LOW - 1, -STEP):
        cand = [c for c in matched if lo <= c["score"] < lo + STEP]
        if not cand:
            empty.append(lo); continue
        best = max(cand, key=lambda c: c["score"])
        chosen.append(best)
        print(f"  {lo}-{lo+STEP-1:<9}{best['ref'][:50]:<52}{best['score']:>7.0f}")
    print(f"\n  covered {len(chosen)} of {(HIGH-LOW)//STEP + 1} bands")
    if empty:
        print(f"  NO public notebook: {empty}")
    json.dump(chosen, open("data/chosen_ladder.json", "w"), indent=1)
    print(f"  -> data/chosen_ladder.json  ({time.time()-t0:.0f}s)")
