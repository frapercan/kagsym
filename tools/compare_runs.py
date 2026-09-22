"""Compare two training runs on HEALTH and CAPACITY, aligned by elapsed updates.

Comparing a single checkpoint against a baseline answers almost nothing: the
per-seed sd is around $13,300 and any one point is inside the noise. What a run
does have is a TRAJECTORY, and two trajectories can be compared even when the
absolute numbers cannot, because each is read against its own
policy-independent floor.

Two blocks, kept apart on purpose:

  HEALTH    is the machinery learning at all -does the step size hold the KL
            target, do the heads travel, is the critic predicting-. These are
            properties of the optimizer, and they are comparable between runs
            regardless of opponent or scale.

  CAPACITY  is it getting better at the game. Money is quoted as a multiple of
            doing nothing, which is $3,000 at every horizon, so a 24-turn rung
            and a 720-turn one can sit in the same column. Margin against the
            opponent is NOT reported: in a shared market our own behaviour
            moves their score, so it is not comparable across runs.

Runs are aligned by updates ELAPSED since each one started, not by absolute
update number, so a run resumed at update 435 can be compared with one that
starts at 1.

Usage:  python tools/compare_runs.py RUN_A RUN_B [--db data/mlflow.db]
"""
import argparse, os, sqlite3, statistics as st, sys

# The metric names changed when the code moved to English. A run recorded
# before that rename has to stay readable, so every metric is looked up under
# both spellings.
KEYS = {
    "kl/dim":        ("2_health/kl_per_dim", "2_salud/kl_por_dim"),
    "lr_trunk":      ("4_diag/lr_trunk", "4_diag/lr_tronco"),
    "lr_micro":      ("4_diag/lr_micro",),
    "lr_macro":      ("4_diag/lr_macro",),
    "micro_w_norm":  ("4_diag/micro_w_norm", "4_diag/micro_w_norma"),
    "macro_w_norm":  ("2_health/macro_w_norm", "2_salud/macro_w_norma"),
    "critic_r2":     ("2_health/critic_r2", "2_salud/critico_r2"),
    "sigma_micro":   ("2_health/sigma_micro_value", "2_salud/sigma_micro_valor"),
    "saturation":    ("2_health/saturation", "2_salud/saturacion"),
    "grad_norm":     ("2_health/grad_norm", "2_salud/grad_norma"),
    "epochs_run":    ("2_health/epochs_run", "2_salud/epocas_corridas"),
    "money":         ("1_result/money", "1_resultado/dinero"),
    "x_inaction":    ("1_result/x_inaction", "1_resultado/x_inaccion"),
    "win_rate":      ("1_result/win_rate", "1_resultado/win_rate"),
}
LR_FLOOR = 1.5e-6          # the optimizer's floor is 1e-6; this is "at it"


def _con(db):
    if not os.path.exists(db):
        sys.exit(f"no database at {db}")
    return sqlite3.connect(db)


def series(cur, uuid, name):
    for k in KEYS[name]:
        cur.execute("SELECT step,value FROM metrics WHERE run_uuid=? AND key=?"
                    " ORDER BY step", (uuid, k))
        d = cur.fetchall()
        if d:
            return d
    return []


def resolve(cur, name):
    cur.execute("SELECT run_uuid,start_time FROM runs WHERE name=?"
                " ORDER BY start_time DESC LIMIT 1", (name,))
    r = cur.fetchone()
    if not r:
        sys.exit(f"no run named {name!r}")
    return r[0]


def summarize(cur, uuid, window=None):
    """`window` limits the comparison to the first N updates of each run, so a
    long baseline does not get an advantage a short challenger cannot have."""
    out = {}
    base = None
    for name in KEYS:
        d = series(cur, uuid, name)
        if not d:
            continue
        if base is None:
            base = d[0][0]
        if window is not None:
            d = [(s, v) for s, v in d if s - base < window]
        if not d:
            continue
        vals = [v for _, v in d if v == v]
        if not vals:
            continue
        out[name] = {"n": len(vals), "median": st.median(vals),
                     "first": d[0][1], "last": d[-1][1],
                     "span": d[-1][0] - d[0][0]}
    out["_elapsed"] = (series(cur, uuid, "kl/dim") or [(0, 0)])[-1][0] - (base or 0)
    return out


def line(label, a, b, fmt="{:.4g}", key="median"):
    fa = fmt.format(a[key]) if a else "-"
    fb = fmt.format(b[key]) if b else "-"
    print(f"  {label:<16}{fa:>14}{fb:>14}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b")
    ap.add_argument("--db", default="data/mlflow.db")
    ap.add_argument("--window", type=int, default=None,
                    help="compare only the first N updates of each run")
    a = ap.parse_args()
    cur = _con(a.db).cursor()
    ua, ub = resolve(cur, a.run_a), resolve(cur, a.run_b)
    A, B = summarize(cur, ua, a.window), summarize(cur, ub, a.window)

    win = f"first {a.window} updates" if a.window else "whole run"
    print(f"\n{a.run_a}  vs  {a.run_b}   ({win})")
    print(f"  {'':<16}{a.run_a[:14]:>14}{a.run_b[:14]:>14}")
    print(f"  {'updates':<16}{A['_elapsed']:>14}{B['_elapsed']:>14}")

    print("\n  --- HEALTH: is the machinery learning ---")
    line("kl/dim", A.get("kl/dim"), B.get("kl/dim"), "{:.4f}")
    line("lr_trunk", A.get("lr_trunk"), B.get("lr_trunk"), "{:.2e}")
    line("lr_micro", A.get("lr_micro"), B.get("lr_micro"), "{:.2e}")
    # the floor fraction needs the raw series, not the summary
    fa = series(cur, ua, "lr_micro"); fb = series(cur, ub, "lr_micro")
    if a.window:
        fa = [(s, v) for s, v in fa if s - (fa[0][0] if fa else 0) < a.window]
        fb = [(s, v) for s, v in fb if s - (fb[0][0] if fb else 0) < a.window]
    pa = 100*sum(1 for _, v in fa if v <= LR_FLOOR)/len(fa) if fa else float("nan")
    pb = 100*sum(1 for _, v in fb if v <= LR_FLOOR)/len(fb) if fb else float("nan")
    print(f"  {'% lr at floor':<16}{pa:>13.0f}%{pb:>13.0f}%")
    for nm in ("micro_w_norm", "macro_w_norm"):
        # Travel per 100 updates, not total: a run that logged 2,600 updates
        # would otherwise look like it moved more than one that logged 200.
        def per100(s):
            if not s or not s["first"] or not s["span"]:
                return "-"
            return f"{100*100*(s['last']-s['first'])/s['first']/s['span']:+.2f}%"
        print(f"  {nm+'/100u':<16}{per100(A.get(nm)):>14}{per100(B.get(nm)):>14}")
    line("critic_r2", A.get("critic_r2"), B.get("critic_r2"), "{:.3f}")
    line("sigma_micro", A.get("sigma_micro"), B.get("sigma_micro"), "{:.4f}")
    line("saturation", A.get("saturation"), B.get("saturation"), "{:.3f}")
    line("grad_norm", A.get("grad_norm"), B.get("grad_norm"), "{:.0f}")
    line("epochs_run", A.get("epochs_run"), B.get("epochs_run"), "{:.2f}")

    print("\n  --- CAPACITY: is it getting better at the game ---")
    line("x_inaction med", A.get("x_inaction"), B.get("x_inaction"), "{:.2f}")
    line("x_inaction last", A.get("x_inaction"), B.get("x_inaction"), "{:.2f}", key="last")
    line("money med", A.get("money"), B.get("money"), "{:.0f}")
    line("money last", A.get("money"), B.get("money"), "{:.0f}", key="last")
    line("win_rate med", A.get("win_rate"), B.get("win_rate"), "{:.3f}")
    print("\n  margin against the opponent is deliberately not reported: in a "
          "shared\n  market our own behaviour moves their score, so it does not "
          "compare.\n")


if __name__ == "__main__":
    main()
