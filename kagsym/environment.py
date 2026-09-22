"""N episodes in parallel, advanced DAY by day.

One step = one day. This is the design decision that makes credit assignment
tractable: an episode is 30 steps instead of 720, and today's action falls in
the same window as its consequence. Measured: with a per-turn decision and a
net-worth reward, PPO converged to $2,794, below the $3,000 you get by not
playing.

It also instruments the USEFUL FRACTION, the share of unit-turns spent on
something other than moving or passing: the dominant term of the gap against
the expert (23.3% ours against 50.6% theirs).
"""
from __future__ import annotations

import numpy as np

from . import obs as O
from . import spec
from .symbolic.executor import Agent
from .fastenv import FastEnv
from .macro import Macro
from .reward import ProductionLedger
from .reward import RIVAL_WEIGHT as _RIVAL_W
from .reward import ILLEGAL_WEIGHT as _ILLEGAL_W

# spec.TURNS_PER_DAY is read at call time (see spec.set_turns_per_day): as a
# module alias it froze at import and stopped following `turnsPerDay`, silently
# desynchronising the executor from the engine.
MOV = {"NORTH", "SOUTH", "EAST", "WEST"}


# Opponent ladder. The real competition criterion is win/loss against
# opponents at your level (Elo plus a final Bradley-Terry tournament), not mean
# money against a passive agent.
#
# THE REAL GRADATION IS THE HAND CAP, not the choice of agent. Measured against
# a passive opponent, the strong public agents all land within 5% of each other
# -$179,514, $171,878, $171,392- so swapping one for another changes nothing.
#
# Attenuating them (passing at random) does NOT work: at 80% of its actions v48
# falls from $179,514 to $312. They are executors of COUPLED PLANS, not
# reactive policies. Capping the hands does preserve coherence, because their
# own logic sizes itself to the number of units. Measured against passive:
#   cap  3 ->  $16,746      cap  8 ->  $79,908
#   cap  5 ->  $42,867      cap 15 -> $179,514
# Below 3 it collapses (cap 2 -> $1): it cannot even start the farm.
LADDER_CAPS = [None, 3, 5, 8, None]

LADDER = [
    None,                                              # pasivo
    "v48-fast-routes",
    "v48-fast-routes",
    "v48-fast-routes",
    "the-2945-farm-96-vs-the-top-10-public-bots",
    # Measured on the difficulty ladder: at full power v16-rc5 makes LESS money
    # than v48 ($133,912 against $153,720) and yet it is the one that leaves US
    # the least ($36,139 against $40,416). The hardest opponent for our policy
    # is not the highest-scoring one, and training only against v48 left out
    # precisely the one we do worst against.
    "v16-rc5-high-score-8c-4s-premium-market-lead",
    # ---- ten more public agents, downloaded with `herramientas/baja_escalera.py`.
    #
    # WHY THEY WERE NEEDED. With six rungs we won eight of eleven at 88-100% and
    # lost the last one 100%: the jump was too large and there was no filler.
    # 598 public notebooks were downloaded and each one validated by playing a
    # full episode; 63 play, and of those only 40 are distinct -the rest are
    # exact republications of the same code, giving the same dollar-.
    #
    # ORDERED BY MEASURED STRENGTH against a passive opponent, not by score:
    # the correlation between leaderboard score and absolute money is r=+0.15.
    # They all farm almost identically; the 1,000 points of difference come
    # from head-to-head play. That is why the REAL gradation comes from the
    # hand cap -measured: 3->$16,825, 4->$24,003, 5->$42,099, 7->$61,361,
    # 9->$92,281, uncapped->$177,315- and these ten contribute STYLE DIVERSITY,
    # which the cap cannot give.
    "testkaggriculture-hamburger",                      #   80613 $, score 789
    "kaggriculture-adaptive-land-allocator",            #   90046 $, score 404
    "kaggriculture-weedproof-clone-market",             #  163933 $, score 1823
    "best-market-agent-high-strategy",                  #  166815 $, score 2498
    "kaggle-frontier-lab-strategy-improvement",         #  169238 $, score 2127
    "v29-r1-adaptive-market-hysteresis",                #  175069 $, score 1842
    "kaggriculture-conservative-market-router-v5",      #  180783 $, score 2042
    "kaggriculture-v6",                                 #  185827 $, score 2269
    "kaggriculture-v45-first-turn-wheat-round-trip",    #  186625 $, score 2498
    "your-market-list-is-an-order-book",                #  188274 $, score 2671
]

def public_with_cap(name_, max_hands: int):
    """A public agent limited in how many hands it may hire.

    ATTENUATING DOES NOT WORK: measured, passing at random on 20% of actions
    sinks v48 from $179,514 to $312. It is not a continuum, it is a cliff
    -these agents are executors of COUPLED PLANS, not reactive policies: they
    move a unit towards a tile over several turns and then act, so losing one
    turn breaks the whole chain-.

    Capping the hands does preserve the plan's coherence: their own logic sizes
    itself to the number of units they have.
    """
    base = load_public(name_)

    def jugar(obs):
        a = base(obs)
        ya = len(obs["farms"][int(obs["player"])]["hands"])
        room = max(0, max_hands - ya)
        orders = []
        for o in (a.get("market") or []):
            if o[0] == "HIRE":
                n_ = int(o[2]) if len(o) > 2 else 1
                n_ = min(n_, room)
                room -= max(0, n_)
                if n_ > 0:
                    orders.append([o[0], o[1], n_] if len(o) > 2 else o)
            else:
                orders.append(o)
        return dict(a, market=orders)
    return jugar


def public_attenuated(name_, p: float, seed: int = 0):
    """A public agent that only ACTS with probability `p`; otherwise passes.

    KEPT FOR REFERENCE, NOT IN USE. Attenuating looks preferable to inventing
    an intermediate bot -the behaviour is still that of a real, good agent,
    just acting less often- but it was measured and it does not work: at 80% of
    its actions v48 falls from $179,514 to $312. These agents execute coupled
    plans, so dropping one turn breaks the whole chain. The hand cap is what
    grades them without breaking them (see `public_with_cap`).
    """
    import random as _rnd
    base = load_public(name_)
    rng = _rnd.Random(seed)

    def jugar(obs):
        a = base(obs)
        if rng.random() < p:
            return a
        return {"farmer": ["PASS"],
                "hands": [["PASS"] for _ in (a.get("hands") or [])],
                "market": []}
    return jugar


_PUBLICOS = {}


def load_public(name_):
    """Load a public agent from `agents_pub/` as a function obs -> action.

    CACHED. Without the cache, `spec_from_file_location` + `exec_module`
    recompiled the module on EVERY call: measured, 553 ms, 64% of the cost of a
    36-turn episode (858 ms total, of which engine and executor are 250 and the
    network only 57). Since it is called once per episode, one CEM iteration
    with 40 candidates x 3 seeds did 120 recompilations.
    """
    ag = _PUBLICOS.get(name_)
    if ag is None:
        import importlib.util
        import os
        path_ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents_pub", name_ + ".py")
        spec_ = importlib.util.spec_from_file_location(
            "pub_" + name_.replace("-", "_"), path_)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        ag = mod.agent
        _PUBLICOS[name_] = ag
    return ag


def _parte_micro(m):
    """The micro map carries 1+N_OPS channels: channel 0 value, rest verbs.

    An agent without a network returns (10,10) or nothing, so the same
    pipeline serves both without branching elsewhere.
    """
    if m is None:
        return None
    m = np.asarray(m, dtype=np.float32)
    return (m[0], m[1:]) if m.ndim == 3 else m


class DayEnv:
    def __init__(self, n, steps=720, seed0=1, scale=None, macro=None,
                 idx0=0, n_total=None,
                 rival_fn=None, level=0, potential=True, gamma=0.995):
        from .reward import SCALE as _ESC
        self.n, self.steps, self.seed0 = n, steps, seed0
        self.scale = float(_ESC if scale is None else scale)
        self.idx0 = int(idx0)
        self.n_total = int(n_total) if n_total else int(n)
        self.macro_fijo = macro
        self.rival_fn = rival_fn
        # Ladder rung. Public agents keep state across turns, so a NEW one has
        # to be built per episode: reusing it drags in the previous episode's
        # farm.
        self.level = level
        self.results = []
        # Potential-based shaping (Ng, Harada & Russell 1999):
        #     r' = r + gamma*Phi(s') - Phi(s)
        # with Phi = my exact liquidation minus the OPPONENT's. Validated: the
        # terminal bias |Phi(s_T) - (my_cash - their_cash)| is 0.7%, against
        # the $611-917 net worth had. That is why it does not change which
        # policy is optimal: it only moves credit earlier.
        #
        # The opponent's negative term makes DESTROYING their value score the
        # same as creating mine, which is what a sign-based scoreboard implies.
        #
        # And being left with unsold produce at the close already penalises
        # itself: Phi counts only what HAS TIME to become cash, so the last
        # jump lands on the cash. A separate term would count it twice.
        # OPPONENT statistics. Without them "$19,000" means nothing: it could
        # be a tie or a thrashing. Measured against v48: us $36,566 with 26
        # productive tiles of 60, them $89,037 with 55 of the same 60.
        self.rival_finals = []
        self.riv_cult = []
        self.riv_anim = []
        self.rival_units = []
        self._riv_uds_max = [1] * n
        # Mask of the dimensions that actually decided, one day per env. PPO
        # uses it so the importance ratio ignores the ~1,570 Gaussian
        # dimensions that cannot change any action.
        self._masks = [None] * n
        self.potential = potential
        self.gamma = gamma
        self._phi = [0.0] * n
        self.unsold = []
        self.envs, self.agents, self.counter, self.obs = [None] * n, [None] * n, [None] * n, [None] * n
        self._rival = [None] * n
        self.ep = 0
        self.finals, self.useful_frac = [], []
        self._micro = [None] * n
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        # DISJOINT SEEDS BY CONSTRUCTION. Each worker used to start at
        # `seed0 + off*1000` and advance by `ep*n` inside, so worker 0 reached
        # worker 1's range at episode 1,000 and several processes played THE
        # SAME episode: the batch silently stopped being independent.
        # Partitioning by residue modulo the total number of envs, two
        # different envs can never coincide.
        s = self.seed0 + self.ep * self.n_total + self.idx0 + i
        self.envs[i] = FastEnv(configuration={"episodeSteps": self.steps}, seed=s)
        self.obs[i] = self.envs[i].reset()
        mac = self.macro_fijo if self.macro_fijo is not None else Macro.default()
        self.agents[i] = Agent(episode_steps=self.steps, macro=mac,
                                micro=lambda ob, k=i: _parte_micro(self._micro[k]))
        self.counter[i] = ProductionLedger()
        self._micro[i] = None
        self._rival[i] = self._new_rival()
        self._phi[i] = self._compute_phi(i) if self.potential else 0.0

    def _compute_phi(self, i):
        from .potential import phi as _phi_fn
        try:
            ob = self.obs[i][0]
            return _phi_fn(ob, 0, ob["private"]) / self.scale
        except Exception:
            return 0.0

    def set_rival_policy(self, factory):
        """Opponent = one of OUR policies (self-play).

        Why it is needed. Training against a full-power public agent we lose
        100% of episodes, so the terminal reward term is CONSTANT at -1: it
        carries not one bit. All the gradient came from the shaping, and the
        criterion that actually scores -winning- was invisible to learning.

        Against a copy of ourselves the win rate is ~50% by construction, which
        is where the win/loss signal has maximum variance and therefore maximum
        information. That is also why self-play needs a FIXED quota rather than
        competing for one: p(1-p) would hand it the whole budget.
        """
        self._rival_factory = factory
        for i in range(self.n):
            self._rival[i] = factory()

    def set_rival_macro(self, vector):
        """Opponent = OUR exact executor with a given macro vector.

        The public agents are 30-day specialists: measured, in 5-10 day games
        they make $42-2,629 while our heuristic makes ~$3,000 and wins 1.000
        against EVERY level. As short-league opponents they are useless
        -win=1.000 carries no more gradient than win=0.000-. A macro vector
        optimised by CEM FOR THAT HORIZON does know how to play those days.
        """
        self._rival_macro = None if vector is None else list(vector)
        self.results.clear()
        # REASSIGN NOW. `_reset` ran in the constructor and the opponents of
        # in-flight episodes are the old ones: changing only the attribute
        # changes nothing until the next reset, and the measurement comes out
        # identical to the previous one without warning.
        for i in range(self.n):
            self._rival[i] = self._new_rival()

    def _new_rival(self):
        if getattr(self, "_rival_macro", None) is not None:
            from .symbolic.executor import Agent as _Ag
            from .macro import Macro as _Mac
            # A NEW agent per episode: it keeps state across turns
            # (`_destinations`, `turns_per_tile`) and reusing it contaminates.
            # `episode_steps=None` on purpose: do not touch the global, which
            # our own agent already set to this league's step count.
            return _Ag(macro=_Mac.from_vector(self._rival_macro))
        if getattr(self, "_rival_factory", None) is not None:
            return self._rival_factory()
        if self.rival_fn is not None:
            return self.rival_fn
        name_ = LADDER[min(self.level, len(LADDER) - 1)]
        if name_ is None:
            from kaggle_environments.envs.kaggriculture import kaggriculture as E
            return E.pass_agent
        try:
            cap = LADDER_CAPS[min(self.level, len(LADDER_CAPS) - 1)]
            if cap is not None:
                return public_with_cap(name_, cap)
            return load_public(name_)
        except Exception:
            from kaggle_environments.envs.kaggriculture import kaggriculture as E
            return E.pass_agent

    def raise_level(self):
        self.level = min(self.level + 1, len(LADDER) - 1)
        self.results.clear()
        return LADDER[self.level]

    def win_rate(self, last=60):
        r = self.results[-last:]
        return float(np.mean(r)) if r else float("nan")

    def encode(self):
        """Rejilla, vector global y OFERTA INMINENTE DEL RIVAL.

        The third output feeds the network's `hist` input, which existed from
        the start -N_HIST = 4 x N_PRODUCTS- and the trainer filled with ZEROS.
        It is what causally precedes our income in a shared market, and it is
        predictable from the opponent's board, which is observable.
        """
        G = np.zeros((self.n, *O.SHAPES["grid"]), dtype=np.float32)
        B = np.zeros((self.n, O.N_GLOBAL), dtype=np.float32)
        Hf = np.zeros((self.n, O.N_HIST_RIVAL), dtype=np.float32)
        for i, o in enumerate(self.obs):
            G[i], B[i] = O.encode_obs(o[0])
            Hf[i] = O.rival_flow(o[0])
        return G, B, Hf

    def step_day(self, maps=None, macros=None):
        """Play 24 turns. `maps` (n,1+N_OPS,10,10) is the day's micro output."""
        from kaggle_environments.envs.kaggriculture import kaggriculture as E
        rec = np.zeros(self.n, dtype=np.float32)
        fin = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            if maps is not None:
                self._micro[i] = np.asarray(maps[i], dtype=np.float32)
            if macros is not None:
                self.agents[i].macro = Macro.from_vector(macros[i])
            env, counter = self.envs[i], self.counter[i]
            from .symbolic import tasks as _Tm
            _Tm.enable_mask()
            util = total = 0
            for _ in range(spec.TURNS_PER_DAY):
                if env.done:
                    break
                ob = self.obs[i][0]
                acc = self.agents[i](ob)
                _invs = (ob.get("private", {}).get("inventories") or []
                         if _ILLEGAL_W else [])
                for _j, l in enumerate([acc.get("farmer")]
                                       + list(acc.get("hands") or [])):
                    if l:
                        total += 1
                        util += 1 if (l[0] not in MOV and l[0] != "PASS") else 0
                        if _ILLEGAL_W and l[0] not in MOV and l[0] != "PASS":
                            _iv = (_invs[_j] if _j < len(_invs)
                                   and isinstance(_invs[_j], dict) else {})
                            if not _Tm._can_do(_iv, l):
                                rec[i] -= _ILLEGAL_W / self.scale
                counter.harvested(ob, acc, 0)
                income, bonus = counter.sold(ob, acc)
                rec[i] += income / self.scale + bonus
                try:
                    rival = self._rival[i](self.obs[i][1])
                except Exception:
                    rival = {"farmer": ["PASS"], "hands": [], "market": []}
                self.obs[i], d = env.step([acc, rival])
                # INSIDE the turn loop, on purpose. It used to be taken once
                # per day, outside, and there the hands have already been
                # cleared: it always gave 1.0 and made it look as if the
                # opponent played with no workforce. Measured by sampling all
                # 719 turns: v48 sustains 8.94 on average -and we 9.88- not 1.
                # The comment below claimed this was fixed and it was not.
                try:
                    self._riv_uds_max[i] = max(
                        self._riv_uds_max[i],
                        1 + len(self.obs[i][1]["farms"][1]["hands"]))
                except Exception:
                    pass
                if d or env.done:
                    break
            self._masks[i] = _Tm.collect_mask()
            if total:
                self.useful_frac.append(util / total)
            if self.potential and not env.done:
                nuevo = self._compute_phi(i)
                rec[i] += self.gamma * nuevo - self._phi[i]
                self._phi[i] = nuevo
            if env.done:
                r = env.rewards()
                if self.potential:
                    # Phi(s_T) = MY cash by construction (no opponent term)
                    fin_phi = float(r[0]) / self.scale
                    rec[i] += self.gamma * fin_phi - self._phi[i]
                self.unsold.append(
                    int(sum(self.obs[i][0]["private"]["shed"].values())))
                # COMPETITIVE TERM, in the OBJECTIVE and not in the shaping.
                # The one below (+-1) already looks at the opponent, but it is
                # a SIGN: winning by $1 scores the same as winning by $50,000,
                # so it does not say which way to push. This one is continuous
                # in the MARGIN.
                # It is not dense per day on purpose: that is the term
                # `potential.phi` documents as measured and withdrawn -the
                # opponent accounted for 99.1% of the daily variance and the
                # critic fell to R2 -2.535-. Shaping can only carry what the
                # state predicts; jumps in their cash are not that. At the
                # close it is one scalar per episode, the same shape as the
                # familiar +-1.
                if _RIVAL_W:
                    rec[i] -= _RIVAL_W * float(r[1]) / self.scale
                res = 1.0 if r[0] > r[1] else (0.5 if r[0] == r[1] else 0.0)
                rec[i] += 2.0 * res - 1.0
                self.results.append(res)
                self.finals.append(float(r[0]))
                self.rival_finals.append(float(r[1]))
                fr = self.obs[i][1]["farms"][1]
                # CAREFUL: at the close the hands have already been cleared,
                # so counting them here always gives 1 and makes it look as if
                # the opponent plays crippled. Verified by measuring
                # separately: it hires 87 times and sustains 3.78 units on
                # average. The same hour-boundary trap bites repeatedly; the
                # maximum seen during the episode is used instead.
                self.rival_units.append(max(1, self._riv_uds_max[i]))
                self._riv_uds_max[i] = 1
                self.riv_cult.append(sum(1 for fl in fr["tiles"] for t in fl
                                         if isinstance(t, dict) and t.get("kind") == "PLANT"))
                self.riv_anim.append(sum(1 for fl in fr["tiles"] for t in fl
                                         if isinstance(t, dict) and t.get("animal")))
                fin[i] = 1.0
                self.ep += 1
                self._reset(i)
        return rec, fin

    def masks(self):
        """(n, 1+N_OPS, B, B): which micro dimensions decided today.

        None when there is nothing to mask.
        """
        from .symbolic import tasks as _Tm
        z = np.zeros((1 + _Tm.N_OPS, spec.BOARD, spec.BOARD), dtype=np.float32)
        return np.stack([m if m is not None else z for m in self._masks])

    def mean_money(self, last=50):
        return float(np.mean(self.finals[-last:])) if self.finals else float("nan")

    def rival_stats(self, last=50):
        m = lambda v: float(np.mean(v[-last:])) if v else float("nan")
        return {"dinero": m(self.rival_finals), "cultivos": m(self.riv_cult),
                "animales": m(self.riv_anim), "unidades": m(self.rival_units)}

    def mean_unsold(self, last=50):
        """Units left in the shed at the close. They should be 0."""
        v = self.unsold[-last:]
        return float(np.mean(v)) if v else float("nan")

    def mean_useful(self, last=400):
        return float(np.mean(self.useful_frac[-last:])) if self.useful_frac else float("nan")
