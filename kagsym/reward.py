"""Dense reward: selling OWN produce, not accumulating net worth.

Why not net worth, which is what it used to be. Measured: PPO with a net-worth
reward converged to $2,794, below the $3,000 you get by doing nothing. The
reason is structural, not a tuning failure: standing still preserves the
starting capital, so passivity is a local optimum with positive reward. Any
action costs before it pays, and the gradient punishes it.

With "income from selling own produce", passivity is worth exactly 0 and stops
being a refuge.

Why *own* and not any sale: the engine quotes `BUY_PRODUCT` at `inventory - 1`
precisely so that a buy-and-sell round trip within the same turn yields 0
(verified: $3,000 -> $3,000). Rewarding sales outright would invite the agent
to trade with the market against itself, which produces nothing. Attribution is
by provenance: only units that came out of a harvest count.

The bonus for each first product sold attacks the other known local optimum,
monoculture -the "melon farm" several competitors describe-: without it the
policy finds a single profitable product and never tries the others.
"""
from __future__ import annotations

from . import spec
from .symbolic.market_ops import marginal_prices

# THE TWO SHAPING NUMBERS. The rest of the design is reasoned and measured (see
# the module docstring); these two are set by hand.
#
# They are read from the ENVIRONMENT at import because the parallel env spawns
# child processes, and threading them through three layers of signatures would
# be more invasive than this. An outer loop sets them via os.environ before
# launching each training run.
#
# The policy does NOT learn them: that would let the agent choose its own exam.
# They are moved by an OUTER level evaluated against the TRUE objective -money
# and wins- on seeds the inner loop has never seen.
import os

FIRST_PRODUCT_BONUS = float(os.environ.get("KAG_BONUS", "0.2"))
# WEIGHT OF THE DENSE TERM AGAINST THE SHAPING. Measured over a real episode,
# the return decomposes as 54.6% dense sales, 41.9% potential shaping, 1.9%
# first-product bonus and 1.6% the win itself. Selling is the act that closes
# the SHORTEST cycle in the game, and it is the majority of what the policy is
# paid for; a five-day play -buy an animal, build, place, feed daily, harvest-
# is paid only through the potential.
#
# `SCALE` cannot express this: it divides the dense term and the potential
# alike, so it moves the size of the reward and not its balance. This is the
# knob that moves the balance, and at 0 the objective becomes "maximise what
# has time to turn into cash", with selling paid only through the value it
# realises.
#
# It belongs to the exam, so it is set from OUTSIDE and never learned: a
# policy that could weigh its own reward would choose the easy one.
DENSE_WEIGHT = float(os.environ.get("KAG_DENSO", "1.0"))
# WEIGHT OF THE WIN ITSELF. Measured over a real episode, the +-1 for winning
# is 1.6% of the return: the objective is already almost entirely "make money",
# and the competitive part is a rounding error that nonetheless makes the
# objective RELATIVE -- which matters, because what has been measured four
# times is a policy improving against its training opponents while getting
# worse against a fixed one.
#
# At 0 the objective stops referring to the opponent at all and becomes "the
# money I end with". Like the other two, it belongs to the exam and is set
# from outside.
WIN_WEIGHT = float(os.environ.get("KAG_PESO_WIN", "1.0"))
SCALE = float(os.environ.get("KAG_ESCALA", "2000.0"))

# COMPETITIVE TERM, continuous in the MARGIN.
#
# The reward already HAD an opponent term -the terminal +-1 for win/loss-. What
# it lacked is GRADATION: a sign scores a $1 win the same as a $50,000 win and
# says nothing about which direction to push.
#
# What motivates it, measured: training against an exact copy of ourselves, our
# money rises 9% and theirs 17% -while being a FIXED vector that does not
# learn-. All of their improvement comes from us changing: as we stop competing
# for products, their marginal prices stay high. Win rate falls from 0.59 to
# 0.22 while we improve in absolute terms.
#
# WHERE IT GOES, and it is not indifferent: in the terminal objective, NOT
# dense per day. The dense version was tried and withdrawn with numbers -see
# potential.phi-: the opponent's liquidation accounted for 99.1% of the daily
# reward variance and the critic fell to R2 = -2.535, worse than predicting the
# mean. Shaping can only carry what the state predicts, and jumps in their cash
# are not that.
#
# RIVAL_WEIGHT = 0 recovers the previous design exactly. At 1 the objective is
# the pure margin; in between it interpolates between own money and scoreboard.
#
# A risk to WATCH, not to assume: rewarding the other side's losses can
# degenerate into destroying value -crashing prices hurts whoever sells less-.
# It may be correct play or self-harm. It gets measured.
RIVAL_WEIGHT = float(os.environ.get("KAG_PESO_RIVAL", "0.0"))


class ProductionLedger:
    """Tracks the provenance of every unit. One instance per episode."""

    def __init__(self):
        self.available = {p: 0 for p in spec.PRODUCTS}
        self.sold_kinds = set()
        # POR PRODUCTO, contado y no estimado. El 84% del hueco contra v48
        # son DOS lineas de negocio -fresa 0 unidades contra 430, leche 72
        # contra 335- y ninguna metrica del panel lo mostraba: se juzgaban los
        # cambios contra un agregado que promedia sobre todo el problema.
        self.vendidas = {p: 0 for p in spec.PRODUCTS}
        self.ingreso = {p: 0.0 for p in spec.PRODUCTS}
        self.cosechadas = {p: 0 for p in spec.PRODUCTS}

    def harvested(self, obs, action, me: int) -> int:
        """Units harvested THIS turn, read from the pre-step state.

        Nothing is estimated: for each HARVEST it reads the actual
        `yield_units` of the tile the unit stands on, which is exactly what the
        engine is about to move into the inventory.
        """
        farm = obs["farms"][me]
        pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
        ops = [action.get("farmer")] + list(action.get("hands") or [])
        total = 0
        for (x, y), op in zip(pos, ops):
            if not op:
                continue
            # FERTILIZANTE: es produccion PROPIA -sale de tus animales, igual
            # que la leche- pero NO entra por HARVEST sino por su propia
            # operacion, asi que este ledger nunca lo acreditaba.
            #
            # Consecuencia, y no era solo de metrica: `sold` recorta las
            # ordenes a `available[p]`, que para FERTILIZER valia 0 siempre.
            # Vender fertilizante daba CERO ingreso contado y por tanto CERO
            # recompensa densa. Medido: ~9.247 $ por partida = 4,6 unidades de
            # retorno nunca pagadas sobre un retorno tipico de 50, y
            # concentradas en la cadena animal->recoger->vender, que es
            # precisamente la que no aprendemos.
            #
            # Recorrido exhaustivo de `_inv_add` en el motor: compra (excluida
            # a proposito, no es produccion), cosecha de cultivo, cosecha de
            # animal y esta. Las tres primeras ya estaban.
            if op[0] == "COLLECT_FERTILIZER":
                t = farm["tiles"][y][x]
                if (isinstance(t, dict) and t.get("animal")
                        and t.get("fertilizer_available")):
                    self.available["FERTILIZER"] = self.available.get("FERTILIZER", 0) + 1
                    self.cosechadas["FERTILIZER"] = self.cosechadas.get("FERTILIZER", 0) + 1
                    total += 1
                continue
            if op[0] != "HARVEST":
                continue
            t = farm["tiles"][y][x]
            if not isinstance(t, dict):
                continue
            n = int(t.get("yield_units", 0))
            if n <= 0:
                continue
            if t.get("kind") == "PLANT":
                cd = spec.CROPS[t["crop"]]
                if int(obs["day"]) - int(t["planted_day"]) < cd["first_yield_day"]:
                    continue
                prod = t["crop"]
            elif t.get("animal"):
                prod = spec.ANIMALS[t["animal"]]["product"]
            else:
                continue
            self.available[prod] = self.available.get(prod, 0) + n
            self.cosechadas[prod] = self.cosechadas.get(prod, 0) + n
            total += n
        return total

    def sold(self, obs, action) -> tuple[float, float]:
        """(income from own produce, exploration bonus) for this turn.

        Income is valued at the engine's MARGINAL price over the previous
        inventory: selling n units does not pay n times the headline price,
        because each one lowers the next.
        """
        income = 0.0
        bonus = 0.0
        for orden in (action.get("market") or []):
            if not isinstance(orden, (list, tuple)) or len(orden) < 3:
                continue
            if orden[0] != "SELL":
                continue
            p = orden[1]
            if p not in self.available:
                continue
            n = min(int(orden[2]), int(self.available[p]))
            if n <= 0:
                continue
            _ing = float(sum(marginal_prices(obs, p, n)))
            income += _ing
            self.available[p] -= n
            self.vendidas[p] = self.vendidas.get(p, 0) + n
            self.ingreso[p] = self.ingreso.get(p, 0.0) + _ing
            if p not in self.sold_kinds:
                self.sold_kinds.add(p)
                bonus += FIRST_PRODUCT_BONUS
        return income, bonus


def reward(obs, action, ledger: ProductionLedger, me: int, scale: float):
    """Dense, per turn. Passivity yields exactly 0.0."""
    ledger.harvested(obs, action, me)
    income, bonus = ledger.sold(obs, action)
    return income / scale + bonus


# WASTED TURN: actions the engine silently IGNORES.
#
# The engine does not punish the impossible, it turns it into a no-op. So
# "trying something I cannot do" and "passing on purpose" are
# INDISTINGUISHABLE in the return, and with 719 turns and eleven units a wasted
# turn does not show up in the final cash. It is not that it costs little: it
# is that it cannot be attributed.
#
# Why penalise and not mask. Masking is better in general -removing probability
# mass beats punishing it- and indeed `tile_options` only proposes legal plays.
# But the INVENTORY filter was measured and came out badly for a concrete
# reason: the tile asking for FEED DISAPPEARS when nobody carries wheat, and
# then nobody goes to fetch wheat. With the filter, 3.8 tasks/turn and $4,484;
# without it, 10.7 tasks/turn and $6,727 but 66.9% unexecutable tasks and 86%
# PASS. Penalising keeps the task VISIBLE -preparing stays learnable- and
# charges for the lost turn.
#
# It lives in the OUTER loop, never in the macro vector: it is a term of the
# objective. If the policy could move its own penalty, the first thing it would
# learn is to set it to zero.
#
# WHEN IT BITES, and this must be known before tuning it. Measured at
# championship scale: ZERO wasted actions in a whole episode, because
# `assign_units` receives the inventories and discards the unit-task pair when
# `_can_do` fails, so the illegal never reaches the engine. Nor are there
# no-ops from a stale destination -watering what is already watered, harvesting
# with no fruit, planting on an occupied tile-: 0 of 1,563 actions. In those
# modes this weight CANNOT have any effect however high it is set.
#
# In dollars per wasted action; 0 recovers the previous design exactly.
#
# COVERAGE, and it should not be overstated: the INVENTORY class is detected
# -FEED without wheat, FERTILIZE without fertiliser, PLACE without the animal,
# DROP with nothing- which is the dominant one in what was measured. Other
# actions also fall into no-ops -watering the watered, harvesting with no
# fruit- and those are NOT counted yet.
ILLEGAL_WEIGHT = float(os.environ.get("KAG_PESO_ILEGAL", "0.0"))
