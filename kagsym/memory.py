"""Non-parametric value memory: the long horizon by retrieval, not by propagation.

WHY. The value target in the trainer is `ret = adv + Vn`: it bootstraps from the
critic. The money made on day 25 reaches the decision of day 3 through a chain of
~22 learned estimates, each with its own bias. That chain IS the long-horizon
problem -- nothing about 30 days is hard, what is hard is that the signal has to
survive 22 approximations on the way back.

A memory of (state, return ACTUALLY OBSERVED) crosses the horizon in one hop: day
3 is worth what the days that looked like it turned out to be worth, measured to
the end of their episodes. No bootstrap, no accumulated bias. Capacity grows with
experience instead of being fixed by a parameter count, which is the whole point
of doing this non-parametrically.

NOTHING HERE IS HAND-SET. Every dial is read off the data on each update, because
the batch just collected carries the realised returns and is out-of-sample for a
memory that only holds PAST episodes:

    k, how many neighbours     the k that minimises the batch's error
    kernel bandwidth           the distance to the k-th neighbour, so the width
                               follows the local density and needs no constant
    staleness half-life        the half-life that minimises the batch's error;
                               it measures representation drift instead of
                               declaring a horizon for it
    alpha, how much to trust   least squares against the realised return, AS A
                               FUNCTION of the neighbour distance -- "trust the
                               memory when its neighbours are close" is fitted,
                               not declared

The two grids below are search RANGES, not constants: what acts is the argmin.
They stop being ranges and start being constants the moment the argmin lands on
an endpoint, so `info["edge"]` reports exactly that and the caller logs it.

If the memory is useless -- early on, off-distribution, or because the idea is
wrong -- alpha comes out at 0 by measurement and the trainer is the one of
always. That is the cheap falsification: one run settles it.

LEAKAGE. `DayEnv` gives every episode a fresh seed (`seed0 + ep*n_total + ...`),
so no two episodes ever share one. That matters because with 15,132 $ of standard
deviation per seed, a neighbour from the same episode would let the memory
"predict" that episode's luck rather than its state's worth. The batch is queried
BEFORE its own entries are added, which is what makes the fit honest.

INFERENCE IS UNTOUCHED. This improves the learning signal; the critic distils it
through its own target. The agent that plays on Kaggle carries no memory, so none
of this spends any of the one second per turn.
"""
import math

import torch
import torch.nn.functional as F

# Search ranges. See the note above: the argmin is what acts, and landing on an
# endpoint is reported as `edge` so it can never silently become a constant.
KS = (1, 2, 4, 8, 16, 32, 64)
HALF_LIVES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, float("inf"))


class EpisodicValue:
    """Kernel regression over observed returns, keyed by the critic's own latent.

    The key is `z`, the same vector the critic reads, on purpose: it makes the
    comparison between the two estimators a fair one -- same information, one
    parametric and one not- instead of a comparison between two encoders.
    """

    def __init__(self, dim: int, capacity: int, device):
        self.dim, self.cap, self.dev = int(dim), int(capacity), device
        self.K = torch.zeros(self.cap, self.dim, dtype=torch.float16, device=device)
        self.G = torch.zeros(self.cap, dtype=torch.float32, device=device)
        self.T = torch.zeros(self.cap, dtype=torch.float32, device=device)
        self.n, self.ptr, self.upd = 0, 0, 0.0
        self.fit = None          # (k, half_life, a, b, mu_x, sd_x) of the last fit

    # ---------------------------------------------------------------- storage
    def add(self, z, g):
        """z (B, dim) raw latents, g (B,) realised discounted return-to-go."""
        b = int(z.shape[0])
        if b == 0:
            return
        q = F.normalize(z.detach().float(), dim=-1).to(torch.float16)
        g = g.detach().float()
        if b > self.cap:                       # keep the most recent ones
            q, g, b = q[-self.cap:], g[-self.cap:], self.cap
        end = self.ptr + b
        if end <= self.cap:
            sl = slice(self.ptr, end)
            self.K[sl], self.G[sl], self.T[sl] = q, g, self.upd
        else:
            c = self.cap - self.ptr
            self.K[self.ptr:], self.G[self.ptr:], self.T[self.ptr:] = q[:c], g[:c], self.upd
            self.K[:b - c], self.G[:b - c], self.T[:b - c] = q[c:], g[c:], self.upd
        self.ptr = end % self.cap
        self.n = min(self.cap, self.n + b)

    # -------------------------------------------------------------- retrieval
    def _neighbours(self, z, kmax):
        """Cosine top-k against the whole memory, in blocks that fit in VRAM."""
        q = F.normalize(z.detach().float(), dim=-1).to(torch.float16)
        K = self.K[:self.n]
        step = max(1, min(int(z.shape[0]), (1 << 24) // max(1, self.n)))
        sims, idx = [], []
        for i in range(0, q.shape[0], step):
            v, j = torch.topk((q[i:i + step] @ K.T).float(), kmax, dim=1)
            sims.append(v); idx.append(j)
        return torch.cat(sims), torch.cat(idx)

    @staticmethod
    def _estimate(d, gnb, age, k, hl):
        """Kernel-weighted mean of the k nearest returns.

        The bandwidth is the distance to the k-th neighbour, so the kernel is as
        wide as the local density requires: in a dense region it is narrow, in a
        sparse one it opens up, and neither case needs a number chosen by hand.
        """
        dk, gk, ak = d[:, :k], gnb[:, :k], age[:, :k]
        h = dk[:, -1:].clamp(min=1e-6)
        w = torch.exp(-(dk / h) ** 2)
        if math.isfinite(hl):
            w = w * torch.exp(-math.log(2.0) * ak / hl)
        w = w + 1e-12
        return (w * gk).sum(1) / w.sum(1)

    # ------------------------------------------------------------------ blend
    def apply(self, z, v_theta, g, valid, upd, half=None):
        """Return (blended value, info). `valid` marks the steps whose episode
        closed inside the window, i.e. the ones whose return is actually known.

        `half` splits those steps in two BY EPISODE: k, the half-life and the
        two coefficients of alpha are chosen on one half and the verdict -the
        error the blend actually achieves- is read off the other. Without that
        split `gain` would be the minimum of 63 combinations measured on the
        same points that chose them, i.e. optimistic by construction, and
        `gain` is the whole go/no-go of this idea.
        """
        self.upd = float(upd)
        info = {"n": self.n, "k": 0, "hl": 0.0, "alpha": 0.0, "share": 0.0,
                "r_mem": float("nan"), "gain": 0.0, "edge": ""}
        if half is None:
            half = torch.zeros_like(valid)
            half[::2] = True
        fit_s, ev_s = valid & half, valid & (~half)
        kmax = min(max(KS), self.n)
        # 2 coefficients are fitted; asking for 32 points on each side is the
        # loosest bar that still makes the least squares mean anything.
        if self.n < 2 or kmax < 1 or int(fit_s.sum()) < 32 or int(ev_s.sum()) < 32:
            return v_theta, info

        sims, idx = self._neighbours(z, kmax)
        d = torch.sqrt(torch.clamp(2.0 - 2.0 * sims, min=0.0))
        gnb = self.G[:self.n][idx]
        age = (self.upd - self.T[:self.n][idx]).clamp(min=0.0)

        gf = g[fit_s]
        ge, te = g[ev_s], v_theta[ev_s]
        mse_t = float(((te - ge) ** 2).mean())           # the critic, held out
        best, bk, bhl = None, KS[0], HALF_LIVES[0]
        for k in KS:
            if k > kmax:
                break
            for hl in HALF_LIVES:
                e = float(((self._estimate(d[fit_s], gnb[fit_s], age[fit_s], k, hl)
                            - gf) ** 2).mean())
                if best is None or e < best:
                    best, bk, bhl = e, k, hl
        info["k"], info["hl"] = bk, (bhl if math.isfinite(bhl) else -1.0)
        edge = []
        if bk == KS[-1] and kmax >= KS[-1]:
            edge.append("k")
        if bhl == HALF_LIVES[0]:
            edge.append("hl-")
        self.fit = None

        v_mem = self._estimate(d, gnb, age, bk, bhl)
        # HOW MUCH TO TRUST IT, as a function of how close the neighbours are.
        # alpha = a + b*x with x the standardised log distance to the k-th
        # neighbour; both coefficients come out of the least squares, so the
        # rule "trust it when they are close" is measured rather than asserted.
        info["r_mem"] = float(((v_mem[ev_s] - ge) ** 2).mean()) / max(mse_t, 1e-9)
        x_raw = torch.log(d[:, bk - 1].clamp(min=1e-6))
        mu_x, sd_x = float(x_raw[fit_s].mean()), float(x_raw[fit_s].std()) + 1e-6
        x = (x_raw - mu_x) / sd_x
        u, r = (v_mem - v_theta)[fit_s], (g - v_theta)[fit_s]
        xu = x[fit_s] * u
        s11 = float((u * u).sum()); s12 = float((xu * u).sum())
        s22 = float((xu * xu).sum())
        b1 = float((u * r).sum()); b2 = float((xu * r).sum())
        det = s11 * s22 - s12 * s12
        if abs(det) > 1e-6 * max(1.0, s11 * s22):
            aa, bb = (b1 * s22 - b2 * s12) / det, (b2 * s11 - b1 * s12) / det
        elif s11 > 1e-9:
            aa, bb = b1 / s11, 0.0        # the distance term is degenerate
        else:
            return v_theta, info          # the memory says exactly what the critic says

        alpha = (aa + bb * x).clamp(0.0, 1.0)
        blend = v_theta + alpha * (v_mem - v_theta)
        mse_b = float(((blend[ev_s] - ge) ** 2).mean())
        # A MEASURED GUARD, not a constant, and read on the HELD-OUT half:
        # alpha=0 is inside the feasible set, so a fit that loses there is
        # telling us the fit does not transfer. Dropped for this update, logged.
        if mse_b > mse_t:
            info["gain"] = 0.0
            return v_theta, info
        info["alpha"] = float(alpha[ev_s].mean())
        info["share"] = float((alpha[ev_s] > 0.5).float().mean())
        info["gain"] = 1.0 - mse_b / max(mse_t, 1e-9)
        info["edge"] = ",".join(edge)
        self.fit = (bk, bhl, aa, bb, mu_x, sd_x)
        return blend, info

    def predict(self, z, v_theta):
        """Blend with the LAST fit, for states with no realised return of their
        own -- the bootstrap at the end of a truncated window."""
        if self.fit is None or self.n < 2:
            return v_theta
        bk, bhl, aa, bb, mu_x, sd_x = self.fit
        sims, idx = self._neighbours(z, min(bk, self.n))
        d = torch.sqrt(torch.clamp(2.0 - 2.0 * sims, min=0.0))
        gnb = self.G[:self.n][idx]
        age = (self.upd - self.T[:self.n][idx]).clamp(min=0.0)
        v_mem = self._estimate(d, gnb, age, min(bk, self.n), bhl)
        x = (torch.log(d[:, -1].clamp(min=1e-6)) - mu_x) / sd_x
        return v_theta + (aa + bb * x).clamp(0.0, 1.0) * (v_mem - v_theta)
