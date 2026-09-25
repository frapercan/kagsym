"""Liga: pool compartido de instantaneas, elegidas por lo que ENSEÑAN.

Sustituye a `torneo.py`, que elegia por cercania de Elo y desalojaba al mas
debil. Dos motivos medidos para cambiarlo, los dos del 2026-09-23:

 1. **El Elo puede estar puntuando el partido equivocado y no se nota.** El
    nuestro se actualizaba con la tasa AGREGADA de los once trabajadores -o
    sea sobre todo como nos iba contra v48- mientras el duelo del torneo
    ocurria en UNO. Bajaba de 1000 a 988 mientras ganabamos ese peldaño el
    100%. Una `p` medida directamente contra esa instantanea es robusta a esa
    clase de error; un escalar resumen no.

 2. **El Elo supone transitividad**, y la transitividad es exactamente lo que
    falla en el regimen piedra-papel-tijera que una liga existe para destapar.
    Sigue siendo util como PRIOR -propaga informacion sin jugar todos los
    pares- pero no como criterio unico.

**QUE SIGNIFICA "INTERESANTE": `p(1-p)`**, la informacion de Bernoulli. Es el
MISMO criterio que este codigo ya usa para repartir trabajadores entre
peldaños, asi que no mete ninguna constante nueva. Un rival que aplastas no
enseña y uno que te aplasta tampoco -- medido en los dos extremos de la
escalera: win 0,98 transfiere -444 $ (t -0,58) y win 0,00 degrada de 29.732 a
25.739.

**Y SUS DOS FALLOS CONOCIDOS, protegidos explicitamente.** `p(1-p)` vale 0
cuando siempre pierdes, asi que tiraria justo a las instantaneas que nos ganan
-- el mismo agujero que dejo el curriculo ciego. Por eso:

  * el CAMPEON (Elo maximo) no se desaloja nunca: es el liston
  * las N mas RECIENTES tampoco: son la frontera
  * y no se desaloja lo que aun no se ha medido, porque no se puede juzgar

**TAMAÑO: lo limita la MEDIDA, no la memoria.** Una instantanea son 30 MB y
hay 46 GB libres; caben 50 de sobra. Pero estimar `p` con sd 0,10 pide ~25
partidas contra cada una, y a 11 emparejamientos por update eso son 114
updates para medir 50 una sola vez -- y se quedan rancias mientras entrenamos.
24 es el tamaño que se puede mantener medido.

**LA DIVERSIDAD VIENE DE LOS LINAJES, no del tamaño.** 50 instantaneas de un
run son 50 puntos de la MISMA trayectoria. Por eso cada entrada lleva su
`linaje`: el pool esta pensado para compartirse entre varios entrenamientos
con inicializacion independiente.
"""
from __future__ import annotations

import json
import os

import numpy as np

ELO_INICIAL = 1000.0
K = 24.0                      # cuanto mueve un resultado la puntuacion
INFO_MAX = 0.25               # max de p(1-p), en p=0,5


class Liga:
    def __init__(self, maximo: int = 24, recientes: int = 3,
                 min_partidas: int = 8, temperatura: float = 150.0):
        self.pool: list[dict] = []
        self.maximo = int(maximo)
        self.recientes = int(recientes)
        self.min_partidas = int(min_partidas)
        self.temperatura = float(temperatura)
        self.elo_actual = ELO_INICIAL

    def __len__(self):
        return len(self.pool)

    # ---------------------------------------------------------------- alta
    def añadir(self, sd: dict, upd: int, linaje: int = 0):
        """Entra la politica actual, heredando SU elo -- no el inicial.

        Con el inicial se le regalaria o quitaria historia y la eleccion
        dejaria de significar nada.
        """
        self.pool.append({
            "sd": {k: v.detach().cpu().clone() for k, v in sd.items()},
            "elo": float(self.elo_actual),
            "upd": int(upd),
            "linaje": int(linaje),
            "g": 0,          # partidas jugadas contra ella
            "w": 0.0,        # suma de nuestros resultados (1/0,5/0)
        })
        self._desaloja()
        return len(self.pool)

    # ------------------------------------------------------------- medidas
    def p(self, i: int):
        """Nuestra tasa de victoria contra la instantanea i, o None."""
        q = self.pool[i]
        return (q["w"] / q["g"]) if q["g"] > 0 else None

    def _peso(self, i: int) -> float:
        """PFSP: informacion medida, o el prior de Elo si no hay bastante.

        El prior se escala a `INFO_MAX` para que una instantanea sin medir
        compita de tu a tu con la mejor medida: asi se la juega y se la mide
        en vez de quedarse fosilizada en el pool.
        """
        q = self.pool[i]
        if q["g"] >= self.min_partidas:
            pr = q["w"] / q["g"]
            return max(1e-6, pr * (1.0 - pr))
        d = abs(q["elo"] - self.elo_actual)
        return max(1e-6, INFO_MAX * float(np.exp(-d / max(self.temperatura, 1e-6))))

    # ------------------------------------------------------------ eleccion
    def elegir(self, rng=None):
        if not self.pool:
            return None
        w = np.array([self._peso(i) for i in range(len(self.pool))], dtype=float)
        w = w / w.sum()
        r = rng or np.random
        return int(r.choice(len(self.pool), p=w))

    def resultado(self, i: int, gane: float):
        """`gane` en [0,1]: 1 ganamos, 0,5 empate, 0 perdimos."""
        if not (0 <= i < len(self.pool)):
            return
        q = self.pool[i]
        esperado = 1.0 / (1.0 + 10.0 ** ((q["elo"] - self.elo_actual) / 400.0))
        self.elo_actual += K * (gane - esperado)
        q["elo"] += K * ((1.0 - gane) - (1.0 - esperado))
        q["g"] += 1
        q["w"] += float(gane)

    # ----------------------------------------------------------- desalojo
    def _desaloja(self):
        if len(self.pool) <= self.maximo:
            return
        n = len(self.pool)
        prot = {max(range(n), key=lambda i: self.pool[i]["elo"])}   # campeon
        for i in sorted(range(n), key=lambda i: -self.pool[i]["upd"])[:self.recientes]:
            prot.add(i)
        cand = [i for i in range(n) if i not in prot] or list(range(n))
        medidos = [i for i in cand if self.pool[i]["g"] >= self.min_partidas]
        if medidos:
            fuera = min(medidos, key=self._peso)
        else:
            # nada medido entre las desalojables: se va la mas antigua, que es
            # la que mas ha derivado respecto a lo que somos ahora
            fuera = min(cand, key=lambda i: self.pool[i]["upd"])
        self.pool.pop(fuera)

    # ------------------------------------------------------------ informes
    def tabla(self):
        return sorted(
            [(q["upd"], q["linaje"], round(q["elo"], 1), q["g"],
              round(q["w"] / q["g"], 2) if q["g"] else None)
             for q in self.pool], key=lambda x: -x[2])

    def resumen(self):
        if not self.pool:
            return "liga vacia"
        e = [q["elo"] for q in self.pool]
        med = sum(1 for q in self.pool if q["g"] >= self.min_partidas)
        ps = [q["w"] / q["g"] for q in self.pool if q["g"] >= self.min_partidas]
        pm = f"  p medias {np.mean(ps):.2f}" if ps else ""
        return (f"{len(self.pool)} instantaneas ({med} medidas)  "
                f"elo actual {self.elo_actual:.0f}  pool [{min(e):.0f}..{max(e):.0f}]"
                + pm)


    # ------------------------------------------------------- persistencia
    #
    # La liga vive en DISCO porque los linajes se entrenan por turnos, en
    # procesos distintos: el pool y las puntuaciones tienen que sobrevivir
    # entre ellos. Los pesos van en ficheros aparte y el indice en json, para
    # poder leer el estado de la liga sin cargar 700 MB de tensores.
    def guardar(self, carpeta: str):
        import torch
        os.makedirs(carpeta, exist_ok=True)
        idx = {"elo_actual": self.elo_actual, "maximo": self.maximo,
               "recientes": self.recientes, "min_partidas": self.min_partidas,
               "temperatura": self.temperatura, "inst": []}
        vivos = set()
        for q in self.pool:
            nom = f"inst_l{q['linaje']}_u{q['upd']}.pt"
            vivos.add(nom)
            ruta = os.path.join(carpeta, nom)
            if not os.path.exists(ruta):
                torch.save(q["sd"], ruta)
            idx["inst"].append({k: q[k] for k in ("elo", "upd", "linaje", "g", "w")}
                               | {"fichero": nom})
        with open(os.path.join(carpeta, "indice.json"), "w") as f:
            json.dump(idx, f, indent=1)
        # los desalojados dejan de ocupar disco
        for f_ in os.listdir(carpeta):
            if f_.startswith("inst_") and f_.endswith(".pt") and f_ not in vivos:
                try:
                    os.remove(os.path.join(carpeta, f_))
                except OSError:
                    pass
        return len(self.pool)

    @classmethod
    def cargar(cls, carpeta: str, **kw):
        import torch
        ruta = os.path.join(carpeta, "indice.json")
        if not os.path.exists(ruta):
            return cls(**kw)
        with open(ruta) as f:
            idx = json.load(f)
        L = cls(maximo=kw.get("maximo", idx.get("maximo", 24)),
                recientes=kw.get("recientes", idx.get("recientes", 3)),
                min_partidas=kw.get("min_partidas", idx.get("min_partidas", 8)),
                temperatura=kw.get("temperatura", idx.get("temperatura", 150.0)))
        L.elo_actual = float(idx.get("elo_actual", ELO_INICIAL))
        for e in idx.get("inst", []):
            fp = os.path.join(carpeta, e["fichero"])
            if not os.path.exists(fp):
                continue
            L.pool.append({"sd": torch.load(fp, map_location="cpu",
                                            weights_only=False),
                           "elo": float(e["elo"]), "upd": int(e["upd"]),
                           "linaje": int(e["linaje"]), "g": int(e["g"]),
                           "w": float(e["w"])})
        return L

    def elo_por_linaje(self):
        """Elo medio de las instantaneas de cada linaje: la aptitud para PBT."""
        out = {}
        for q in self.pool:
            out.setdefault(q["linaje"], []).append(q["elo"])
        return {k: float(np.mean(v)) for k, v in out.items()}
