"""Potencial para shaping: valor de liquidacion EXACTO, mio menos del rival.

    r'_t  =  r_t  +  gamma * Phi(s_{t+1})  -  Phi(s_t)

Con Phi(s_T) igual al valor terminal verdadero, el shaping es *policy-invariant*
(Ng, Harada & Russell 1999): no cambia que politica es optima, solo adelanta el
credito. Esa condicion es la que fallo antes y hay que respetarla al dolar.

## Lo que fallo con patrimonio neto

PPO con recompensa de patrimonio convergio a 2 794 $, por debajo de los 3 000 $
que da no jugar. El diagnostico fino no es que el patrimonio este mal: es que
**Phi(s_T) != caja**. El cobertizo, los cultivos en pie y los animales valen
CERO al cerrar la partida, asi que un potencial que los cuenta miente justo en
el instante que puntua. Medido: +611 a +917 $ de sesgo medio, hasta 4 131 $.

Aqui solo se cuenta lo que DA TIEMPO a convertirse en caja antes del turno 720.
Al llegar a T no queda tiempo para nada, asi que Phi(s_T) = caja exactamente y
el sesgo desaparece por construccion.

## Por que la DIFERENCIA y no mi nivel absoluto

El marcador es `sign(dinero_yo - dinero_rival)`: suma cero en el signo. Asi que
destruir valor del rival vale exactamente lo mismo que crear el mio. Sin el
termino negativo, la recompensa no ve la jugada mas rentable del juego: volcar
producto en un mercado fino hunde su precio un 98 % (LECHE 169 -> 5), y con ello
toda la produccion ganadera del rival.

## Asimetria de informacion, asumida

De mi lado se ve todo. Del rival solo lo publico: su dinero y su tablero. Su
cobertizo, sus semillas y lo que cargan sus unidades estan ocultos -se despejan
exacto un turno despues, pero el potencial se evalua AHORA-. Se usa lo
observable, que es una funcion bien definida del estado y por tanto un potencial
valido; simplemente ignora una parte de su valor.
"""
from __future__ import annotations

from . import spec
from .exacto.mercado import marginal_prices

# spec.TURNS_PER_DAY se lee en tiempo de llamada (ver spec.set_turns_per_day):
# como alias de modulo se congelaba al importar y no seguia a
# `turnsPerDay`, desincronizando el ejecutor del motor sin avisar.


def _valor_lote(obs, producto: str, n: int) -> float:
    """Lo que rendirian n unidades vendidas ahora, a precio MARGINAL del motor.

    Nominal sobrestima: cada unidad vendida baja el precio de la siguiente.
    """
    n = int(n)
    if n <= 0:
        return 0.0
    return float(sum(marginal_prices(obs, producto, min(n, 200))))


def liquidacion(obs, pid: int, privado=None) -> float:
    """Caja + todo lo que da tiempo a convertirse en caja antes del cierre."""
    farm = obs["farms"][pid]
    total = float(farm["money"])
    dias = max(0, (spec.EPISODE_STEPS - 1 - int(obs["step"])) // spec.TURNS_PER_DAY)

    if privado is not None:
        for item, n in (privado.get("shed") or {}).items():
            if item in spec.PRODUCTS and n:
                total += _valor_lote(obs, item, n)
        for inv in (privado.get("inventories") or []):
            for item, n in inv.items():
                if item in spec.PRODUCTS and n:
                    total += _valor_lote(obs, item, n)

    dia = int(obs["day"])
    for fila in farm["tiles"]:
        for t in fila:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                cd = spec.CROPS[t["crop"]]
                edad = dia - int(t["planted_day"])
                # solo cuenta si llega a dar y da tiempo a venderlo
                if edad + dias < cd["first_yield_day"]:
                    continue
                uds = int(t.get("yield_units", 0))
                if cd["ongoing"]:
                    restantes = max(0, min(cd["max_yield"] - uds,
                                           dias // max(1, cd["interval"])))
                    uds += restantes
                total += _valor_lote(obs, t["crop"], uds)
            elif t.get("animal"):
                a = spec.ANIMALS[t["animal"]]
                uds = int(t.get("yield_units", 0))
                por_venir = max(0, dias - a["first_yield_day"]) // max(1, a["interval"])
                uds = min(a["max_held"] + por_venir, uds + por_venir)
                total += _valor_lote(obs, a["product"], uds)
    return total


def phi(obs, me: int = 0, privado=None, con_rival: bool = False) -> float:
    """Potencial relativo: lo mio menos lo del rival.

    Del rival solo se cuenta lo publico (dinero y tablero); su cobertizo no es
    observable. Eso infravalora su posicion de forma sistematica, pero de forma
    CONSISTENTE, que es lo que importa para que la diferencia telescopica siga
    siendo valida.
    """
    mio = liquidacion(obs, me, privado)
    if not con_rival:
        # POR DEFECTO, SIN EL RIVAL. Medido: su termino aporta el 99.1 % de la
        # varianza de la recompensa diaria (sd 1.93 de 1.94 total), porque su
        # liquidacion da saltos cuando cosecha y vende y nuestro estado no puede
        # anticiparlo. Con el, el critico da R2 = -2.535 (peor que predecir la
        # media); sin el, R2 = +0.736. Sin critico no hay ventaja, y sin ventaja
        # PPO es ruido centrado en cero.
        #
        # La idea de premiar la destruccion de su valor NO se pierde: vive en el
        # terminal (+-1 de victoria/derrota), que es de signo y por tanto ya
        # cuenta su dinero tanto como el nuestro. El rival debe estar en el
        # OBJETIVO, no en el shaping: el shaping solo puede llevar lo que el
        # estado predice.
        return mio
    return mio - liquidacion(obs, 1 - me, None)
