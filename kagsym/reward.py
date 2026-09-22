"""Recompensa densa: vender producto PROPIO, no acumular patrimonio.

Por que no patrimonio neto, que es lo que habia. Medido el 2026-09-19: PPO con
recompensa de patrimonio convergia a 2794 $, por debajo de los 3000 $ que da no
hacer nada. La razon es estructural, no un fallo de ajuste: quieto se conserva
el capital inicial, asi que la pasividad es un optimo local con recompensa
positiva. Cualquier accion cuesta antes de rendir, y el gradiente la castiga.

Con "ingresos por venta de produccion propia" la pasividad vale exactamente 0 y
deja de ser un refugio.

Por que *propia* y no cualquier venta: el motor cotiza `BUY_PRODUCT` a
`inventory - 1` justo para que un round-trip de compra y venta en el mismo turno
rinda 0 (verificado: 3000 $ -> 3000 $). Premiar ventas a secas invitaria a jugar
con el mercado consigo mismo, que no produce nada. Se atribuye por procedencia:
solo cuentan las unidades que salieron de una cosecha.

Y el bonus por primer producto vendido ataca el otro optimo local conocido, el
monocultivo -la "granja de melon" que describen varios competidores-: sin el, la
politica encuentra un solo producto rentable y nunca prueba los demas.
"""
from __future__ import annotations

from . import spec
from .symbolic.market_ops import marginal_prices

# LOS DOS NUMEROS DE LA CONFORMACION. El resto del diseno esta razonado y
# medido (ver el docstring); estos dos estan puestos a ojo.
#
# Se leen del ENTORNO al importar porque `EntornoParalelo` lanza procesos hijos
# y pasarlos por tres capas de firmas seria mas invasivo que esto. Un bucle
# exterior los fija con os.environ antes de lanzar cada entrenamiento.
#
# Y no los aprende la politica: eso seria dejar que el agente elija su examen.
# Los mueve un nivel EXTERIOR que se evalua con el objetivo VERDADERO -dinero y
# victoria- sobre semillas que el interior no ha visto.
import os

BONUS_PRIMER_PRODUCTO = float(os.environ.get("KAG_BONUS", "0.2"))
ESCALA = float(os.environ.get("KAG_ESCALA", "2000.0"))

# TERMINO COMPETITIVO, continuo en el MARGEN.
#
# Correccion de lo que dije antes de mirar el codigo: la recompensa SI tenia
# un termino sobre el rival -el +-1 terminal de victoria/derrota-. Lo que no
# tenia es GRADUACION: es un signo, asi que ganar por 1 $ puntua igual que
# ganar por 50.000 y no dice en que direccion apretar.
#
# Lo que lo motiva, medido el 2026-09-21: entrenando contra una copia exacta
# de nosotros mismos, nuestro dinero sube un 9 % y el suyo un 17 % -siendo un
# vector FIJO que no aprende-. Toda su mejora viene de que nosotros cambiamos:
# al dejar de disputarle productos, sus precios marginales se quedan altos. El
# win rate cae de 0,59 a 0,22 mientras mejoramos en absoluto.
#
# DONDE VA, y no es indiferente: en el objetivo terminal, NO denso por dia.
# El denso ya se probo y se retiro con numeros -ver potencial.phi-: la
# liquidacion del rival aportaba el 99,1 % de la varianza de la recompensa
# diaria y el critico caia a R2 = -2,535, peor que predecir la media. El
# shaping solo puede llevar lo que el estado predice, y los saltos de su caja
# no lo son.
#
# PESO_RIVAL = 0 recupera exactamente el diseno anterior. Con 1 el objetivo es
# el margen puro; entre medias, interpola entre dinero propio y marcador.
#
# Riesgo a VIGILAR, no a suponer: premiar que el otro pierda puede degenerar en
# destruir valor -hundir precios hace menos dano a quien menos vende-. Puede
# ser jugada correcta o autolesion. Se mide.
PESO_RIVAL = float(os.environ.get("KAG_PESO_RIVAL", "0.0"))


class ContadorProduccion:
    """Lleva la procedencia de cada unidad. Una instancia por partida."""

    def __init__(self):
        self.available = {p: 0 for p in spec.PRODUCTS}
        self.vendidos = set()

    def harvested(self, obs, action, me: int) -> int:
        """Unidades cosechadas ESTE turno, leidas del estado previo al paso.

        No se estima: para cada HARVEST se mira el `yield_units` real de la
        casilla sobre la que esta la unidad, que es exactamente lo que el motor
        va a mover al inventario.
        """
        farm = obs["farms"][me]
        pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
        ops = [action.get("farmer")] + list(action.get("hands") or [])
        total = 0
        for (x, y), op in zip(pos, ops):
            if not op or op[0] != "HARVEST":
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
            total += n
        return total

    def sold(self, obs, action) -> tuple[float, float]:
        """(ingreso de produccion propia, bonus de exploracion) de este turno.

        El ingreso se valora con el precio MARGINAL del motor sobre el
        inventario previo: vender n unidades no cobra n veces el precio de
        portada, porque cada una baja la siguiente.
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
            income += float(sum(marginal_prices(obs, p, n)))
            self.available[p] -= n
            if p not in self.vendidos:
                self.vendidos.add(p)
                bonus += BONUS_PRIMER_PRODUCTO
        return income, bonus


def reward(obs, action, contador: ContadorProduccion, me: int, scale: float):
    """Densa por turno. La pasividad da exactamente 0.0."""
    contador.harvested(obs, action, me)
    income, bonus = contador.sold(obs, action)
    return income / scale + bonus


# TURNO REGALADO: acciones que el motor IGNORA en silencio.
#
# El motor no castiga lo imposible, lo convierte en no-op. Asi que "intentar
# algo que no puedo" y "pasar turno a proposito" son INDISTINGUIBLES en el
# retorno, y con 719 turnos y once unidades un turno tirado no se ve en la caja
# final. No es que cueste poco: es que no se puede atribuir.
#
# Por que penalizar y no enmascarar. Enmascarar es mejor en general -quitar la
# masa de probabilidad bate a castigarla- y de hecho `tile_options` solo
# propone jugadas legales. Pero el filtro por INVENTARIO se midio y salio mal
# por un motivo concreto: la casilla que pide FEED DESAPARECE cuando nadie
# lleva trigo, y entonces nadie va a por trigo. Con filtro 3,8 tareas/turno y
# 4.484 $; sin filtro 10,7 tareas/turno y 6.727 $ pero 66,9 % de tareas
# inejecutables y 86 % de PASS. Penalizar deja la tarea VISIBLE -prepararse
# sigue siendo aprendible- y cobra el turno perdido.
#
# Va en el bucle EXTERIOR, nunca en el vector macro: es un termino del
# objetivo. Si la politica pudiera mover su propia penalizacion, lo primero
# que aprenderia es a ponerla a cero.
#
# CUANDO MUERDE, y hay que saberlo antes de ajustarlo. Medido el 2026-09-21 en
# campeonato con la via heuristica Y con la hibrida: CERO acciones regaladas en
# toda la partida, porque `assign_units` recibe los inventarios y descarta el
# par unidad-tarea cuando `_puede` falla, asi que lo ilegal no llega al motor.
# Tampoco hay no-ops por destino caducado -regar lo ya regado, cosechar sin
# fruto, plantar en ocupada-: 0 de 1.563 acciones. En esos modos este peso NO
# PUEDE tener efecto por mucho que se suba.
#
# Solo muerde en `ops` sin el filtro de inventario, que es donde se midio el
# 66,9 % de tareas inejecutables y el 86 % de PASS.
#
# Y el corolario importante: nuestro tiempo muerto NO es ruido de acciones
# imposibles, es PASS deliberado -52,8 %- por falta de tarea que asignar. La
# senal no esta contaminada, esta ausente; se arregla generando tareas, no
# castigando.
#
# En dolares por accion regalada; 0 recupera exactamente el diseno anterior.
#
# COBERTURA, y conviene no exagerarla: se detecta la clase de INVENTARIO -FEED
# sin trigo, FERTILIZE sin fertilizante, PLACE sin el animal, DROP sin nada-,
# que es la dominante en lo medido. Otras acciones tambien caen en no-op
# -regar lo ya regado, cosechar sin fruto- y esas NO se cuentan todavia.
PESO_ILEGAL = float(os.environ.get("KAG_PESO_ILEGAL", "0.0"))