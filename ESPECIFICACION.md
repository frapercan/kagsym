# Especificación exacta de Kaggriculture

Todo lo de aquí está **derivado del motor** (`kaggle_environments.envs.kaggriculture`),
no supuesto. Cada afirmación es verificable ejecutando el intérprete.

---

## 1. La recompensa

**Sólo puntúa el dinero en caja en el turno final.** Nada más.

```
reward = farm["money"]   en el ultimo paso
score  = victoria / empate / derrota     (el MARGEN no puntua)
```

Consecuencias exactas, no interpretaciones:

- El cobertizo vale **0**. El inventario de las unidades vale **0**. Los cultivos
  en pie valen **0**. Los animales valen **0**. La tierra comprada vale **0**.
  Todo lo que no sea caja al cerrar es pérdida total.
- El experto 2945 termina con cobertizo 0, manos 0, cultivos 0: liquida exacto.
- Medido sobre 224 partidas reales de la ladder: **50 % se deciden por <0.5 %**,
  **26 % por menos de 41 $** (el valor de UNA acción útil) y **16.5 % son empates
  exactos**. La exactitud no es higiene: es la variable de decisión.

---

## 2. Espacio de acciones

### 2.1 Por unidad — una acción por unidad y turno

Unidades = 1 granjero + N peones. **Los peones duran UN día**: al cierre
`farm["hands"] = []` y `hires_today = 0`.

| op | precondición exacta | efecto |
|---|---|---|
| `PASS` | siempre | nada |
| `NORTH/SOUTH/EAST/WEST` | destino dentro del tablero (LOCKED es transitable) | mueve 1 |
| `PLANT <crop>` | casilla `None`, no LOCKED, `seeds[crop] > 0` | −1 semilla, nace planta |
| `WATER` | casilla PLANT, `not watered_today` | riega; si en ventana, +1 (+2 si fertilizada) |
| `HARVEST` | dict con `yield_units > 0`; si PLANT, `day − planted_day >= first_yield_day` | pasa a inventario de la unidad |
| `FERTILIZE` | casilla PLANT y **1 FERTILIZER en el inventario DE LA UNIDAD** | activa días `day..day+2` |
| `BUILD_COOP` / `BUILD_PASTURE` | casilla `None`, no LOCKED | crea estructura |
| `PLACE <animal>` | sobre estructura compatible vacía **y animal en la unidad** | coloca animal |
| `PLACE <item> [n]` | en casilla de acceso al cobertizo | deja n en el cobertizo |
| `PICKUP <item> [n]` | en acceso al cobertizo, `shed[item] > 0` | coge n |
| `DROP` | en acceso al cobertizo | vuelca TODO el inventario |
| `FEED` | animal `not fed_today` y **1 WHEAT en el inventario DE LA UNIDAD** | marca comido |
| `CARE` | animal `not cared_today` | bonus diferido de producción |
| `COLLECT_FERTILIZER` | animal con `fertilizer_available` | +1 FERTILIZER a la unidad |
| `DIG` | casilla no vacía y **sin animal** | deja la casilla a `None` |

Toda acción ilegal es **no-op silencioso**: el motor devuelve el mismo estado.
Medido: sin máscara, el 99 % de los `PLANT` de nuestra red eran no-ops.

### 2.2 Mercado — hasta `maxMarketOrdersPerTurn` órdenes por turno

`SELL`, `BUY_PRODUCT`, `BUY_SEED`, `BUY_ANIMAL` (por unidades, en paso a paso
alternando jugadores) y `HIRE`, `BUY_LAND` (atómicas, una por orden).

- `HIRE`: coste `mult · fib(hires_today)` con **mult = 1**. Los 10 primeros peones
  del día cuestan **143 $ en total**; el 15.º solo ya cuesta 610 $.
- `BUY_ANIMAL` deja el animal **en el cobertizo**, no en el tablero.
- Round-trip comprar/vender en el mismo turno rinde **exactamente 0**: el motor
  cotiza `BUY_PRODUCT` a `inventory − 1` justo para impedir el arbitraje.
  *(verificado: 3000 $ → 3000 $)*

---

## 3. Mecánica de exactitud: dónde se pierde por uno

1. **Plantar sin regar el mismo día = planta muerta.** `consecutive_unwatered`
   nace en **1**; al cierre pasa a 2 y la casilla se convierte en `WEED`.
   *(verificado: sin regar → WEED; regando → PLANT)*. `PLANT` y `WATER` son
   inseparables dentro del día.
2. **Dos días sin regar = WEED.** Se pierde semilla, casilla y trabajo.
3. **Dos días sin comer = el animal se escapa**, la estructura queda. Cada
   animal vale ~1 700 $ netos. `FEED` consume 1 trigo **del inventario de la
   unidad**: hay que llevárselo. *(medido: vendiendo el trigo se nos escaparon
   17 de 25 animales)*.
4. **Cobertizo: 100 unidades.** Al cierre del día los inventarios se vuelcan y
   **lo que no cabe se tira**.
5. **Decadencia:** pasado `max_lifespan_step`, −1 `yield_units` cada 2 pasos
   hasta 0, y entonces `WEED`. Cosechar tarde pierde unidades.
6. **Ventana de riego de los cultivos no-`ongoing`:** sólo suman rendimiento si
   se riegan entre `(max_yield_day+1)//2` y `max_yield_day`. Regar fuera sólo
   mantiene viva la planta.
7. **El bonus de fertilizante sólo cuenta en días regados**; el de `CARE` sólo
   se consume en un día de producción **y** alimentado.
8. **El pueblo drena en pasos fijos:** tiendas cada 4, centro cada 24. El precio
   es entero y tiene suelo en 1.
9. **Liquidación final:** el cobertizo no puntúa, así que hay un instante exacto
   a partir del cual toda retención es pérdida.

---

## 4. Lo derivable frente a lo decidible

Principio del proyecto: *predecir sólo lo que no se puede derivar ni muestrear*.
Aplicado al **espacio de acciones**, no sólo al estado.

**Derivable — el ejecutor, exacto, sin aprendizaje:**
ruta mínima (Manhattan; no hay obstáculos), asignación unidad→tarea, máscara de
legalidad, emparejar PLANT+WATER el mismo día, reserva de trigo = animales ×
días, no desbordar el cobertizo, calendario de liquidación, viabilidad por fecha
(no plantar lo que no madura antes del final).

**Decidible — lo que depende del rival y del riesgo, y sólo esto:**
cuánta superficie, qué mezcla de cultivos, cuántos animales, cuántos peones,
cuándo expandir, y sobre todo **acumular o volcar al mercado compartido**.

El world model entra exactamente ahí: su señal medida es de **nivel** (dinero del
rival +19 %, gasto +35 %), no de temporizado (flujo por turno: peor que no
corregir). Y estas decisiones son de nivel.

---

## 5. El MDP propuesto

**Estado:** observación codificada + predicción del world model (flujo del rival,
su dinero derivado) + historia de producción propia.

**Acción — un vector macro por día, no 17 ops × 24 unidades × 720 turnos.** Son
las constantes que hoy están escritas a mano:

```
superficie objetivo · mezcla de cultivos · animales objetivo · peones objetivo
agresividad de venta (acumular <-> volcar) · momento de expandir
```

**Recompensa.** La medida de ayer explica por qué la actual falla: con patrimonio
neto, *no hacer nada* conserva 3 000 $ y es un óptimo local — PPO convergía a
2 794 $, por debajo de no jugar. La recompensa debe dar **0 a la pasividad**:

```
denso     : ingresos por venta de producto PROPIO menos los del rival
explorar  : +bonus la PRIMERA vez que se vende cada producto (una vez por partida)
terminal  : victoria/derrota, que es lo unico que puntua
```

Lo de "producto propio" no es un detalle: excluye jugar al mercado consigo mismo,
que es donde el motor ya garantiza beneficio 0. El bonus por producto es lo que
rompe el mínimo local del monocultivo.
