# Formas escritas a mano -- barrido del 2026-09-24

El proyecto tiene una regla dura -ninguna constante a ojo- y SE CUMPLE: 39
parametros que eran constantes viven hoy en el macro y se aprenden.

Pero la regla solo cubre los NUMEROS. No cubre las ECUACIONES. Seis agentes
barrieron el repo buscando reglas de decision cuya FORMA es una creencia
escrita a mano y que ningun dial puede cambiar. Dos escepticos por hallazgo,
con dos filtros duros: si el motor impone la forma es un HECHO y no cuenta,
y si algun dial del macro ya la puede cambiar en la practica, tampoco.

Sobrevivieron 3 de 24. Las tres tienen `hay_dial = false`.

Una cuarta -el PASS como opcion de reserva de la unidad que pierde la
subasta- entro con un solo voto por un corte de sesion y fue REFUTADA al
completar el escrutinio. Queda listada abajo.


## Resumen

| impacto | fichero:linea | tipo | forma |
|---|---|---|---|
| alto | `market_ops.py:159` | horizonte | El trigo de pienso no se puede reponer mas tarde: hay que tenerlo TODO en el almacen des |
| alto | `tasks.py:873` | agregador | Una casilla solo puede rendir una operacion por turno, y la que gana el argmax es la uni |
| alto | `potential.py:94` | horizonte | Un animal vivo siempre tiene por delante su incubacion entera; no hay tal cosa como un a |

---


## 1. market_ops.py:159 -- horizonte

**LA FORMA.** `reserve = {"WHEAT": n_animals * max(1, (remaining + 1) // spec.TURNS_PER_DAY)}`: reserva trigo por TODOS los dias que quedan de temporada. El lado comprador (`feed_orders`, linea 731) usa `min(stock_days, days)` con FEED_STOCK_DAYS ~2,5 dias, que ademas es aprendido. Dos horizontes distintos para la misma magnitud, uno escrito a mano a la temporada entera y otro aprendido. Y la precedencia de las lineas 223-231 remata: el `if n <= 0: continue` se evalua ANTES de la rama de liquidacion, asi que la 'liberacion de la reserva' que anuncia el comentario de la linea 228 no ocurre nunca cuando la reserva cubre todo el stock.

**LA CREENCIA.** El trigo de pienso no se puede reponer mas tarde: hay que tenerlo TODO en el almacen desde hoy hasta el final de la temporada.

**ALTERNATIVA.** Reservar el mismo horizonte que se compra (FEED_STOCK_DAYS, ya aprendido), o reservar el COSTE de reponer el pienso en dinero en vez de las unidades fisicas, ya que el mercado vende trigo todos los turnos. Y evaluar la rama de liquidacion antes del descarte por reserva.

**POR QUE ES PORTANTE.** Cierra un canal de mercado entero. Medido: en 476 de los 484 turnos con trigo en el almacen (98,3%) la reserva superaba al stock, con la reserva >= almacen en 708 de 719 turnos; 74 unidades de trigo vendidas en todo el episodio frente a 179 de huevo. En la ventana final de liquidacion el trigo se salta en 123 turnos, contra lo que dice el comentario. El trigo es el producto mas inelastico del mercado (25 $ -> 21 $ vendiendo 160 unidades, frente a la leche que se hunde a 1 $ con 80) y el que mas drena el pueblo (0,625 por tienda), o sea el unico que se puede volcar sin romper el precio: el dial aprendido m_wheat queda muerto en el lado vendedor mientras viva un solo animal. Contrafactual con politica congelada (reserva a 2,5 dias, 24 semillas): +588 $, se 1.174, t 0,50 -- una politica que ya no cultiva trigo no puede medir la estrategia que esta forma le prohibe.


## 2. tasks.py:873 -- agregador

**LA FORMA.** El diccionario `tasks` se indexa por CASILLA, no por par (casilla, operacion): `k, op = max(options, key=lambda o: verb_map[o[0]][y][x])` colapsa todas las operaciones legales de la casilla a UNA sola, y `tasks[(x,y)] = (v, op)` (linea 905) la convierte en UNA columna del humgaro. Una casilla = una accion = una unidad por turno.

**LA CREENCIA.** Una casilla solo puede rendir una operacion por turno, y la que gana el argmax es la unica que merece competir por una unidad-turno.

**ALTERNATIVA.** Columnas por PAR (casilla, operacion): cada opcion legal de `tile_options` entra como su propia columna con su propio valor. El motor lo permite -verificado en kaggle_environments/envs/kaggriculture/kaggriculture.py: el movimiento no comprueba colision y cada accion se aplica a la casilla donde esta la unidad-, asi que FEED y HARVEST sobre el mismo establo, o WATER y FERTILIZE sobre la misma planta, son legales el mismo turno y dan efectos distintos.

**POR QUE ES PORTANTE.** Es la forma mas portante del modulo, y es la que DE VERDAD decide: la cascada de `tile_task` que motiva el barrido esta MUERTA en juego -el `continue` de la linea 908 se ejecuta siempre que hay verb_map, y el ejecutor lo pasa en los 700/700 turnos medidos-, asi que el unico camino vivo es `tile_options` + este argmax. Medido con prod (runs/partida_v5.pt, 48.387 $, 1 episodio, 700 turnos, semilla reservada 0): 25.056 casillas-turno con operacion legal, de las que 15.192 (60,6 %) tienen MAS DE UNA y solo entra una; 61 pares legales por turno se quedan en 35,8 columnas. Verbos tirados por episodio: FERTILIZE 4.225, CARE 2.072, HARVEST 1.776, COLLECT_FERTILIZER 1.610, DROP 1.504, WATER 713, FEED 511. Y el coste se ve en las unidades ociosas: de los 1.086 unidad-turnos en PASS, 906 (83 %) tenian una operacion descartada por este argmax que ESA unidad podia ejecutar con lo que llevaba encima, a 3,66 pasos de media. Con desdeprod.pt.ultimo (30.156 $): 1.686 de 1.936 PASS, a 3,14 pasos.


## 3. potential.py:94 -- horizonte

**LA FORMA.** `to_come = max(0, days - a["first_yield_day"]) // interval`. El periodo de incubacion se resta de los dias RESTANTES, todos los dias, como si el animal se acabase de colocar. La casilla lleva `placed_day` (lo escribe `_new_animal` en el motor y obs.py:186 y obs.py:505 ya lo leen) y aqui no se usa. Los cultivos, dos ramas mas arriba, SI se envejecen con `age = day - planted_day`.

**LA CREENCIA.** Un animal vivo siempre tiene por delante su incubacion entera; no hay tal cosa como un animal maduro.

**ALTERNATIVA.** Envejecerlo igual que el cultivo: `edad = day - t["placed_day"]`, y proyectar `to_come = (days - max(0, first_yield_day - edad)) // interval`. El campo ya esta en la observacion; es la misma ley que ya se aplica a los cultivos en la rama de al lado.

**POR QUE ES PORTANTE.** Una vaca madura con 15 dias por delante: phi proyecta 3 unidades de leche, la real son 7. Con 8 dias restantes phi proyecta CERO y la real son 4; con 5 dias, 0 contra 2. Oveja con 8 dias: 0 contra 2. Ganso con 5 dias: 1 contra 5. O sea: durante los ultimos `first_yield_day` dias de la temporada -8 para vaca, 6 para oveja, 4 para ganso- phi valora todo animal vivo SOLO por lo que tiene en la ubre. Consecuencias directas: (a) el shaping deja de pagar FEED y CARE justo en el tramo final, y dos dias sin comer = el animal se escapa (~1.700 $ netos segun SPECIFICATION.md) con un castigo en phi de casi nada; (b) comprar un animal siempre parece tardio, porque el credito se calcula desde cero. Y cae exactamente sobre la linea animal, que es la que el propio comentario de reward.py:135 llama 'precisamente la que no aprendemos' y que la memoria del proyecto mide como parte del 84% del hueco contra v48 (leche 72 uds contra 335).


---

## Descartadas por los escepticos (21)

Listadas para que nadie las vuelva a levantar sin datos nuevos.

- **`if unused >= max(SEED_FLOOR, SEED_STOCK_PER_UNIT * n_units): return []`, donde `unused` (linea 513) es la SUMA de semillas sobre **
  - La FORMA existe literalmente. Verificada en /home/xaxi/farm/kagsym/symbolic/market_ops.py: linea 513 `unused = sum(int(v) for v in obs["private"].get("seeds", {}).values())` suma sobre TODOS los cultivos; linea 518 `if unused >= max(SEED_FLOOR, SEED_STOCK_PER_UNIT * n_units): return []` aborta la compra entera; linea 539 `missing_total = max(0, tile_target - sum(... for k in viable))` agrega la mi

- **`budget = max(LABOUR_FLOOR, money * LABOUR_BUDGET_FRACTION)`, y luego `if cost > budget: break` (381). Un `max` entre una constant**
  - La forma existe literal y no es refutable en ninguno de los tres criterios.

(1) LITERALIDAD. market_ops.py:369 contiene exactamente `budget = max(LABOUR_FLOOR, money * LABOUR_BUDGET_FRACTION)`, y el consumo es el descrito: 380 `if cost > budget:` / 381 `break`, con `budget -= cost` en 384 dentro del bucle `for n in range(hired, n_max)`. `money` es la caja del instante (farm["money"] leida en 318 

- **`fert = prod_days * float(prices_.get("FERTILIZER", 0)) * MANURE_CREDIT`: el credito del estiercol se cuenta solo sobre los dias P**
  - La forma existe literalmente en market_ops.py:605 (`fert = prod_days * float(prices_.get("FERTILIZER", 0)) * MANURE_CREDIT`, con prod_days = days - first_yield_day en la 592), y el motor impone lo CONTRARIO: docs/env_src/kaggriculture.py:831 pone `tile["fertilizer_available"] = True` incondicionalmente en _daily_refresh_animals, fuera del bloque de produccion, para todo animal vivo -- son `days`, 

- **`row[j] = v * (STEP_DISCOUNT ** dist(pos, tile))`: el coste de andar es MULTIPLICATIVO, es decir PROPORCIONAL al valor de la propi**
  - La forma existe literal en /home/xaxi/farm/kagsym/symbolic/tasks.py:1131 -- `row[j] = v * (STEP_DISCOUNT ** dist(pos, tile))`, con `dist` Manhattan puro (linea 59), y repetida en la rama encadenada (1120: `_CV * v * (STEP_DISCOUNT ** _dtot)`). NO es un hecho del motor: STEP_DISCOUNT se declara en la linea 507 con el comentario 'a step costs a turn: the value is discounted' y justo debajo 'Two more

- **El valor de la columna sale SOLO del canal de valor por casilla: `r = value_map[y][x]`, `v = copysign(expm1(min(MAP_CAP, |MAP_GAIN**
  - REFUTADA como "forma" (aunque el literal de la línea sí existe).

Lo que sí existe (/home/xaxi/farm/kagsym/symbolic/tasks.py:883-885): `r = float(value_map[y][x])` y `v = copysign(expm1(min(MAP_CAP, abs(MAP_GAIN*r))), r)`. En esa expresión el logit del verbo ganador (línea 873-874) no entra. Hasta ahí, correcto.

Por qué queda refutada tal y como se describe:

1) El verbo SÍ tiene consecuencia de 

- **La opcion de reserva de una unidad es `actions.append(["PASS"])` sobre columnas ficticias de valor 0 (lineas 1058-1059, 1244): la **
  - La forma esta ahi, literal y en las lineas citadas. tasks.py:1058-1059 construye `m = len(tiles) + n` con `value = [[0.0] * m ...]` (las n columnas ficticias de valor 0, documentadas en el docstring 1050-1052); 1244 es la rama `if j >= len(tiles) or value[i][j] <= 0.0:`; y 1266 es exactamente `actions.append(["PASS"])`, seguido de 1267-1268 `previous[i] = None`, que ademas le borra el destino. El 

- **free = max(0, sustainable - planted), con sustainable = n_units * TURNS_PER_DAY / turns_per_tile (:49). El techo de plantado es un**
  - La forma existe literalmente y no la impone el motor. executor.py:181 es exactamente `free = max(0, sustainable - planted)`, con `planted` contando tiles kind=='PLANT' ACTUALES (:179-180, sin ningun termino de maduracion ni de futuro), y executor.py:49 es `max(1, int(n_units * spec.TURNS_PER_DAY / max(1.0, turns_per_tile)))` — presupuesto de turnos-unidad de UN dia dividido por UN escalar de coste

- **orders += by_category[c]() recorriendo cats en orden y luego orders[:_max]. La competencia por los diez huecos del motor se resuel**
  - REFUTADA: la "forma escrita a mano" ya es un dial vivo del macro, y ademas la linea :252 no es la que corre en juego.

1) La linea 252 esta dentro de la rama `if mac is None or PRIORITY_SPLIT <= 0.0` (/home/xaxi/farm/kagsym/symbolic/executor.py:250). Con el checkpoint desplegado runs/partida_v5.pt la red emite priority_split en 0,00048-0,00138 (719 llamadas a apply_params en la semilla 1, mediana 

- **turns_per_tile *= 1 + min(COST_RISE, dry/planted) cuando hay plantas secas, y turns_per_tile = max(TURNS_PER_TILE_MIN, turns_per_t**
  - La forma SÍ está escrita tal cual en /home/xaxi/farm/kagsym/symbolic/executor.py:143-148 (subida `self.turns_per_tile *= 1.0 + min(COST_RISE, dry/planted)` en la 145; bajada `max(TURNS_PER_TILE_MIN, turns_per_tile * COST_DECAY)` en la 147-148; una sola muestra al día por la guarda `obs["hour"] != spec.TURNS_PER_DAY - 1` en la 130). El motor no la impone. Pero se refuta por los otros dos criterios.

- **flow = self._opponent_flow(obs, provisional) devuelve None salvo que USE_RIVAL_FLOW este encendido (:26, variable de entorno KAG_F**
  - La forma existe tal y como se describe. executor.py:26 lee USE_RIVAL_FLOW de KAG_FLUJO_MERCADO con defecto '0' en tiempo de import; _opponent_flow (executor.py:79-117) devuelve None cuando self.rival is None y no esta encendido, y self.rival solo se asigna en executor.py:63 desde el constructor: NINGUN llamador del repo pasa rival= (environment.py:355 y 366, policy.py:194, tests/test_fastenv.py:53

- **target_animals: `room = max(0.0, capacity - planted) / max(1e-6, ACTIONS_PER_ANIMAL)` y acto seguido `return max(0, int(macro.anim**
  - La forma EXISTE literalmente (macro.py:539-540), pero la creencia alegada no se sostiene y la ALTERNATIVA ya está implementada.

1) YA HAY UN DIAL QUE LA CAMBIA EN LA PRÁCTICA — y es exactamente el "dial simétrico" que pide la alternativa. `target_animals = animals * (capacity - planted) / ACTIONS_PER_ANIMAL`, y `ACTIONS_PER_ANIMAL` es borderless `3.0*exp(logit(f))` -> (0,+inf) (macro.py:342 `("ac

- **Dos contabilidades del MISMO recurso a 18 lineas de distancia. `target_tiles` (518): `watering_cap = n_units * spec.TURNS_PER_DAY **
  - La forma existe textualmente (macro.py:518 `watering_cap = n_units*TURNS_PER_DAY*WATERING_FACTOR` * exp(_logit(macro.tiles)) borderless, sin descontar animales; macro.py:536-540 `capacity = n_units*TURNS_PER_DAY` sin fraccion de viaje, menos `planted`, por `macro.animals` en [0,1]). Pero cae por las dos clausulas de refutacion.

(1) DIALES VIVOS QUE YA LA CAMBIAN. `WATERING_FACTOR` se lee en exact

- **`target_hands` = `max(0, int(round(7.5 * math.exp(_logit(macro.hands)))))`: no mira el estado -ni caja, ni dia, ni trabajo pendien**
  - La forma existe literalmente en la línea (macro.py:480 no lee `obs`; executor.py:177 usa `max` y market_ops.py:364 usa `min`; target_tiles/target_animals dimensionan sobre `1 + target_hands`), pero la lectura como "forma escrita a mano que ignora el estado" se cae por cuatro vías. (1) El dial YA mira el estado: `policy.py:218-236` (`plan_day`) reemite `macro.hands` cada día desde `macro_mu(z)` con

- **`sell_horizon` devuelve UN escalar -`max(1, int(round(spec.TURNS_PER_DAY * math.exp(_logit(macro.selling)))))`- que vale para los **
  - REFUTADA. El núcleo literal existe (`/home/xaxi/farm/kagsym/macro.py:543-555`: `sell_horizon` ignora `obs`, no recibe producto y devuelve un escalar diario), pero la FORMA tal y como se describe —"un solo número fija si se guarda o se suelta para leche, fresa y trigo a la vez", y "la agresividad se expresa EXCLUSIVAMENTE como ventana de previsión"— es falsa, y la ALTERNATIVA propuesta ya está impl

- **Cada lote se valora con una llamada INDEPENDIENTE a `_lot_value`, y `marginal_prices` lee siempre `obs["market"]["inventory"][p]` **
  - La forma existe literalmente en kagsym/potential.py:90 y no es un hecho del motor ni la toca ningun dial. VERIFICADO EN EL FICHERO: `_lot_value` (l.48-56) devuelve `sum(marginal_prices(obs, product, min(n,200)))`, y `marginal_prices` (symbolic/market_ops.py:126) hace `inv = obs["market"]["inventory"][product]` en CADA llamada, sin acumular. Hay cuatro puntos de acumulacion con `+=` sobre el mismo 

- **`if cd["ongoing"]: remaining = min(max_yield - units, days // interval)` -y nada en la rama contraria-. Los cultivos `ongoing` (TO**
  - La forma existe tal cual en kagsym/potential.py:86. `if cd["ongoing"]: remaining = max(0, min(cd["max_yield"] - units, days // max(1, cd["interval"]))); units += remaining` y NADA en la rama contraria: WHEAT/CARROT/MELON solo aportan sus `yield_units` actuales (el motor les siembra 1 en linea 223 y los sube solo al regar dentro de la ventana, motor 438-443). Y el `days` de los cultivos no resta in

- **`_ing = float(sum(marginal_prices(obs, p, n)))` con `obs` = el estado PREVIO al `env.step`. El termino denso se paga con un precio**
  - La FORMA existe literalmente: /home/xaxi/farm/kagsym/reward.py:191 es `_ing = float(sum(marginal_prices(obs, p, n)))`, y `obs` es efectivamente el estado PREVIO al step (/home/xaxi/farm/kagsym/environment.py:556 llama a `counter[i].sold(ob, acc)`; `env.step([acc, rival])` no ocurre hasta la :582). Ningun dial del macro toca la recompensa (macro.py solo pesa la DECISION de vender: `w["rival"]`, `se

- **`for k,v in enumerate(RIVAL_WINDOWS): if missing <= v: out[k,j] += max(listo, 1.0)`, con `missing = max(0, first_yield_day - age)`**
  - NO REFUTADA: la forma esta literalmente en obs.py:511-513, `out[k, j] += max(listo, 1.0)` bajo `if missing <= v`, con `missing` calculado SOLO desde `first_yield_day` (498 y 504-505). (a) `interval` no aparece en ninguna linea del cuerpo de `rival_flow` (477-514); el unico campo de rendimiento que se lee es `first_yield_day`. (b) el suelo `max(listo, 1.0)` es identico para los 9 productos, y el mo

- **Tres reglas de horizonte DISTINTAS, escogidas a mano, en ocho lineas: `ciclo = first_yield_day if ongoing else max_yield_day` (241**
  - La forma existe literalmente en kagsym/obs.py:238-246: `ciclo = cd["first_yield_day"] if cd["ongoing"] else cd["max_yield_day"]` (241), escalon `1.0 if dias_q >= ciclo else 0.0` (242) y `first_yield_day + interval` para animales (246). Tres reglas de horizonte distintas colapsadas a 0/1, sin cambios desde el commit inicial e0764a2 (git log -S"viables_cultivo"): nunca medida. NO la impone el motor:

- **`holgura = 0.0 if sin_regar == 0 else clamp((action_q/3.0 - sin_regar)/max(1,sin_regar), -1, 1)`. Tres decisiones de forma juntas:**
  - NO refutada. (1) La forma existe literal en kagsym/obs.py:271-272: `holgura = 0.0 if sin_regar == 0 else max(-1.0, min(1.0, (action_q / 3.0 - sin_regar) / max(1.0, sin_regar)))`. El 3.0 es el UNICO literal 3.0 del fichero; la colision en 0.0 es real (numerador exactamente cero cuando action_q/3 == sin_regar, el mismo valor que la rama sin_regar==0); la saturacion +/-1 esta. (2) El motor NO impone 

- **`urgencia = max(0.0, 1.0 - dias_q / 2.0)`: rampa lineal de liquidacion con horizonte fijado en 2 dias, plana en 0 el resto del epi**
  - NO refutada: la forma existe literalmente en kagsym/obs.py:248 como `urgencia = max(0.0, 1.0 - dias_q / 2.0)`, con el 2.0 escrito a mano dentro de la expresion. (1) El motor no la impone: sus hechos cercanos son que el cobertizo puntua 0 al cierre, shedCapacity=100 (spec.py:82), maxMarketOrdersPerTurn=10 y el precio marginal decreciente; ninguno produce el numero 2, que es una eleccion de horizont

