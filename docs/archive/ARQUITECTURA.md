# Arquitectura: RL neurosimbolico adversarial

Plan derivado de lo medido, no de lo deseable. Cada fase tiene una puerta con
numero. Si una puerta no se pasa, la fase se declara muerta y se documenta.

---

## 1. El corte neurosimbolico

El principio del proyecto -*predecir solo lo que no se puede derivar ni
muestrear*- aplicado al espacio de ACCIONES, no solo al de estados.

```
SIMBOLICO (exacto, sin aprendizaje, sin parametros)
  motor            clon bit-a-bit, 38 567 pasos/s
  legalidad        precondiciones exactas de las 18 ops
  aritmetica       fechas limite, 1 trigo/animal/dia, tope 100 del cobertizo,
                   PLANT+WATER el mismo dia (si no, hierba esa noche)
  combinatoria     asignacion humgara optima unidad->tarea (0.34 ms)
  liquidacion      calendario de venta con precio marginal exacto

NEURAL (solo lo que depende de juicio o del rival)
  micro            VALOR de cada tarea del tablero    (10x10 x n_ops)
  macro            objetivos del dia                  (6 componentes en [0,1])
  rival            nivel acumulado de vertido a 12-48 turnos
```

Lo que NO se aprende nunca, y por que: aprender la asignacion optima o la ruta
minima gastaria todo el presupuesto de muestras en redescubrir, peor, algo que
corre exacto en 0.34 ms. Medido: sin mascara de legalidad el 99 % de los PLANT
eran no-ops, y el 42 % de las decisiones de PPO eran navegacion sin obstaculos.

---

## 2. Las dos politicas

### Micro: valores aprendidos + solver exacto

Hoy `tile_task` devuelve estimaciones en dolares **escritas a mano**. Eso es
juicio de valor disfrazado de mecanica, y es donde esta la brecha:

```
              acciones   util%   $/accion util
nosotros         5 072   23.3%        31.1
experto          6 770   50.6%        41.2
                 1.33x x 2.17x x     1.32x  =  3.81x
```

El 2.17x de fraccion util es el termino dominante. La red emite el tensor de
valor; el humgaro lo consume. Gradiente a traves del solver por **optimizadores
perturbados** (Berthet et al. 2020): se anade ruido Gumbel al tensor, se resuelve
el problema combinatorio EXACTO, y la solucion esperada es diferenciable. No hay
relajacion ni solver aproximado: el argmax sigue siendo exacto en ejecucion.
Alternativas si falla: Implicit MLE (Niepert 2021) o diferenciacion de caja negra
(Vlastelica 2020).

### Macro: objetivos diarios

Ya construido. Beta sobre [0,1]^6, soporte exacto, arranque uniforme. Un paso =
un dia -> episodio de 30 pasos en vez de 720. El credit assignment a 30 dias es
lo que el foro describe como el muro de esta competicion.

Jerarquia: el macro fija el presupuesto (cuantas casillas, cuantos animales,
cuantos peones, agresividad de venta); el micro reparte el trabajo dentro de ese
presupuesto. El macro NO ve casillas; el micro NO ve la partida entera.

---

## 3. Modelo del RIVAL, no del mundo

El mundo no hay que modelarlo: el motor es exacto y lo tenemos. Lo unico no
simulable es el rival. Medido sobre 8360 transiciones retenidas:

```
dinero del rival (nivel) .... +19.0 %   util
gasto fuera del mercado ..... +34.8 %   util
flujo turno a turno ......... -26.0 %   PEOR QUE NO CORREGIR
```

Acierta el NIVEL, no el TEMPORIZADO. Consecuencia de diseno: el modelo predice
**el vertido acumulado a 12-48 turnos**, nunca el turno concreto. Formulacion
JEPA legitima: predecir el LATENTE del estado futuro del rival, no sus
observaciones -sus observaciones son en su mayoria publicas (`money`, `tiles`,
`hands`), asi que predecirlas es redundante; lo oculto es cobertizo, semillas e
inventarios, y de ahi sale el vertido-.

**Puerta de admision, ya construida** (`trust.py`): el flujo real del rival se
despeja EXACTO del inventario de mercado un turno despues. Si el modelo no bate
a (a) flujo cero, (b) EMA del flujo observado y (c) autoclon simulado en
`fastenv`, se apaga solo. Con trust=0 el sistema ES el motor exacto.

---

## 4. Adversarial y replay

**Por que es obligatorio, cuantificado:**

```
experto vs pasivo ....... 169 287 $
experto vs experto ...... 101 385 $     -40 %
```

Y el canal de sabotaje es brutal en los productos caros: si el rival vuelca 100
unidades antes que tu, tus 10 de LECHE pasan de 1 506 $ a 10 $ (-99.3 %), y las
de LANA de 1 983 $ a 10 $. WHEAT y EGG apenas se mueven (-11 %). El mercado
compartido es el UNICO canal de interaccion del juego: ninguna accion de unidad
toca la granja del rival.

**Por que NO todavia:** a nuestra escala (20 k$) no disputamos nada, el drenaje
del pueblo nos absorbe. Por eso el autojuego rinde mas que contra pasivo. La
liga se activa cuando pasemos de 100 k$ contra pasivo.

**Cuando se active:** PSRO / double oracle sobre una poblacion, no autojuego
ingenuo -hay no-transitividad medida (ciclos en la matriz de pagos), que es
justo el caso donde el autojuego ingenuo cicla y la poblacion no-. Replay: los
661 replays de los mejores como prior de rivales, mas el banco de versiones
propias con promocion por win rate (>60 % al banco, >70 % a rival activo).

**Comparaciones siempre con numeros aleatorios comunes (CRN)** y lados
intercambiados: la desviacion por rollout es de 222-2108 $ y la pareada de 78 $.
Sin CRN no se distingue una mejora de 41 $, y el 26 % de las partidas se deciden
por menos de eso.

---

## 5. Fases, con puerta medible

| # | Que | Puerta | Coste |
|---|-----|--------|-------|
| 1 | Micro: valor aprendido + humgaro perturbado | util% de 23.3 -> **>35 %**, y $ > 22 298 (CEM) | ~4 h CPU |
| 2 | Macro y micro entrenados juntos | $ > **40 000** contra pasivo | ~8 h CPU |
| 3 | Liquidacion exacta + desempate | gana >=50 % de los empates exactos (16.5 % de la ladder) | ~2 h |
| 4 | Modelo del rival, tras la puerta de falsacion | bate a flujo-cero y a EMA en dinero final | ~4 h |
| 5 | Liga PSRO con replay | solo si fase 2 supera 100 k$ | dias |

Listones vigentes: no hacer nada 3 000 $ · capa a mano 10 097 $ · CEM fijo
22 298 $ · experto 169 287 $ (101 385 $ contra si mismo).

Criterio unico de aceptacion en todas: `scripts/league.py` contra los 4 agentes
publicos reales. Precision de validacion, win rate de curriculo y R2 agregado ya
han enganado una vez cada uno.

---

## 6. Lo que este plan NO hace, y por que

- **No aprende dinamica del entorno.** El motor es exacto y 102x mas rapido que
  cualquier latente. Un world model de dinamica aqui es redundante por
  construccion.
- **No planifica online con MPC de cola larga.** `from_observation` sortea la
  semilla y vacia el `private` del rival: la varianza por rollout (222-2108 $)
  supera con creces la separacion entre macros razonables (~25 $). SNR < 1.
- **No usa shaping por patrimonio.** Medido: convierte la pasividad en optimo
  local con premio positivo (quieto conservas 3 000 $) y PPO convergia a 2 794 $.
- **No clona al experto al 99 %.** Medido: 99.67 % de paridad de acciones dio
  0 % de victorias. El competidor que llego a 80 k clono deliberadamente solo
  hasta el 7 %. (Matiz anadido el 2026-09-21: ese 99,67 % es `acc_op` sobre el
  buffer ACUMULADO, no sobre rollouts autonomos, y el acierto de las decisiones
  de MERCADO -contratar, comprar- nunca se midio. Ver ROADMAP §3.)
- **No migra a JAX.** Nuestro motor ya hace 38 567 pasos/s frente a los 10 000
  que reporta el mejor competidor del foro. El cuello era el ejecutor Python, y
  un cambio de 10 lineas le saco un 50 %. Queda paralelizar por procesos: 12
  cores, usamos 1.
