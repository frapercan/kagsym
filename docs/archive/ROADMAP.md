# Roadmap

Documento vivo. Regla de casa: **ninguna cifra de dinero significa nada sin decir
contra quien se midio.** Toda la sesion del 2026-09-20 mezclo cifras medidas
contra rivales distintos y eso invalido varias conclusiones.

---

## 1. Estado actual (2026-09-20, 23:5x)

Todo contra `v48-fast-routes`, rival entero:

```
no hacer nada .....................    245 $
nuestro e2e (backbone6, en curso) ~ 21 000 $
nuestra heuristica guionizada .....  30 065 $   (sd enorme: 13k-54k)
v48 contra nosotros ............... 135 000 $
v48 contra un pasivo .............. 179 514 $
```

La misma heuristica hace **64 273 $** contra v48 con tope de 3 peones y
**32 083 $** contra v48 entero. El rival cambia el resultado por un factor 2.

**Arquitectura viva:** codificador CNN+FiLM (2,17M par.) con cabezas macro
(14 dims, 1/dia), micro (1 canal de valor + 15 logits de verbo por casilla,
1/dia), critico y auxiliar de rival. PPO con shaping potencial. Legalidad,
rutas, asignacion humgara y aritmetica exacta se quedan simbolicas.

---

## 2. Auditoria: bugs encontrados

| bug | evidencia | estado |
|---|---|---|
| Critico mal **escalado** | `smooth_l1` beta=1.0 sobre retornos de media 20 y sd 8 -> regimen L1 puro. Regresion de juguete: R2 **-16,58** asi vs **+0,826** normalizando | **ARREGLADO** |
| Cociente de PPO = ruido | 1.614 dims gaussianas de las que ~40 deciden; **99,4 %** de cocientes saturados -> **0,1 %** enmascarando | **ARREGLADO** |
| Critico estrangulado por el controlador de KL | `lr` tronco 1,57e-6 vs cabezas 1,57e-4. R2 cayo de 0,675 a **-0,405** al bajar el lr del tronco | **ARREGLADO** (100:1 + normalizacion) |
| Optimizador no persistido | Reanudar reiniciaba Adam y el lr: 20 957 $ -> 4 717 $ en 5 updates | **ARREGLADO** |
| `lr` explicito machacado al reanudar | El estado del optimizador pisaba el `--lr` pedido, en silencio | **ARREGLADO** |
| `dinero_rival` siempre 0 | Leia `env.riv_fin`, que solo existe en `EntornoDia`, no en `EntornoParalelo` | **ARREGLADO** |
| `liga.py:154` `bc` sin importar | `NameError` con `--challenger`: el checkpoint de DAgger llevaba un dia sin poder evaluarse | **ARREGLADO** |
| ESCALERA sin escalones | Los 3 publicos que funcionan valen 171-180k; `shop-router-0909` **ROTO** (falta `agents_pub/actions.json`) | **ARREGLADO** (topes de peones) |
| Ceguera de la hora 0 | **31 de 88** dims globales constantes; `carried` = 0 exacto siempre, porque a la hora 0 hay 0 peones | **ABIERTO** |
| `version.py` no cubre `redes/` ni `cli/` | 4 huellas distintas en 6 h y la huella no detecta un cambio de red | **ABIERTO** |
| MLflow: 38/78 corridas en `RUNNING`, BD fantasma por ruta relativa, `dinero` con 3 nombres | | **PARCIAL** (metricas nuevas si) |

**Dispersion medida** (8.000 pasos del buffer): rejilla **94,3 % ceros**;
**10 de 25** canales por jugador muertos; accion 2.200 logits con **3,18** vivos
(densidad 0,14 %).

---

## 3. Research: veredictos

### Descartado, con razon

- **JEPA / modelos del mundo aprendidos.** Ni como dinamica (tenemos simulador
  exacto a 38.567 pasos/s; MuZero **igualo** a AlphaZero, no lo supero) ni como
  representacion (la clonacion acierta el verbo **99,4 %** — no hay problema de
  representacion que arreglar). Tasa base: Li et al. NeurIPS 2022, **11 de 19**
  configuraciones SSL no baten a solo-aumentos.
- **Aprender la recompensa (IRL, RLHF, modelos de preferencia).** Tenemos la
  recompensa exacta del motor. Aprenderla es sustituir la verdad por una
  aproximacion peor.
- **RUDDER / return decomposition / HCA.** Atacan MDPs con recompensa unica al
  final. La nuestra ya es densa por dia.
- **Sparse coding / representaciones dispersas.** Es la solucion **al reves**:
  existen para *crear* dispersion; nuestra entrada ya es dispersa al 5,69 %. El
  problema es gastar computo en los ceros, y eso lo arreglan las entidades.
- **Slot attention** (descubre objetos en pixeles; tenemos la lista exacta) y
  **GNNs explicitas** (un transformer sobre 45 tokens ya es una GNN completa).
- **Imitacion como camino.** Refutado **dos veces por vias independientes**:
  clonacion 99,4 % de acierto -> 3.036-6.727 $; DAgger **99,67 %** de acuerdo
  -> **-99 % de margen, 0 victorias en 24**. El acuerdo con el experto **no
  predice el rendimiento**.
  - **Corregido el 2026-09-21.** El 99,67 % NO es "sobre sus propios estados":
    es `acc_op` sobre el **buffer ACUMULADO** de todas las rondas, incluida la
    ronda 0 que es experto puro (`scripts/train_dagger.py:215`, `losses(pred, b)`
    sobre el conjunto acumulado). Y es el acierto de **una sola cabeza**: el
    entrenamiento supervisa `op_target`, `crop_target`, `item_target` y
    `market_target`, pero `kagworld/policy_net.py:183` solo registra `acc_op`.
    El mercado se registra como PERDIDA, nunca como acierto, asi que
    **contratar y comprar no tienen acierto medido**. Justo donde otros reportan
    sus primeras divergencias.

### Vivo

- **Ventajas pareadas con numeros aleatorios comunes.** Medido: reduce la sd
  **2,2x** (4,8x en muestras). Pero como sustituto directo cuesta ~23x por
  update -> **perdida neta de 5x**. Y no se puede abaratar acortando: **el signo
  se invierte** (1 dia: -1509 con |t|=20; 3 dias: +3838), porque con truncamiento
  Phi deja de ser shaping y pasa a ser valor terminal, y Phi es **optimista con
  el cultivo** (proyecta que la planta madura; nosotros regamos 1,8 veces por
  plantada frente a 4,0 del experto). **Requiere primero un valor terminal fiable.**
- **Entidades (transformer + cabezas puntero).** Justificacion **debilitada**:
  el DAgger fallido ya era una politica por unidad. Queda el argumento de
  eficiencia (94 % de ceros). Aviso honesto: RogueNet **iguala** a IMPALA en
  Procgen; la ganancia es de computo, no de eficiencia de muestra a igual
  expresividad.
- **ELO sobre el pool de rivales.** El termino de victoria debe ser `res - E`
  con `E = 1/(1+10^((R_riv-R_nos)/400))`, no `2*res-1`. Media cero bajo
  calificaciones correctas (no sesga) y **no satura en ningun peldano**.

---

## 4. Escenarios de juego

### Partidas cortas: si, pero >= 15 dias

Tiempos del motor:

```
WHEAT/CARROT  primer rendimiento dia  2     GOOSE  dia 4, intervalo 1
TOMATO                            dia  8     SHEEP  dia 6, intervalo 3
MELON/STRAWBERRY                  dia 10     COW    dia 8, intervalo 2
```

A **10 dias** COW da 1 cosecha, MELON ninguna, y el macro actual
(`cultivo=0.832` -> MELON) deja de significar lo que significa: es **otro
juego**. A partir de **~15 dias** todo amortiza al menos una vez.

**Clave:** una partida corta **completa** no es lo mismo que truncar una larga.
El terminal es liquidacion real y exacta, no Phi optimista — asi que esquiva
justo el problema del signo invertido.

### Escalera por tope de peones (implementada)

Atenuar NO sirve: al **80 %** de sus acciones v48 cae de 179.514 $ a **312 $**
(son ejecutores de planes acoplados, no politicas reactivas). Limitar peones si,
porque su propia logica se dimensiona sola.

```
nivel 1 (tope 3):  heuristica 64 273  rival   3 638   3/3 ganadas
nivel 2 (tope 5):  heuristica 50 233  rival  27 360   3/3 ganadas
nivel 3 (tope 8):  heuristica 34 216  rival  49 189   0/3
nivel 4 (entero):  heuristica 32 083  rival 134 189   0/3
```

Por debajo de tope 3 el publico se derrumba (tope 2 -> 1 $).

### Pool con diversidad de COMPORTAMIENTO

Hoy la escalera son 4 copias de v48: diversidad de fuerza, cero de
comportamiento. Pool objetivo: **3 comportamientos x 4 topes = 12 rivales**
(`v48`, `v16-rc5`, `2945`; `shop-router` esta roto), mas instantaneas propias.
Medido contra pasivo:

```
                tope3   tope5   tope8  entero
v48-fast-routes 16 049  43 094  80 932 176 422
v16-rc5          2 392  29 142  58 686 169 072
2945             (revienta publico_con_tope: formato de orden distinto)
```

### Reutilizacion de muestras

Hoy: 48 entornos x 720 turnos = **34.560 pasos de motor -> 4 pasos de
gradiente** (4 epocas a lote completo, sin minilotes, y se tira el lote). PPO es
on-policy, por eso no hay replay. Es la ineficiencia de fondo.

---

## 5. Orden de trabajo

Por informacion por minuto, no por elegancia. Cada etapa con criterio numerico.

**E0 — Confirmar el critico arreglado.** `critico_r2` en MLflow sobre
`backbone6`. **Exito: > +0,4 en 50 updates.** Si sigue negativo, correr la sonda
ridge (`z -> retorno`, cerrada, 5 min): R2 >= 0,50 -> la culpa es la cabeza;
< 0,10 -> el tronco, y hace falta critico con tronco propio (IDAAC/PPG).

**E1 — Curriculo corto + topes progresivos.** 15 dias, tope de peones nuestro y
del rival subiendo juntos. **Exito: >= 3x mas episodios/hora y que la politica
entrenada a 15 dias no pierda contra la entrenada a 30 en partidas de 30.**
Riesgo nombrado: los cultivos lentos no amortizan y el macro cambia de
significado.

**E2 — ELO sobre el pool de 12.** Sustituir `2*res-1` por `res - E`. **Exito:
que la tasa de victoria se mantenga en 0,3-0,7 sola**, sin ajustar el nivel a
mano.

**E3 — Ventajas pareadas.** Solo cuando E0 este verde. Con valor terminal
aprendido en vez de Phi. **Exito: |t| del efecto plantar/no-plantar >= 8 con
1/5 del computo actual.**

**E4 — Entidades.** Solo si E1-E3 dejan de mover la aguja. Es optimizacion.

---

## 6. Riesgos

- **El indicador que mas ha enganado es el acuerdo con el experto.** Dos veces
  cerca del 100 % con dinero cercano a cero. No volver a usarlo como criterio.
- **Leer derivadas con pocos puntos.** Me ha pasado tres veces en una sesion.
  Minimo 4-5 medidas antes de declarar meseta o giro.
- **Cambiar dos cosas a la vez.** Paso con sigma y proporcion de lr: la mejora
  no se pudo atribuir.
- **La trampa de la hora 0 lleva 7 apariciones.** Cualquier cosa que muestree
  estado tiene que declarar en que hora lo hace.
- **Cifras sin rival.** Ver la cabecera de este documento.

---

## 7. Noche del 2026-09-21: tres bugs, una rejilla calibrada, y un rival que no existia

Regla de casa nueva, ganada a base de perderla: **antes de construir un
escenario sobre una cota, comprobar que la cota es ALCANZABLE.**

### 7.1 Bugs que invalidaban medidas

| bug | efecto | arreglo |
|---|---|---|
| carga tolerante silenciosa | `N_GLOBAL` 88->90 dejo `mundo.glob_enc.0` y `mundo.resumen.0` **al azar** en todo `backbone*.pt` y `e2e_ops*.pt`. Sintoma: el mismo ckpt con la misma semilla daba resultados distintos en cada proceso | `kagsym/migrar_ckpt.py`: migra insertando columnas de cero en su sitio; `carga_estricta` revienta, `carga_tolerante` nombra lo que queda al azar. Aplicado en los 6 sitios |
| `--kl-max` 1.8e-4 | en unidades de antes de normalizar el KL por dimension. El KL sano es 1e-2..4e-2, o sea **90x el umbral**: las 4 epocas abortaban tras la primera SIEMPRE, y el controlador estrangulaba el lr hasta 6.6e-6 | defecto 0.12, medido por A/B (ret 8.54 contra 5.84 en 11 updates). Guarda que aborta si `kl_max < kl_objetivo` |
| `_fast_copy(obs)` por turno | `ejecutor.py:197` copiaba la observacion entera cada turno para `self.prev`, que solo consume `_update_gate`, que se rinde si no hay modelo de rival. Ningun camino de entrenamiento pasa `rival=` | guardado tras `if self.rival is not None`. **1,16x** a 30 dias, medido con A/B en el mismo proceso |

### 7.2 El rival no existia fuera de la competicion

Medido, dinero de `v48-fast-routes` jugando solo:

```
24h x 30d   60 426 $        13h x 13d      30 $
13h x 21d        0 $         8h x 13d       0 $
 8h x 21d       29 $         6h x  6d       7 $
```

Los agentes publicos son **reproductores de cinta** indexados por paso, con la
cinta construida para 719 pasos a 24 h/dia (`the-2945...py:221`, "route-replay
chassis", `router(observation, step, state_dict)`). Fuera de esa escala la cinta
se desincroniza y el agente acaba sin nada.

Consecuencias: en toda la escalera reducida el termino de victoria (+-1) era
**gratis**, `win_rate` no significaba nada, y el mercado compartido no lo vaciaba
nadie -o sea que se entrenaba a granjear en un mundo vacio para luego competir en
uno lleno-. Arreglo: `--rival-propio` pone NUESTRO ejecutor con el macro
calibrado de esa escala como rival en los peldanos reducidos, y deja a v48 en la
escala de competicion, que esta exenta por codigo.

Esto tambien explica por que imitar al experto nunca transfirio (BC 99,4 % ->
3 036 $): imitar una cinta es imitar un calendario, y una cinta que no reacciona
no puede ensenar a reaccionar.

### 7.3 Rejilla CALIBRADA (`runs/ligas/calibra_rejilla.py`)

Techo ALCANZABLE por CEM sobre el macro con verbos de la heuristica, no la cota
analitica. Valida: reproduce 6h x 6d = 3 241 $ contra los 3 237 que costaron 150
iteraciones.

```
  h   d  turnos  alcanzable  x_inaccion   % de la cota
  2  13      26       3 000       1.000          16 %
  3   8      24       3 000       1.000          44 %
  5   5      25       3 000       1.000          54 %
  5   8      40       3 000       1.000          30 %
  5  13      65       3 220       1.073          13 %
  6   6      36       3 241       1.080          45 %    <- L0 de toda la vida
  8   8      64       3 744       1.248          26 %
  5  21     105       5 444       1.815          18 %
  8  13     104      15 340       5.113          55 %    <- el minimo viable REAL
  8  21     168      18 912       6.304          40 %
 13  13     169      26 515       8.838          61 %
 13  21     273      30 327      10.109          40 %
```

Dos lecciones:

1. **Las cotas analiticas mienten a pocas horas.** `2h x 13d` tiene cota 6,23x y
   alcanzable 1,000x: con dos acciones por unidad y dia, moverse se come el
   presupuesto entero y la relajacion que ignora distancias es fantasia. Una
   rejilla construida sobre esas cotas entreno 145 updates hasta **0,82x** -o
   sea destruyendo valor- antes de que lo detectara.
2. **El minimo viable no es pequeno: son 104 turnos, `8h x 13d`.** Ni 9 turnos
   ni 36. Hacen falta >= 13 dias para que la cadena de produccion cierre y >= 8
   horas para que una unidad pueda moverse Y actuar. El acantilado del dia 12 al
   13 que llevabamos sin explicar aparece aqui como 1,248x (8d) -> 5,113x (13d).
   Y L0, la base de la liga desde siempre, solo tenia un **8 %** de margen.

### 7.4 Arquitectura: medida y NO adoptada

Sonda supervisada, 17 001 transiciones, 40 epocas con coseno, base 0,521:

```
cnn-sep (3x3 separable)   26,8 MFLOPs  0,977      convnext    48,8  0,977
cnn-sep7 (7x7)            32,9         0,978      CNN densa  182,7  0,979
entidades (transformer)   90,5         0,974      deepsets    11,6  0,904
                                                  MLP          3,4  0,896
```

La mezcla que falta es **geometria local**, no contexto global. La atencion no
aporta y ademas no era la palanca (a T=101, D=128 el termino por pares es el
12 % del coste). La CNN no era cara por ser CNN sino por usar 3x3 **densas**.

No se adopta: reloj **1,03x** (el tronco no es el cuello de botella), memoria GPU
**+25 %** por muestra (las activaciones dominan), y destilarla cuesta **-913 $ a
-1,9 ee** en duelo emparejado a 30 dias. Disponible en `MundoConfig(conv="sep")`.

### 7.5 La run

```
--rejilla "8,13,0;13,13,0;13,21,0;24,30,0;8,21,0" --rival-propio
--envs 88 --procs 11 --kl-objetivo 0.02 --init runs/ligas/macro_13h13d.npy
```

Mas: ventaja normalizada POR PELDANO (sin eso el peldano de 720 turnos se lleva
todo el modulo del gradiente, porque `escala` del moldeado es una constante y el
potencial terminal vale ~20 alli y ~2,5 a 105 turnos), reparto de episodios por
COSTE (si no, diez trabajadores esperan al lento en cada update), y `x_inaccion`
por peldano en consola y en MLflow -el promedio entre escalas no significa nada-.

### 7.6 El macro es una funcion CONSTANTE del estado

Diagnosticado en el update 110 de la run de rejilla. La red emite el mismo
vector macro a `8h x 13d` que a `24h x 30d` -distancia media **0.000**- y ese
vector unico vale 12.625 $ a escala de competicion contra los 40.468 del macro
que el CEM encontro para esa escala: **3,2x de dinero** tirado.

Medido el porque, en cadena:

```
dims de horizonte      8h x 13d [0.144, 1.0]   24h x 30d [1.0, 1.0]
global: difieren        2 de 90 dims        grid: 0 de 5000 celdas
z (256):                coseno 0.999987     |diff| medio 0.00697
macro_mu.weight:        |W| medio 0.0016    max 0.015
contribucion de z al logit del macro:  0.00113  (los logits van de -9 a +9)
```

El problema no es solo la escala: **`macro_mu.weight` es numericamente cero**, o
sea que el macro no depende del estado EN ABSOLUTO. La red emite catorce numeros
fijos, que es exactamente una politica guionizada sin adaptacion al dia, al
dinero ni al mercado. Dos causas que se suman:

1. `inicializa_macro_en` hace `nn.init.zeros_(macro_mu.weight)` para que el
   macro inicial sea EXACTAMENTE el vector dado. La cabeza nace constante y
   tiene que viajar desde cero, igual que la del verbo (documentado: ~4.000
   pasos con el sigma de entonces).
2. El horizonte son 2 de 90 dimensiones globales, pasan por `symlog` y dos
   capas, y a `z` llegan atenuadas a 0,007 de diferencia media.

**Intento fallido, registrado para no repetirlo:** ajustar `macro_mu` por minimos
cuadrados sobre (z, logit del macro calibrado de cada peldano) da r2 = 0,90 y
politicas PEORES (1.500-2.450 $ contra los 5.000-6.900 de la red actual). Varias
dimensiones del macro viven en los extremos (0,0 y 0,99), sus logits valen +-9, y
el 10 % de varianza residual de un mapa lineal son +-2,8 de logit: suficiente
para invertir el comportamiento. `runs/ligas/macro_consciente.py` queda como
registro del negativo.

Siguiente: A/B de `--lr-cabezas` (3e-4 / 1e-3 / 3e-3) midiendo `|W|`, la
distancia entre macros de escalas distintas y el dinero por peldano. Si la cabeza
no llega a despegar, la alternativa preparada es darle al macro una via DIRECTA
al horizonte -`macro_mu` sobre `cat([z, dims_de_horizonte])`- en vez de
esperar a que el tronco propague 2 de 90 dimensiones.

**A/B de `--lr-cabezas`** (60 updates cada uno desde el mismo checkpoint):

```
lr-cabezas    |W|      dist escalas   24h x 30d   13h x 13d
3e-4        0.00189      0.0004          6.07        4.47
1e-3        0.00239      0.0005          5.99        4.15
3e-3        0.00386      0.0006          6.38        4.93
```

Diez veces el paso duplica |W| y **no mueve la distancia entre escalas**: lo que
ata es el presupuesto de KL, no el lr -el controlador multiplica los dos grupos
por 0,7 cuando el KL sube, asi que devuelve lo que le des-. Misma trampa que con
sigma. Se adopta 3e-3 por el dinero, no por la dependencia del estado.

Y amplificar W a mano (x10/x30/x100) DERRUMBA el dinero -14.667 a 4.038 $ en
8h x 13d- sin crear dependencia de escala: la W aprendida es ruido en direcciones
que no correlacionan con el horizonte. **El macro constante es el optimo de PPO,
no un fallo**: el macro vive en los extremos y cualquier dependencia del estado
que no sea exactamente la correcta cuesta dinero.

Lo que queda por decidir, y hay experimento corriendo: si la red no puede
condicionar la estrategia por escala, mezclar escalas compra eficiencia de
muestra al precio de un macro de COMPROMISO justo donde importa. `mezcla` contra
`solo 24h x 30d`, igualdad de reloj, mismo termometro externo.

### 7.7 Macro y verbo NO son separables, y el modo `ops` tiene un techo

Las cuatro combinaciones cruzadas, 24h x 30d, 8 semillas fijas, rival v48:

```
macro de la red  + verbo de la red         24 220 $   8,07x   <- par emparejado
macro calibrado  + verbo heuristico        42 249 $  14,08x   <- par emparejado
macro de la red  + verbo heuristico         8 063 $   2,69x
macro calibrado  + verbo de la red            419 $   0,14x
macro calibrado  + verbo recien nacido         24 $   0,01x
```

**Solo funcionan los dos pares.** La cabeza de verbo de la red esta co-adaptada a
su propio macro -le suma 3x, de 8.063 a 24.220- pero con un macro bueno colapsa
por debajo de la inaccion: aprendio a compensar un macro malo. Y simetricamente,
el macro del CEM esta co-adaptado a los verbos de la heuristica. No hay trasplante
posible en ninguna direccion, asi que no se puede arrancar la red desde los
42.249 pegando las dos mitades buenas.

**Y el modo `ops` no llega a la heuristica ni con un ORACULO.** Dandole a la red
el verbo que `tile_task` elegiria y su valor exacto, replicando el lazo del
ejecutor -misma capacidad libre inicial, mismo decremento al plantar-:

```
heuristico (modo residuo)          42 249 $   14,08x
ORACULO ops: verbo + valor         26 536 $    8,85x   -37 %
ORACULO ops: verbo, valor plano     6 239 $    2,08x
```

Diagnostico del oraculo sobre 143.800 casillas: **`no encaja 0,0 %`**, o sea que
`tile_options` SIEMPRE contiene lo que `tile_task` elige y el mapeo de verbos es
exacto. El hueco del 37 % no es de enumeracion y no esta aislado todavia.

Pero la tercera fila dice algo que cambia la prioridad: **el canal de VALOR
aporta 4,25x y el verbo es lo secundario** (26.536 contra 6.239 con valor plano).
La asignacion humgara se ordena por valor. Llevamos la sesion optimizando 15
logits por casilla cuando lo que decide es el UNICO numero de valor por casilla.

Decision que queda para la manana, con los numeros encima: seguir e2e en `ops`
-que es el diseno pedido y esta aprendiendo, aunque plano en ~19.000- o volver a
`--modo residuo`, que arranca en los 42.249 de la heuristica y solo puede subir.
El argumento a favor de `residuo` ya no es de comodidad: es que `ops` tiene un
techo medido por debajo del punto de partida de `residuo`.

**Donde NO esta el hueco del 37 %** (descartado con medicion, 2026-09-21):

```
vocabulario de verbos   0,0 % de tareas de `tile_task` que `tile_options` no enumere
                        0,0 % de verbos del oraculo ausentes al reenumerar el ejecutor
valor centinela         identico con -1,72 y con -4,85e8 (26.306 $ las cuatro veces)
```

**Donde SI se ve el sintoma:** contando las tareas que construye cada modo,

```
heuristico     11,7 tareas/turno, 11,7 con valor>0, valor medio 157 $
oraculo ops    17,4 tareas/turno,  6,1 con valor>0, valor medio 274 $
```

El camino `ops` pone MAS casillas en la tabla de tareas pero solo la MITAD con
valor positivo, o sea que las unidades tienen la mitad de cosas que merezca la
pena hacer. La causa de esa mitad no esta aislada. Queda como la pregunta
abierta mas importante del proyecto: si se cierra, `ops` deja de tener techo.

### 7.8 El techo de `ops` era un argumento perdido, y esta arreglado

Aislado comparando las DOS tablas de tareas construidas sobre EL MISMO estado
(719 turnos, 8.441 pares casilla-tarea):

```
identicas                 96,8 %
casilla en una sola        0,0 %
mismo sitio, VERBO dist.   3,2 %   <- todo el hueco
mismo verbo, VALOR dist.   0,0 %
```

Y el 3,2 % era, literalmente:

```
heuristica  ['PICKUP', 'WHEAT', 2]
ops         ['PICKUP', 'WHEAT', 1]
```

`OPS_VOCAB` trata `PICKUP_WHEAT` como un verbo SIN argumento, asi que
`tile_options` emitia siempre 1 mientras `tile_task` coge
`min(hambrientos, trigo_en_cobertizo)` y `min(fert, fertilizables, 4)`. Una
unidad trayendo un trigo por viaje hace el DOBLE de viajes al cobertizo, y el
ganado -que es la economia entera a esta escala- come a media velocidad.

Arreglado en `tile_options`: la cantidad la fijan la necesidad y el stock, no es
una decision estrategica, asi que va del lado simbolico igual que la legalidad y
el enrutado. La red sigue eligiendo el VERBO.

```
                       antes      despues
ORACULO ops           26 536 $   40 763 $     -37 % -> -3,5 %
casillas con tarea      26,2 %     46,7 %
```

Antes de dar con ello se descartaron, con medicion: el vocabulario de verbos
(0 % de huerfanas con contador decreciente, 0 % de verbos perdidos al reenumerar),
la magnitud del valor centinela (identico de -1,72 a -4,85e8) y la completitud de
la enumeracion. Lo que lo encontro fue **diffear las dos tablas sobre el mismo
estado** en vez de comparar resultados de partidas.

Consecuencias: `ops` YA NO tiene techo por debajo de la heuristica, asi que el
argumento a favor de `--modo residuo` se debilita mucho; la run A se relanzo con
el arreglo (`runs/A_ops2.log`); y clonar el oraculo en la cabeza micro pasa a ser
un arranque legitimo, que es lo unico que queda para llegar a los 42.249 con la
arquitectura e2e -el trasplante sigue cerrado: macro calibrado + verbo de la red
da 419 $ tambien despues del arreglo-.

### 7.9 Clonar el oraculo: tres intentos, los tres peores cuanto mejor el ajuste

Con el techo de `ops` ya levantado (7.8), clonar el oraculo en la cabeza micro
pasaba a ser un arranque legitimo: el objetivo vale 40.763 $. Resultado:

```
                                   mse(valor)  acierto(verbo)   jugando
tronco congelado, centinela -20       22,61        0,9948        5 086 $
tronco congelado, centinela -3          1,77        0,9943       12 080 $
tronco suelto,    centinela -3          0,21        0,9990        2 278 $
```

El centinela importaba mucho: el oraculo marca las casillas sin tarea con -20 en
symlog -que son -4,85e8 $- y regresar sobre eso se come el error. Comprimirlo a
-3 (medido aparte: la magnitud NO cambia el juego, 26.306 $ identico de -1,72 a
-4,85e8) baja el mse 12x y el dinero sube 2,4x.

Pero soltar el tronco baja el mse otras 8,5x, sube el acierto al 99,90 %... y el
dinero cae 5x. **Cuarta vez en el proyecto que mas acierto de imitacion da menos
dinero**, y ahora con tres maestros distintos -la cinta publica, la valoracion
heuristica y el oraculo exacto-. El patron ya no es una anecdota: la politica se
ejecuta a traves de una asignacion combinatoria sobre 720 turnos, y un error
pequeno de ORDEN se compone. Via abandonada.

### 7.10 Estado al cerrar la noche

Termometro externo (determinista, 24h x 30d contra v48):

```
B (residuo, upd 170)   37 380 $  12,46x   margen -68,6 %
A (ops,     upd  50)   17 446 $   5,82x   margen -88,8 %
guionizada             40 743 $  13,58x   margen -68,4 %
```

Y EMPAREJADO contra la heuristica, aislando solo el residuo (16 semillas, mismo
macro en los dos lados): **B 45.263 $ contra 40.743, +4.519 +- 3.804 (+1,2 ee),
gana 9 de 16**. Nominalmente +11 % y todavia no distinguible; hace falta mas
entrenamiento o mas semillas.

El macro de B ha derivado 0,075 de media respecto al del CEM y la deriva es
BUENA: con el ejecutor heuristico puro, el macro de B vale 43.700 $ contra los
42.249 del CEM. PPO si aporta en la capa de estrategia cuando parte de un
optimo del CEM, aunque no lo encuentre solo.

> **REFUTADO el 2026-09-21** por `runs/ligas/dos_sillas.py` (24 semillas
> emparejadas, determinista, AMBAS sillas, 24h x 30d contra v48):
> la red hace **36.695 / 38.489 $** y el ejecutor con el macro del CEM hace
> **38.011 / 41.226 $**. PPO **no** le gana a CEM sobre los mismos 14 numeros:
> esta dentro del ruido, y si acaso por debajo. El "+4.519 +- 3.804 (+1,2 ee)"
> de arriba es exactamente el tamano de efecto que la §8.1 demuestra que no se
> puede resolver con 16 semillas. Se deja escrito como ejemplo del error, no
> como resultado.

### 7.11 PPO DEGRADA desde un optimo local fuerte, con la maquinaria sana

La run B (residuo) hizo pico en el update 123 y bajo durante 267 updates. No es
ruido ni es cosa del muestreo: medido con el termometro DETERMINISTA,

```
B upd 123 (mejor guardado)   39 349 $   13,12x   margen -69,0 %
B upd 390 (ultimo)           29 754 $    9,92x   margen -77,0 %
```

Y toda la maquinaria estaba sana en ese tramo:

```
critico r2          0,88    (techo medido 0,900)
kl/dim              0,023   (objetivo 0,02)
saturacion          5,4 %
epocas corridas     4 de 4
micro_w_norma       0,011 -> 0,090   la cabeza SI viaja, x8
```

O sea que no es un fallo mecanico. La explicacion que encaja: **B arrancaba
exactamente en un optimo local fuerte -residuo cero ES la heuristica- y con la
senal de ventaja debil frente al ruido, la politica se pasea y cualquier
movimiento es cuesta abajo.** `ret` tambien bajo (33,35 -> 26,27), asi que PPO ni
siquiera mejoraba su propio objetivo: es varianza, no desalineacion de la
recompensa.

Eso NO se ataca con pasos mas pequenos -solo frena el paseo- sino con menos
varianza de gradiente, o sea mas episodios por lote. Relanzado como `B2` desde
el checkpoint del pico con 72 entornos en vez de 48, region de confianza mas
apretada (kl 0,008) y minilotes para el critico. `A3` sigue en `ops` con el resto
de la maquina.

Y una leccion de proceso: **guardar el mejor por retorno salvo el experimento**.
Sin `runs/B_residuo.pt` (upd 123) se habrian perdido 267 updates de deriva y el
unico punto bueno con el.

**Y el resultado emparejado, que es el sensible, dice que NO la batimos:**

```
B  upd 170   heuristica 40 743 $   red 45 263 $   +4 519 +- 3 804  (+1,2 ee)  9/16
B2 upd 200   heuristica 40 743 $   red 38 279 $   -2 464 +- 3 017  (-0,8 ee)  5/16
```

Los dos dentro del ruido y con el signo cambiado, asi que la lectura honesta es
que el residuo **orbita** la heuristica, no la supera. Con 16 semillas y un error
tipico de ~3.000 $ no se detecta nada por debajo del 15 %: para afirmar una
mejora hacen falta bastantes mas semillas emparejadas, no mas updates.

Herramienta para eso: `runs/ligas/pareado.py` (mismo macro en los dos lados, asi
que aisla SOLO el residuo).

**La degradacion se repite con mas lote y region mas apretada.** B2 arranco del
pico de B (upd 123) con 72 entornos en vez de 48, kl 0,008 en vez de 0,02 y
minilotes para el critico. Aguanto hasta el update ~220 -frente a los ~125 de
B- y volvio a caer:

```
B    pico upd 125   36 165 $  ->  upd 375   30 090 $
B2   pico upd 220   37 292 $  ->  upd 305   32 638 $
```

O sea que bajar la varianza de gradiente y apretar la confianza **duplica el
tiempo hasta la degradacion pero no la evita**. Dos configuraciones distintas,
mismo final: PPO se aleja del optimo de la heuristica y no vuelve.

Picos preservados: `runs/B_pico_upd123.pt` (39 349 $ determinista) y
`runs/B2_pico.pt` (upd 209). La maquina pasa entera a la run `ops` (`A4`), que
es la unica que sube de forma sostenida -19 816 -> 23 821 $ en 300 updates- y
cuyo techo, tras el arreglo de la cantidad, es 40 763 $.

## 8. CEM sobre el residuo: NEGATIVO LIMPIO (2026-09-21)

`runs/ligas/cem_residuo.py`: CEM sobre los **129 parametros del residuo del
valor** -una conv 1x1 sobre 128 canales mas el sesgo-, arrancando en CERO, que
es la heuristica exacta (residuo multiplicativo exp(0) = 1), a 24h x 30d contra
v48.

Validado en `runs/ligas/valida_residuo.py` con 16 semillas que el CEM NO ha
visto (500-515), emparejado -mismo macro y mismas semillas en los dos lados, asi
que aisla SOLO el residuo-:

```
retenido        heuristica    CEM       delta          ganadas
500-515           33 975    40 199   +6 223 +- 2 635    12/16
700-715           39 172    40 992   +1 819 +- 3 981    10/16
900-931           32 059    33 835   +1 776 +- 2 178    23/32
---------------------------------------------------------------
AGRUPADO (64)                        +3 315 +- 1 547    45/64
```

Y un CUARTO retenido, medido despues con el protocolo corregido:

```
2000-2011         40 023    37 179   -2 844             -
```

**La mejora NO esta establecida.** Tres conjuntos positivos y uno negativo, con
sd de ~14.000 $ por semilla: ninguno resuelve un efecto del orden de 3.000.

DOS CORRECCIONES, las dos mias y las dos por lo mismo. Primero reporte +18,3 %
del retenido 500-515, que era la suerte de ESE conjunto -el error contra el que
acababa de advertir a proposito del `mejor` del CEM, un nivel mas arriba-.
Despues reporte +10,3 % agrupando tres conjuntos... que eran los tres que habia
medido hasta entonces, todos positivos; el cuarto salio negativo. Contar solo
los conjuntos ya vistos es la misma seleccion otra vez.

**MEDICION DEFINITIVA, dimensionada ANTES de mirar** (200 semillas emparejadas
frescas, 3000-3199, 400 partidas):

```
heuristica   36 320 $      CEM   36 083 $
delta         -237 +- 942  (-0,3 ee)     gana 104/200 (52 %)
```

**Cero.** El efecto esta dentro de +-942 $, o sea +-2,6 %. El CEM sobre el
residuo del valor NO aporta nada.

### 8.1 La leccion, que es el verdadero resultado

Reporte **+18,3 %** (16 semillas), luego **+10,3 %** (agrupando los tres
conjuntos que habia mirado, todos positivos), y la verdad es **0 %**. Las dos
veces el mecanismo fue el mismo: medir, mirar, y decidir a partir de lo visto.

El numero que lo explica: la sd EMPAREJADA es ~13.300 $ por semilla. Con 16
semillas el error tipico es 3.300 -del tamano del efecto que buscabamos-, asi
que cada conjunto pequeno es esencialmente un sorteo.

**Regla para este proyecto, a partir de ahora:**

```
efecto a detectar   semillas emparejadas necesarias (2 ee)
        20 %  (7 200 $)        ~15
        10 %  (3 600 $)        ~55
         5 %  (1 800 $)       ~220
         2 %  (  700 $)     ~1 450
```

Cualquier afirmacion por debajo del 10 % necesita >=200 semillas emparejadas, y
el tamano de muestra se fija ANTES de ver el resultado. `runs/ligas/pareado.py` y
`runs/ligas/valida_residuo.py` hacen el emparejado; lo que faltaba era el
dimensionado.

**El cuello de botella del bucle de investigacion de este proyecto es la
varianza del juego, no los algoritmos.** Una noche entera de conclusiones se
apoyo en muestras de 8-16 semillas.

Y ojo con que la evidencia fuerte es el TEST DE SIGNOS, no la media: gana en el
70 % de las semillas con p = 0,0008, mientras la media tiene un error tipico de
1.547 $ sobre una mejora de 3.315. Con sd de ~14.000 $ por semilla, la media es
demasiado ruidosa para resolver un 10 % aunque sea real.

Por que funciono esto y no PPO, con todo lo medido delante:

1. **El canal de VALOR es lo que manda**, no el verbo: con el oraculo, verbo +
   valor da 40.763 $ y verbo con valor plano 6.668. La asignacion humgara se
   ordena por valor. El CEM ataca exactamente esos 129 numeros.
2. **PPO se aleja del optimo y el CEM no puede.** El CEM se queda con la elite y
   el incumbente compite cada generacion, asi que el suelo esta garantizado por
   construccion; PPO da 4 pasos por lote en la direccion de una ventaja ruidosa
   y se pasea cuesta abajo (medido dos veces, -17 % y -13 % desde el pico).
3. **Cuarta victoria del CEM sobre el gradiente en este proyecto**, tras el macro
   global, el verbo en L0 y el macro por escala. Ya no es anecdota: en este
   problema, busqueda global sobre pocos parametros bate a gradiente sobre
   muchos.

**Cuidado con el numero que imprime el CEM.** `mejor` es el maximo sobre 32
candidatos y 5 semillas -maldicion del ganador- y llego a marcar 55.456 $ cuando
la validacion fuera de muestra daba 40.199. La `elite` esta menos inflada pero
tambien es seleccionada. El unico numero que vale es el emparejado con semillas
no vistas.

---

## 9. 2026-09-21 (manana): la barrera es una ESCALERA, y la medida definitiva

### 9.1 El paisaje de recompensa es constante a trozos

`runs/ligas/valle.py`. La inaccion es un estado absorbente que vale **3.000 $
exactos a cualquier horizonte** (macro todo-ceros, sd 0, verificado). Recorriendo
el segmento recto del macro de inaccion al optimo del CEM, 48 semillas por punto:

```
8h x 13d    t=0,000 .. 0,925   ->  0,98 - 1,16x        MESETA PLANA
            t=0,945 -> 0,950   ->  1,131x -> 5,106x    peones 8 -> 9
5h x 21d    t=0,000 .. 0,700   ->  1,000x              plano en la inaccion
            t=0,969 -> 0,972   ->  1,074x -> 1,795x    animales 1 -> 2
8h x  8d    t=0,950 -> 0,975   ->  1,111x -> 1,246x
```

> **CORREGIDO el mismo dia, dos veces.** Lo de arriba describe UNA CURVA -el
> segmento mueve las 14 dimensiones a la vez-, no el paisaje, y ademas atribuye
> mal el salto.
>
> (a) **El acantilado es el CULTIVO, no los peones.** Rejilla de 0,0025 con 6
> semillas: en t=0,9425 -> 0,9450 el dinero va de 1,139x a 4,841x y lo unico que
> cambia al decodificar es `cultivo_objetivo`: **CARROT -> MELON**, porque
> `int(macro.cultivo * len(por_valor))` (macro.py:153) cruza 0,75. `peones 8->9`
> ocurre DESPUES, en t=0,9475, y aporta 0,26x. El peon es el 7 % del acantilado.
>
> (b) **Por EJE el paisaje no es plano.** `runs/ligas/por_eje.py`, 12 semillas,
> 17 puntos, desde el optimo de 8h13d:
> ```
> peones     1.00 1.50 1.37 2.29 2.75 3.16 3.59 3.59 4.83 4.83 5.10 ...  10 SUBIDAS
> cultivo    1.00 0.39 0.39 0.39 1.04 1.04 1.08 1.08 1.03 ... 1.03 5.10  la aguja
> casillas   0.70 1.16 4.30 5.10 4.98 ... 2.37 0.00 0.00 0.00             y un precipicio
> animales   5.10 2.73 1.85 1.73 1.53 ...                                monotono a peor
> venta, fertilizar y las 6 prioridades: 5.10 constante, UN nivel
> ```
> `peones` tiene **diez escalones de mejora**: hay gradiente de sobra. La aguja
> es `cultivo` y solo `cultivo`, y plantar el cultivo equivocado da **0,39x**,
> peor que no hacer nada.
>
> (c) **8 de 14 dimensiones estan muertas** en este peldano. Es el bug del
> Trick 6 -dimensiones que no pueden cambiar la accion entrando en el cociente
> de importancia- pero en la cabeza MACRO, donde nunca se enmascaro.

Causa, verificada en el codigo: **todas** las dimensiones del macro se colapsan
a entero en `kagsym/macro.py` — `int(round(peones*15))` (l.83),
`int(casillas*...)` (l.106), `int(animales*...)` (l.126),
`int(round(venta*...))` (l.136), `int(cultivo*...)` (l.153). El gradiente vale
**cero casi en todo punto, por construccion**.

Esto explica hacia atras lo que no se explicaba:

- **CEM cruza** porque muestrea con sd 0,28 y cae al otro lado por sorteo.
- **PPO no cruza y ademas empeora**: su region de confianza ENCOGE la sigma al
  converger, destruyendo lo unico que encontraba el escalon. El estancamiento es
  estructural, no de hiperparametros.
- ~~**`macro_mu.weight` es numericamente cero**~~ **FALSO, y §7.6 entera cae
  con ello.** Medido: `runs/A3.pt` tras 485 updates da absmax 6,08e-2 y sd
  7,67e-3, **el mismo orden que `micro.weight`** (sd 6,90e-3), que es la cabeza
  que si funciona. Quien midio "cero" midio una red recien inicializada:
  `inicializa_macro_en` hace `nn.init.zeros_(self.macro_mu.weight)` A PROPOSITO
  (`kagsym/redes/mundo.py:210`), y se llama en seis sitios. La cabeza macro SI
  depende del estado; lo que pasa es que la dependencia que aprende es peor que
  un vector fijo. Es otro problema, y mas dificil.
- Explicacion alternativa del fallo macro, NO refutada y mas simple que la
  cuantizacion: `eps` se sortea UNA vez por entorno y solo se refresca al cerrar
  episodio (`entrenar_e2e.py:535` y `609`). Con `--dias 30` eso son **48
  muestras macro independientes por update** contra 48x30x100 = 144.000 sorteos
  del micro, desde el mismo tronco y el mismo optimizador: **3.000x** de
  diferencia en tamano de muestra efectivo.
- El control ya estaba corrido dentro del propio proyecto: la cabeza de **verbo
  es CATEGORICA** y funciona; la de **macro es gaussiana sobre reales** y es una
  funcion constante. Mismo tronco, mismo entrenador, resultados opuestos.

Hipotesis derivada, **no medida todavia**: la barrera es autoinfligida por la
parametrizacion, y una cabeza categorica sobre el entero la elimina. Test barato
y discreto (cruza / no cruza, binomial, sin el problema de la sd de 13.300 $).

### 9.2 La medida definitiva: ambas sillas, y cero victorias

`runs/ligas/dos_sillas.py`. 24 semillas emparejadas, determinista, 24h x 30d
contra `v48-fast-routes`, NUESTRO agente en las dos sillas:

```
                        silla   nosotros       sd      rival    margen   ganadas  x_inac
red (PPO)                   0     36 695   16 099    123 032    -70,2 %    0/24    12,23
red (PPO)                   1     38 489   14 769    123 333    -68,8 %    0/24    12,83
ejecutor + macro a mano     0      7 951    3 401    158 135    -95,0 %    0/24     2,65
ejecutor + macro a mano     1      8 219    3 390    157 370    -94,8 %    0/24     2,74
ejecutor + macro CEM        0     38 011   18 560    123 037    -69,1 %    0/24    12,67
ejecutor + macro CEM        1     41 226   19 086    125 474    -67,1 %    0/24    13,74
inaccion                    0      3 000        0    145 080    -97,9 %    0/24     1,00
```

Tres cosas, ninguna comoda:

1. ~~**Cero victorias, nunca se ha batido a un agente publico.**~~ **FALSO.**
   Cierto solo contra el rival SIN TOPE. Contra la escalera por tope del §4,
   con el vector del CEM que ya estaba en disco y sin entrenar
   (`runs/ligas/vs_topes.py`, 12 semillas):
   ```
   tope v48   nosotros      v48     margen   ganadas
          3      62 556    4 229  +1379,2 %   12/12
          5      52 174   27 882     +87,1 %   12/12
          8      49 106   58 765     -16,4 %    5/12   <- cruce real
         11      39 990  110 859     -63,9 %    0/12
       None      37 673  124 276     -69,7 %    0/12
   ```
   La escalera por tope FUNCIONA y nunca se habia corrido el mejor vector propio
   contra ella. El peldano informativo es **tope 8**, que es un cruce limpio.
2. **PPO no le gana a CEM** sobre los mismos 14 numeros. Condicionar en el estado
   no ha comprado nada medible sobre un vector fijo. Toda la tesis del proyecto
   descansa en que deberia comprarlo, y eso sigue sin medirse a favor.
3. **Efecto de silla: nulo.** La silla 1 es mayor en los cuatro brazos, por
   1.794-3.215, contra un error tipico pareado de ~3.300-3.900. Se habia jugado
   siempre de silla 0 sin comprobarlo.

El macro escrito a mano vale 7.951 $ porque **planta** (`casillas=0.5`), y
plantar cuesta 41-50k a 8-11 sigma. Es el resultado estrella del proyecto
reapareciendo por su cuenta.

### 9.3 Regla nueva: el margen contra el rival esta CONTAMINADO

Mirando la columna del rival: v48 gana **123k** contra nuestras politicas
activas, **145k** contra la inaccion y **158k** contra la que planta. Nuestra
propia conducta le mueve la puntuacion al rival un **28 %**, porque al vender
menos producto sus precios marginales se quedan altos.

Consecuencia: **el margen contra el rival no es comparable ni siquiera entre
nuestros propios brazos.** La regla antigua ("nunca compares dinero medido
contra rivales distintos") se queda corta. Lo que se reporta es dinero absoluto
contra una referencia INDEPENDIENTE DE LA POLITICA, y la que tenemos es
`x_inaccion` = dinero / 3.000.

---

## 10. 2026-09-21 (tarde): el techo no era del problema, era de lo que alguien expuso

### 10.1 El hallazgo: catorce constantes que nunca entraron en una busqueda

Toda la sesion se midio contra un techo de **962 $** en la celda de trabajo
(12h x 5d, tope 3, caja 400), obtenido con 8.000 sorteos sobre 18 dimensiones
-14 del macro y 4 del mapa de valor- y **saturado**: desde el sorteo 769 hasta
el 8.000 no mejoraba ni un dolar. Se concluyo que la clase de politicas estaba
agotada.

Estaba agotado el **subconjunto que alguien decidio exponer**. Auditando las
cuatro capas aparecieron catorce constantes fijadas a mano que ninguna busqueda
habia tocado nunca:

```
mercado.py   PRESUPUESTO_MANO_OBRA 0.15   FRACCION_SEMILLA 0.5   MARGEN_PEON 3.0
             DIAS_STOCK_PIENSO 3   SAT_ALTA 0.85   SAT_BAJA 0.60
             LAND_RETORNO 2.0   LAND_CAJA 1.5
tareas.py    DESCUENTO_POR_PASO 0.82   VALOR_DIG 0.9   HORIZONTE_FERTILIZAR 3
ejecutor.py  TURNOS_CASILLA_INI 3.0   TURNOS_CASILLA_MIN 2.0
macro.py     FACTOR_RIEGO 0.5
```

Exponiendolas a la misma busqueda:

```
 18 dims (14 macro + 4 mapa)      939 $
 21 dims (+3 de mercado)        1.059 $
 30 dims (+12)                  1.201 $
 32 dims (+14)                  1.258 $        +34 % sobre el "techo"
```

Y los valores aprendidos contradicen los escritos de forma sistematica: **mucha
mas mano de obra y mucha menos semilla** (0.15 -> 0.40-0.85 y 0.50 -> 0.09-0.14
segun la corrida), y **la distancia penaliza mas** de lo que suponia el 0.82.
Se estaba comprando semilla que no habia manos para regar.

El docstring de `macro.py` dice que ese fichero existe porque las constantes a
ojo "violan el principio del proyecto -nada adivinado- y, peor, ocupan justo el
sitio donde deberia decidir el aprendizaje". Movio catorce de veintiocho.

**Leccion transferible: antes de medir el techo de una clase de politicas,
audita que decisiones quedaron fuera de ella.** La frontera entre "buscado" y
"congelado por un humano" no estaba escrita en ningun sitio.

### 10.2 Salvedades de medida, que son la mitad del resultado

**Presupuesto por dimension.** Las cuatro filas de arriba usan las MISMAS 2.500
muestras con dimension creciente, asi que el presupuesto por dimension cae en
cada paso. Las subidas son reales -superan la dilucion- pero sus magnitudes no
son comparables entre si. Anadiendo dos dimensiones mas que no atan en esta
celda (`GANANCIA_MAPA`, `TOPE_MAPA`) el resultado BAJA a 1.158: ahi la dilucion
gana.

**Ruido de seleccion.** El mejor de 2.500 sorteos en 32 dimensiones se elige
sobre UNA semilla. Reevaluado con semillas frescas, una corrida dio 1.255 ->
1.258 (sin maldicion) y otra 1.255 -> 1.084 (1,16x). El techo honesto es
**~1.100-1.260**, no "1.258": en la primera corrida hubo suerte.

**Dimensiones inertes.** Seis de las catorce (`dias_pienso`, `sat_*`, `land_*`)
no atan a 5 dias -no hay ganaderia viable ni caja para cuadrante-, asi que sus
valores "aprendidos" son ruido, no hallazgos. Mismo patron que las 8 de 14
dimensiones planas del §9.1.

### 10.3 Cableado: los catorce viven ya en el macro

```
Macro                14 -> 28 campos; los nuevos con defectos que reproducen
                     EXACTAMENTE los valores antiguos (verificado al 3er decimal)
parametros()         lleva los [0,1] a sus rangos reales
aplica_parametros()  los escribe en mercado/tareas/ejecutor, una vez por turno
N_MACRO              14 -> 28; la cabeza de la red emite 28
migrar_ckpt          cabeza 14->28: peso a CERO, sesgo a logit(defecto), sigma
                     a log(0.35). Verificado en un checkpoint `residuo` y en uno
                     `ops`: carga estricta, nada al azar.
compatibilidad       un vector de 14 da dinero IDENTICO (709 = 709)
cableado             cambiar solo los parametros nuevos mueve el dinero +13,6 %
```

Filas nuevas del peso a cero a proposito: un checkpoint migrado se comporta
igual que antes -las dimensiones nuevas no dependen del estado, como cuando
eran constantes- y el entrenamiento puede darles peso si le sirve.

### 10.4 Mecanismos del aprendizaje, aislados uno a uno

**Estrechar la busqueda PERJUDICA.** Dos brazos, 66 corridas, hipotesis
declarada antes, juicio con semillas frescas: ancho (muestreo uniforme) alcanza
el 99 % del techo en el **44 %** de las corridas; estrecho (CEM, que colapsa una
gaussiana) en el **18 %**. Diferencia +26 puntos +- 8 = **3,3 ee**. Y con cinco
mecanismos el orden es monotono con cuanto estrecha cada uno: azar 68 %,
escalada 50 %, escalada+region de confianza 32 %, CEM 18 %.

Consecuencia de diseno: el controlador de KL debe perseguir un **suelo**, no un
techo, y **la entropia cayendo es una alarma**, no una senal de convergencia.

**Tres cuantizaciones encadenadas matan la derivada**, cada una en su nivel:

```
1  int() en el macro        -> categoricas sobre el entero
2  argmax en el verbo       -> categorica MUESTREADA
3  el orden en el DESPLIEGUE-> entrenar y desplegar la misma politica
```

La (3) es la traicionera: sobrevive a las otras dos. Con argmax al desplegar, el
objetivo desplegado solo depende del ORDEN de los logits, asi que el gradiente
puede estar sano y no tocar nunca la politica que juegas -medido: la columna de
despliegue valio 776 durante 110 iteraciones, sin moverse-.

Arreglando las tres a la vez SI aprende (435 -> 568, +30,6 %), cosa que ninguna
por separado consiguio. Y el aprendizaje conjunto macro+micro llega a 922 en 704
episodios (524 de partida): por partes topa en 865 y 776.

**El gradiente no cruza CONJUNCIONES.** Partiendo POR DEBAJO de la inaccion
(360 $ contra 400), 1.760 episodios dan **+0,0 %**. No es el punto ciego de la
ventaja normalizada -el lote tiene varianza y |grad| esta vivo-: es que la
primera accion rentable exige peones **y** cultivo correcto **y** parcelas
alcanzables a la vez, y las mejoras marginales no cruzan conjunciones. El azar
si cruza porque **un sorteo ES un salto coordinado**.

=> **mecanismo hibrido**: muestrear por debajo del suelo, gradiente por encima.
El disparador entre regimenes es exacto y gratis: la inaccion vale la caja
inicial con varianza cero.

**Pero el hibrido no bate al muestreo puro a igual presupuesto** (929 = 929 con
1.800 episodios cada uno), y esa demostracion fallo por eleccion de escenario:
en esta celda 11 sorteos dan el 90 % del techo y 769 el 99 %, asi que se le pide
al gradiente que gane a un rival que ya ha terminado. **El banco barato sirve
para entender mecanismos y no para demostrar que aprender paga.**

### 10.5 Las cotas mienten en la direccion que las favorece

Dos veces el mismo error en un dia:

```
2.609 $  cota original            regala el desplazamiento entero
1.543 $  cobrando viaje POR VISITA y con liquidez
```

Y el "hueco de ejecucion" del 39 % (62 tareas hechas de 102 que caben) resulto
ser un artefacto de **ignorar la disponibilidad temporal** de las tareas: no se
puede regar dos veces el mismo dia ni cosechar antes de tiempo. De ahi se
dedujo un +58 a +67 % que **no es valido**. Implementado el valor de RUTA en la
asignacion -mirar la secuencia y no una tarea suelta- el resultado es **944 vs
942, +0,2 %**: la via del MIP se cierra con un numero.

**Una cota solo vale si sabes que esta regalando.**

### 10.6 Como elegir la celda de trabajo

Criterio, medido: la celda util esta limitada por **acciones**, no por capital
(`PASS` ~3 %), y tiene un escalon real entre lo evidente y lo profesional.

```
 8h x 5d caja 100    limitada por CAPITAL   36,7 % de los sorteos clavan el maximo
12h x 5d caja 400    limitada por ACCIONES   90 % del techo en 11 sorteos,
                                             99 % en 769  -> factor 40
```

El caso base es demasiado facil para estudiar nada: con 200 evaluaciones,
cualquier mecanismo que muestree lo resuelve el 100 % de las veces, y la
absorcion en la inaccion sale **0 %** en los cuatro mecanismos probados.

### 10.7 Codigo nuevo o tocado

```
kagsym/macro.py           +14 campos, parametros(), aplica_parametros(),
                          FACTOR_RIEGO, N_MACRO 14->28
kagsym/exacto/mercado.py  8 constantes expuestas
kagsym/exacto/tareas.py   DESCUENTO_POR_PASO/VALOR_DIG/HORIZONTE_FERTILIZAR/
                          GANANCIA_MAPA/TOPE_MAPA expuestas; METODO_ASIGNACION
                          con valor de ruta; MUESTREA_VERBO + acumulador de
                          gradiente de score-function
kagsym/exacto/ejecutor.py TURNOS_CASILLA_INI/MIN expuestas; aplica_parametros
kagsym/migrar_ckpt.py     migra_salida_macro(): cabeza 14->28
kagsym/obs.py             +TURNS_PER_DAY/24 absoluto (ver §9.1); N_GLOBAL 90->91
runs/ligas/              constantes.py macro28.py estrechar.py aprendices.py
                          conjunto.py todo_discreto.py macro_discreto.py
                          reinforce_verbo.py muestrear_verbo.py aprendible.py
                          mezcla_exp.py profesional.py micro_celda.py
                          micro_escala.py verbo_celda.py techo_clase.py
                          minimo.py mas_pequeno.py exacto_dia.py ab_ruta.py
                          cota_con_viaje.py trampa.py escape.py valle.py
                          por_eje.py vs_topes.py dos_sillas.py simetria_silla.py
                          pool.py mundos_pequenos.py frontera.py
```

Todo lo nuevo esta **apagado por defecto**: `METODO_ASIGNACION = "humgaro"`,
`MUESTREA_VERBO = False`, y los catorce parametros con sus valores antiguos.
Ningun camino existente cambia de comportamiento.

