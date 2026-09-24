# Estado, 2026-09-20

Lo que esta medido, lo que esta abierto, y lo que se ha descartado. Las cifras
son contra `v48-fast-routes` salvo donde se diga.

## Cifras firmes

```
heuristica escrita a mano (macro del CEM) ......... 39 558 $
e2e con la red emitiendo el valor, sin tabla ...... 37 471 $   (91 %)
mejor vector fijo (CEM, 20 semillas) .............. 75 157 $   contra PASIVO
rival v48 ........................................ ~135 000 $
experto publico 2945 ............................. 141 286 $   contra pasivo
                                                   101 385 $   contra si mismo

VICTORIAS EN LIGA .................................. 0 %       <- el criterio
```

## La brecha, descompuesta

Misma tierra (2.4 cuadrantes los dos), mismo rendimiento por cosecha
(3.02 uds contra 3.32). **La diferencia es el numero de casillas produciendo.**

```
              cultivos   cosechas   uds cosechadas   util%   PASS%
nosotros        19.0        232          701        38.4%   14.4%
v48             43.2        488         1622        45.6%    5.3%
```

Descartado como causa, midiendo:
- **No es riego**: nuestras plantas mueren a los 4-5 dias, que es fin de ciclo,
  no de sed. Ninguna muere joven.
- **No es cosecha tardia**: solo se pierde el 5 % de lo producido por decadencia
  (ellos 1 %), y lo perdido tiene 0.39 unidades de media.
- **No es rendimiento por planta**: 3.02 contra 3.32 unidades por cosecha.

**Contradiccion abierta**: forzar `casillas` hacia arriba EMPEORA el dinero
(66 425 $ con casillas~0 contra 43 253 con 0.15 y 35 729 con 0.6), y sin embargo
mas casillas es justo lo que falta. Algo hace que sostenerlas cueste mas de lo
que rinden en NUESTRO ejecutor y no en el suyo. Sin resolver.

## Eslabones que faltan

```
operacion      nosotros   v48
FERTILIZE             0    240     disponible solo en modo directo (ver abajo)
DROP                  0    417     NO esta en nuestro espacio de acciones
```

`DROP` vuelca todo el inventario de golpe. Sin el se descarga de uno en uno con
`PLACE` (21 veces contra sus 135). Es capacidad pura, sin juicio de valor:
deberia anadirse.

## La maquinaria de aprendizaje, validada pieza a pieza

```
representacion ...... +67 % sobre persistencia prediciendo el vertido del rival
exploracion ......... correlacionada por episodio; la independiente por dia
                      costaba el 54 % del rendimiento (40 972 -> 18 967 $)
recompensa .......... potencial exacto, sesgo terminal 0.7 %
critico ............. R2 0.675 en vuelo, techo medido 0.900
micro ............... la red emite el valor; preentreno supervisado al 91 %
```

## Descartado por medicion (no reintentar a ciegas)

- **El rival en el shaping diario**: aporta el 99.1 % de la varianza de la
  recompensa. Con el, el critico da R2 = -2.535; sin el, +0.736. El rival va en
  el OBJETIVO (el +-1 terminal), no en el shaping: **el shaping solo puede
  llevar lo que el estado predice**.
- **Fertilizar con valor escrito a mano**: tres intentos, los tres peores.
  Comparacion pareada sobre 8 semillas: **-21 998 $ +- 4 563** (4.8 sigma), con
  dos partidas hundidas a 568 $ y 33 $. La capacidad hace falta pero no se como
  ponerle precio: queda disponible solo en modo directo, donde la red la valora.
- **La cabeza auxiliar del rival dentro del critico**: lo empeora (R2 0.900 sola
  contra 0.852 con ella). Compiten por el tronco.
- **Optimizar contra rival pasivo**: lleva a ganaderia pura, que es la estrategia
  MAS fragil. Volcar 100 uds hunde LECHE un 97 % y LANA un 98 %; TRIGO y HUEVO
  apenas se mueven. El optimo contra pasivo esta anticorrelacionado con la
  fuerza real.

## Auto-juego (2026-09-20, tarde)

Entrenar contra v48 era el peor curriculo posible: **perdemos el 100 %**, asi que
el termino terminal vale -1 SIEMPRE y no aporta un bit sobre lo unico que puntua.
Y no hay peldano intermedio entre los publicos (a shop-router le ganamos el
100 %, a los otros tres les perdemos el 100 %).

Auto-juego contra una instantanea congelada de uno mismo, refrescada cada N
updates, con banco de versiones antiguas. Win rate ~0.5 por construccion = maxima
informacion por partida.

**Resultado contra la escalera (el unico termometro absoluto):**

```
                 vs v48    vs v16-rc5   vs 2945    media
preentreno      -84.4 %     -88.2 %     -98.7 %   -90.4 %
16 updates      -75.2 %     -86.7 %     -82.8 %   -81.6 %
```

Tres arreglos que hicieron falta, todos medidos:

1. **El rival decide una vez al DIA**, no cada turno. 24x mas barato y, sobre
   todo, simetrico: con otra frecuencia de decision no seria auto-juego.
2. **`torch.set_num_threads(1)` en los trabajadores.** 10 procesos pedian 60
   hilos sobre 12 nucleos. **230 s -> 11 s por update, 21x.** La senal para
   detectarlo fue un numero que no cuadraba: v48 tambien es un agente completo
   y costaba 12 s.
3. **Exploracion simetrica.** Con el rival determinista comparabamos "nosotros
   con ruido" contra "nosotros sin ruido", y el ruido cuesta mas de la mitad del
   rendimiento (40 972 -> 18 967 $ con el mismo vector). Perdiamos por handicap.

**Y el error de lectura mas caro del dia**: en auto-juego el dinero es RELATIVO.
El mercado es compartido y el rival tambien mejora, asi que "nuestro dinero baja"
no significa empeorar. Mientras el dinero de auto-juego caia de 56 717 a 42 240,
el margen contra el 2945 MEJORABA de -98.7 % a -82.8 %. Estuve a punto de parar
un entrenamiento que funcionaba. **Solo la escalera de publicos mide nivel.**

Corolario: guardar "el mejor por retorno de auto-juego" deja el checkpoint
congelado -100 updates sin guardar nada- porque el retorno deja de subir en
cuanto el rival se fortalece. Hay que guardar el ULTIMO estado siempre y
seleccionar entre tramos con la escalera.

## Siguiente, por valor medido

1. **Resolver la contradiccion de la densidad.** Es la brecha entera: 19 casillas
   contra 43. Todo lo demas es ruido al lado.
2. **Anadir `DROP`** al espacio de acciones. Capacidad pura, barata.
3. **Subir el critico de 0.675 a 0.900** con epocas propias (`--epocas-valor`) y
   mas peso. Ya esta implementado, sin medir.
4. **Fertilizar en modo directo**, dejando que la red le ponga precio.

## Reglas de casa que han costado tiempo

1. **Nada adivinado.** Once constantes a ojo rendian lo mismo que un dado.
2. **Toda capa guionizada debe batir a no-hacer-nada** antes de entrenar encima.
3. **Nunca muestrear ni decidir en la hora 0**: `len(farm["hands"])` vale
   SIEMPRE 0 ahi. Ha mordido dos veces, una midiendo y otra decidiendo.
4. **Comparar EMPAREJADO**, misma semilla. Sin eso, con sd de ~9 000 $, tres
   semillas producen conclusiones aleatorias: paso con `venta` (lei un efecto
   del 12 % que era ruido) y con el fertilizante (lei una semilla como media).
5. **Una busqueda que converge es un pesimo detector de errores.** El CEM
   encontro cuatro veces el optimo EXACTO dentro de una jaula que yo habia
   puesto sin saberlo, y devolvio un vector interpretable que lei como
   preferencia informada. Forzar cada dimension a mano es la unica deteccion.
6. **Solo cuenta `liga.py`.** Dinero medio contra pasivo, precision de
   validacion, win rate de curriculo y R2 agregado han enganado, cada uno.
