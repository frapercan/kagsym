# kagsym

Agente neurosimbólico para [**Kaggriculture**](https://www.kaggle.com/competitions/kaggriculture)
(Kaggle × Google): dos jugadores, 720 turnos —30 días de 24 horas—, tablero de
10×10 y **un mercado compartido** donde vender hunde el precio para los dos.

Lee [`ESPECIFICACION.md`](ESPECIFICACION.md) primero: es el motor derivado
exactamente —espacio de acciones con precondiciones, recompensa y las trampas
de exactitud ya verificadas—.

---

## La idea

El reparto entre lo simbólico y lo aprendido no es de gusto, es de competencia:

| | quién decide | por qué |
|---|---|---|
| **legalidad** | el motor | es un hecho, no una opinión |
| **reparto de unidades** | asignación húngara | el óptimo exacto, 0,23 ms para 16×116 |
| **qué hacer y cuánto vale** | la red | es preferencia, y la preferencia se aprende |

La red emite, **una vez al día**, dos cosas: un vector `macro` de 58
dimensiones y un mapa `micro` de 10×10 con un canal de valor y uno por verbo.
La capa simbólica juega las 24 horas con eso.

**Un paso de RL es un día.** Es la decisión que hace tratable la asignación de
crédito, y condiciona todo lo demás: llamar a la red por turno multiplicaría
por 24 el coste en una competición con un segundo por turno.

## El principio: nada adivinado

Todo valor que no se derive del motor es **aprendible**. No es una aspiración;
es lo que más dinero ha dado del proyecto:

```
exponer 14 constantes de módulo            techo de la celda:  939 → 1.258 $  (+34 %)
exponer 4 literales dentro de funciones    dos de ellas valían ±22 % cada una
exponer 8 decisiones estructurales         cuatro vivas, entre +23 % y +71 %
```

Buscarlas con `grep` no basta: eso encuentra `SAT_ALTA = 0.85`, pero no un
`min` que es un techo infranqueable, ni un bucle que reparte presupuesto y
**tira lo que sobra**. Para eso está
[`herramientas/auditoria_cascada.py`](herramientas/auditoria_cascada.py), que
recorre el árbol de sintaxis y saca **todos** los puntos de decisión.

Y una vez expuestas, **perturbar y medir**: leer el código no distingue una
constante viva de una muerta. En modo `ops`, multiplicar por mil el valor de
desbrozar no mueve un solo dólar; mover la hora límite de contratación da +38 %.

### Parametrización sin bordes

```python
positivo   valor = defecto · exp(logit(f))            → (0, +∞)
fracción   valor = sigmoid(logit(defecto) + logit(f))  → (0, 1)
aditivo    peso  = logit(f)                            → (−∞, +∞)
```

Con `f = 0.5` sale el defecto exacto y los extremos alcanzan todo el dominio.
No queda ninguna constante de escala: como la red emite `macro_mu` y en todas
partes se hace `f = sigmoid(macro_mu)`, resulta que `logit(f) == macro_mu`, así
que el factor que multiplicaría es 1 **porque sigmoide y logit se cancelan**,
no porque alguien lo eligiera.

## Un solo modo

Hubo tres —`residuo`, `directo`, `ops`— para poder medirse entre sí. Ya están
medidos: `ops`, donde la red emite valor **y** verbo, es el único con libertad
de aprendizaje completa. Quitar los otros dos, el asignador voraz y el de rutas
deja el mismo resultado **al dólar** en seis semillas: era código muerto.

## Dónde estamos, medido

Contra un rival pasivo, 24h × 30d, caja 3.000, seis semillas:

```
capa simbólica, macro neutro          21.545 $
capa simbólica, macro CEM             51.905 $
la red e2e                            72.796 $     +40 % sobre CEM
mediana de 63 agentes públicos       186.594 $     2,6× sobre nosotros
```

El aprendizaje funciona. El techo de la arquitectura, todavía no: **estamos al
39 % del campo**, y el hueco es de producción, no de temporizado —cosechan 2×,
riegan 2,3× y mantienen 44 plantas vivas frente a nuestras 15,4 sembrando lo
mismo—.

## Entrenar

```bash
python -m kagsym.cli.train \
  --updates 4000 --envs 44 --procs 11 \
  --rejilla "24,30,9;24,30,7;24,30,5;24,30,0;24,30,0;24,30,5" \
  --niveles "6,7,8,12,15,1" \
  --rival-macros="-,-,-,-,-,vector.npy" \
  --curriculo-auto --cuota-autojuego 2 --cuota-objetivo 3 \
  --modo ops --ctx-micro 3x3 --kl-objetivo 0.01 --kl-max 0.06
```

**El currículo** reparte los trabajadores por información de Bernoulli
`p·(1−p)`, con dos cuotas fijas **fuera** de ese reparto:

- **objetivo**: `p(1−p)` vale 0 cuando se pierde siempre, así que el criterio
  abandonaría justo al rival que nos mide.
- **autojuego**: una copia congelada da `p ≈ 0,5` por construcción, o sea
  información máxima, y sin cuota se llevaría el presupuesto entero.

**El autojuego** lleva la red completa —macro y micro—, no un vector: con el
mismo macro en los dos lados, la red hace 75.009 $ y el vector 33.164.

## Verificar

```bash
pytest kagsym/tests -q          # el húngaro == scipy; FastEnv == el motor real
python herramientas/vara.py ckpt.pt   # vara fija: v48 sin tope, 8 semillas
python herramientas/auditoria_cascada.py
```

`FastEnv` tiene que ser **indistinguible** del motor turno a turno. Si esa
prueba falla, todo lo que hay encima miente.

## Disposición

```
kagsym/
  spec.py  fastenv.py      el motor y su clon exacto
  obs.py                   codificación de la observación
  macro.py                 el vector de 58 dimensiones y sus 30 parámetros
  reward.py            el objetivo (fijo) y la conformación
  potential.py
  environment.py               un paso = un día
  parallel_env.py           N partidas en paralelo
  symbolic/                  la capa simbólica: tasks, market, Hungarian
  nets/                   el world model and the heads
  cli/entrenar_e2e.py      el único entrenador
  tests/
herramientas/              medida y construcción de la escalera
```

`archivo/` guarda 84 scripts de medida históricos y 16 herramientas heredadas.
No son producción y no entran en el paquete, pero son el rastro de por qué cada
decisión es la que es.

## Licencia

MIT. No incluye agentes de terceros: la escalera de rivales se construye
descargándolos con `herramientas/baja_escalera.py`.
