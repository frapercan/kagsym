# De pesos aleatorios a batir a `partida_v5`

Escrito el 2026-09-24 para que otro agente reproduzca y supere el estado
actual desde una inicializacion completa, sin depender de nadie. Todo numero
de aqui esta medido; los que no sobrevivieron a una remedida estan marcados.

## 0. El criterio CAMBIA, y esta es la medida que lo justifica

La ladder puntua Bradley-Terry sobre VICTORIAS contra los otros agentes
subidos. Hoy se midio que optimizar dinero casi no mueve esa variable:

    partida_v2   win medio 0,056   bate a 4 de 74   (el pool son 76, dos no cargan)
    partida_v5   win medio 0,077   bate a 6 de 74   +13.115 $ sobre v2 (t +8,38)

Un 25 % mas de dinero compro DOS PUNTOS de win rate. Con 6 semillas por rival
ese +0,021 es ~1,3 sigma, asi que ni siquiera es solido -pero su magnitud si-.

**El objetivo es `tools/banda.py`: win medio y rivales batidos.** El dinero
contra v48 es una segunda columna util para diagnosticar, nunca el criterio.

    .venv312/bin/python tools/banda.py <ckpt> --n 6 --procs 12

## 1. Lo que YA FUNCIONA y no hay que reconstruir

- `kagsym/fastenv.py` -- clon exacto del motor. Arranca desde cualquier turno.
- `kagsym/symbolic/` -- legalidad, asignacion hungara, liquidacion.
- `tools/cotejo_submission.py` -- carga `submit_kagsym/main.py` COMO LO HACE
  KAGGLE (compile + exec, ultimo invocable), lo conduce contra el motor y lo
  compara con el bucle de medida semilla a semilla. **Esta es la puerta que
  mas cara ha salido no tener.**
- `kagsym/rampa.py` -- la UNICA definicion del desplazamiento del macro. La
  importan la busqueda, el validador y el agente que se sube.
- `tools/cem_diales.py` + `tools/valida_delta.py` -- la busqueda que produjo
  todo el dinero de hoy.
- `tools/paired_yardstick.py` -- vara pareada, instrumento independiente.

## 2. El procedimiento, en orden, con puerta numerica

**Paso 0 -- puertas, ANTES de cualquier corrida larga (5 minutos).**

    .venv312/bin/python tools/cotejo_submission.py <ckpt> 4     # 4/4 al dolar
    .venv312/bin/python tools/diales_vivos.py <ckpt> 3          # cuantos operan

Si el cotejo no da identidad exacta, PARAR: el bucle de medida no mide al
agente que juega, y cualquier numero que salga describe a otro agente. Eso ha
invalidado una noche entera DOS veces.

**Paso 1 -- e2e desde cero, solo para salir del azar.** Con autojuego de dos
asientos, desde cero alcanza el 68 % de `prod` en 8 minutos. Es la unica fase
donde el gradiente aporta algo. NO esperar mas de el: esta medido que PPO no
mejora desde ningun punto de partida -degrada el mejor checkpoint -6.787
(t -4,96) y desde uno 10.000 $ peor da +904 (t +0,75), y `ret`, su propio
objetivo, lleva 420 updates plano-.

**Paso 2 -- CEM sobre el desplazamiento CONSTANTE de los diales vivos.**
Primero el constante y no la rampa, porque un nulo aqui es interpretable.

    CEM_POP=32 CEM_ELITE=8 CEM_SEM=16 CEM_OUT=runs/d.npy \
      .venv312/bin/python tools/cem_diales.py <ckpt> 45

**Paso 3 -- CEM en RAMPA `a + b*progreso`** desde el resultado del paso 2.
Con `b=0` reproduce el constante al dolar (verificado), asi que es un
superconjunto estricto. Valio +5.570 (t +4,75).

    CEM_RAMPA=1 CEM_POP=32 CEM_ELITE=8 CEM_SEM=16 ... 60

**Paso 4 -- validar y hornear.** El optimo sobre las semillas de busqueda NO
es un resultado, es una hipotesis.

    .venv312/bin/python tools/valida_delta.py runs/d_mu.npy <ckpt> --hornear runs/nuevo.pt

Se hornea SOLO si DOS instrumentos coinciden en signo y el reservado pasa de
t >= +2. Cuando discrepan EN SIGNO es un artefacto: paso con +8.436 contra
-636, y la causa era el historico a ceros.

## 3. Trampas medidas -- cada una costo horas

- **El historico.** `torch.zeros(1, N_HIST)` en vez de `O.rival_flow(obs)`
  mutila al agente: el MISMO episodio en el MISMO tablero pasa de 76.607 $ a
  25.335. Entrenamiento y despliegue lo rellenan los dos.
- **`_destinations`.** `encode_obs(obs)` sin el segundo argumento no es lo que
  juega el agente subido.
- **`hands` vale SIEMPRE 0 en la hora 0.** Muestrear ahi da plantilla cero.
- **Los modulos se congelan al importar**: una corrida viva no ejecuta el
  codigo que hay en disco.
- **`pkill -f <patron>` casa con su propia linea de comandos** y mata el shell.
  Usar PID explicito.
- **En un CEM, la media de la POBLACION no mide el centro.** Es `mu` mas
  ruido, y donde perturbar sale caro queda por debajo de la base aunque `mu`
  este muy por encima: por leer eso declare nulo un +6.524 (t +4,35). El
  buscador ya evalua `mu` como candidato.
- **Un instrumento ciego dice "no hay efecto" con las mismas palabras que un
  nulo real.** La vara devolvio `+0 $ +- 0` con sd 0 comparando una rampa
  contra su base, porque no leia `ck["delta_rampa"]`.
- **Semillas.** Busqueda 9000+, validacion reservada 7101-7300, y para un
  contraste limpio usar 30000+. Nunca comparar numeros de conjuntos distintos.
  sd por semilla 15.000-22.000 $: menos de 100 pareadas no decide nada.

## 4. Ya refutado con medida -- no reabrir sin datos nuevos

Sopa de checkpoints (-7.893, t -4,89). SWA. ASP. Estrechar la politica.
Desajuste muestreado/determinista (bajan lo mismo). Elo como criterio (sube de
988 a 1730 mientras el rendimiento real CAE). 18 de los 67 diales del macro no
cambian el episodio ni un centimo: el mapa de valor de la red sustituyo el
valor en dolares de la heuristica y nadie retiro los diales.

## 5. Donde esta el hueco, medido

    casillas con cultivo   nosotros 10,2 de media   v48 42,1
    DIAS 0-1    HIRE 12 / BUY_ANIMAL 7 / BUY_SEED 1     v48: 5 / 1 / 7
    dinero dia 10   nosotros 10 $        v48 4.738 $

Los dos nos arruinamos el dia 5: el error no es cuanto se gasta, es EN QUE. La
simiente compone -el trigo hace x1,78 diario- y la plantilla y el ganado no.
Y NO es: los peones (8,57 contra 9,08), las rutas (v48 se mueve MAS que
nosotros, 48,7 % contra 42,4 %; lo que nos sobra es PASS, 21,4 % contra 4,9 %)
ni `sustainable_tiles` (nunca ata).
