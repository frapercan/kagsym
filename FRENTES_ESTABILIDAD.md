# Frentes de estabilidad -- barrido del 2026-09-24

Seis agentes barrieron el repo por ejes de fragilidad operativa; dos
escepticos por hallazgo, uno comprobando que existe en esa linea y otro
DESCARTANDO cualquier consejo generico que no saliera de este repo.
Sobrevivieron 13 de 24.

**Los trece son silenciosos.** Ese es el resultado, mas que la lista: en
este proyecto los fallos no revientan, FIRMAN. Producen un numero
plausible. Un `nan` en una puerta, un rival que se convierte en
`pass_agent`, un JSON global que redefine el espacio de busqueda -- todos
dejan el log mas limpio que nunca.

El barrido incluye el codigo escrito hoy mismo, asi que varios frentes
afectan a una corrida que estaba viva mientras se escribia esto.

Orden: por gravedad. El coste en horas es la estimacion del agente que lo
encontro.


## Resumen

| gravedad | fichero:linea | eje | h | frente |
|---|---|---|---|---|
| alta | `tools/search.py:152` | ciclo | 2 | Los ficheros de salida de una búsqueda no tienen identidad ni marca de terminado: se pisan entr |
| alta | `/home/xaxi/farm/tools/search.py:152` | concurrencia | 3 | El prefijo --out de la búsqueda se reescribe en sitio, sin id de run: dos ejecuciones comparten |
| alta | `/home/xaxi/farm/kagsym/symbolic/tasks.py:525` | deriva | 2.5 | Nueve variables de entorno KAG_* cambian cómo juega el agente, no entran en el ledger, y la ver |
| alta | `/home/xaxi/farm/tools/search.py:99` | deriva | 1.5 | `runs/live_dials.json` es una ruta global única que DEFINE el espacio de búsqueda, sin comproba |
| alta | `/home/xaxi/farm/kagsym/evaluate.py:359` | deriva | 2 | La huella de código de `version.py` -escrita para el fallo exacto que está ocurriendo hoy- solo |
| alta | `/home/xaxi/farm/tools/search.py:51` | estado | 2 | runs/live_dials.json es una ranura global compartida y las cachés de checkpoint se indexan solo |
| alta | `tools/validate_offset.py:56` | reproducible | 3 | El offset buscado es un vector desnudo: su significado vive en otro fichero, mutable y ordenado |
| alta | `kagsym/evaluate.py:351` | reproducible | 2 | La procedencia se escribe y NUNCA se lee: `version.check()` es codigo muerto y el ledger apunta |
| alta | `tools/evalua.py:73` | reproducible | 2 | `tools/evalua.py` es un SEGUNDO bucle de juego, ciego a la rampa, y decide que checkpoint se gu |
| alta | `kagsym/cli/train.py:302` | reproducible | 2 | Nada siembra torch: `--seed0` solo siembra el entorno, y dos entrenamientos con el mismo comand |
| alta | `tools/validate_offset.py:87` | silencio | 2 | Un `nan` no bloquea la puerta de horneado: la medida ciega de hoy no dice "no hay efecto", AUTO |
| alta | `kagsym/evaluate.py:359` | silencio | 2.5 | La procedencia es de sólo escritura, y `runs/live_dials.json` es estado global compartido entre |
| media | `kagsym/environment.py:433` | silencio | 1.5 | El rival puede desaparecer sin que nada lo diga: `except Exception: return E.pass_agent` en la  |

---


## 1. Los ficheros de salida de una búsqueda no tienen identidad ni marca de terminado: se pisan entre corridas y la validación no comprueba de dónde vienen

**`tools/search.py:152`** -- eje *ciclo*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** `np.save(a.out + "_mu.npy")`, `_best.npy` y `_history.json` se reescriben en CADA generación, sin comprobar que el prefijo esté libre y sin escribir nada que diga «terminé». Un `_mu.npy` de una búsqueda muerta en la generación 3, uno de una búsqueda viva a medio converger y uno de 40 generaciones completas son el mismo fichero con el mismo aspecto. Y tools/validate_offset.py:50-57 solo saca `live` del `_history.json` de al lado CUANDO ese fichero existe: si copias o renombras el npy -`cp X_mu.npy runs/mejor_offset.npy`, que es lo natural-, cae al default `runs/live_dials.json`. Hoy ese fichero tiene 50 diales y la búsqueda que corre usa 51 (añadió el dial 12 porque el checkpoint lo toca, search.py:64-67). Entonces `Offset.is_ramp` en kagsym/policy.py:57 evalúa `len(delta)=102 >= 2*50=100` -> True, y `vector()` (policy.py:59-65) calcula `d[:50] + d[50:100]*progress`: la parte constante truncada, la rampa ENTERA desplazada un dial, los dos últimos valores tirados. Sin excepción, sin aviso, y devuelve una diferencia pareada en dólares con su t. Además validate_offset.py no compara nunca `history["checkpoint"]` con el checkpoint del argumento ni mira si la búsqueda llegó a `args["gens"]`.

**Arreglo propuesto.** Que search.py cree el directorio `a.out` y falle si ya existe salvo `--force`; que grabe `meta.json` al empezar (digest del checkpoint, lista `live`, ramp, gens pedidas, commit, pid) y `done.json` solo al terminar; y que validate_offset.py cargue ese meta, exija que el digest coincida con el checkpoint del argumento y que `done.json` exista, y aborte si no. Como red: en `Offset.vector`, exigir `len(delta) in (n, 2n)` exacto en vez de `>=`.


## 2. El prefijo --out de la búsqueda se reescribe en sitio, sin id de run: dos ejecuciones comparten tres ficheros y nadie lo nota

**`/home/xaxi/farm/tools/search.py:152`** -- eje *concurrencia*, gravedad **alta**, SILENCIOSO, ~3 h.

**Modo de fallo.** `--out` es un prefijo literal. Cada generación reescribe `<out>_mu.npy`, `<out>_best.npy` y `<out>_history.json` EN SITIO (líneas 152-157), sin id de run, sin pid, sin comprobar si el fichero ya existe y sin lock. Tres consecuencias, todas silenciosas: (a) relanzar la misma búsqueda con el mismo prefijo -que es lo normal cuando arreglas un bug y repites- deja en disco un `_mu.npy` del run VIEJO hasta que el nuevo cierre su primera generación (aquí 365 s); quien ejecute `tools/validate_offset.py <out>_mu.npy` en esa ventana valida el vector del run roto y obtiene un número en dólares perfectamente plausible; (b) `validate_offset.py:50-58` reconstruye la lista `live` desde `<out>_history.json` y la empareja con `<out>_mu.npy` COMO SI vinieran del mismo run — si el `_mu.npy` es del run A y el `_history.json` del run B con otra lista de diales, `Offset.vector()` (kagsym/policy.py) coloca cada delta en un dial distinto: no lanza, no avisa, devuelve dinero; (c) si el otro agente lanza una segunda `search.py` con el mismo `--out`, cada proceso vuelca su historia COMPLETA en memoria en cada generación, así que el JSON alterna entre dos trayectorias CEM enteras y coherentes y leerlo te da una u otra al azar.

**Arreglo propuesto.** En `search.py`, derivar el directorio de salida de `--out` + un id de run corto (timestamp + pid) y crearlo con `os.makedirs(..., exist_ok=False)`: si existe, aborta. Escribir los tres artefactos dentro, y escribir `_mu.npy` a `.tmp` + `os.replace` para que nunca se lea a medias. Guardar el `live` DENTRO del `_mu.npy` (np.savez con `delta` y `live`) para que el vector y su mapa de diales no puedan separarse nunca, y que `validate_offset.py` lea los dos del mismo fichero en vez de inferir el compañero por `str.replace`. Dejar que la herramienta escriba ella el `.pid` y el `.log`.


## 3. Nueve variables de entorno KAG_* cambian cómo juega el agente, no entran en el ledger, y la verja no puede verlas por construcción

**`/home/xaxi/farm/kagsym/symbolic/tasks.py:525`** -- eje *deriva*, gravedad **alta**, SILENCIOSO, ~2.5 h.

**Modo de fallo.** Hay una superficie de configuración por entorno leída EN TIEMPO DE IMPORT dentro de la capa simbólica: `_VALOR = environ.get('KAG_VALOR','mapa')` (tasks.py:525, conmuta mapa de la red contra heurística: son dos agentes distintos), `_COORD` (tasks.py:660), `KAG_CADENA` (tasks.py:1112), `KAG_COMMIT` (tasks.py:1220, pisa el dial que emite el macro), `USE_RIVAL_FLOW = KAG_FLUJO_MERCADO` (executor.py:26), `KAG_CAJA` en spec.py:79 -que `FastEnv.__init__` mezcla como base de TODO episodio en fastenv.py:92-, `KAG_GAMMA` y `KAG_POTENCIAL` (environment.py:225,270) y seis pesos de recompensa (reward.py:40,56,67,68,95,245). `evaluate.record` (kagsym/evaluate.py:356-366) guarda commit, digest del checkpoint y familia de semillas, y NINGUNA de ellas. Modo de fallo: un agente exporta KAG_VALOR=heuristica para un A/B, la variable se queda en su shell, y cada medida posterior en esa shell describe otro agente mientras escribe filas de ledger indistinguibles de las del otro agente. Peor: en Kaggle ninguna está puesta, así que cualquiera de ellas es una bifurcación entre el agente medido y el subido. Y `tools/check_submission.py` NO puede detectarlo: ejecuta `exec(compile(src))` en el MISMO intérprete (línea 27) y luego `E.play` en ese mismo proceso (línea 68), así que los dos lados ven idénticas las variables y la verja da 6/6 idéntico con el entorno envenenado. Su docstring dice «Any difference means one of the two lies» -esta clase entera es invisible para ella-.

**Arreglo propuesto.** Una función `game_env()` que devuelva el dict de los nueve KAG_* con su valor efectivo; llamarla desde `evaluate.record` y meterla en cada fila del ledger y en el `_history.json` de la búsqueda. En `tools/check_submission.py`, abortar si alguna KAG_* difiere de su default antes de comparar (y decir cuál). Y que `spec.verify()` coteje DEFAULT_CONFIG contra los `default` del `kaggriculture.json` instalado en vez de confiar en la copia.


## 4. `runs/live_dials.json` es una ruta global única que DEFINE el espacio de búsqueda, sin comprobar a qué checkpoint pertenece, y con dos agentes escribiéndola a la vez

**`/home/xaxi/farm/tools/search.py:99`** -- eje *deriva*, gravedad **alta**, SILENCIOSO, ~1.5 h.

**Modo de fallo.** `tools/search.py` toma los diales vivos de un único fichero fijo (`--live` por defecto `runs/live_dials.json`, línea 99, consumido en la 112) y `tools/live_dials.py:37` escribe a esa misma ruta fija por defecto. El JSON lleva dentro `checkpoint` y `commit`, y NADIE los lee: `live_dials()` en search.py:48-51 solo hace `d['live']`. Modo de fallo A: se busca sobre runs/nuevo.pt con la lista de diales de partida_v5; los diales muertos para el checkpoint nuevo entran como dimensiones de ruido puro -que es justo lo que live_dials.py:6-7 y README.md:89 declaran dañino- y los vivos que la sonda no vio quedan CONGELADOS a cero, así que la búsqueda no puede encontrarlos nunca. Modo de fallo B, hoy mismo: el agente A lanza una búsqueda de 12 núcleos a las 12:00 leyendo la lista de v5; el agente B corre live_dials.py sobre otro checkpoint a las 12:05 y sobrescribe la misma ruta. Modo de fallo C, el caro: `tools/validate_offset.py:56` cae a esa misma ruta compartida cuando no encuentra el `_history.json`; el vector mu se indexa POSICIONALMENTE contra `live` (Offset.vector, policy.py:58-64), así que una lista distinta aplica cada desplazamiento al dial equivocado y valida otro agente. En los tres casos sale un número plausible y ninguna excepción.

**Arreglo propuesto.** Que `live_dials.py` escriba por defecto a `runs/live_dials.<digest-del-checkpoint>.json` y que `search.py:112` y `validate_offset.py:56` aborten si el `checkpoint`/`checkpoint_digest` del JSON no es el del checkpoint que se está buscando. Corregir las tres cifras de README:89 y PROCEDURE:21 para que citen el fichero en vez de un número copiado, y borrar `runs/diales_vivos.json`.


## 5. La huella de código de `version.py` -escrita para el fallo exacto que está ocurriendo hoy- solo está cableada al entrenador, que es el bucle que ya nadie corre

**`/home/xaxi/farm/kagsym/evaluate.py:359`** -- eje *deriva*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** `kagsym/version.py` existe para «un proceso que sobrevive a una edición: un número perfectamente válido dentro del mundo equivocado» y su docstring (línea 13) afirma «The fingerprint is stored next to every vector and every checkpoint». No lo está. `fingerprint()` y `model_fingerprint()` solo se importan en `kagsym/cli/train.py:42` -y README.md:101 dice que después de salir de la inicialización aleatoria «everything after that is search and validation»-. El ledger (`evaluate.record`, evaluate.py:359) guarda `git_commit()`, que es `git rev-parse --short HEAD` (evaluate.py:343-348) y NO mira si el árbol está sucio. `tools/search.py:157` y `tools/validate_offset.py:101` guardan lo mismo. `version.check()` (version.py:77-88), que existe para imprimir el aviso, no se llama desde ningún sitio. Modo de fallo: dos filas del ledger con el mismo commit y el mismo digest de checkpoint son dos juegos distintos, y la comparación entre ellas es inválida sin que nada lo diga. Con dos agentes editando el repo mientras corren búsquedas de 12 núcleos, es el estado por defecto, no un caso raro.

**Arreglo propuesto.** Añadir `fingerprint()` y un flag `dirty` (`git status --porcelain` no vacío) a `evaluate.record` (evaluate.py:356-366), al `_history.json` de search.py:155 y al `offset_provenance` de validate_offset.py:100; y que `tools/ledger.py` marque en rojo toda fila con `dirty` o con huella distinta de la actual. Derivar GAME_FILES recorriendo `kagsym/symbolic/*.py` en vez de listarlos, para que no vuelva a quedarse corta.


## 6. runs/live_dials.json es una ranura global compartida y las cachés de checkpoint se indexan solo por ruta, con dos agentes reescribiendo esas rutas

**`/home/xaxi/farm/tools/search.py:51`** -- eje *estado*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** search.py:48-51 `live_dials()` lee SOLO `d["live"]` y tira `d["checkpoint"]`, que live_dials.py:82-85 sí escribe. El fichero por defecto es único: search.py:99 y live_dials.py:37 apuntan los dos a `runs/live_dials.json`. Con dos agentes a la vez, el segundo sondea otro checkpoint, pisa ese JSON, y cualquier búsqueda lanzada después optimiza el subconjunto de diales vivo en OTRO modelo. No revienta: los offsets son aditivos sobre el conjunto vivo, así que la búsqueda simplemente gasta dimensiones muertas y deja fuera dimensiones vivas, su `sd` no colapsa nunca y la columna CENTRE-BASE (search.py:159-162) sigue siendo internamente coherente. Parece una búsqueda sana que no encuentra nada. Ahora mismo hay un `tools/search.py runs/partida_v5.pt --gens 40` corriendo con 11 procesos contra un live_dials.json de las 12:09 y un checkpoint de las 11:37; nada impide que un `--bake` o un `torch.save` reescriba esa ruta a mitad de las 40 generaciones, y entonces `base` de la gen 20 y `base` de la gen 0 son redes distintas con la misma etiqueta, porque `_history.json` (search.py:154-157) guarda la ruta relativa y el commit pero NO el digest, que evaluate.py:335 `file_digest` ya sabe calcular. Segunda cara del mismo eje: evaluate.py:140-144 `_POLICY_CACHE` se indexa por `spec.path` y además evaluate.py:145-151 MUTA `pol.offset` sobre el objeto compartido; tools/evalua.py:40/48/85 hace lo mismo con `_CACHE[ckpt]`; evaluate.py:90 `_OPPONENT_CACHE` usa la ruta absoluta como clave para rivales de tipo checkpoint. Hoy solo es seguro porque `play()` fija el offset y juega en serie: nada lo garantiza.

**Arreglo propuesto.** (1) En `live_dials()`, comparar `d["checkpoint"]` (y un `digest` nuevo) con el checkpoint que se va a buscar y abortar con SystemExit si no coinciden —el mismo patrón ruidoso de parallel_env.py:251-255—; añadir `digest` al JSON en live_dials.py:82-85 y a `_history.json` en search.py:155. (2) Al arrancar una búsqueda larga, copiar el checkpoint a `runs/search/<nombre>_frozen.pt` y buscar sobre la copia, para que nadie pueda moverlo bajo los pies. (3) Cambiar la clave de `_POLICY_CACHE`, `_CACHE` y `_OPPONENT_CACHE` a `(ruta, st_mtime_ns, st_size)`.


## 7. El offset buscado es un vector desnudo: su significado vive en otro fichero, mutable y ordenado por EFECTO, no por indice

**`tools/validate_offset.py:56`** -- eje *reproducible*, gravedad **alta**, SILENCIOSO, ~3 h.

**Modo de fallo.** `tools/search.py` guarda el resultado como un `.npy` pelado (solo numeros). La lista `live` que dice A QUE DIAL corresponde cada numero se guarda aparte, en `_history.json`, y `validate_offset.py` la busca por NOMBRE de fichero (linea 50). Si ese history no esta, o se paso `--live`, cae en silencio a `runs/live_dials.json` (linea 56), que es un fichero COMPARTIDO que cualquiera regenera con `tools/live_dials.py`. Peor: `live_dials.py:75` ordena la lista por TAMAÑO DE EFECTO medido en 3 semillas, no por indice, asi que re-ejecutar la sonda devuelve el MISMO conjunto en OTRO ORDEN, y el mapeo delta->dial es posicional (`policy.py:63`, `out[list(self.live)] = v`). Y `is_ramp` (`policy.py:56`) se INFIERE de una comparacion de longitudes, asi que una rampa de 102 con 50 diales sigue leyendose como rampa, y una de 98 con 50 se lee como CONSTANTE. Nada lanza. Sale un numero en dolares perfectamente plausible, se compara contra la base en semillas reservadas, y si pasa el t>=2 se hornea en el bias de `macro_mu` (linea 98) de forma irreversible.

**Arreglo propuesto.** Que la busqueda escriba UN solo artefacto autodescriptivo por corrida (`.npz` o json) con {delta, live, ramp explicito, checkpoint_digest, fingerprint, commit, rng} y un id de corrida en el nombre; que `validate_offset.py` LEA el live de ahi y ABORTE si no esta, en vez del fallback de la linea 56; que `live` se escriba siempre ordenado por indice (`live_dials.py:75` ya ordena por efecto: guardar el efecto aparte, en el dict `effect` que ya existe); y que `is_ramp` sea un campo, no una comparacion de longitudes.


## 8. La procedencia se escribe y NUNCA se lee: `version.check()` es codigo muerto y el ledger apunta a un commit que ignora el arbol sucio

**`kagsym/evaluate.py:351`** -- eje *reproducible*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** `record()` guarda commit + digest del checkpoint, pero NO guarda `fingerprint()` ni `model_fingerprint()`, que es justo lo que `kagsym/version.py` existe para dar. `git_commit()` (linea 345) es un `git rev-parse --short HEAD` sin comprobar suciedad: con DOS agentes editando el repo mientras corren procesos largos, todas las lineas del ledger declaran el mismo commit mientras el codigo bajo ellas cambia. Y `version.check()` -la funcion que imprime 'la comparacion NO es valida'- no la llama NADIE: `grep` da un unico import en `train.py:42`, y solo de `fingerprint`/`model_fingerprint`. `policy.load_network()` lee `ck['cfg']` y `ck['sd']` y jamas mira `ck['fingerprint']`. Resultado: dado un numero en un log, NO se puede reconstruir el mundo que lo produjo, y una comparacion entre dos numeros medidos con ejecutores distintos sale sin un solo aviso. Es exactamente el fallo que el docstring de `version.py` describe en sus primeras 15 lineas, con la herramienta para detectarlo desconectada.

**Arreglo propuesto.** Añadir a `record()` (linea 351) `fingerprint`, `model_fingerprint`, `dirty` (salida de `git status --porcelain` no vacia), `argv` y `pid`; llamar a `version.check(ck.get('fingerprint'))` desde `policy.load_network()` para que cada carga avise; y meter la llamada a `record()` en `evalua.py`, `matriz.py` y en cada generacion de `search.py`. Es cableado, no diseño: los dos modulos ya existen.


## 9. `tools/evalua.py` es un SEGUNDO bucle de juego, ciego a la rampa, y decide que checkpoint se guarda como el mejor

**`tools/evalua.py:73`** -- eje *reproducible*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** `evalua.py` reimplementa el episodio entero (lineas 39-80: carga la red a mano, monta su propio `Agent`, su propio `_split_micro`, su propio bucle) en vez de usar `kagsym.policy.Policy` + `kagsym.evaluate.play`. En la linea 73 fija `ag.macro` directamente desde `macro_mu` y jamas mira `ck['offset']` ni `ck['delta_rampa']`. Es el bug 3 de hoy, todavia en el arbol. Modo de fallo: apuntalo a un checkpoint horneado con rampa y mide la politica SIN rampa; si lo comparas contra su propia base, las dos corridas son la misma politica y devuelve '+0 $ +- 0' con sd 0 -un nulo perfecto, indistinguible de un nulo verdadero-. Y esta cableado al entrenador: `train.py:2577` lo invoca por subproceso y `train.py:2585-2591` copia `.ultimo` -> `a.out` cuando su numero supera a `best`, es decir la seleccion de checkpoint se decide con un instrumento que no ve una de las dos cosas que el checkpoint lleva. Agravante del mismo eje: `train.py` no contiene NI UNA referencia a `offset`/`delta_rampa` (grep vacio), asi que un `--resume` desde un checkpoint horneado con rampa la pierde en cada `torch.save` posterior, sin aviso, porque `load_tolerant` solo audita tensores.

**Arreglo propuesto.** Vaciar `_uno()` y dejar `evalua.py` como un envoltorio de 20 lineas sobre `E.run(E.PolicySpec(ckpt), [E.V48], S.RESERVED.seeds(n))` -el `ProductionLedger` se conserva pasandolo como tag, o se mueve a `kagsym/evaluate.py`-; y que `train.py` propague `d0.get('offset')` a los tres `torch.save` (lineas 2569, 2616, 2698) para que reanudar no borre una rampa horneada.


## 10. Nada siembra torch: `--seed0` solo siembra el entorno, y dos entrenamientos con el mismo comando dan redes distintas

**`kagsym/cli/train.py:302`** -- eje *reproducible*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** `--seed0` (linea 302) fija la primera semilla del flujo de TABLEROS y nada mas; su ayuda habla de 'semillas de entrenamiento', lo que da la impresion de que la corrida es reproducible. No lo es: no hay un solo `torch.manual_seed` / `np.random.seed` / `random.seed` en todo el paquete. La inicializacion de la red y TODOS los sorteos de exploracion -`parallel_env.py:95` y `:116` (`torch.randn` del macro y del micro), `train.py:1013, 1038, 1219, 1225` (el ruido antitético) y los `torch.randperm` de `train.py:1008, 1405, 1787`- salen del generador global sin sembrar. Modo de fallo: no revienta, degrada la capacidad de decidir. Un A/B de dos configuraciones de entrenamiento confunde la configuracion con el sorteo de inicializacion, y con sd de 13.300 $/semilla eso es exactamente el tamaño del efecto que se busca; una corrida que dio un buen checkpoint no se puede repetir; y como `runs/` esta en `.gitignore`, si el `.pt` se pisa -el `--out` por defecto es `runs/e2e.pt` (linea 386), asi que dos entrenadores lanzados sin `--out` comparten fichero, `.ultimo` y `.mejor.json`- el resultado no es recuperable de ninguna forma.

**Arreglo propuesto.** Un `--seed` que llame a `torch.manual_seed` / `np.random.seed` / `random.seed` al arrancar `main()` y que los trabajadores de `parallel_env.py` deriven como `seed + rank`; guardarlo en el checkpoint junto a `upd` y `fingerprint`; y hacer que `--out` sin valor explicito falle o incorpore el `--run-name`, para que dos entrenadores simultaneos no compartan fichero.


## 11. Un `nan` no bloquea la puerta de horneado: la medida ciega de hoy no dice "no hay efecto", AUTORIZA escribir el checkpoint

**`tools/validate_offset.py:87`** -- eje *silencio*, gravedad **alta**, SILENCIOSO, ~2 h.

**Modo de fallo.** `kagsym/evaluate.py:328` devuelve `t = nan` exactamente en los dos casos que importan: n<=1 (se=nan, línea 326) y sd=0, que es el fallo nº3 de hoy -- una rampa comparada contra su propia base da diferencia idéntica cero en todos los tableros, "+0 $ +- 0". La puerta de horneado es `if d["t"] < a.t_min or not win_ok`. En Python `nan < 2.0` es False. Con `--band-n 0`, `win_ok` vale True por defecto (línea 73), la condición entera es False, no se imprime "NOT baked", no hay `sys.exit(1)`, y se escribe el checkpoint horneado como si hubiera pasado una validación a t>=2. Una vara que se quedó ciega no es que calle: firma. Compañero en el mismo eje: `tools/search.py:85-87` puntúa con `np.nanmean`, así que un candidato cuyos episodios revientan sólo en los tableros difíciles puntúa con la media de los fáciles y sube al elite -- la presión de selección apunta al candidato que revienta. Y `E.paired` (evaluate.py:321-323) construye las claves por intersección: si la mitad de los episodios falló, `n` encoge en silencio y el par sigue pareciendo pareado.

**Arreglo propuesto.** En `paired`, devolver `n_esperado` y marcar el caso degenerado: si `se == 0` y `money_diff == 0`, eso no es t=nan sino "las dos ramas son la misma política" y debe ser un error explícito. En validate_offset, exigir `np.isfinite(d["t"]) and d["n"] == len(seeds)` ANTES de comparar con `t_min`. En search.py, sustituir `nanmean` por media sobre conjunto completo y descartar (no promediar) todo candidato con cualquier `ep.error`.


## 12. La procedencia es de sólo escritura, y `runs/live_dials.json` es estado global compartido entre los dos agentes

**`kagsym/evaluate.py:359`** -- eje *silencio*, gravedad **alta**, SILENCIOSO, ~2.5 h.

**Modo de fallo.** `record()` sella cada línea del ledger con `git_commit()` = `git rev-parse --short HEAD`, sin marca de árbol sucio y sin `fingerprint()`. Dos evaluaciones separadas por ediciones no commiteadas se archivan con el MISMO commit y el ledger afirma una comparabilidad que no existe; la fila sale perfectamente formada. El mecanismo construido precisamente para esto -- kagsym/version.py, cuyo docstring dice "a perfectly valid number inside the wrong world" -- no lo llama NADIE: `check()` (version.py:77) no tiene un solo llamador en el repo, y `record()` no guarda `fingerprint()` ni `model_fingerprint()`. Peor, `_digest` (version.py:58-59) traga `OSError` con `h.update(b"?")`: si un fichero de `GAME_FILES` se renombra, la huella sale igual de plausible -- 12 hex -- en vez de fallar. Segundo vector, el que muerde hoy con dos agentes: `runs/live_dials.json` es un único fichero compartido, escritura por defecto en tools/live_dials.py:37 y lectura por defecto en tools/search.py:99. `live_dials` escribe ahí el campo `"checkpoint"` (live_dials.py:82) y NADIE lo lee: tools/search.py:51 toma sólo `d["live"]` y tools/validate_offset.py:54 sólo `["live"]`. Si el otro agente lanza `tools/live_dials.py` sobre otro checkpoint, redefine en silencio qué 50 de los 67 diales optimiza toda búsqueda posterior -- para cualquier checkpoint, sin una línea de aviso, y el resultado sigue siendo un vector de la dimensión correcta.

**Arreglo propuesto.** `record()` añade `fingerprint()`, `model_fingerprint()` y `dirty = bool(git status --porcelain)`; `tools/ledger.py` marca la fila cuando la huella difiere de la actual. `version._digest` lanza en vez de escribir `b"?"`. `search.py` y `validate_offset.py` comparan `d["checkpoint"]` con el checkpoint que reciben y abortan si no coincide. `live_dials.py` escribe por defecto junto al checkpoint (`<ckpt>.dials.json`) en vez de a un global compartido.


## 13. El rival puede desaparecer sin que nada lo diga: `except Exception: return E.pass_agent` en la construcción del oponente

**`kagsym/environment.py:433`** -- eje *silencio*, gravedad **media**, SILENCIOSO, ~1.5 h.

**Modo de fallo.** `_new_rival` envuelve la construcción del oponente del peldaño en un `try` que, ante CUALQUIER excepción -- fichero de `agents_pub/` ausente o renombrado, import del módulo público que peta, `LADDER_CAPS` más corto que `LADDER`, `spec_from_file_location` devolviendo None --, devuelve `E.pass_agent`. A partir de ese instante el entrenamiento juega contra un agente que pasa siempre: `win_rate` sube a ~1.0, `raise_level` promociona, el currículo escala peldaños que nunca se jugaron, y el retorno sube de verdad -- contra nada. No hay contador, no hay traza, y el log se ve mejor que nunca. El contraste está 130 líneas más abajo en el MISMO fichero: el except del rival por turno (environment.py:564-576) sí incrementa `_RIV_FALLOS` e imprime el traceback las 3 primeras veces, con un comentario que dice "NUNCA EN SILENCIO". El camino de construcción no recibió ese tratamiento. Compañero: `_compute_phi` (environment.py:373) devuelve 0.0 ante cualquier excepción; como la recompensa es `rec[i] += phi_w*(gamma*nuevo - self._phi[i])`, un fallo intermitente no apaga el shaping, inyecta un salto espurio de `-phi_viejo` (o `+gamma*phi_nuevo`) en la recompensa de ese día. Recompensa envenenada, no ausente.

**Arreglo propuesto.** En `_new_rival`, dejar propagar la excepción; si se quiere tolerancia, contar y anunciar como ya hace el camino por turno. Añadir una validación al arrancar el entrenador: comprobar que todos los nombres de `LADDER` existen y cargan en `agents_pub/` antes del primer update, y que `len(LADDER_CAPS) == len(LADDER)`. En `_compute_phi`, propagar; si hace falta tolerancia, devolver el phi ANTERIOR, nunca 0.0.


---

## Descartados por los escepticos (11)

Se listan para que nadie los vuelva a levantar sin datos nuevos.

- **El instrumento que ELIGE qué checkpoint se guarda es un segundo bucle de juego: ignora la rampa, carga en modo** -- REFUTADO: el frente describe con exactitud el código de HEAD (3cae2ed), pero NO el árbol de trabajo actual. `tools/evalua.py` está borrado en disco (`git status`: `D  tools/evalua.py`, junto a `tools/liga.py` y
- **apply_params escribe 39 constantes en TRES módulos globales, sin dueño, sin restaurar y en el orden equivocado** -- El frente existe exactamente donde se alega. macro.py:412 es apply_params; escribe 37 atributos (18 market_ops, 14 tasks, 5 executor) mas tres global propios (PRIORITY_TEMP 464-465, ACTIONS_PER_ANIMAL 466-467, 
- **El controlador exterior mide con HAND_CAP del currículo, hist a ceros y sin _destinations: los tres defectos q** -- REFUTADO por la lente exacta ("¿existe de verdad en esa línea y en ese fichero?"): el fichero ya no existe.

1) `/home/xaxi/farm/kagsym/outer.py` NO está en disco. `ls` da "No existe el archivo o directorio" y 
- **Diez variables KAG_* leídas en tiempo de import cambian al agente, se heredan por fork y no aparecen en ningún** -- No refutado: cada linea citada existe literalmente. spec.py:79 es `int(__import__("os").environ.get("KAG_CAJA", "3000"))` dentro del dict DEFAULT_CONFIG a nivel de modulo; executor.py:26, tasks.py:660 y reward.
- **El listón de «mejor» vive en un json suelto, sin unidades ni dueño: una corrida entera puede entrenar toda la ** -- El titular existe literalmente donde dice: kagsym/cli/train.py:1043-1049 hereda `best` de `a.out + ".mejor.json"` solo por la RUTA (la 1048 es el `json.load(...)["mejor"]`), sin digest, sin commit y sin unidade
- **Nadie es dueño de un proceso: trabajadores que quedan huérfanos para siempre, un `cerrar()` que no puede matar** -- NO refutado: el frente existe exactamente en esa linea y en ese fichero, y lo reproduje. kagsym/parallel_env.py:310 es literalmente `padre, hijo = ctx.Pipe()`; :321 es `p.start()` y no hay `hijo.close()` ni `pa
- **Checkpoints escritos sin atomicidad sobre el destino final, y un lector que absorbe la lectura rota como «meno** -- REFUTADO por obsolescencia: el frente está escrito contra un árbol anterior a HEAD, y la mitad de su cadena causal apunta a ficheros que ya no existen.

LO QUE SÍ EXISTE, verbatim:
- /home/xaxi/farm/kagsym/cli/
- **`macro.apply_params` escribe globales de módulo que NO se restauran entre episodios: un candidato contamina al** -- NO refutado: el frente existe exactamente en esa linea y en ese fichero, y cada ancla citada es correcta.

CADENA VERIFICADA LEYENDO EL CODIGO:
1. executor.py:75 `self.turns_per_tile = TURNS_PER_TILE_INIT` lee 
- **El listón `.mejor.json` sobrevive al proceso y se indexa por --out, cuyo default es el mismo para todos: el se** -- NO refutado: el codigo existe verbatim en /home/xaxi/farm/kagsym/cli/train.py. Confirmado: `best = -1e18` y `_MEJOR_JSON = (a.out + ".mejor.json")` con herencia dentro de `except Exception: pass` en 1056-1063 (
- **El timeout de 600 s de la evaluación va dentro de un `except Exception` genérico: con la máquina sobresuscrita** -- El código existe pero el frente no. Cinco comprobaciones lo tumban.

1) Las líneas citadas no son las reales. En /home/xaxi/farm/kagsym/cli/train.py la 2579 es `if a.eval_cada > 0 and (upd % a.eval_cada == 0 or
- **`runs/live_dials.json` es un global compartido que no está atado al checkpoint, y el commit se muestrea al esc** -- El frente EXISTE en esas líneas exactas, pero DOS de los tres trozos de "evidencia medida ahora" son falsos, y eso baja la gravedad.

Lo que se confirma, línea a línea:
- /home/xaxi/farm/tools/search.py:99 → `-
