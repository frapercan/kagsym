"""Prueba de capacidad de CONTROL: ¿mejora el e2e sobre su inicializacion?

Arranca CLAVADO en el vector que encontro el CEM contra rival real -25 % de
victorias en la liga- y entrena contra ese mismo rival. La pregunta es limpia:

    si sube del 25 %, hay capacidad de control y toca escalar
    si no se mueve, el fallo esta en la SENAL y hay que arreglarla antes
    de gastar computo masivo

Por que esta prueba y no "entrenar y ver que sale": el CEM ya convergio
(sigma 0.011) dentro del espacio de 7 numeros y sigue perdiendo todas las
partidas contra 3 de los 4 publicos. O sea, el mejor vector FIJO no basta. Lo
unico que el e2e anade es dependencia del estado: cambiar de plan segun lo que
el rival este haciendo. Si eso no mueve la aguja, no la mueve nada de lo que
hay construido.

Las dos acciones se muestrean UNA VEZ AL DIA y se mantienen fijas las 24 horas:
    macro  Beta^7   objetivos del dia
    micro  10x10    residuo de valor sobre la valoracion exacta
Sus log-probabilidades se suman: es una sola politica sobre una accion compuesta.
El episodio queda en 30 pasos, que es lo que hace tratable el credit assignment.
"""
import argparse
import os
import os
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from kagsym import obs as O
from kagsym.environment import LADDER
from kagsym.parallel_env import ParallelEnv
from kagsym.macro import N_MACRO
from kagsym.version import fingerprint, model_fingerprint
from kagsym.nets.world import E2EAgent, WorldConfig, N_HIST


# Dinero que hace cada peldano de la ESCALERA contra un rival PASIVO, medido
# 2026-09-20 sobre 3 semillas. Sirve de referencia para la merma: la escalera no
# es tal, los tres niveles que funcionan valen casi lo mismo (171-180k) y el
# nivel 1 (shop-router-0909) esta ROTO -le falta agents_pub/actions.json-.
_BASE_RIVAL = [0.0, 0.0, 179514.0, 171878.0, 171392.0]

# Lo que hace v48 contra un PASIVO, por (dias, nivel). Sin esto la merma no se
# puede calcular con horizontes mezclados: dividir por la marca de 30 dias hace
# que una partida de 15 parezca una paliza nuestra cuando solo es corta.
# Medido el 2026-09-21, 2 semillas. nivel: 1=tope3 2=tope5 3=tope8 4=entero.
BASE_PER_HORIZON = {
    15: {1:  2918, 2:  7727, 3: 14837, 4:  17078},
    20: {1:  6106, 2: 24157, 3: 34184, 4:  61028},
    30: {1: 16049, 2: 43094, 3: 80932, 4: 176422},
}

# LIGAS: (pasos_del_episodio, nivel_de_ESCALERA). Se asciende ganando y se
# DESCIENDE perdiendo -el descenso importa tanto como el ascenso, porque es lo
# que impide el olvido catastrofico-.
#
# Por que por ligas y no eligiendo el peldano a mano: medido, el peldano bueno
# depende del horizonte. A 30 dias la heuristica gana 3/3 en el nivel 2 y 0/3 en
# el 3; a 15 dias gana 3/3 en el 2 y 0,667 en el 3. Elegirlo a ojo cada vez es
# lo que nos dejo entrenando con win=0,000 (sin gradiente en el termino de
# victoria) y con win=1,000 (idem).
#
# Y el horizonte corto NO es mas barato por muestra -el coste por decision son
# 24 turnos en los dos casos- pero si da 1,61x mas TERMINALES por segundo a
# lote igual, que es el recurso escaso: la sd por semilla es de ~9.000 $ y solo
# el terminal la reduce.
# (pasos, nivel_publico, ruta_macro_rival). Si hay ruta, el rival es NUESTRO
# ejecutor con ese macro -optimizado por CEM para ESE horizonte- y el nivel se
# ignora. Medido: a 5 dias el especialista-5d bate al especialista-30d 3.843 vs
# 2.920 (+31 %, 100 % de victorias, simetrico por los dos lados), mientras el
# publico mas fuerte hace 723 $. Sin rival propio, las ligas cortas dan
# win=1,000, que no tiene mas gradiente que win=0,000.
# ESCALERA COMPLETA: (horas_por_dia, dias, tope_de_peones_del_rival).
#
# Una DIAGONAL por la rejilla fibonacci: primero crece el juego (L0-L3), y ya a
# escala de competicion crece el rival (L4-L8). Los dos ejes por separado estan
# medidos:
#
#   horas  2 y 3 dan x_inaccion 0,73 -> actuar DESTRUYE valor, se excluyen
#   dias   por debajo de 5 pasa lo mismo
#   tope   1 y 2 son degenerados (el publico hace 1-2 $) y 13 es indistinguible
#          de no tener tope (175.984 vs 176.422), asi que fibonacci muestrea MAL
#          este eje: todo el gradiente vive en 9-12 y lo salta entero.
#          Por eso aqui van 3/5/8/11, que son los medidos.
#
# A escala reducida el publico se derrumba a ~0 $, asi que L0-L3 no se ganan
# "compitiendo": son rampa de escala y se cruzan rapido. El criterio que importa
# ahi es `x_inaccion`, no la victoria.
# (horas_por_dia, dias, tope_de_peones_del_rival).
#
# EL JUEGO MAS PEQUENO QUE ENSENA ALGO ES 13h x 13d. Medido el 2026-09-21 con
# la heuristica y el mejor de 8 vectores al azar, 3 semillas, techo alcanzable
# como multiplo de la inaccion (3.000 $):
#     5h x  5d  ->  0,99x   IMPOSIBLE: nada supera a no hacer nada en 25 turnos
#     8h x  8d  ->  1,09x   marginal, indistinguible del ruido
#    13h x 13d  ->  4,98x   jugable
#    21h x 21d  ->  9,21x   jugable
# Con un techo de 0,99 y un umbral de ascenso de 1,2 la liga era un BLOQUEO:
# no se podia cruzar ni jugando perfecto. Las dos primeras se descartan.
#
# Del eje del tope: 1 y 2 son degenerados y 13 es indistinguible de sin tope
# (175.984 vs 176.422), asi que fibonacci muestrea mal ese eje -todo el
# gradiente vive en 9-12- y van los valores medidos.
# (horas_por_dia, dias, tope_del_rival, TECHO en multiplos de la inaccion).
#
# El techo es lo MEJOR alcanzable en esa liga, medido con la heuristica y el
# mejor de 8 vectores al azar. Existe porque un umbral de ascenso FIJO es un
# bloqueo: a 5h x 5d el techo es 0,99x -la politica optima es NO HACER NADA,
# no da tiempo a que nada madure ni a amortizar una contratacion- y exigir 1,2x
# hacia la liga imposible de cruzar jugando perfecto.
#
# Esa liga no sobra: es la unica donde la respuesta correcta es la contencion,
# que es justo el error que el agente comete a escala grande (comprar semilla
# que no planta, contratar peones que no amortizan).
#
# Se asciende al alcanzar el 70 % del techo de la liga, no un numero absoluto.
# (horas, dias, tope_del_rival, margen_alcanzable, x_inaccion_alcanzable).
#
# LA MINIMA VIABLE SON 36 TURNOS (6h x 6d). Barrido medido el 2026-09-21,
# 3 semillas, mejor de heuristica + 6 vectores al azar:
#     5h x  5d   25 turnos  x_inaccion 0,99  <- NADA supera no hacer nada
#     6h x  6d   36 turnos             1,03  <- primera jugada rentable
#     8h x  8d   64                    1,10
#    10h x 10d  100                    1,20
#    11h x 11d  121                    1,10
#    13h x 13d  169                    3,54  <- salto x3: discontinuidad
#
# El MARGEN alcanzable es lo mejor demostrado, no un maximo teorico: un techo
# que nadie ha alcanzado hace la liga incruzable. En las ligas duras el margen
# de la heuristica es NEGATIVO (tope 8: -14.973; sin tope: -102.106), asi que
# ahi el objetivo es simplemente GANAR -margen 0-, alcanzable por definicion
# porque el rival lo consigue.
LIGAS = [
    ( 6,  6,    3,   3088,  1.03),   # L0  minima viable
    ( 8,  8,    3,   3291,  1.10),   # L1
    (10, 10,    3,   3602,  1.20),   # L2
    (13, 13,    5,  10565,  3.54),   # L3  tras la discontinuidad
    (21, 21,    5,  27639,  9.21),   # L4
    (24, 30,    3,  60635, 21.40),   # L5  escala de COMPETICION
    (24, 30,    5,  22873, 16.70),   # L6
    (24, 30,    8,      0, 11.40),   # L7  la heuristica pierde -> objetivo: ganar
    (24, 30,   11,      0, 11.00),   # L8
    (24, 30, None,      0, 10.70),   # L9  competicion completa
]
FRAC_TECHO = 0.90
ASCENSO, DESCENSO, PERMANENCIA = 0.60, 0.30, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=1,
                    help="trozos por epoca. 1 = lote completo (como hasta ahora). "
                         "Mismo computo, N veces mas pasos de optimizador.")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--sigma", type=float, default=0.15, help="sigma del canal de VALOR")
    ap.add_argument("--sigma-verb", type=float, default=0.03,
                    help="sigma de los canales de VERBO (logits)")
    ap.add_argument("--level", type=int, default=2, help="peldano de ESCALERA")
    ap.add_argument("--selfplay-quota", type=int, default=2,
                    help="trabajadores SIEMPRE en autojuego, fuera del reparto "
                         "por informacion")
    ap.add_argument("--target-quota", type=int, default=3,
                    help="trabajadores SIEMPRE en los peldanos sin tope (el "
                         "rival de competicion)")
    ap.add_argument("--auto-curriculum", action="store_true",
                    help="reparte los trabajadores entre los peldanos en "
                         "proporcion a la INFORMACION que da cada uno, p*(1-p) "
                         "de su tasa de victoria: maxima en el empate, CERO en "
                         "los extremos. Sustituye a --desliza y su umbral de "
                         "0,85, que era otra constante a ojo. Un peldano nunca "
                         "probado arranca en p=0,5 -informacion maxima-, asi "
                         "que la exploracion no necesita otro numero.")
    ap.add_argument("--rival-macros", default=None,
                    help="rivales POR TRABAJADOR: rutas .npy separadas por coma, "
                         "con '-' donde se quiera dejar el agente publico. "
                         "Permite mezclar publicos con politicas NUESTRAS "
                         "congeladas en la misma escalera. Sin esto, el banco "
                         "sembrado solo entra por la via de --autojuego y con "
                         "--rejilla se queda sin usar: sembrado y mudo.")
    ap.add_argument("--slide", type=float, default=0.0,
                    help="tasa de victoria a partir de la cual un peldano se "
                         "considera EXPRIMIDO y la escalera se desliza hacia "
                         "arriba, quitandolo del muestreo. 0 = desactivado. "
                         "Sin esto seguimos gastando presupuesto en rivales que "
                         "ganamos siempre, y ahi el termino de victoria esta "
                         "clavado en +1 y no da gradiente -el mismo problema de "
                         "saturacion, pero autoinfligido-.")
    ap.add_argument("--levels", default=None,
                    help="peldanos de ESCALERA por TRABAJADOR, separados por "
                         "coma: '2,4,5,2'. Mezcla ESTILOS de rival, no solo "
                         "dificultad. Los publicos no se parecen -medido: v48 "
                         "puntua mas (153.720) pero v16-rc5 es el que menos nos "
                         "deja (36.139)-, asi que entrenar contra uno solo "
                         "arriesga aprender a batir a ESE en vez de a jugar.")
    ap.add_argument("--force-macro", action="store_true",
                    help="re-aplica --init DESPUES de --resume. Sin esto el "
                         "checkpoint machaca el macro pedido, en silencio.")
    ap.add_argument("--rival-macro", default=None,
                    help="ruta .npy: el rival es NUESTRO ejecutor con ese macro")
    ap.add_argument("--own-rival", action="store_true",
                    help="en los peldanos reducidos el rival es NUESTRO ejecutor "
                         "con el macro calibrado de esa escala "
                         "(runs/ligas/macro_{h}h{d}d.npy). Necesario porque los "
                         "agentes publicos son cintas de 719 pasos a 24h/dia y "
                         "hacen ~0 $ en cualquier otra escala.")
    ap.add_argument("--grid", default=None,
                    help="rejilla MEZCLADA de escalas: 'h,d,tope;h,d,tope;...'. "
                         "Cada trabajador juega una. Excluye --mezcla y --ligas.")
    ap.add_argument("--mix", default=None,
                    help="horizontes MEZCLADOS en el mismo lote, en dias: '15,20,30'. "
                         "Cada trabajador juega uno. Excluye --ligas.")
    ap.add_argument("--leagues", action="store_true",
                    help="currículo: asciende ganando, desciende perdiendo")
    ap.add_argument("--league0", type=int, default=0, help="liga inicial")
    ap.add_argument("--hand-cap", type=int, default=None,
                    help="limita NUESTROS peones (curriculo); None = sin tope")
    ap.add_argument("--init", default="runs/macro_vs_v48.npy")
    ap.add_argument("--init-net", default=None,
                    help="checkpoint preentrenado del micro (modo directo)")
    ap.add_argument("--promote-rival", type=float, default=0.0,
                    help="tasa de victoria a partir de la cual el rival se "
                         "sustituye por una copia congelada de la politica "
                         "actual. 0 = desactivado. Ataca un fallo medido: al "
                         "saturar, el termino terminal de victoria se vuelve "
                         "CONSTANTE y deja de informar -igual que perdiendo el "
                         "100 %% contra v48-. En L1 la tasa llego a 0,988 en el "
                         "update 120 y los 480 updates siguientes no anadieron "
                         "nada. Es un control de CURRICULO, no un parametro de "
                         "la politica: no lo aprende el modelo porque no es una "
                         "decision del juego. 0,9 deja la varianza de la senal "
                         "binaria en 0,09 de su maximo 0,25.")
    ap.add_argument("--verb-head", action="store_true",
                    help="da a la red la cabeza de VERBOS aunque el modo no sea "
                         "'ops'. Es lo que permite el hibrido: la heuristica "
                         "sigue proponiendo sus casillas (modo residuo) y la red "
                         "decide verbo y valor en las que la heuristica declara "
                         "vacias y el motor considera legales -5,81 por turno "
                         "frente a 8,94 ofrecidas-. Sin esta cabeza esas "
                         "casillas son invisibles al aprendizaje y se pasa turno "
                         "el 52,8 %% de las veces, contra el 5,3 %% de v48.")
    ap.add_argument("--mode", default="residuo", choices=("residuo", "directo", "ops"))
    ap.add_argument("--resume", default=None,
                    help="continuar desde un checkpoint de RL en vez de reempezar")
    ap.add_argument("--value-epochs", type=int, default=0,
                    help="epocas extra SOLO para el critico (0 = como antes)")
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=None,
                    help="lr del TRONCO; si se omite, el del checkpoint (o 3e-4)")
    ap.add_argument("--lr-heads", type=float, default=None,
                    help="lr de las cabezas; por defecto = --lr")
    ap.add_argument("--kl-target", type=float, default=0.0,
                    help="si >0, el lr se ajusta solo para mantener este kl/dim")
    # 1,8e-4 era el umbral de cuando el KL se media SIN normalizar por
    # dimension. Al normalizarlo, el KL sano de este problema vive en 1e-2 a
    # 4e-2, o sea noventa veces por encima: las epocas abortaban SIEMPRE tras
    # la primera -"[KL corto 1]" en cada update de la sesion entera- y a la vez
    # el controlador seguia bajando el lr para alcanzar un objetivo 111 veces
    # mayor que el umbral de aborto. Los dos mandos tirando en sentidos
    # opuestos: el lr del tronco acabo en 6,6e-6, 45 veces por debajo del
    # nominal, y cada lote daba UN solo paso de gradiente.
    #
    # Ahora por defecto es 2x el objetivo, que es la practica habitual en PPO:
    # el aborto es una red de seguridad para el lote raro, no el regimen normal.
    # A/B medido el 2026-09-21 (11 updates, 8h x 13d + 13h x 13d, rival propio
    # calibrado): con 0,04 el retorno llega a 5,84 y aborta epocas en casi todos
    # los updates; con 0,12 llega a 8,54 y ya no aborta ninguna. El motivo es
    # que la epoca 1 ya mide kl/dim ~0,05 -hay un desfase entre la log-prob del
    # rollout y la recalculada-, asi que un umbral de 2x el objetivo corta antes
    # de dar el primer paso util.
    ap.add_argument("--jepa-warmup", type=int, default=0,
                    help="updates iniciales en los que SOLO se entrena la "
                         "representacion (JEPA + critico), con la politica "
                         "quieta. El codificador es el 82 %% de los parametros "
                         "y hoy se forma unicamente con el gradiente de "
                         "politica, que es escaso y ruidoso. Se alimenta de los "
                         "mismos rollouts, asi que no cuesta interaccion extra.")
    ap.add_argument("--freeze-trunk", type=int, default=0,
                    help="updates tras el calentamiento en los que el TRONCO "
                         "queda congelado y solo se mueven las cabezas, para "
                         "que aprendan sobre una representacion estable en vez "
                         "de sobre un objetivo movil.")
    ap.add_argument("--jepa-weight", type=float, default=0.0,
                    help="peso de la perdida JEPA: predecir el EMBEDDING futuro "
                         "del propio codificador a 1,2,3,5,8,13 dias, con el "
                         "objetivo detenido. A diferencia de --peso-aux, que "
                         "predice unidades crudas del rival, aqui lo "
                         "impredecible desaparece del objetivo porque el "
                         "embedding solo retiene lo que el codificador juzga "
                         "relevante. Vigilar `2_salud/jepa_sd`: si cae a cero, "
                         "colapso.")
    ap.add_argument("--ctx-micro", default="3x3",
                    choices=("1x1", "3x3", "3x3x2", "5x5", "attn"),
                    help="forma del contexto espacial de la cabeza micro. Aqui "
                         "vino el salto medido: 1x1 -> 3x3 dio 1.397 -> 1.949 $.")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="peso de la perdida AUXILIAR: predecir la oferta del "
                         "rival de MANANA desde el estado de HOY. 0 = apagada. "
                         "Obliga al codificador a modelar como crece su granja, "
                         "que es lo que el critico no puede anticipar hoy -su "
                         "liquidacion aporta el 99,1 %% de la varianza diaria-. "
                         "El config traia `peso_aux: 0.1` para una perdida que "
                         "nunca se cableo.")
    ap.add_argument("--rival-flow", action="store_true",
                    help="alimenta la entrada `hist` con la OFERTA INMINENTE "
                         "del rival por producto en 4 horizontes, en vez de "
                         "ceros. Esa entrada existia (N_HIST = 4 x productos) y "
                         "nunca se conecto. Es predecible desde su tablero, que "
                         "se observa entero, y es lo causalmente anterior a "
                         "nuestro ingreso en un mercado compartido.")
    ap.add_argument("--factored-ratio", action="store_true",
                    help="un cociente de importancia POR CABEZA (macro y micro) "
                         "en vez de uno solo sobre la suma de las 1.632 "
                         "dimensiones. Evita que el ruido de exploracion del "
                         "macro -cuyo condicionamiento aporta +1 $ de 953- "
                         "recorte el gradiente del micro, que aporta el resto.")
    ap.add_argument("--sigma-floor-micro", type=float, default=0.0,
                    help="suelo del sigma APRENDIDO de la cabeza micro. 0 = sin "
                         "suelo. Se deja apagado a proposito: el suelo del macro "
                         "existe porque se midio que estrechar perjudica alli; "
                         "para el micro no hay esa medida todavia, y meter el "
                         "numero antes de tenerla seria inventarse una constante.")
    ap.add_argument("--sigma-floor", type=float, default=0.12,
                    help="suelo de log_sigma del MACRO. Medido el 2026-09-21: "
                         "estrechar la busqueda cuesta 26 puntos de tasa de "
                         "acierto (3,3 ee) y el gradiente se extingue solo en "
                         "44 iteraciones si se deja saturar la distribucion.")
    ap.add_argument("--kl-max", type=float, default=0.12,
                    help="si la politica se aleja mas que esto (kl/dim), se "
                         "abortan las epocas. Medido: 6x --kl-objetivo.")
    ap.add_argument("--selfplay", action="store_true",
                    help="rival = instantanea congelada de uno mismo")
    ap.add_argument("--refresh", type=int, default=25,
                    help="cada cuantos updates se congela una version nueva")
    ap.add_argument("--bank-seed", type=str, default=None,
                    help="rutas .npy separadas por coma con las que SEMBRAR el "
                         "banco de rivales. Sin esto el banco solo guarda fotos "
                         "de uno mismo, que son el mismo linaje: si la politica "
                         "se mete en un nicho, sus propias copias se lo premian. "
                         "Sembrarlo con vectores que produjeron optimizadores "
                         "DISTINTOS (CEM, coordenadas, calibraciones de otra "
                         "escala) da rivales que fallan de formas distintas.")
    ap.add_argument("--bank", type=int, default=4,
                    help="cuantas versiones antiguas se guardan como rivales")
    ap.add_argument("--out", default="runs/e2e.pt")
    ap.add_argument("--run-name", default="e2e-control")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WorldConfig(device=dev, sigma_micro=a.sigma, con_ops=(a.mode == "ops" or a.verb_head),
                      sigma_ops=a.sigma_verb, ctx_micro=a.ctx_micro)
    net = E2EAgent(cfg).to(dev)
    vec0 = list(np.load(a.init))
    net.inicializa_macro_en(vec0)
    from kagsym.symbolic import tasks as _T
    _T.MICRO_MODE = a.mode
    if a.kl_target > 0 and a.kl_max < a.kl_target:
        raise SystemExit(f"--kl-max {a.kl_max:g} es MENOR que --kl-objetivo "
                         f"{a.kl_target:g}: el controlador subiria el KL "
                         f"hasta el objetivo y el aborto lo cortaria en cada "
                         f"epoca. Usa --kl-max >= 2x --kl-objetivo.")
    if a.hand_cap is not None:
        from kagsym import macro as _M
        _M.HAND_CAP = a.hand_cap
        print(f"tope de NUESTROS peones: {a.hand_cap}", flush=True)
    # ESCALA DEL OBJETIVO DE VALOR. `fret` va en unidades crudas (media ~20,
    # sd ~8) y la perdida era smooth_l1 con beta=1.0: TODO error mayor de 1
    # unidad caia en regimen L1 puro, con gradiente +-1 que no lleva el tamano
    # del error. Medido en regresion de juguete con senal lineal perfecta y
    # esta misma escala: R2 = -16,58 asi, contra +0,826 normalizando. Explica
    # el critico en -0,405 sin culpar al tronco.
    #
    # El critico predice NORMALIZADO y se desnormaliza para GAE, que necesita
    # unidades crudas porque `ret = adv + Vn` bootstrapea de V.
    _vmu, _vsd, _vn = 0.0, 1.0, 0
    _n_reusados = _n_total = 0
    d0 = {}
    if a.resume:
        # CONTINUAR, no reempezar. Cada run arrancaba del preentreno supervisado
        # y tiraba todo el RL acumulado. Solo es valido si la arquitectura no ha
        # cambiado: hoy N_GLOBAL paso de 75 a 88 y N_MACRO de 7 a 13, y con eso
        # las formas no encajan. Se carga lo que encaje y se dice que se reusa.
        d0 = torch.load(a.resume, map_location="cpu", weights_only=False)
        from kagsym.migrate_ckpt import load_tolerant
        _n_reusados, _n_total, _azar = load_tolerant(net, d0["sd"], a.resume)
        net.to(dev)
        print(f"reanudado desde {a.resume}: {_n_reusados}/{_n_total} tensores "
              f"reusados (update {d0.get('upd','?')}, "
              f"retorno {d0.get('ret', float('nan')):.2f})", flush=True)
    elif a.init_net:
        # Arrancar del micro que YA reconstruye la valoracion por supervision.
        # En modo directo la tabla escrita a mano sale del lazo: la red emite
        # el valor de cada casilla y la heuristica no se usa.
        # TOLERANTE, como `--resume`. El preentreno puede tener otra cabeza de
        # macro (hoy 13 dimensiones contra 14 al anadir `fertilizar`), pero lo
        # que interesa de el -el tronco y el mapa de valor- si encaja. Cargar
        # solo lo compatible evita rehacer el preentreno con cada dimension
        # nueva, y se informa de cuanto se reutiliza para que no pase inadvertido.
        d0 = torch.load(a.init_net, map_location="cpu", weights_only=False)
        from kagsym.migrate_ckpt import load_tolerant
        current = net.state_dict()
        _nok, _ntot, _ = load_tolerant(net, d0["sd"], a.init_net)
        net.inicializa_macro_en(vec0)        # la cabeza de macro, desde el vector
        net.to(dev)
        print(f"micro preentrenado desde {a.init_net}: "
              f"{_nok}/{_ntot} tensores reusados", flush=True)
    # DOS RITMOS. Medido en el arranque desde cero: tras 40 updates
    # `micro.weight` valia 0.013 y los logits de verbo 0.008, contra un sigma de
    # muestreo de 0.15 -el ruido aplastaba a la senal aprendida 19 a 1-, o sea
    # que la cabeza estaba limitada por NUMERO DE PASOS, no por gradiente. Y a
    # la vez el KL por dimension crecia de 0.036 a 0.363 porque el crítico,
    # que comparte tronco, zarandeaba el codificador: la politica se movia
    # muchisimo sin haber aprendido nada, y el dinero hacia pico en el update 15
    # y bajaba desde ahi.
    #
    # Una cabeza lineal tiene que VIAJAR desde cero; el codificador solo tiene
    # que afinarse. Un unico lr no puede servir para las dos cosas.
    # CUATRO GRUPOS, no dos. El KL total se descompone en el del MACRO mas el
    # del MICRO porque las dos politicas son gaussianas independientes, asi que
    # cada una puede tener su propio lazo de control. Con un solo controlador,
    # el macro -que aporta +1 $ de 953 y cuyas pesos crecen x4,3 contra x2,8
    # del micro- gasta el presupuesto de divergencia y frena a la cabeza que si
    # decide: medido en campeonato, saturacion subiendo a 0,16 mientras el lr
    # de TODO bajaba de 3e-4 a 3,6e-5.
    #
    # El grupo 3 (critico, auxiliares) no es politica: no genera KL y no se
    # controla por KL.
    _g_macro, _g_micro, _g_otros, _tronco = [], [], [], []
    for _n, _p in net.named_parameters():
        _b = _n.split(".")[0]
        if _b in ("macro_mu", "log_sigma"):
            _g_macro.append(_p)
        elif _b in ("micro", "micro_ctx", "log_sigma_micro"):
            _g_micro.append(_p)
        elif _b in ("critico", "aux_rival", "jepa_proy", "jepa_pred"):
            _g_otros.append(_p)
        else:
            _tronco.append(_p)
    _cabezas = _g_macro + _g_micro + _g_otros
    _rm_actual = None
    # PELDANOS DE AUTOJUEGO CON RED ENTERA. La identidad del rival vive en el
    # PELDANO, no en el trabajador: el curriculo reasigna trabajadores y sin
    # esto el rival congelado se perderia en la primera reasignacion.
    _pool_red = set()
    _red_congelada = None
    _pool, _pool_p, _asig = [], [], []
    _niveles = ([int(x) for x in a.levels.split(',')] if a.levels else None)
    if _niveles:
        from kagsym.environment import LADDER as _ESC
        print('rivales por trabajador: ' + ', '.join(
            str(_ESC[min(n, len(_ESC)-1)]) for n in _niveles), flush=True)
    if a.lr is None:
        a.lr = 3e-4
        _lr_explicito = False
    else:
        _lr_explicito = True
    _lrc = a.lr_heads if a.lr_heads is not None else a.lr
    # `--lr-cabezas` A SOLAS tambien manda. Sin esto, `_lr_explicito` solo se
    # activaba con `--lr`, asi que al reanudar se restauraban los dos lr del
    # checkpoint y el cambio pedido se perdia EN SILENCIO -misma clase de fallo
    # que `--init` machacado por `--resume`, que costo `--forzar-macro`-. Y aqui
    # lo que importa es el RATIO cabezas/tronco: el controlador de KL reescala
    # los dos grupos a la vez, asi que el ratio es lo unico que sobrevive.
    if a.lr_heads is not None:
        _lr_explicito = True
    # 0 tronco | 1 macro | 2 micro | 3 critico+auxiliares
    opt = torch.optim.Adam([{"params": _tronco, "lr": a.lr},
                            {"params": _g_macro, "lr": _lrc},
                            {"params": _g_micro, "lr": _lrc},
                            {"params": _g_otros, "lr": _lrc}])
    print(f"lr tronco {a.lr:.1e} ({len(_tronco)} tensores) | "
          f"lr cabezas {_lrc:.1e} ({len(_cabezas)} tensores)", flush=True)
    # ESTADO DEL OPTIMIZADOR, no solo los pesos. Sin esto cada reanudacion
    # empieza con los momentos de Adam a cero y el lr de fabrica, asi que el
    # primer paso es enorme: medido, un reanudado de 20 957 $ caia a 4 717 $ en
    # cinco updates, con kl/dim de 16.9 en el primero. Se perdia mas de lo que
    # se recuperaba.
    # Solo si la ARQUITECTURA no cambio. `load_state_dict` del optimizador no
    # valida formas: acepta los momentos viejos y revienta despues, en el primer
    # step(), con "size of tensor a (124) must match b (126)". Si `--resume` no
    # pudo reusar TODOS los tensores, la red es otra y los momentos no valen.
    _arq_igual = (not a.resume) or (_n_reusados == _n_total)
    if a.resume and "opt" in d0 and _arq_igual:
        try:
            # MOMENTOS POR TENSOR, no todo-o-nada. Cuando se expone una
            # constante nueva el vector macro crece, y con el la cabeza:
            # `macro_mu.weight`, `macro_mu.bias` y `log_sigma` cambian de
            # forma. Los otros 66 tensores son identicos y sus momentos siguen
            # siendo validos -son justamente los que evitan el primer paso
            # enorme-. Tirarlos todos por tres que cambiaron es lo que costo
            # 20.957 -> 4.717 $ en cinco updates.
            #
            # `load_state_dict` del optimizador NO valida formas: acepta los
            # momentos viejos y revienta despues, dentro de step(). Asi que se
            # filtran ANTES, comparando con el parametro que les toca. Los que
            # se caen los reinicia Adam solo, en su primer paso.
            _est = dict(d0["opt"])
            _ps = [q for g in opt.param_groups for q in g["params"]]
            _fuera = []
            _nuevo_est = {}
            for _i, _v in (_est.get("state") or {}).items():
                _j = int(_i)
                _ok = _j < len(_ps)
                if _ok:
                    for _c in ("exp_avg", "exp_avg_sq"):
                        _t = _v.get(_c)
                        if _t is not None and tuple(_t.shape) != tuple(_ps[_j].shape):
                            _ok = False
                if _ok:
                    _nuevo_est[_i] = _v
                else:
                    _fuera.append(_j)
            _est["state"] = _nuevo_est
            opt.load_state_dict(_est)
            if _fuera:
                print(f"  momentos reiniciados en {len(_fuera)} de {len(_ps)} "
                      f"tensores (cambiaron de forma); el resto conserva Adam",
                      flush=True)
            # Los MOMENTOS de Adam siempre se restauran -son lo que evita el
            # primer paso enorme-, pero un lr pedido a mano MANDA sobre el del
            # checkpoint. Sin esto, restaurar el optimizador machacaba el
            # cambio de lr que el operador venia a hacer, en silencio.
            if _lr_explicito:
                opt.param_groups[0]["lr"] = a.lr
                opt.param_groups[1]["lr"] = _lrc
                print(f"  lr EXPLICITO ({a.lr:.1e} / {_lrc:.1e}, ratio "
                      f"{_lrc/a.lr:.1f}x), manda sobre el del checkpoint",
                      flush=True)
            print(f"  optimizador restaurado: lr tronco "
                  f"{opt.param_groups[0]['lr']:.2e} cabezas "
                  f"{opt.param_groups[1]['lr']:.2e}", flush=True)
        except Exception as e:
            print(f"  AVISO: no se pudo restaurar el optimizador ({e}); "
                  f"se sigue con los lr de fabrica", flush=True)
    elif a.resume and "opt" in d0:
        print(f"  optimizador NO restaurado: la arquitectura cambio "
              f"({_n_reusados}/{_n_total} tensores). Momentos a cero.", flush=True)
    if a.resume and _n_reusados < _n_total and getattr(net, "n_ops", 0):
        # RESCATE DE LA INACCION. Si la arquitectura cambio, las capas de
        # entrada reiniciadas mandan ruido a la cabeza de valor, que emite
        # negativos; el humgaro prefiere su columna ficticia y TODAS las
        # unidades hacen PASS. Y la inaccion es un ESTADO ABSORBENTE: si nadie
        # actua no hay variacion de la que aprender. Medido: 70 updates
        # clavados en 3.000 $ exactos -el dinero inicial- a 5 dias, contra los
        # 3.746 que hace el mismo macro sin red.
        # El sesgo del canal de valor se repone a 1.0 = "actuar vale algo
        # positivo", que es el unico arranque del que se puede salir.
        with torch.no_grad():
            net.micro.bias[0] = 1.0
        print("  sesgo de valor repuesto a 1.0 (rescate de la inaccion)",
              flush=True)
    print(f"huella del codigo: {fingerprint()}", flush=True)
    print(f"dispositivo: {dev} | rival: {LADDER[a.level]} | init: {a.init}", flush=True)

    from kagsym import spec
    from kagsym.macro import Macro
    # Rollouts repartidos: medido 632 pasos/s en serie contra 10 069 en paralelo
    # con 48 entornos y 10 procesos (15.9x). El arranque son 1-3 s, una vez.
    if a.force_macro:
        # DESPUES del resume a proposito: `inicializa_macro_en` corre antes y el
        # checkpoint lo pisa. Medido: pedi el macro del CEM de 5 dias
        # ([0.121, 0.754, 0.057, ...]) y la red emitia el de backbone6
        # ([0.01, 0.24, 0.02, ...]), o sea el de 30 dias. El experimento no
        # probaba lo que yo creia y nada avisaba.
        net.inicializa_macro_en(vec0)
        print(f"macro FORZADO a {a.init}", flush=True)
    _pasos = a.steps
    if a.mix:
        # MEZCLADO, no secuencial. En secuencia la politica entrena solo a un
        # horizonte y olvida el anterior -medido dos veces: margen real de
        # -82,7 % a -98,3 %-. Se evitan los horizontes muy cortos a proposito:
        # a 5 dias no-hacer-nada da 3.000 $ y nuestra heuristica 2.920, o sea
        # que actuar DESTRUYE valor y la leccion que se aprende ahi es "no
        # hagas nada".
        _dias = [int(x) for x in a.mix.split(",")]
        _pasos = [d * 24 for d in _dias]
        a.days = max(_dias)
        print(f"MEZCLA de horizontes: {_dias} dias -> {_pasos} pasos", flush=True)
    _horas = None
    if a.grid:
        # REJILLA FIBONACCI MEZCLADA. Los tres ejes a la vez, un peldano por
        # trabajador, todos en el mismo lote. Lo que lo hace posible: `spec` y
        # `TOPE_PEONES` son globales POR PROCESO, y la observacion lleva dos
        # dimensiones ABSOLUTAS -EPISODE_STEPS/720 y tope/HANDS_REF- asi que
        # la red sabe en que peldano juega y puede condicionar la politica en
        # vez de promediar los tres.
        #
        # Mezclado y no secuencial, otra vez por lo medido: el curriculo en
        # secuencia dio -98,3 % dos veces porque la politica olvida el peldano
        # anterior; la mezcla dio -75,4 %, el mejor resultado de la sesion.
        #
        # Y con la matriz de transferencia medida: ascendente 0-5 %, descendente
        # 43-96 %. Entrenar SOLO arriba no baja; entrenar mezclado sube Y baja
        # (mezcla 15/20/30 dio 1,22x a 10 dias y 2,66x a 13, no vistos, y
        # ademas gano a 30 dias: 12,78x contra 10,65x del entrenado solo ahi).
        _rej = []
        for t in a.grid.split(";"):
            h, d, tp = (int(x) for x in t.split(","))
            _rej.append((h, d, tp))
        _pasos = [h * d for h, d, _ in _rej]
        _horas = [h for h, _, _ in _rej]
        # TERCER EJE = TOPE DEL RIVAL, no el nuestro. `--tope-peones` limita
        # NUESTROS peones (es una restriccion del espacio de accion propio);
        # el eje de dificultad de la escalera siempre fue el rival. Y es como
        # se calibraron los peldanos en runs/ligas/calibra_rejilla.py, asi que
        # tienen que significar lo mismo o los techos medidos no aplican.
        # 0 = sin tope (rival a plena potencia).
        _rivtope = [(tp if tp > 0 else None) for _, _, tp in _rej]
        a.days = max(d for _, d, _ in _rej)
        print(f"REJILLA mezclada, {len(_rej)} peldanos:", flush=True)
        for h, d, tp in _rej:
            print(f"   {h:>3}h x {d:>3}d = {h*d:>4} turnos, tope {tp}", flush=True)
    env = ParallelEnv(a.envs, n_procs=a.procs, steps=_pasos,
                          macro=Macro.from_vector(vec0),
                          level=(_niveles if _niveles else a.level),
                          mode=a.mode, tope_peones=a.hand_cap, hours=_horas)
    if a.grid:
        env.set_rival_cap(_rivtope)
        print(f"tope del RIVAL por peldano: {_rivtope}", flush=True)
        if a.own_rival:
            # Un rival de verdad en cada peldano. Medido: v48 hace 60.426 $ a
            # 24h x 30d y 0-30 $ a cualquier otra escala, porque su cinta esta
            # indexada por paso para 719 pasos a 24h/dia. Sin esto, en diez de
            # los once peldanos el termino de victoria es gratis y el mercado
            # compartido no lo vacia nadie: se entrena a granjear en un mundo
            # vacio y luego se compite en uno lleno.
            # NUNCA se sustituye en la escala de competicion: ahi v48 SI juega
            # -60.426 $ medidos- y es el unico peldano donde ganar significa
            # algo de verdad. La sustitucion es para los peldanos donde el
            # publico esta muerto, no para ahorrarse al rival de verdad.
            _COMPET = (24, 30)
            _riv = []
            for _h, _d, _ in _rej:
                _f = f"runs/ligas/macro_{_h}h{_d}d.npy"
                if (_h, _d) == _COMPET or not os.path.exists(_f):
                    _riv.append(None)
                else:
                    _riv.append(list(np.load(_f)))
            env.set_rival_macro(_riv)
            _n = sum(1 for v in _riv if v is not None)
            print(f"rival = NUESTRO ejecutor calibrado en {_n}/{len(_rej)} "
                  f"peldanos; publico en los demas", flush=True)
            for (_h, _d, _), _v in zip(_rej, _riv):
                print(f"   {_h}h x {_d}d: "
                      + ("ejecutor con macro calibrado" if _v is not None
                         else "v48-fast-routes"), flush=True)
    # Rangos de indices de entorno que pertenecen a cada trabajador, o sea a
    # cada peldano de la rejilla. `EntornoParalelo` reparte los entornos en
    # orden, `por_proc[k]` seguidos por trabajador.
    _GRUPOS = None
    if a.grid and len(set(env.pasos_proc)) > 1:
        _GRUPOS, _o = [], 0
        for _m in env.por_proc:
            _GRUPOS.append((_o, _o + _m))
            _o += _m
        print(f"ventaja normalizada por peldano: {len(_GRUPOS)} grupos "
              f"{[b - a_ for a_, b in _GRUPOS]}", flush=True)
    if a.rival_macros:
        _rm = []
        for _t in a.rival_macros.split(","):
            _t = _t.strip()
            _rm.append(None if _t in ("-", "") else list(np.load(_t)))
        env.set_rival_macro(_rm)
        _rm_actual = list(_rm)
        # El POOL de peldanos: (nivel, tope, macro). Arranca con la asignacion
        # inicial y el curriculo automatico redistribuye los trabajadores entre
        # ellos segun cuanta informacion da cada uno.
        if _niveles:
            _pool = [(_niveles[i % len(_niveles)],
                      _rivtope[i % len(_rivtope)],
                      _rm[i % len(_rm)]) for i in range(len(_rm))]
            _pool_p = [None] * len(_pool)
            _asig = list(range(len(_pool)))
        print("rivales por trabajador (macro): " + ", ".join(
            ("publico" if x is None else "NUESTRO") for x in _rm), flush=True)
    elif a.rival_macro:
        env.set_rival_macro(list(np.load(a.rival_macro)))
        print(f"rival = nuestro ejecutor con {a.rival_macro}", flush=True)
    _liga = a.league0
    _en_liga = 0
    _RIV_ULT = [0.0]

    def _monta_liga(idx):
        """Crea el entorno de la liga `idx`: horas, dias y tope del rival.

        Cambiar horas o dias obliga a recrear los trabajadores, porque
        `spec.TURNS_PER_DAY` y `spec.EPISODE_STEPS` son globales POR PROCESO.
        """
        hours, days, cap, _marg, _techo = LIGAS[idx]
        spec.set_turns_per_day(hours)
        steps = days * hours
        spec.set_episode_steps(steps)
        a.steps, a.days = steps, days
        e = ParallelEnv(a.envs, n_procs=a.procs, steps=steps,
                            macro=Macro.from_vector(vec0), level=4,
                            mode=a.mode, tope_peones=a.hand_cap,
                            hours=hours)
        e.set_rival_cap(cap)
        print(f"LIGA {idx}/{len(LIGAS)-1}: {hours}h x {days}d, rival v48 "
              f"tope {cap if cap is not None else 'sin tope'} "
              f"({steps} pasos)", flush=True)
        return e

    if a.leagues:
        env.cerrar()
        env = _monta_liga(_liga)

    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
        mlflow.set_experiment("kaggriculture-world-model")
        mlflow.start_run(run_name=a.run_name)
        mlflow.log_params(vars(a))
        # REFERENCIAS. Una curva de dinero sin sus lineas no se puede leer.
        mlflow.log_params({
            "ref_inaccion": 3000,            # medido, a cualquier horizonte
            "ref_azar_legal": 8692,          # verbo legal al azar cada turno
            "ref_heuristica_v48tope5": 30065,
            "ref_backbone6_v48tope5": 32645,
            "ref_termometro_backbone6": -82.7,
            "ref_v48_tope5": 34905,
            "ref_experto_2945_vs_v48": 82382,
        })
        usar_ml = True
    except Exception:
        usar_ml = False

    # PERTURBACION POR EPISODIO, no por dia. Medido: remuestrear cada dia
    # cuesta el 54 % del rendimiento (40 972 $ fijo -> 18 967 $ muestreado),
    # porque 30 dias de temblor aleatorio destruyen la coherencia de la granja.
    eps = torch.randn(a.envs, N_MACRO, device=dev)
    _NC = 1 + getattr(net, "n_ops", 0)
    _FORMA_U = (10, 10) if _NC == 1 else (_NC, 10, 10)
    # Un sigma por canal: valor y verbos viven en escalas distintas.
    # El sigma de la cabeza MICRO ya no es una constante del config: es un
    # `nn.Parameter` de la red y lo aprende PPO, igual que el del macro. Se
    # recalcula en cada uso porque tras `opt.step()` el valor cambia; guardarlo
    # en una variable dejaria un tensor rancio y, en la actualizacion, un grafo
    # que ya no corresponde.
    def SIG():
        return net.log_sigma_micro.exp()
    eps_u = torch.randn(a.envs, *_FORMA_U, device=dev)
    best = -1e18
    # RED DE SEGURIDAD ante colapso. Medido esta noche: un update con kl 25,7
    # llevo la caja de 34.900 $ a 3 $ en quince updates y el controlador
    # reacciono tarde. Guarda el ultimo estado BUENO y lo restaura si el
    # retorno se desploma, bajando ademas el ritmo a la mitad.
    _salvavidas = {"ret": None, "sd": None, "opt": None, "rescates": 0}
    _grad_normas = []
    _ultima_promo = -10**9
    ret_ep = []          # retornos de EPISODIOS CERRADOS, no sumas por update
    acum = np.zeros(a.envs, dtype=np.float64)

    if a.selfplay:
        # Contra v48 perdemos el 100 %: el terminal vale -1 SIEMPRE y no aporta
        # un solo bit sobre lo unico que puntua. Contra uno mismo la tasa ronda
        # el 0.59 con los dos haciendo el mismo dinero (36 385 vs 36 295), que
        # es donde una senal binaria tiene maxima informacion.
        env.set_selfplay(net)
        print("rival: AUTO-JUEGO (instantanea congelada de la politica)", flush=True)

    import copy
    bank = []
    # Semillas del banco: rivales DIVERSOS, no del propio linaje.
    _seed_vectors = []
    for _p in [x.strip() for x in (a.bank_seed or "").split(",") if x.strip()]:
        try:
            _v = np.load(_p)
            if _v.ndim != 1 or _v.shape[0] > N_MACRO:
                print(f"  semilla ignorada {_p}: dims {_v.shape}, se esperaba "
                      f"({N_MACRO},)", flush=True)
                continue
            if _v.shape[0] < N_MACRO:
                # Vector de una version con menos parametros. Se RELLENA con
                # los valores por defecto del `Macro`, que son exactamente los
                # que la constante tenia cuando estaba a ojo: asi el vector
                # viejo sigue describiendo la misma politica. Dejarlo caer
                # -lo que hacia antes- vaciaba el banco en silencio y la run
                # se quedaba en auto-juego puro sin diversidad.
                from kagsym.macro import Macro as _Mc
                _d = np.array(_Mc().to_vector(), dtype=np.float64)
                _d[: _v.shape[0]] = _v
                print(f"  semilla {os.path.basename(_p)}: {_v.shape[0]} dims "
                      f"-> {N_MACRO}, rellenada con los defectos", flush=True)
                _v = _d
            _seed_vectors.append((os.path.basename(_p)[:-4], [float(x) for x in _v]))
        except Exception as _e:
            print(f"  semilla ignorada {_p}: {_e}", flush=True)
    if _seed_vectors:
        print(f"banco sembrado con {len(_seed_vectors)} rivales diversos: "
              f"{', '.join(n for n, _ in _seed_vectors)}", flush=True)

    t0 = time.time()
    # CONTINUIDAD DEL EJE. Al reanudar, el contador volvia a 1 y MLflow pintaba
    # la continuacion SOLAPADA sobre el tramo anterior en vez de a continuacion:
    # la curva de aprendizaje quedaba partida en dos y no se podia leer como
    # una. El desplazamiento la vuelve a unir.
    _upd0 = int(d0.get("upd", 0) or 0) if a.resume else 0
    if _upd0:
        print(f"el eje de pasos continua desde {_upd0}", flush=True)
    for upd in range(1, a.updates + 1):
        if a.selfplay and upd % a.refresh == 0:
            # Banco de versiones antiguas: sin el, la politica puede olvidar
            # como batir a lo que ya sabia batir y ciclar. Se alterna entre la
            # version mas reciente y una del banco elegida por turno.
            bank.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            bank[:] = bank[-a.bank:]
            # La poblacion es semillas diversas MAS fotos propias. Rotar solo
            # entre fotos propias es jugar contra el mismo linaje.
            # Las semillas ocupan UN solo puesto porque se lanzan todas a la
            # vez -una por trabajador-. Con un puesto por semilla la rotacion
            # gastaba turnos identicos.
            _pool = (([("macro", "diversos", None)] if _seed_vectors else [])
                     + [("foto", f"auto{i}", sd) for i, sd in enumerate(bank)])
            _tipo, _nom, _obj = _pool[(upd // a.refresh) % len(_pool)]
            if _tipo == "macro":
                # Todos los rivales diversos A LA VEZ, uno por trabajador:
                # `pon_rival_macro` lo admite de forma nativa. Rotar de uno en
                # uno hace que el gradiente los vea en serie, que es como se
                # olvida de batir a lo que ya sabia batir.
                env.set_rival_macro([v for _, v in _seed_vectors])
                _nom = f"{len(_seed_vectors)} diversos a la vez"
            else:
                rival = E2EAgent(cfg)
                rival.load_state_dict(_obj)
                env.set_selfplay(rival)
            print(f"  [upd {upd}] rival -> {_nom} ({_tipo}), poblacion de "
                  f"{len(_pool)}", flush=True)
        G, B, H, AM, AU, LP, V, R, D, MS = [], [], [], [], [], [], [], [], [], []
        LPMA, LPMI = [], []          # log-prob por cabeza, para el cociente factorizado
        HF = []                      # oferta del rival por dia (objetivo auxiliar)
        JZ = []                      # proyecciones JEPA por dia (objetivo, detenido)
        for _ in range(a.days):
            g, b, hf = env.encode()
            # La entrada `hist` llevaba CEROS desde siempre pese a estar
            # disenada y conectada al codificador. Con --flujo-rival se le da
            # lo que le corresponde: la oferta inminente del adversario, que
            # es el mecanismo por el que nos afecta -vuelca genero, cae el
            # precio marginal, baja nuestro ingreso-.
            h = (hf if a.rival_flow
                 else np.zeros((a.envs, N_HIST), dtype=np.float32))
            HF.append(torch.from_numpy(np.asarray(hf, dtype=np.float32)).to(dev))
            tg = torch.from_numpy(g).to(dev)
            tb = torch.from_numpy(b).to(dev)
            th = torch.from_numpy(h).to(dev)
            with torch.no_grad():
                s = net(tg, tb, th)
                am, lp_m = net.macro_from(s, eps)
                au = s["micro"] + SIG() * eps_u
                _lpu = torch.distributions.Normal(
                    s["micro"], SIG()).log_prob(au)
                lp = lp_m + _lpu.flatten(1).sum(-1)   # provisional; se corrige abajo
                _lp_mi = _lpu.flatten(1).sum(-1)
                if a.jepa_weight > 0:
                    JZ.append(s["jepa_z"].detach())
            rec, fin = env.step_day(mapas=au.cpu().numpy(), macros=am.cpu().numpy())
            # La mascara solo se conoce DESPUES de jugar el dia: dice que
            # dimensiones influyeron de verdad en alguna decision. Se recalcula
            # la log-prob con ella para que el cociente de PPO ignore el resto.
            _ms = env.masks()
            if _ms is not None:
                mt = torch.from_numpy(_ms).to(dev)
                lp = lp_m + (_lpu * mt).flatten(1).sum(-1)
                _lp_mi = (_lpu * mt).flatten(1).sum(-1)
                MS.append(mt)
            # retorno por episodio cerrado, que si es comparable entre updates
            acum += rec
            for k in np.nonzero(fin)[0]:
                ret_ep.append(float(acum[k])); acum[k] = 0.0
            if fin.any():
                idx = torch.from_numpy(np.nonzero(fin)[0]).to(dev)
                eps[idx] = torch.randn(len(idx), N_MACRO, device=dev)
                eps_u[idx] = torch.randn(len(idx), *_FORMA_U, device=dev)
            G.append(g); B.append(b); H.append(h)
            AM.append(am); AU.append(au); LP.append(lp)
            LPMA.append(lp_m); LPMI.append(_lp_mi)
            V.append(s["valor"] * _vsd + _vmu)
            R.append(rec); D.append(fin)
        with torch.no_grad():
            g, b, hf = env.encode()
            ult = (net(torch.from_numpy(g).to(dev),
                      torch.from_numpy(b).to(dev))["valor"] * _vsd + _vmu)

        R = np.array(R); D = np.array(D); Vn = torch.stack(V).cpu().numpy()
        adv = np.zeros_like(R); acc = 0.0; u = ult.cpu().numpy()
        for t in reversed(range(a.days)):
            next_ = u if t == a.days - 1 else Vn[t + 1]
            delta = R[t] + a.gamma * next_ * (1 - D[t]) - Vn[t]
            acc = delta + a.gamma * a.lam * (1 - D[t]) * acc
            adv[t] = acc
        ret = adv + Vn
        _b_mu = float(ret.mean()); _b_sd = float(ret.std()) + 1e-6
        _vn += 1
        _w = 1.0 / min(_vn, 100)
        _vmu = (1 - _w) * _vmu + _w * _b_mu
        _vsd = (1 - _w) * _vsd + _w * _b_sd
        # R2 DEL CRITICO en vuelo. Medido aparte, su techo con datos suficientes
        # es 0,900 y en vuelo iba por 0,675: el hueco es de entrenamiento, no de
        # ruido. Sin registrarlo no hay forma de ver si se cierra.
        _vr = ret.reshape(-1); _vp = Vn.reshape(-1)
        _sse = float(((_vr - _vp) ** 2).sum())
        _sst = float(((_vr - _vr.mean()) ** 2).sum())
        _r2 = 1.0 - _sse / _sst if _sst > 0 else 0.0
        # VENTAJA NORMALIZADA POR PELDANO cuando hay rejilla.
        #
        # `escala` del moldeado es una constante (2.000 $) que NO depende del
        # peldano, asi que el potencial terminal vale ~20 en una partida de 720
        # turnos y ~2,5 en una de 105. Normalizando la ventaja sobre el lote
        # ENTERO, las muestras del peldano grande se llevan casi todo el modulo
        # y las de los pequenos quedan aplastadas contra cero: mezclar dejaria
        # de servir para nada, que es justo lo contrario de lo que se busca.
        #
        # Normalizar por peldano pone a los once a competir en igualdad. Es la
        # practica estandar en RL multitarea y no cambia el signo de ninguna
        # ventaja, solo su escala relativa entre tareas.
        if _GRUPOS is not None:
            for _a, _b in _GRUPOS:
                _sl = adv[:, _a:_b]
                if _sl.size:
                    adv[:, _a:_b] = (_sl - _sl.mean()) / (_sl.std() + 1e-8)
        else:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        fg = torch.from_numpy(np.concatenate(G)).to(dev)
        fb = torch.from_numpy(np.concatenate(B)).to(dev)
        fh = torch.from_numpy(np.concatenate(H)).to(dev)
        fam, fau, flp = torch.cat(AM), torch.cat(AU), torch.cat(LP)
        flpma, flpmi = torch.cat(LPMA), torch.cat(LPMI)
        # OBJETIVO AUXILIAR, horizontes de Fibonacci. Para el estado del dia t,
        # lo que el rival tendra LISTO en t+1, t+2, t+3, t+5, t+8 y t+13. Mas
        # alla del ultimo dia disponible se repite el ultimo: no inventa
        # informacion y mantiene cuadradas las longitudes.
        #
        # Predecir solo t+1 seria casi trivial -el crecimiento de un dia es
        # determinista-; los ciclos del juego viven entre 2 y 13 dias y es ahi
        # donde el codificador tiene que aprender algo.
        # OBJETIVO JEPA: para el dia t, la proyeccion del PROPIO codificador
        # en t+k, detenida y congelada del rollout. Si se recalculara durante
        # las epocas, el objetivo se moveria con el predictor, que es una via
        # directa al colapso.
        _fjz = None
        if a.jepa_weight > 0 and JZ:
            from kagsym.obs import AUX_HORIZONS as _HZ
            _njz = len(JZ)
            _fjz = torch.cat([
                torch.stack([JZ[min(t + k, _njz - 1)] for k in _HZ], dim=1)
                for t in range(_njz)])
        _fhf = None
        if HF:
            from kagsym.obs import AUX_HORIZONS, VENTANAS_RIVAL
            _nd = len(HF)
            # de (dias, envs, 4*P) a la primera ventana: lo LISTO hoy
            _listo = [x.reshape(x.shape[0], len(VENTANAS_RIVAL), -1)[:, 0, :]
                      for x in HF]
            _fhf = torch.cat([
                torch.cat([_listo[min(t + k, _nd - 1)] for k in AUX_HORIZONS],
                          dim=-1)
                for t in range(_nd)])
        fms = torch.cat(MS) if MS else None
        fadv = torch.from_numpy(adv.reshape(-1).astype(np.float32)).to(dev)
        fret = torch.from_numpy(ret.reshape(-1).astype(np.float32)).to(dev)
        kl_cortes = 0
        # (la red de seguridad se inicializa antes del bucle)
        # RED DE SEGURIDAD. El guardia de KL limita cuanto se mueve la politica
        # en un update, pero no repara lo que ya se rompio: medido esta noche,
        # un solo update con kl 25,7 llevo la caja de 34.900 $ a 3 $ en quince
        # updates, y el controlador reacciono tarde. En una run desatendida eso
        # son horas perdidas sin que nadie se entere.
        #
        # Se guarda el ultimo estado BUENO y, si el retorno se desploma por
        # debajo de la mitad del mejor reciente, se restaura y se baja el ritmo
        # a la mitad. No evita el mal update; evita que se lo lleve la noche.

        # MINILOTES. A lote completo, 4 epocas son 4 PASOS de optimizador sobre
        # ~2.160 muestras. Trocear en N no cambia el computo -el mismo numero de
        # gradientes-muestra- pero da 4*N pasos, y cada uno parte de los pesos
        # que actualizo el anterior. Donde mas se nota es en el CRITICO, que es
        # una regresion: su R2 va por 0,80 con techo medido en 0,900, y ese
        # hueco esta documentado como de entrenamiento, no de ruido.
        _N = len(fadv)
        # La POLITICA siempre a lote completo: trocearla multiplica su KL por 24
        # (medido) y el controlador tendria que deshacerlo recortando el lr.
        # `--minilotes` afecta SOLO al critico, mas abajo.
        _nm, _tam = 1, _N
        for _ep in range(a.epochs):
            _orden = torch.randperm(_N, device=dev) if _nm > 1 else None
            for _j in range(_nm):
                if _nm > 1:
                    _sel = _orden[_j * _tam:(_j + 1) * _tam]
                    if len(_sel) == 0:
                        continue
                    _fg, _fb, _fh = fg[_sel], fb[_sel], fh[_sel]
                    _fam, _fau, _flp = fam[_sel], fau[_sel], flp[_sel]
                    _flpma, _flpmi = flpma[_sel], flpmi[_sel]
                    _fadv, _fret = fadv[_sel], fret[_sel]
                    _fms = fms[_sel] if fms is not None else None
                else:
                    _fg, _fb, _fh = fg, fb, fh
                    _fam, _fau, _flp = fam, fau, flp
                    _flpma, _flpmi = flpma, flpmi
                    _fadv, _fret = fadv, fret
                    _fms = fms
                s = net(_fg, _fb, _fh)
                _lp2 = torch.distributions.Normal(
                    s["micro"], SIG()).log_prob(_fau)
                if _fms is not None:
                    _lp2 = _lp2 * _fms
                _lpma = net.macro_logprob(s, _fam)
                _lpmi = _lp2.flatten(1).sum(-1)
                lp = _lpma + _lpmi
                if a.factored_ratio:
                    # UN COCIENTE POR CABEZA. PPO suma la log-prob sobre TODAS
                    # las dimensiones -32 del macro mas 1.600 del micro- en un
                    # unico cociente, asi que el ruido de exploracion del macro
                    # entra en el cociente de cada muestra y arrastra al micro:
                    # cuando el cociente se sale de banda por culpa del macro,
                    # el gradiente del micro se pierde con el.
                    #
                    # Y el macro no lo merece: medido, su condicionamiento
                    # aporta +1 $ de 953. Separar los cocientes deja que cada
                    # cabeza explore lo suyo sin recortar a la otra.
                    #
                    # La mascara `fms` ya atacaba este problema a medias
                    # -ignorar dimensiones que no deciden nada-; esto es la
                    # version completa.
                    _rma = torch.exp((_lpma - _flpma).clamp(-10, 10))
                    _rmi = torch.exp((_lpmi - _flpmi).clamp(-10, 10))
                    l_pi = -0.5 * (
                        torch.min(_rma * _fadv,
                                  _rma.clamp(1 - a.clip, 1 + a.clip) * _fadv)
                        + torch.min(_rmi * _fadv,
                                    _rmi.clamp(1 - a.clip, 1 + a.clip) * _fadv)
                    ).mean()
                    ratio = _rmi
                else:
                    ratio = torch.exp((lp - _flp).clamp(-10, 10))
                    l_pi = -torch.min(ratio * _fadv,
                                      ratio.clamp(1 - a.clip, 1 + a.clip) * _fadv).mean()
                l_v = torch.nn.functional.smooth_l1_loss(
                    s["valor"], (_fret - _vmu) / _vsd)
                l_aux = torch.zeros((), device=dev)
                if a.aux_weight > 0 and _fhf is not None:
                    # symlog: la oferta va de 0 a >100 unidades y un error de
                    # 80 no puede pesar 80 veces mas que uno de 1.
                    _t = _fhf[_sel] if _nm > 1 else _fhf
                    l_aux = torch.nn.functional.smooth_l1_loss(
                        s["rival"], torch.sign(_t) * torch.log1p(_t.abs()))
                l_jepa = torch.zeros((), device=dev)
                if a.jepa_weight > 0 and _fjz is not None:
                    _tj = _fjz[_sel] if _nm > 1 else _fjz
                    # coseno: solo interesa la DIRECCION del embedding; la
                    # norma la puede inflar el codificador sin aprender nada.
                    _p = torch.nn.functional.normalize(s["jepa_p"], dim=-1)
                    _q = torch.nn.functional.normalize(_tj, dim=-1)
                    l_jepa = (1.0 - (_p * _q).sum(-1)).mean()
                if upd <= a.jepa_warmup:
                    # CALENTAMIENTO: solo representacion. La politica no se
                    # mueve, asi que el codificador se forma con un gradiente
                    # DENSO y de baja varianza antes de que nadie dependa de
                    # el. Ataca la carrera de arranque -codificador, critico y
                    # politica persiguiendose- que es candidata a explicar la
                    # loteria del despegue: el critico entra en R2 -1,7 a -4,1
                    # y hasta que sube, la ventaja es ruido.
                    #
                    # El valor SI se entrena: es lo que la politica necesitara
                    # el primer dia y no cuesta nada tenerlo listo.
                    perdida = a.value_weight * l_v + a.jepa_weight * l_jepa
                else:
                    perdida = (l_pi + a.value_weight * l_v + a.aux_weight * l_aux
                               + a.jepa_weight * l_jepa)
                opt.zero_grad(); perdida.backward()
                if a.jepa_warmup < upd <= a.jepa_warmup + a.freeze_trunk:
                    # OJO al intervalo: se congela DESPUES del calentamiento,
                    # no durante. Durante el calentamiento el tronco es
                    # justamente lo que tiene que aprender; congelarlo alli
                    # dejaria la fase sin efecto y luego mediriamos "JEPA no
                    # sirve" cuando lo que no sirvio fue el montaje.
                    #
                    # Se anula el gradiente en vez de sacar los tensores del
                    # optimizador, para no perder el estado de Adam.
                    for _p in _tronco:
                        if _p.grad is not None:
                            _p.grad = None
                # La norma ANTES de recortar, que es lo que devuelve la
                # funcion. Es el diagnostico directo del fallo que mas nos ha
                # costado: con `log_prob` sobre la muestra sin `detach`, mu se
                # cancela y esto vale 0.000e+00 mientras todo lo demas parece
                # normal -el sintoma indirecto era "la columna de despliegue no
                # se mueve", que tardamos 5.456 episodios en leer-.
                _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                _grad_normas.append(float(_gn))
                opt.step()
                if a.sigma_floor > 0:
                    with torch.no_grad():
                        net.log_sigma.clamp_(min=float(np.log(a.sigma_floor)))
                        if a.sigma_floor_micro > 0:
                            net.log_sigma_micro.clamp_(
                                min=float(np.log(a.sigma_floor_micro)))

            # PARADA POR DIVERGENCIA KL. El recorte de gradiente limita la
            # MAGNITUD del paso, no cuanto se mueve la POLITICA. Medido: 80
            # updates estables (win ~0.5, 44 000 $) y de golpe un precipicio
            # -win 0.015, 7 514 $- del que no se recupera. Un solo update malo
            # destruye la politica; esto lo corta antes de que ocurra.
            with torch.no_grad():
                # Sobre el LOTE COMPLETO, con un forward propio. Con minilotes,
                # `lp` es el del ULTIMO trozo y `flp` el de todo: compararlos
                # seria comparar formas distintas.
                _s = net(fg, fb, fh)
                _l2 = torch.distributions.Normal(_s["micro"], SIG()).log_prob(fau)
                if fms is not None:
                    _l2 = _l2 * fms
                lp = net.macro_logprob(_s, fam) + _l2.flatten(1).sum(-1)
                # POR DIMENSION. La log-prob es una SUMA sobre las dimensiones
                # muestreadas, asi que el KL total escala con el tamano de la
                # cabeza: al pasar el micro de 10x10 a 16x10x10 el mismo
                # movimiento real daba 16x mas KL y el guard cortaba en la
                # epoca 1 de las 4, siempre. Normalizar lo hace comparable
                # entre arquitecturas.
                _nd = (N_MACRO + float(fms.flatten(1).sum(-1).mean())
                       if fms is not None
                       else N_MACRO + int(np.prod(fau.shape[1:])))
                kl = float((flp - lp).mean().clamp(min=0)) / _nd
                # KL POR CABEZA. Las dos politicas son gaussianas
                # independientes, asi que el KL total se descompone en la suma
                # y cada mitad puede controlar su propio lr. Sin esto, el macro
                # -que aporta +1 $ de 953- gasta el presupuesto de divergencia
                # y el controlador frena a TODO, incluido el micro, que aporta
                # el resto.
                _lpma2 = net.macro_logprob(_s, fam)
                _lpmi2 = _l2.flatten(1).sum(-1)
                _ndmi = (float(fms.flatten(1).sum(-1).mean()) if fms is not None
                         else int(np.prod(fau.shape[1:])))
                kl_ma = float((flpma - _lpma2).mean().clamp(min=0)) / N_MACRO
                kl_mi = float((flpmi - _lpmi2).mean().clamp(min=0)) / max(1.0, _ndmi)
                _d = (lp - flp)
                _sat = float((_d.abs() > 10).float().mean())
                _sd = float(_d.std())
            if upd % 5 == 0 or upd == 1:
                print(f"    [diag] kl/dim={kl:.2e}  sd(lp-flp)={_sd:.2f}  "
                      f"saturacion del cociente={_sat:.1%}  dims={_nd}", flush=True)
            # LA CABEZA PEOR, no la media. El KL agregado diluye: medido esta
            # noche, `kl_micro` llego a 0,112 mientras el agregado se quedaba
            # por debajo del corte de 0,06 y el guardia no cortaba, con la
            # saturacion subiendo a 0,22. Es el mismo error que el win rate
            # medio escondiendo el estado por peldano.
            #
            # No anade ningun numero nuevo: usa los KL por cabeza que ya se
            # calculan para los controladores de lr.
            if max(kl, kl_ma, kl_mi) > a.kl_max:
                kl_cortes += 1
                break

        # CONTROL DEL PASO POR KL MEDIDO. Adivinar el lr a mano fallo dos
        # veces seguidas: a 3e-4 el kl/dim crecia de 0.036 a 0.363 en 30
        # updates -el dinero hacia pico en el 15 y caia- y a 3e-3 saltaba a
        # 3.23 con 98 % de saturacion en el primer update. El KL ya esta
        # medido cada epoca, asi que el lazo lo cierra el, no yo.
        if a.kl_target > 0:
            def _factor(_k):
                if _k > 2.0 * a.kl_target:
                    return 0.7
                if _k < 0.5 * a.kl_target:
                    return 1.1
                return 1.0
            _fma, _fmi = _factor(kl_ma), _factor(kl_mi)
            # grupo 1 = macro, grupo 2 = micro: cada uno con SU lazo.
            for _ix, _f2 in ((1, _fma), (2, _fmi)):
                if _f2 != 1.0:
                    opt.param_groups[_ix]["lr"] = min(
                        1e-2, max(1e-6, opt.param_groups[_ix]["lr"] * _f2))
            # El TRONCO alimenta a las dos, asi que se frena con la mas
            # exigente y se suelta solo si las dos lo permiten. Y el grupo 3
            # -critico y auxiliares- no es politica: no genera KL y no se toca.
            # EL MACRO NUNCA MAS RAPIDO QUE EL MICRO. Sin esto, el lazo
            # reparte velocidad en proporcion INVERSA al impacto: una cabeza
            # que no cambia la conducta genera poco KL y se le deja correr,
            # mientras la que decide genera mucho y se le frena. Medido en
            # campeonato: lr del macro x6,7 en 15 updates, `micro_w_norma`
            # PLANA -la cabeza que aporta 952 $ de 953 dejo de aprender- y el
            # dinero cayendo de 33.508 a 30.080.
            #
            # Es una RELACION entre dos cantidades medidas, no un numero
            # elegido: el macro puede ir tan rapido como el micro, nunca mas.
            opt.param_groups[1]["lr"] = min(opt.param_groups[1]["lr"],
                                            opt.param_groups[2]["lr"])
            # EL TRONCO SE CONTROLA CON EL KL TOTAL, que es el que de verdad
            # produce: alimenta a las dos cabezas, asi que su efecto sobre la
            # politica es el conjunto, no el minimo de dos lazos ajenos.
            #
            # Antes era `min(_fma, _fmi)` y eso es un TRINQUETE: frena si
            # CUALQUIERA de las dos frena (0,7) pero solo acelera si las DOS
            # aceleran a la vez (1,1). Con dos cabezas moviendose de forma
            # independiente, bajar es mucho mas probable que subir, asi que el
            # tronco cae al suelo aunque el KL no lo pida.
            #
            # Medido en COMPLETO-50, 2.625 updates: lr del tronco x0,18
            # mientras las cabezas subian x1,35 -la razon cabezas/tronco paso
            # de 1,1x a 8,3x- con el KL total en 0,007 contra un objetivo de
            # 0,010. Estaba POR DEBAJO del objetivo: el lazo tenia que estar
            # acelerando el tronco, no estrangulandolo. La representacion
            # compartida quedo casi congelada y solo aprendian las cabezas
            # encima de ella, que coincide con el estancamiento del dinero
            # desde el update 1744.
            #
            # La propiedad de seguridad se conserva sola: si una cabeza se
            # dispara, el KL total sube con ella y el tronco frena igual.
            _ftr = _factor(kl)
            if _ftr != 1.0:
                opt.param_groups[0]["lr"] = min(
                    1e-2, max(1e-6, opt.param_groups[0]["lr"] * _ftr))
            if upd % 5 == 0 or upd == 1:
                print(f"    [lr] tronco {opt.param_groups[0]['lr']:.2e}  "
                      f"macro {opt.param_groups[1]['lr']:.2e} (kl {kl_ma:.3f})  "
                      f"micro {opt.param_groups[2]['lr']:.2e} (kl {kl_mi:.3f})  "
                      f"kl/dim {kl:.3f} -> objetivo {a.kl_target}", flush=True)

        # Epocas EXTRA solo para el critico. Medido: su techo con datos
        # suficientes es R2 = 0.900 y en vuelo va por 0.675; el hueco es de
        # entrenamiento, no de ruido. Compartir las epocas con la politica hace
        # que esta tire del tronco y el valor no llegue a ajustar.
        # MINILOTES SOLO PARA EL CRITICO. Medido: trocear la POLITICA en 8
        # multiplica su KL por 24 y satura el cociente al 100 % -el controlador
        # tendria que recortar el lr en la misma proporcion y el movimiento neto
        # se quedaria igual-. El critico no tiene ese problema: es una regresion
        # y no entra en el cociente de importancia, asi que mas pasos pequenos
        # solo le hacen bien. Su R2 va por 0,80 con techo medido en 0,900.
        _nmv = max(1, int(a.minibatches))
        _tamv = max(1, _N // _nmv)
        for _ in range(a.value_epochs):
          _ordv = torch.randperm(_N, device=dev) if _nmv > 1 else None
          for _jv in range(_nmv):
            if _nmv > 1:
                _sv = _ordv[_jv * _tamv:(_jv + 1) * _tamv]
                if len(_sv) == 0:
                    continue
                _g, _b, _h, _r = fg[_sv], fb[_sv], fh[_sv], fret[_sv]
            else:
                _g, _b, _h, _r = fg, fb, fh, fret
            v_pred = net(_g, _b, _h)["valor"]
            l = torch.nn.functional.smooth_l1_loss(v_pred, (_r - _vmu) / _vsd)
            opt.zero_grad(); l.backward()
            _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            _grad_normas.append(float(_gn))
            opt.step()

        # AUTOJUEGO POR MERITO, no por calendario. Cada peldano "nuestro"
        # lleva una version congelada de la politica. La rotacion NO es
        # rotatoria: se sustituye SOLO la que ya dominamos.
        #
        # Por que importa: una version de hace mil updates que todavia nos gana
        # el 40 % de las veces es el mejor profesor que tenemos, y
        # sobrescribirla porque "le toca" es tirarla. Y como esto es un
        # simulador, medir contra quien nos cuesta es barato: rotar a ciegas
        # es lo que se hace cuando evaluar es caro.
        #
        # El criterio es el mismo que el del deslizamiento -ganarlo de forma
        # consistente- aplicado por peldano, con su tasa propia.
        # UNA SOLA MEDIDA, COMPARTIDA. El relevo por merito y el curriculo
        # automatico corren en el MISMO update (los dos con `% refresco`), y el
        # relevo llama a `olvida_resultados()`, que pone todo el historial a
        # nan. El curriculo, que va justo despues, volvia a preguntar y SIEMPRE
        # encontraba nan: nunca actualizaba su estimacion, asi que repartia
        # uniforme para siempre.
        #
        # Medido en la run COMPLETO-50: 0 reasignaciones en 2.625 updates,
        # cuota 1,00 en los once peldanos y `win_rate` agregado nan. Con 8 de
        # los 11 trabajadores en peldanos que ganabamos al 88-100 %, o sea el
        # 73 % del computo gastado en rivales que por el propio criterio del
        # curriculo no ensenan nada.
        _wpp_ref = None
        if upd % max(1, a.refresh) == 0:
            _wpp_ref = env.win_rate_per_rung()
        if _rm_actual and upd % max(1, a.refresh) == 0:
            try:
                _wpp3 = list(_wpp_ref)
                _mios = [i for i, v in enumerate(_rm_actual) if v is not None]
                # UMBRAL 0,5, y no es una constante a ojo: es la definicion de
                # "somos mejores que esa version". Antes usaba `a.slide`, que
                # por defecto vale 0, asi que `0 % >= 0` era cierto y relevaba
                # precisamente las versiones que nos estaban GANANDO al 100 %.
                _dominados = [i for i in _mios
                              if i < len(_wpp3) and _wpp3[i] == _wpp3[i]
                              and _wpp3[i] > 0.5]
                if _dominados:
                    _nuevo = fam.mean(0).detach().cpu().numpy().tolist()
                    # el mas dominado de todos deja su sitio
                    _k_ref = max(_dominados, key=lambda i: _wpp3[i])
                    _rm_actual[_k_ref] = _nuevo
                    env.set_rival_macro(_rm_actual)
                    # LA RED ENTERA, no el vector macro. `pon_rival_macro`
                    # manda un vector, o sea NUESTRO EJECUTOR CON LA HEURISTICA
                    # DE TABLERO y sin cabeza micro; en modo ops eso le quita
                    # justo el componente que lleva el valor.
                    #
                    # Medido con el MISMO macro en los dos lados, 5 semillas de
                    # 24h x 30d: la red 64.762 $ contra 29.924 del rival de
                    # vector -+116 %, 5 de 5-. Por eso el relevo lo sustituia
                    # 105 veces en una noche sin que dejase de perder: no podia
                    # ganar, le faltaba la mitad del agente.
                    _red_congelada = E2EAgent(cfg)
                    _red_congelada.load_state_dict(
                        {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()})
                    _j_ref = _asig[_k_ref] if _asig and _k_ref < len(_asig) else _k_ref
                    _pool_red.add(_j_ref)
                    env.set_selfplay_on(
                        [k for k, j in enumerate(_asig or [])
                         if j == _j_ref] or [_k_ref], _red_congelada)
                    env.forget_results()
                    print(f"  [upd {upd}] peldano {_k_ref} lo ganabamos al "
                          f"{_wpp3[_k_ref]:.0%}: se releva por NUESTRA politica "
                          f"actual. Los demas se conservan porque aun ensenan",
                          flush=True)
            except Exception as _e:
                print(f"  aviso: no se pudo relevar el autojuego ({_e})",
                      flush=True)
        # ------- CURRICULO AUTOMATICO -------
        if a.auto_curriculum and upd % max(1, a.refresh) == 0 and _niveles:
            try:
                # la medida de ANTES del relevo; ver `_wpp_ref` arriba.
                _w4 = list(_wpp_ref) if _wpp_ref is not None else env.win_rate_per_rung()
                # la estimacion vive en el PELDANO, no en el trabajador: si un
                # trabajador cambia de peldano, su historia se queda con el
                # peldano que jugaba.
                for _k4, _v4 in enumerate(_w4):
                    if _v4 == _v4 and _k4 < len(_asig):
                        _p4 = _pool_p[_asig[_k4]]
                        _pool_p[_asig[_k4]] = (0.7 * _p4 + 0.3 * _v4
                                               if _p4 is not None else _v4)
                _info = [( (p4 * (1.0 - p4)) if p4 is not None else 0.25 )
                         for p4 in _pool_p]          # sin medir -> p=0.5
                # CUOTAS FIJAS, fuera del reparto por informacion. `p(1-p)`
                # mide donde la ESTIMACION es mas incierta, no donde esta el
                # objetivo, y eso falla en los dos extremos:
                #
                #  * Vale 0 cuando perdemos SIEMPRE. El peldano sin tope es el
                #    rival de competicion y lo perdemos al 100 %, asi que el
                #    criterio abandona justo lo que nos mide.
                #  * Vale 0,25 -el MAXIMO- para el autojuego, porque una copia
                #    congelada de nosotros da p = 0,5 por construccion. Sin
                #    cuota se lo llevaria todo y acabariamos entrenando para
                #    ganarnos a nosotros en vez de para ganarles a ellos.
                #
                # Medido el 2026-09-22 contra rival pasivo: nosotros 72.796 $,
                # mediana de 63 agentes publicos 186.594. La distribucion de
                # examen son ellos, no nuestra copia.
                _n = len(_asig)
                _es_auto = [j for j in range(len(_pool)) if _pool[j][2] is not None]
                _es_obj = [j for j in range(len(_pool))
                           if _pool[j][1] is None and _pool[j][2] is None]
                _qa = min(max(0, a.selfplay_quota), len(_es_auto), _n)
                _qo = min(max(0, a.target_quota), len(_es_obj), _n - _qa)
                _fijos = ([_es_auto[i % len(_es_auto)] for i in range(_qa)]
                          + [_es_obj[i % len(_es_obj)] for i in range(_qo)])
                _libres = _n - len(_fijos)
                # el resto, por informacion, y SOLO sobre los peldanos que no
                # tienen cuota propia
                _resto = [j for j in range(len(_pool))
                          if j not in set(_es_auto) | set(_es_obj)]
                _tot = sum(_info[j] for j in _resto) if _resto else 0.0
                if _libres > 0 and _tot > 1e-9:
                    _nuevo_asig = list(_fijos)
                    for _j4 in _resto:
                        _nuevo_asig += [_j4] * max(0, int(round(_libres * _info[_j4] / _tot)))
                    _mejor = max(_resto, key=lambda j: _info[j])
                    _nuevo_asig = (_nuevo_asig[:len(_fijos)]
                                   + _nuevo_asig[len(_fijos):][:_libres])
                    _nuevo_asig += [_mejor] * (_n - len(_nuevo_asig))
                    _nuevo_asig = _nuevo_asig[:_n]
                elif _libres > 0:
                    _nuevo_asig = list(_fijos) + [_fijos[-1] if _fijos else 0] * _libres
                else:
                    _nuevo_asig = list(_fijos)[:_n]
                if True:
                    if _nuevo_asig != _asig:
                        _asig = _nuevo_asig
                        env.set_rival_cap([_pool[j][1] for j in _asig])
                        env.raise_level([_pool[j][0] for j in _asig])
                        env.set_rival_macro([_pool[j][2] for j in _asig])
                        # Los peldanos de autojuego llevan RED, no vector, y
                        # `pon_rival_macro` acaba de machacarla en todos. Se
                        # devuelve a los trabajadores que caen en uno de ellos.
                        if _pool_red and _red_congelada is not None:
                            _vuelven = [k for k, j in enumerate(_asig)
                                        if j in _pool_red]
                            if _vuelven:
                                env.set_selfplay_on(_vuelven, _red_congelada)
                        env.forget_results()
                        _res = {}
                        for j in _asig:
                            _res[j] = _res.get(j, 0) + 1
                        print(f"  [upd {upd}] CURRICULO: reparto por informacion "
                              f"-> " + ", ".join(
                                  f"p{j}x{n}" for j, n in sorted(_res.items())),
                              flush=True)
            except Exception as _e:
                print(f"  aviso: curriculo automatico ({_e})", flush=True)
        if a.slide > 0 and upd % 25 == 0 and _niveles:
            try:
                _wpp = env.win_rate_per_rung()
                # el peldano mas facil es el primero de la lista
                if _wpp and _wpp[0] == _wpp[0] and _wpp[0] >= a.slide:
                    # Al anadir por arriba se ROTA EL ESTILO en vez de
                    # duplicar el ultimo: si no, a fuerza de deslizar los once
                    # peldanos acabarian siendo el mismo rival y perderiamos la
                    # variedad que es el motivo de tener once.
                    _estilos = sorted(set(_niveles))
                    _sig = _estilos[(_estilos.index(_niveles[-1]) + 1)
                                    % len(_estilos)]
                    _rivtope = _rivtope[1:] + [_rivtope[-1]]
                    _niveles = _niveles[1:] + [_sig]
                    env.set_rival_cap(_rivtope)
                    env.raise_level(_niveles)
                    env.forget_results()
                    print(f"  [upd {upd}] ESCALERA DESLIZADA: el peldano mas "
                          f"facil se ganaba al {_wpp[0]:.0%}; sale del muestreo. "
                          f"topes ahora {_rivtope}", flush=True)
            except Exception as _e:
                print(f"  aviso: no se pudo deslizar la escalera ({_e})", flush=True)
        if (a.promote_rival > 0 and upd % 5 == 0
                and upd - _ultima_promo >= a.refresh):
            # La separacion minima es `--refresco`. Sin ella promociona en
            # cascada: vaciar el historial no basta porque cinco updates
            # despues la ventana nueva tiene poquisimos episodios y, ganando,
            # vuelve a dar ~1,0. Medido: 9 promociones en 45 updates.
            _wr = env.win_rate()
            if _wr == _wr and _wr >= a.promote_rival:
                # El rival se ha quedado pequeno: se sustituye por una copia
                # CONGELADA de la politica de ahora. La copia no aprende, asi
                # que el siguiente tramo se juega contra un adversario que ya
                # sabe lo que nosotros sabiamos, y la senal binaria vuelve a
                # tener varianza.
                _nuevo = E2EAgent(cfg)
                _nuevo.load_state_dict({k: v.detach().cpu().clone()
                                        for k, v in net.state_dict().items()})
                env.set_selfplay(_nuevo)
                # Sin esto promociona en cascada: la ventana de 60 episodios
                # sigue llena de victorias contra el rival viejo.
                env.forget_results()
                _ultima_promo = upd
                print(f"  [upd {upd}] RIVAL PROMOCIONADO (win={_wr:.3f} >= "
                      f"{a.promote_rival}): pasa a ser una copia de la "
                      f"politica actual", flush=True)
        if upd % 5 == 0 or upd == 1:
            wr = env.win_rate()
            rv = env.rival_stats()
            ret = float(np.mean(ret_ep[-80:])) if ret_ep else float("nan")
            print(f"upd {upd:4d}/{a.updates}  win={wr:.3f}  ret={ret:7.2f}  "
                  + (f"[KL corto {kl_cortes}] " if kl_cortes else "")
                  + f"$={env.mean_money():7.0f} vs {rv['dinero']:7.0f}  "
                  f"cult={rv['cultivos']:4.1f}r  uds={rv['unidades']:4.1f}r  "
                  f"macro={[round(float(x),2) for x in fam.mean(0)]}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            _ph = getattr(env, "money_by_horizon", lambda: {})()
            if len(_ph) > 1:
                print("        x_inaccion por peldano: " + "  ".join(
                    f"{k} {v/3000.0:.2f}" for k, v in sorted(
                        _ph.items(), key=lambda kv: kv[1])), flush=True)
            # ASCENSO Y DESCENSO. El descenso importa tanto como el ascenso:
            # es lo que impide que la politica olvide las ligas ya ganadas.
            # `PERMANENCIA` evita el trasiego: una liga se juzga con al menos
            # ese numero de updates dentro.
            if a.leagues:
                _en_liga += 1
                if _en_liga >= PERMANENCIA:
                    # DESCENSO POR INACCION, no por victoria. La victoria no
                    # detecta el colapso: medido, una liga en autojuego ascendio
                    # cuatro peldanos con win=1,00 mientras el dinero se quedaba
                    # en 3.000 $ -la caja inicial- y el margen real caia a
                    # -98,3 %. `x_inaccion` si lo detecta, y a CUALQUIER escala:
                    # 1,0 significa que la politica no hace nada.
                    _inac = env.mean_money() / 3000.0
                    _RIV_ULT[0] = float((env.rival_stats() or {}).get("dinero", 0.0) or 0.0)
                    # Y NO SE ASCIENDE ESTANDO INERTE. Un peldano se pasa
                    # PRODUCIENDO valor, no sobreviviendo a un rival aun peor:
                    # en L0 el publico hace 171 $, asi que se gana 1,00 con
                    # x_inaccion 0,99 -por debajo de no hacer nada-. Ascender
                    # ahi es justo la promocion degenerada que hundio las ligas
                    # anteriores, sin autojuego de por medio.
                    # Umbral RELATIVO al techo de la liga, no absoluto.
                    # ASCENSO = LIGA DOMINADA, no "gano mas de lo que pierdo".
                    # Dos condiciones: el MARGEN alcanza el 90 % de lo mejor
                    # demostrado en esa liga, y la politica esta VIVA. Promover
                    # a medio aprender es lo que hizo cascada en los intentos
                    # anteriores.
                    # SOBRE LA INACCION, no sobre cero. Tercera vez que esta
                    # regla se rompe por la misma clase de fallo: el liston se
                    # cruzaba sin producir nada.
                    #   - por victoria: el rival de L0 hace 4 $, se gana 1,00 quieto
                    #   - por margen:   margen 2.973 con solo conservar la caja
                    #   - por x_inaccion bruto: el techo de L0 es 1,03 y la
                    #     inaccion 1,00, asi que TODO el rango aprendible es un
                    #     3 % y cualquier umbral por debajo de 1,0 lo cruza
                    #     una politica inerte.
                    # Lo que hay que exigir es el 90 % del margen APRENDIBLE,
                    # que es lo que hay por encima de no hacer nada.
                    _techo_l = LIGAS[_liga][4]
                    _obj_inac = 1.0 + FRAC_TECHO * max(0.0, _techo_l - 1.0)
                    _marg_actual = _din - _RIV_ULT[0]
                    if _inac >= _obj_inac and _liga < len(LIGAS) - 1:
                        _liga += 1
                        print(f"  [upd {upd}] ASCIENDE: x_inaccion {_inac:.3f} "
                              f">= {_obj_inac:.3f} (techo {_techo_l:.2f}), "
                              f"margen {_marg_actual:.0f} ->", flush=True)
                    elif _inac < 0.35 * LIGAS[_liga][4] and _liga > 0:
                        _liga -= 1
                        print(f"  [upd {upd}] DESCIENDE: politica inerte "
                              f"(x_inaccion {_inac:.2f}) ->", flush=True)
                    else:
                        _en_liga = PERMANENCIA - 1   # sigue midiendo
                        _liga = _liga
                    if _en_liga >= PERMANENCIA:
                        env.cerrar(); env = _monta_liga(_liga); _en_liga = 0
                        acum[:] = 0.0; ret_ep.clear()
            if usar_ml:
                try:
                    import mlflow
                    # QUE SE REGISTRA Y POR QUE. Durante toda una sesion de
                    # depuracion, win_rate y dinero no diagnosticaron NADA: son
                    # el resultado, no la causa. Cada diagnostico real salio de
                    # estas otras, que estaban solo en el log:
                    #   kl/dim          crecio de 0,036 a 0,363 -> divergencia
                    #   saturacion      99,4% -> el cociente de PPO era ruido
                    #   lr_*            el controlador estrangulando el paso
                    #   micro_w_norma   la cabeza no viajaba (0,013 en 40 updates)
                    #   dims_activas    cuantas dimensiones deciden de verdad
                    # GUARD DE INACCION. `startingMoney` son 3.000 $ y no
                    # hacer NADA acaba exactamente en 3.000. Medido: las ligas
                    # en autojuego convergieron a 3.000 clavados a TODO
                    # horizonte -la politica se sentaba sobre la caja- porque
                    # la instantanea congelada hacia 2.588, o sea PEOR que la
                    # inaccion, y ganarle no exigia hacer nada. `win` decia
                    # 1,00 y el margen real era -98,3 %.
                    _din = env.mean_money()
                    _inercia = _din / float(
                        spec.DEFAULT_CONFIG.get("startingMoney", 3000) or 3000)
                    # Solo cuando el horizonte es UNICO. Con mezcla, `liga` es
                    # constante y sobra, y el dinero promediado entre 15/20/30
                    # dias no significa nada: se desglosa.
                    _extra = {}
                    if a.leagues:
                        _extra["liga"] = float(_liga)
                    _porh = getattr(env, "money_by_horizon", lambda: {})()
                    if len(_porh) > 1:
                        # POR PELDANO, y en multiplos de la inaccion ademas de
                        # en dolares. La inaccion son 3.000 $ a CUALQUIER
                        # escala -no gastar deja la caja quieta-, asi que
                        # x_inaccion SI es comparable entre peldanos mientras
                        # que el dinero crudo no lo es: un peldano de 24 turnos
                        # no puede ganar lo que uno de 720, y promediarlos
                        # esconde exactamente lo que hay que vigilar, que es si
                        # ALGUN peldano se ha vuelto inerte.
                        for _d, _v in _porh.items():
                            _extra[f"dinero_{_d}"] = float(_v)
                            _extra[f"x_inaccion_{_d}"] = float(_v) / 3000.0
                        _xs = [v / 3000.0 for v in _porh.values()]
                        _extra["x_inaccion_min"] = float(min(_xs))
                        _extra["peldanos_inertes"] = float(
                            sum(1 for x in _xs if x < 1.02))
                    else:
                        _extra["sobre_inaccion"] = float(_inercia)
                    if len(_porh) > 1:
                        # Con rejilla, el promedio no sirve de aviso: mezcla
                        # peldanos de 24 y de 720 turnos. Se avisa por peldano.
                        _inertes = [k for k, v in _porh.items()
                                    if v / 3000.0 < 1.02]
                        if _inertes and upd > 20:
                            print(f"  AVISO upd {upd}: INERTES {len(_inertes)}/"
                                  f"{len(_porh)} -> " + ", ".join(
                                      f"{k} {_porh[k]/3000.0:.2f}x"
                                      for k in _inertes), flush=True)
                    elif _inercia < 1.05 and upd > 20:
                        print(f"  AVISO upd {upd}: dinero {_din:.0f} ~ "
                              f"no-hacer-nada ({_inercia:.2f}x). Politica inerte.",
                              flush=True)
                    # DASHBOARD EN CUATRO GRUPOS. Antes eran 19 metricas
                    # planas mezclando "voy ganando" con "la maquinaria esta
                    # sana", sin lineas de referencia y con varias rotas por la
                    # mezcla de horizontes. El prefijo numerado las agrupa en
                    # la interfaz y fija el orden de lectura.
                    _RIV = float(rv.get("dinero", 0.0) or 0.0)
                    _m = {
                        # 1_RESULTADO: lo unico que dice si vamos ganando.
                        "1_resultado/win_rate": float(wr),
                        "1_resultado/margen_pct": (100.0 * (_din - _RIV) / _RIV
                                                   if _RIV > 0 else 0.0),
                        # x1 = no hacer nada. MEDIDO: la inaccion deja los
                        # 3.000 $ iniciales a CUALQUIER horizonte. (El "245 $"
                        # que se cito antes era otra cosa: unidades quietas
                        # pero la capa de mercado comprando, que pierde.)
                        "1_resultado/x_inaccion": _din / 3000.0,
                        # 2_SALUD: si la maquinaria aprende.
                        "2_salud/critico_r2": float(_r2),
                        "2_salud/kl_por_dim": float(kl),
                        "2_salud/saturacion": float(_sat),
                        "2_salud/epocas_corridas": float(a.epochs - kl_cortes),
                        # 3_CONTEXTO: contra que se esta midiendo.
                        "3_contexto/dinero_rival": _RIV,
                        # 4_DIAG: solo se miran cuando algo falla.
                        "4_diag/lr_tronco": float(opt.param_groups[0]["lr"]),
                        "4_diag/lr_cabezas": float(opt.param_groups[1]["lr"]),
                        "4_diag/dims_activas": float(_nd),
                        "4_diag/sd_logratio": float(_sd),
                        "4_diag/recompensa": float(R.sum(0).mean()),
                    }
                    if not a.mix:
                        _m["1_resultado/dinero"] = _din
                    # MERMA POR HORIZONTE: cuanto le quitamos al rival
                    # respecto a lo que hace contra un PASIVO en ESE MISMO
                    # horizonte. Es la unica senal competitiva que se mueve
                    # mientras perdemos todas las partidas. Con la base de 30
                    # dias aplicada a partidas de 15 mentia: leia "paliza"
                    # cuando solo era que la partida era corta.
                    _rporh = getattr(env, "rival_by_horizon", lambda: {})()
                    for _d, _rv_d in _rporh.items():
                        _b = BASE_PER_HORIZON.get(_d, {}).get(a.level, 0.0)
                        if _b > 0:
                            _m[f"3_contexto/merma_{_d}d_pct"] = 100.0 * (1.0 - _rv_d / _b)
                    # Escenario: siempre presentes, para poder seguir el curso.
                    _m["3_contexto/liga"] = float(_liga) if a.leagues else -1.0
                    _m["3_contexto/nivel_rival"] = float(a.level)
                    for _k, _v in _extra.items():
                        _m[("1_resultado/" if _k.startswith("dinero")
                            else "3_contexto/") + _k] = float(_v)
                    with torch.no_grad():
                        _m["4_diag/micro_w_norma"] = float(net.micro.weight.norm())
                        # SIGMA DEL MACRO. Medido el 2026-09-21: estrechar la
                        # busqueda cuesta 26 puntos de tasa de acierto (3,3 ee),
                        # asi que sigma cayendo es una ALARMA, no una senal de
                        # convergencia. Se vigila junto a su suelo.
                        # NORMA DEL GRADIENTE, antes de recortar. Es el
                        # diagnostico directo del fallo que mas nos ha costado:
                        # con log_prob sobre la muestra sin detach, mu se
                        # cancela y esto vale 0.000e+00 mientras el resto
                        # parece normal. El sintoma indirecto -"el despliegue
                        # no se mueve"- tardo 5.456 episodios en leerse.
                        if _grad_normas:
                            _gg = _grad_normas[-50:]
                            _m["2_salud/grad_norma"] = float(np.mean(_gg))
                            _m["2_salud/grad_cero_pct"] = 100.0 * float(
                                np.mean([g < 1e-9 for g in _gg]))
                        if a.jepa_weight > 0 and JZ:
                            # VIGILANTE DE COLAPSO. Si el codificador emite
                            # siempre el mismo vector, predecirlo es trivial y
                            # no se aprende nada. Esto cae a cero si pasa.
                            _z = torch.cat(JZ)
                            _m["2_salud/jepa_sd"] = float(_z.std(0).mean())
                        try:
                            _m["2_salud/kl_macro"] = float(kl_ma)
                            _m["2_salud/kl_micro"] = float(kl_mi)
                            _m["4_diag/lr_macro"] = float(opt.param_groups[1]["lr"])
                            _m["4_diag/lr_micro"] = float(opt.param_groups[2]["lr"])
                        except Exception:
                            pass
                        # WIN RATE POR PELDANO. Cada trabajador juega uno, asi
                        # que sin esto el deslizamiento decide con una cifra que
                        # nadie puede auditar: la media de los once esconde un
                        # peldano ganado al 100 % junto a otro perdido al 100 %.
                        try:
                            # INDEXADO POR PELDANO, no por trabajador. El
                            # currículo reasigna trabajadores a peldanos, asi
                            # que `_wr[k]` es "lo que juega el trabajador k
                            # AHORA", no un peldano fijo. Registrarlo por k
                            # dejaba las etiquetas congeladas mientras el
                            # contenido cambiaba: la tabla parecia decir que
                            # empatabamos contra el rival mas duro cuando ese
                            # peldano ni siquiera se estaba jugando.
                            _wpp2 = env.win_rate_per_rung()
                            _rpp2 = env.rival_per_rung()
                            _map = (_asig if _asig else list(range(len(_wpp2))))
                            for _i2, _v2 in enumerate(_wpp2):
                                if _v2 == _v2 and _i2 < len(_map):
                                    _m[f"5_peldano/win_{_map[_i2]:02d}"] = float(_v2)
                            for _i2, _v2 in enumerate(_rpp2):
                                if _v2 == _v2 and _i2 < len(_map):
                                    _m[f"5_peldano/rival_{_map[_i2]:02d}"] = float(_v2)
                            # cuantos trabajadores hay en cada peldano
                            for _j2 in set(_map):
                                _m[f"5_peldano/cuota_{_j2:02d}"] = float(
                                    sum(1 for x in _map if x == _j2))
                        except Exception:
                            pass
                        _m["2_salud/sigma_macro"] = float(net.log_sigma.exp().mean())
                        _sm = net.log_sigma_micro.detach().exp()
                        _m["2_salud/sigma_micro_valor"] = float(_sm[0].mean())
                        if _sm.shape[0] > 1:
                            _m["2_salud/sigma_micro_verbo"] = float(_sm[1:].mean())
                        _m["2_salud/sigma_suelo"] = float(a.sigma_floor)
                        # x_inaccion: 1.0 = la politica esta INERTE. Cuatro
                        # colapsos distintos se habrian visto de un vistazo.
                        _m["1_resultado/x_inaccion"] = float(_din) / 3000.0
                        # cuanto del macro depende del ESTADO. Si es ~0 la
                        # cabeza emite una constante y no condiciona nada.
                        _m["2_salud/macro_w_norma"] = float(net.macro_mu.weight.norm())
                        if getattr(net, "n_ops", 0):
                            _m["2_salud/verbo_senal_ruido"] = float(
                                net.micro.bias[1:].abs().max()) / max(1e-9, cfg.sigma_ops)
                    mlflow.log_metrics(_m, step=_upd0 + upd)
                except Exception:
                    pass
        # Guardar el MEJOR por retorno de episodio, no el ultimo. En el run
        # anterior el update 40 era mejor que el 55 y lo sobrescribi.
        # ULTIMO estado, sin condiciones. Guardar solo "el mejor por retorno"
        # dejaba el checkpoint congelado en el update 16 durante 100 updates:
        # en auto-juego el retorno es RELATIVO y deja de subir en cuanto el
        # rival se fortalece, aunque el agente siga mejorando -medido, el margen
        # contra el 2945 mejoro de -98.7 % a -82.8 % en ese mismo tramo-.
        # Sin esto, una noche sin supervision no acumula nada.
        if upd % 10 == 0 or upd == a.updates:
            torch.save({"sd": net.state_dict(), "cfg": vars(cfg), "init": vec0,
                        "upd": upd, "huella": fingerprint(), "huella_modelo": model_fingerprint(),
                        "opt": opt.state_dict()}, a.out + ".ultimo")
        # --- red de seguridad: comprobar y, si toca, rescatar ---
        if ret_ep and len(ret_ep) >= 40:
            _r80 = float(np.mean(ret_ep[-80:]))
            _prev = _salvavidas["ret"]
            # CAIDA RELATIVA A LA MAGNITUD, no fraccion del valor: con
            # retornos NEGATIVOS `r < 0.5*prev` se invierte -de -0,8 a -0,7 es
            # una MEJORA y disparaba el rescate-. Asi vale para los dos signos.
            _umbral = _prev - 0.5 * abs(_prev) if _prev is not None else None
            if _prev is not None and _r80 < _umbral and _salvavidas["sd"] is not None:
                net.load_state_dict(_salvavidas["sd"])
                try:
                    opt.load_state_dict(_salvavidas["opt"])
                except Exception:
                    pass
                for _gr in opt.param_groups:
                    _gr["lr"] = max(1e-6, _gr["lr"] * 0.5)
                _salvavidas["rescates"] += 1
                ret_ep.clear()
                print(f"  [upd {upd}] RESCATE {_salvavidas['rescates']}: el retorno "
                      f"cayo de {_prev:.1f} a {_r80:.1f}; se restaura el ultimo "
                      f"estado bueno y se halva el ritmo", flush=True)
            elif _prev is None or _r80 > _prev:
                _salvavidas["ret"] = _r80
                _salvavidas["sd"] = {k: v.detach().cpu().clone()
                                     for k, v in net.state_dict().items()}
                _salvavidas["opt"] = opt.state_dict()

        if ret_ep and len(ret_ep) >= 20:
            r80 = float(np.mean(ret_ep[-80:]))
            if r80 > best:
                best = r80
                torch.save({"sd": net.state_dict(), "cfg": vars(cfg),
                            "init": vec0, "ret": r80, "upd": upd,
                            "huella": fingerprint(), "huella_modelo": model_fingerprint(), "opt": opt.state_dict()}, a.out)


if __name__ == "__main__":
    main()
