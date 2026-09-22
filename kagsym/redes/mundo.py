"""Arquitectura end-to-end: un embedding del mundo, dos politicas y una cabeza auxiliar.

               grid (50 canales: 25 mios + 25 del rival)
               glob (75: tiempo, dinero, mercado, pueblo, mi privado)
               hist (flujo REALIZADO del rival, despejado exacto)
                                  |
                        CodificadorMundo
                                  |
                              h_t  (embedding compartido)
                    +-------------+-------------+-------------+
                    |             |             |             |
              cabeza MACRO   cabeza MICRO    critico     cabeza RIVAL
              Beta^7 / dia   mapa 10x10      V(s)        flujo acumulado
                                                          (auxiliar)

Tres decisiones, cada una atada a algo medido:

1. UN SOLO EMBEDDING para macro y micro. Las dos politicas leen el mismo estado;
   lo que cambia es la resolucion de la decision, no la informacion. Compartir
   tronco es ademas la unica forma de que la senal del micro -densa, cada turno-
   ayude a formar la representacion que usa el macro -escasa, una vez al dia-.

2. EL RIVAL ENTRA POR TRES SITIOS: su tablero ya esta en `grid` (es publico),
   el mercado compartido en `glob`, y su flujo realizado en `hist`. Lo unico
   oculto de verdad es su cobertizo, sus semillas y lo que llevan sus unidades,
   y eso se despeja EXACTO del inventario de mercado un turno despues.

3. LA CABEZA AUXILIAR PREDICE NIVEL ACUMULADO, NO TEMPORIZADO. Medido sobre
   8360 transiciones retenidas: predecir el flujo turno a turno es PEOR que no
   corregir (-26 %), mientras que el nivel de dinero del rival (+19 %) y su
   gasto (+35 %) si se aciertan. Se le pide lo segundo. Las etiquetas son
   exactas, asi que es supervision gratis que da forma al embedding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import obs as O
from ..macro import N_MACRO
from .bloques import BloqueSep, ResBlock, symlog

N_PRODUCTOS = 9
N_HIST = 4 * N_PRODUCTOS      # flujo del rival en 4 ventanas hacia atras


@dataclass
class MundoConfig:
    width: int = 128
    blocks: int = 6
    hidden: int = 256
    sigma_micro: float = 0.15   # se calibra midiendo cuantas asignaciones voltea
    # E2E del tablero: ademas del valor por casilla, la red emite un logit por
    # OPERACION legal. Medido 2026-09-20: elegir el verbo al azar entre los
    # legales da 8 692 $ frente a 245 $ de no hacer nada (35x el suelo), asi que
    # hay gradiente desde cero. Y un ranking FIJO al azar da 397 $: la operacion
    # correcta depende del estado, que es justo lo que una heuristica escrita a
    # mano -un ranking fijo- no puede capturar y una red condicionada si.
    con_ops: bool = False
    # Sigma PROPIO para los canales de verbo. El 0.15 de `sigma_micro` se
    # calibro "midiendo cuantas asignaciones voltea" sobre el canal de VALOR,
    # en dolares-symlog; heredarlo para los logits de una eleccion entre 15
    # verbos no tenia justificacion. Medido: con 0.15 la cabeza habia viajado
    # 0.0059 en 200 pasos, senal/ruido 0.039 -el verbo seguia siendo 96 % ruido
    # tras 50 updates- y llegar a taparlo pedia ~4 000 pasos.
    sigma_ops: float = 0.03
    # Forma de la convolucion del tronco. "denso" = la 3x3 de siempre; "sep" =
    # separada en profundidad + punto, que MIDE lo mismo por 6,8x menos coste
    # (ver BloqueSep). El tronco es el 90,5 % de los parametros -1.960.192 de
    # 2.165.558- y el 51 % del tiempo de juego perfilado, asi que 6,8x ahi son
    # ~1,8x de partidas por hora en total.
    #
    # Cambiarlo INVALIDA los checkpoints densos: las formas no coinciden. El
    # camino que no tira el entrenamiento es destilar (runs/ligas/destila_tronco.py).
    # Forma del contexto de la CABEZA MICRO. Aqui vino el salto medido de la
    # noche (1.397 -> 1.949 $ solo por pasar de 1x1 a 3x3), y coincide con lo
    # que la sonda de arquitecturas ya decia: la mezcla que falta es GEOMETRIA
    # LOCAL, no contexto global. "1x1" = sin contexto, como estaba.
    ctx_micro: str = "3x3"
    conv: str = "denso"
    nucleo: int = 3          # tamano de la 3x3 separable; 7 amplia el campo receptivo
    lr: float = 3e-4
    peso_aux: float = 0.1
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class CodificadorMundo(nn.Module):
    """(grid, glob, hist) -> embedding espacial h y resumen global g."""

    def __init__(self, cfg: MundoConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.stem = nn.Sequential(
            nn.Conv2d(O.N_GRID_CH, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w), nn.GELU())
        if getattr(cfg, "conv", "denso") == "sep":
            self.blocks = nn.Sequential(*[BloqueSep(w, cfg.nucleo)
                                          for _ in range(cfg.blocks)])
        else:
            self.blocks = nn.Sequential(*[ResBlock(w) for _ in range(cfg.blocks)])
        # El vector global modula por FiLM en vez de entrar como canal: el dia,
        # los precios y el dinero afectan a TODAS las casillas a la vez.
        self.glob_enc = nn.Sequential(
            nn.Linear(O.N_GLOBAL + N_HIST, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, 2 * w))
        self.resumen = nn.Sequential(
            nn.Linear(O.N_GLOBAL + N_HIST, cfg.hidden), nn.GELU())

    def forward(self, grid, glob, hist=None):
        if hist is None:
            hist = torch.zeros(glob.shape[0], N_HIST, device=glob.device, dtype=glob.dtype)
        g = symlog(torch.cat([glob, hist], dim=-1))
        h = self.stem(symlog(grid))
        escala, sesgo = self.glob_enc(g).chunk(2, dim=-1)
        h = h * (1 + escala[:, :, None, None]) + sesgo[:, :, None, None]
        return self.blocks(h), self.resumen(g)


class _AtencionCasillas(nn.Module):
    """Auto-atencion sobre las 100 casillas del tablero."""

    def __init__(self, w: int, cabezas: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.att = nn.MultiheadAttention(w, cabezas, batch_first=True)
        self.norm2 = nn.LayerNorm(w)
        self.ffn = nn.Sequential(nn.Linear(w, 2 * w), nn.SiLU(), nn.Linear(2 * w, w))

    def forward(self, h):
        b, c, H, W = h.shape
        x = h.flatten(2).transpose(1, 2)          # (b, 100, c)
        y = self.norm(x)
        x = x + self.att(y, y, y, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x.transpose(1, 2).reshape(b, c, H, W)


class AgenteE2E(nn.Module):
    """Todo conectado: un tronco, dos politicas, un critico y una cabeza auxiliar."""

    def __init__(self, cfg: MundoConfig):
        super().__init__()
        self.cfg = cfg
        w, hid = cfg.width, cfg.hidden
        self.mundo = CodificadorMundo(cfg)
        self.cuerpo = nn.Sequential(
            nn.Linear(2 * w + hid, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU())
        # Gaussiana en espacio LOGIT, no Beta. Motivo medido: remuestrear el
        # macro cada dia es un paseo aleatorio sobre 30 dias y destruye la
        # coherencia de la estrategia -se compra semilla que no se planta, se
        # contratan peones sin trabajo-. El coste medido del ruido, con el MISMO
        # vector: 40 972 $ fijo contra 18 967 $ muestreado (-54 %).
        #
        # Con una gaussiana en logit, la perturbacion `eps` se puede FIJAR POR
        # EPISODIO y aplicar a los 30 dias: la media sigue dependiendo del
        # estado -la politica puede subir `venta` al acercarse el cierre- pero
        # el temblor aleatorio desaparece. Es explorar en el espacio de
        # POLITICAS, no en el de acciones.
        self.macro_mu = nn.Linear(hid, N_MACRO)
        self.log_sigma = nn.Parameter(torch.full((N_MACRO,), float(np.log(0.35))))

        # Canal 0 = valor de actuar en la casilla (symexp -> $).
        # Canales 1.. = un logit por verbo de OPS_VOCAB.
        # Van en UN solo tensor a proposito: PPO ya hace flatten(1).sum(-1)
        # sobre el micro, asi que la perdida no cambia ni una linea.
        from ..exacto.tareas import N_OPS
        self.n_ops = N_OPS if cfg.con_ops else 0
        # CAPACIDAD DE LA CABEZA QUE DECIDE. Medido el 2026-09-21 por
        # descomposicion: el mapa micro aporta +952 $ de 953 y el macro +1, y
        # sin embargo esta cabeza eran 2.064 parametros -el 0,1 % del modelo-
        # contra 1,96 M del codificador. Una sonda LINEAL sobre el tronco para
        # la unica salida que decide algo.
        #
        # El 3x3 no es solo capacidad: da CONTEXTO ESPACIAL. En este tablero la
        # decision de una casilla depende de sus vecinas -a quien riegas antes,
        # donde plantas para no dispersar las unidades- y un 1x1 no puede ver
        # eso por construccion.
        #
        # La ultima capa sigue arrancando en CERO, asi que la propiedad medida
        # se conserva: el residuo empieza neutro y no se tira la valoracion
        # exacta, que ya vale 75.157 $.
        _c = getattr(cfg, "ctx_micro", "3x3")
        if _c == "1x1":
            self.micro_ctx = nn.Identity()
        elif _c == "3x3x2":
            # campo receptivo 5 con dos 3x3: mas barato que un 5x5 y con una
            # no linealidad en medio.
            self.micro_ctx = nn.Sequential(
                nn.Conv2d(w, w, 3, padding=1), nn.SiLU(),
                nn.Conv2d(w, w, 3, padding=1), nn.SiLU())
        elif _c == "attn":
            # ATENCION SOBRE LAS 100 CASILLAS, igualada en parametros al 3x3
            # (~130 k contra 147 k) para que la comparacion aisle MEZCLA GLOBAL
            # contra GEOMETRIA LOCAL y no capacidad.
            #
            # Por que aqui y no en el tronco: la sonda vieja midio el tronco y
            # puntuaba IMITAR al experto -que hoy sabemos que es el cuello de
            # botella-, asi que no responde esta pregunta. Y la cabeza micro es
            # donde la atencion tiene sentido: decide casilla por casilla y las
            # casillas interactuan -una unidad solo hace una tarea, e ir a una
            # es no ir a otra-. El 1x1 no ve vecinas, el 3x3 ve ocho, esto ve
            # las cien.
            self.micro_ctx = _AtencionCasillas(w)
        elif _c == "5x5":
            self.micro_ctx = nn.Sequential(nn.Conv2d(w, w, 5, padding=2), nn.SiLU())
        else:
            self.micro_ctx = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.SiLU())
        self.micro = nn.Conv2d(w, 1 + self.n_ops, 1)
        # SIGMA DE LA CABEZA MICRO, APRENDIDO. Antes era `cfg.sigma_micro` y
        # `cfg.sigma_ops`, dos constantes del config: la cabeza que NO aporta
        # nada -la macro- exploraba de forma adaptativa y la que aporta el
        # +952 $ de 953 lo hacia con un numero elegido a mano. Y ese numero se
        # calibro cuando se creia secundaria.
        #
        # Se inicializa EXACTAMENTE en los valores de antes, asi que al
        # arrancar la conducta es identica; a partir de ahi lo mueve PPO, con
        # el mismo mecanismo que ya usa para el macro.
        _nc = 1 + self.n_ops
        _ini = torch.full((_nc, 1, 1), float(np.log(cfg.sigma_ops)))
        _ini[0] = float(np.log(cfg.sigma_micro))
        self.log_sigma_micro = nn.Parameter(_ini)
        # CRITICO. Era `nn.Linear(hid, 1)`: 257 parametros, otra sonda lineal
        # sobre el tronco. Y de el sale la VENTAJA que guia todo PPO, asi que
        # su error se convierte en ruido de gradiente en las dos politicas.
        # En L1 daba R2 0,96 y no urgia; en campeonato, con rival real y
        # mercado compartido, es donde se espera que se rompa -alli la
        # liquidacion del adversario aporta el 99,1 % de la varianza diaria-.
        self.critico = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(),
                                     nn.Linear(hid, 1))
        # CABEZA AUXILIAR, de vuelta con OBJETIVO NUEVO. La anterior predecia
        # los productos del rival, que ya estan en la entrada -su tablero se
        # codifica entero-, asi que era una tarea trivial que no obligaba al
        # codificador a nada. Y ademas nadie cableo su perdida: no recibia
        # gradiente y se quedaba en su inicializacion.
        #
        # Ahora predice lo que el rival tendra LISTO a 1, 2, 3, 5, 8 y 13 dias
        # vista, desde el estado de HOY. Su oferta de hoy si es entrada, asi que acertar manana exige
        # modelar como crece su granja -que madura, que acaba de plantar-, que
        # es exactamente lo que el codificador no representaba.
        #
        # Por que importa: su liquidacion aporta el 99,1 % de la varianza de la
        # recompensa diaria, y por eso meterla en el shaping hundia el critico
        # a R2 -2,535. La leccion fue "el shaping solo puede llevar lo que el
        # estado predice"; esto ataca la otra mitad, enseñar al estado a
        # predecirlo.
        # JEPA. La cabeza auxiliar de arriba predice UNIDADES CRUDAS del
        # rival, y parte de eso es impredecible desde nuestra observacion -su
        # cobertizo es privado, su politica no la vemos-. Forzar al codificador
        # a predecir ruido gasta capacidad; es el defecto que yo mismo senale
        # al elegir ese objetivo.
        #
        # Aqui se predice el EMBEDDING futuro del propio codificador, con el
        # objetivo DETENIDO (stop-gradient). Lo impredecible desaparece del
        # objetivo porque el embedding solo retiene lo que el codificador
        # considera relevante.
        #
        # Receta sin codificador de momento (predictor + stop-grad). El riesgo
        # es el COLAPSO: si el codificador emite siempre el mismo vector,
        # predecirlo es trivial. Se vigila con la desviacion de las
        # proyecciones, que se registra como `2_salud/jepa_sd`.
        self.d_jepa = 64
        self.jepa_proy = nn.Linear(hid, self.d_jepa)
        from ..obs import HORIZONTES_AUX as _HZ
        self.n_hz = len(_HZ)
        self.jepa_pred = nn.Sequential(
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, self.d_jepa * self.n_hz))
        from ..obs import N_AUX_RIVAL as _NAUX
        self.n_aux = _NAUX
        self.aux_rival = nn.Linear(hid, _NAUX)
        # (nota historica) La version anterior fue retirada: la perdida del
        # entrenador es `l_pi + peso_valor * l_v` y nadie toca su salida, asi
        # que no recibia gradiente de nada -se quedaba en su inicializacion
        # para siempre- y solo costaba tiempo en cada pasada. Queda ademas
        # `peso_aux: 0.1` en el config, que era el peso de esa perdida que
        # nunca se cableo: el fosil de un diseno a medias.
        # El micro arranca en residuo CERO: la valoracion exacta ya vale 75 157 $
        # y empezar por debajo de ese punto seria tirar la busqueda.
        nn.init.zeros_(self.micro.weight)
        nn.init.zeros_(self.micro.bias)
        if self.n_ops:
            # En modo "ops" el cero NO es neutro: symexp(0) = 0, el humgaro
            # prefiere su columna ficticia y TODAS las unidades hacen PASS.
            # Medido: 225 $, que es exactamente el suelo de no hacer nada.
            # El arranque correcto es "actuar vale algo positivo y todos los
            # verbos son igual de probables", que es la politica aleatoria
            # legal: 8 692 $ +- 4 003 sobre 6 semillas, 35x ese suelo.
            # No es una heuristica: no dice QUE hacer, solo que hacer algo
            # legal bate a quedarse quieto.
            with torch.no_grad():
                self.micro.bias[0] = 1.0

    def tronco(self, grid, glob, hist=None):
        h, g = self.mundo(grid, glob, hist)
        z = self.cuerpo(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), g], -1))
        return h, z

    def forward(self, grid, glob, hist=None):
        h, z = self.tronco(grid, glob, hist)
        return {
            "macro_mu": self.macro_mu(z),
            "rival": self.aux_rival(z),
            "jepa_p": self.jepa_pred(z).reshape(-1, self.n_hz, self.d_jepa),
            "jepa_z": self.jepa_proy(z),
            "micro": (self.micro(self.micro_ctx(h)) if self.n_ops
                      else self.micro(self.micro_ctx(h)).squeeze(1)),
            "valor": self.critico(z).squeeze(-1),
        }

    def macro_desde(self, salida, eps):
        """Accion macro y su log-prob, con la perturbacion `eps` dada.

        `eps` se muestrea UNA VEZ por episodio y se reutiliza los 30 dias. La
        log-probabilidad se evalua sobre la marginal gaussiana de cada paso, que
        es lo que PPO necesita; la correlacion temporal solo cambia COMO se
        exploran las trayectorias, no la distribucion de cada accion.
        """
        mu = salida["macro_mu"]
        sigma = self.log_sigma.exp()
        pre = mu + sigma * eps
        a = torch.sigmoid(pre)
        lp = (torch.distributions.Normal(mu, sigma).log_prob(pre)
              - (torch.log(a + 1e-8) + torch.log1p(-a + 1e-8))).sum(-1)
        return a, lp

    def logprob_macro(self, salida, a):
        """Recalcula la log-prob de una accion ya tomada (fase de actualizacion)."""
        mu = salida["macro_mu"]
        sigma = self.log_sigma.exp()
        a = a.clamp(1e-6, 1 - 1e-6)
        pre = torch.log(a) - torch.log1p(-a)
        return (torch.distributions.Normal(mu, sigma).log_prob(pre)
                - (torch.log(a) + torch.log1p(-a))).sum(-1)

    def inicializa_macro_en(self, vector, concentracion: float = 6.0) -> None:
        """Centra la Beta inicial en un vector conocido (el que encontro el CEM)."""
        with torch.no_grad():
            nn.init.zeros_(self.macro_mu.weight)
            for i, m in enumerate(list(vector)[:N_MACRO]):
                m = min(0.995, max(0.005, float(m)))
                self.macro_mu.bias[i] = float(np.log(m) - np.log1p(-m))
