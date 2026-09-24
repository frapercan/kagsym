"""Envoltorio de submission del agente neurosimbolico.

Interfaz de Kaggle: `def agent(obs, config)` devuelve la accion del turno.

TRES COSAS QUE ESTE ENVOLTORIO TIENE QUE HACER BIEN, y las tres estan medidas
en este proyecto como fuente de fallo:

 1. **Estado entre turnos.** El ejecutor guarda `_destinations`, `prev` y
    `turns_per_tile` -una estimacion autocalibrada-. Crear un `Agent` nuevo
    cada turno lo tira: costo enterrar una medida entera (-342 $, t -2,5) hasta
    que se vio.
 2. **Cargar el modelo UNA vez.** El primer callback es donde se pierden las
    submissions por timeout. Se carga perezosamente y se reutiliza.
 3. **Decidir UNA vez al dia.** El mapa micro es un PLAN DEL DIA; reemitirlo
    cada turno costo -13.346 $ (t -21,5) y ademas multiplica por 24 el coste.

Y el presupuesto: 1 s POR LLAMADA, que NO se acumula, mas 60 s de bolsa por
episodio (`remainingOverageTime`).
"""
import os
import sys

# NADA DE `__file__`. Kaggle NO importa este fichero como modulo: compila su
# fuente y hace `exec(code, {})`, asi que `__file__` no esta definido y
# cualquier uso revienta en la primera linea -- el agente ni llega a cargarse
# y la validacion falla con "Invalid raw Python: NameError".
#
# Lo que SI hace el runner (kaggle_environments/agent.py):
#   * añade `os.path.dirname(path)` a `sys.path`, asi que `import kagsym`
#     funciona sin que hagamos nada;
#   * devuelve la ULTIMA funcion invocable del fichero, asi que `agent` tiene
#     que quedar definida la ultima;
#   * pone la ruta del propio agente en `configuration["__raw_path__"]`.
#
# El directorio se resuelve PEREZOSAMENTE desde el paquete ya importado, que
# es lo unico que no depende de como nos hayan cargado.
# IMPORTAR `kagsym` AQUI, EN TIEMPO DE EXEC. El runner añade el directorio
# del agente a `sys.path` SOLO mientras ejecuta este fuente y lo quita justo
# despues (`sys.path.pop()`). Un import perezoso dentro de `agent()` llega
# tarde: `ModuleNotFoundError: No module named 'kagsym'`.
#
# Una vez importado el paquete, sus submodulos se resuelven por
# `kagsym.__path__` y ya no dependen de `sys.path`.
_RAIZ = None
for _p in list(sys.path):
    try:
        if _p and os.path.isdir(os.path.join(_p, "kagsym")):
            _RAIZ = _p
            break
    except Exception:
        pass
try:
    import kagsym as _KAGSYM          # fija __path__ mientras el path vale
except Exception:
    _KAGSYM = None

_ESTADO = {"net": None, "ag": None, "dia": None, "mapa": None, "ep": None}


def _donde(config):
    """Directorio donde vive `modelo.pt`, sin depender de `__file__`."""
    env_ = os.environ.get("KAGSYM_CKPT")
    if env_ and os.path.exists(env_):
        return env_
    cands = []
    try:
        import kagsym as _k
        cands.append(os.path.dirname(os.path.dirname(os.path.abspath(_k.__file__))))
    except Exception:
        pass
    try:
        raw = (config or {}).get("__raw_path__")
        if raw:
            cands.append(os.path.dirname(os.path.abspath(raw)))
    except Exception:
        pass
    cands += [os.getcwd()] + [p for p in sys.path if p]
    for d in cands:
        r = os.path.join(d, "modelo.pt")
        if os.path.exists(r):
            return r
    raise FileNotFoundError("modelo.pt no encontrado; buscado en " + repr(cands[:4]))


def _arranca(obs, config):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from kagsym import spec
    from kagsym.symbolic.executor import Agent
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_tolerant
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M

    pasos = int(getattr(config, "episodeSteps", None)
                or (config or {}).get("episodeSteps", 720))
    horas = int(getattr(config, "turnsPerDay", None)
                or (config or {}).get("turnsPerDay", 24))
    spec.set_turns_per_day(horas)
    spec.set_episode_steps(pasos)
    _M.HAND_CAP = None

    _ckpt = _donde(config)
    d = torch.load(_ckpt, map_location="cpu", weights_only=False)
    cfg = WorldConfig(**d["cfg"]) if isinstance(d["cfg"], dict) else d["cfg"]
    net = E2EAgent(cfg)
    load_tolerant(net, d["sd"], _ckpt, verbose=False,
                  macro_fields=d.get("macro_fields"))
    net.eval()
    _ESTADO["net"] = net
    _ESTADO["ag"] = Agent(episode_steps=pasos,
                          macro=Macro.from_vector([0.5] * N_MACRO))
    from kagsym.environment import _split_micro
    _ESTADO["ag"].micro = lambda ob: _split_micro(_ESTADO["mapa"])
    _ESTADO["ep"] = pasos
    # LA RAMPA, si el checkpoint la trae. Un checkpoint sin el campo da
    # (None, None) y el desplazamiento es un vector de ceros, o sea la
    # identidad: el comportamiento de siempre, bit a bit.
    #
    # Los DIAS salen de la configuracion del motor, no de un 30 escrito a
    # mano, porque el mismo agente juega episodios de otra longitud.
    from kagsym.rampa import del_checkpoint
    _ESTADO["rampa"], _ESTADO["vivos"] = del_checkpoint(d)
    _ESTADO["dias"] = max(1, pasos // max(1, horas))


def agent(obs, config=None):
    # TODO dentro del try, imports incluidos. Estaban fuera, asi que un
    # ModuleNotFoundError escapaba y la submission quedaba en ERROR en vez de
    # jugar PASS -- que al menos habria dado una partida legible de 3.000 $.
    try:
        if _RAIZ and _RAIZ not in sys.path:
            sys.path.insert(0, _RAIZ)
        import numpy as np
        import torch
        from kagsym import obs as O
        from kagsym.macro import Macro
        if _ESTADO["net"] is None:
            _arranca(obs, config)
        net, ag = _ESTADO["net"], _ESTADO["ag"]
        dia = int(obs["day"])
        if _ESTADO["dia"] != dia:
            g, b = O.encode_obs(obs, getattr(ag, "_destinations", None))
            hf = np.asarray(O.rival_flow(obs), dtype=np.float32)
            with torch.no_grad():
                out = net(torch.from_numpy(g).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0),
                          torch.from_numpy(hf).unsqueeze(0))
                # DETERMINISTA: la media, sin ruido de exploracion.
                from kagsym.rampa import offset as _off
                from kagsym.macro import N_MACRO as _NM
                _o = _off(_ESTADO.get("rampa"), _ESTADO.get("vivos") or [],
                          dia / _ESTADO.get("dias", 30), _NM)
                ag.macro = Macro.from_vector(torch.sigmoid(
                    out["macro_mu"][0] + torch.from_numpy(_o)).numpy())
                _ESTADO["mapa"] = out["micro"][0].numpy()
            _ESTADO["dia"] = dia
        return ag(obs)
    except Exception:
        # El motor convierte lo imposible en no-op, asi que un fallo aqui
        # seria una partida silenciosa de 3.000 $. Se deja rastro en stderr.
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"farmer": ["PASS"], "hands": [], "market": []}
