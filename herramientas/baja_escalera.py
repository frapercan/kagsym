"""Baja los cuadernos elegidos y los convierte en rivales jugables.

LA FUENTE ES LA SALIDA DEL CUADERNO, no su codigo. Las competiciones de
simulacion se envian como `submission.tar.gz` con un `main.py` dentro, y el
cuaderno lo GENERA: su codigo fuente suele ser un blob base64 que lo escribe,
asi que extraer las celdas daba un modulo sin `agent` -7 de 15 fallos- o con
un `SyntaxError`. `kernels_output` lo entrega ya construido, y ademas evita
ejecutar codigo descargado solo para extraerlo.

El tar trae tambien datos (`model.json`, `actions.json`) que `main.py` abre por
ruta RELATIVA: de ahi los `FileNotFoundError`. Por eso el modulo que se escribe
en `agents_pub/` es un enganche que importa `main.py` desde SU directorio.

VALIDACION OBLIGATORIA: cada rival juega una partida completa de 24h x 30d
contra el agente que no hace nada. Un rival que revienta o se cuelga a mitad
contamina el peldano entero sin avisar. Referencia: v48 juega 720 turnos en
0,1 s, asi que el tope de 300 s es 3.000 veces lo normal.
"""
import json, os, re, subprocess, sys, tarfile, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents_pub")
CRUDO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "public_agents")
ENGANCHE = '''"""Generado por runs/ligas/baja_escalera.py — no editar a mano.

Carga el `main.py` del submission.tar.gz del cuaderno {ref!r}.
El `chdir` es para que sus lecturas de fichero relativas (model.json,
actions.json) encuentren sus datos; solo dura la importacion.
"""
import importlib.util, os

_DIR = {dir_!r}
_spec = importlib.util.spec_from_file_location({mod!r}, os.path.join(_DIR, "main.py"))
_mod = importlib.util.module_from_spec(_spec)
_ant = os.getcwd()
os.chdir(_DIR)
try:
    _spec.loader.exec_module(_mod)
finally:
    os.chdir(_ant)
agent = _mod.agent
'''


def cuerpo(ipynb_path):
    """Respaldo: codigo del cuaderno, si no hubo salida utilizable."""
    nb = json.load(open(ipynb_path, encoding="utf-8"))
    celdas = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    escritos, rest = [], []
    for c in celdas:
        src = "".join(c.get("source", []))
        m = re.match(r"\s*%%writefile\s+(\S+\.py)\s*\n", src)
        if m:
            escritos.append(src[m.end():])
        elif not re.match(r"\s*%%", src):
            rest.append("\n".join(l for l in src.splitlines()
                                   if not re.match(r"\s*[%!]", l)))
    return "\n\n".join(escritos or rest)


def desde_salida(api, ref, dst):
    """Baja la salida del cuaderno y deja listo `main.py`. Devuelve el dir o None."""
    sal = os.path.join(dst, "salida")
    sub = os.path.join(dst, "sub")
    if os.path.isfile(os.path.join(sub, "main.py")):
        return sub
    os.makedirs(sal, exist_ok=True)
    api.kernels_output(ref, path=sal)
    tgz = [os.path.join(r, f) for r, _, fs in os.walk(sal)
           for f in fs if f.endswith((".tar.gz", ".tgz"))]
    for t in tgz:
        try:
            os.makedirs(sub, exist_ok=True)
            with tarfile.open(t) as tf:
                tf.extractall(sub, filter="data")
        except Exception:
            continue
        if os.path.isfile(os.path.join(sub, "main.py")):
            return sub
    # sin tar: a veces la salida trae el .py suelto
    for r, _, fs in os.walk(sal):
        for f in fs:
            if f == "main.py" or (f.endswith(".py") and re.search(
                    r"^\s*def agent\s*\(",
                    open(os.path.join(r, f), encoding="utf-8",
                         errors="replace").read(), re.M)):
                os.makedirs(sub, exist_ok=True)
                import shutil
                for g in os.listdir(r):
                    shutil.copy(os.path.join(r, g), sub)
                os.rename(os.path.join(sub, f), os.path.join(sub, "main.py"))
                return sub
    return None


def valida(nombre, segundos=300):
    """Carga el modulo y juega una partida entera. (ok, dinero, nota)."""
    prueba = f'''
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kagsym import spec
from kagsym.entorno import carga_publico
from kagsym.fastenv import FastEnv
spec.set_turns_per_day(24); spec.set_episode_steps(24*30)
ag = carga_publico({nombre!r})
env = FastEnv(configuration={{"episodeSteps":24*30,"turnsPerDay":24,
                             "startingMoney":3000}}, seed=4242)
o = env.reset()
while not env.done:
    try:
        a = ag(o[0])
    except Exception as e:
        print("REVIENTA", type(e).__name__, str(e)[:60]); raise SystemExit(2)
    o, _ = env.step([a, {{"farmer":["PASS"],"hands":[],"market":[]}}])
print("OK", int(env.rewards()[0]))
'''
    try:
        p = subprocess.run([sys.executable, "-c", prueba],
                           cwd=R if "R" in dir() else None, capture_output=True,
                           text=True, timeout=segundos)
    except subprocess.TimeoutExpired:
        return False, 0, f"se cuelga (>{segundos}s; v48 tarda 0,1 s)"
    sal = (p.stdout or "").strip().splitlines()
    if sal and sal[-1].startswith("OK"):
        return True, int(sal[-1].split()[1]), "ok"
    err = (p.stderr or p.stdout or "").strip().splitlines()
    return False, 0, (err[-1][:60] if err else "sin salida")
