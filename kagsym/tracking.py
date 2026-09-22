"""Registro de experimentos con MLflow, opcional y sin acoplar.

Regla de diseno: el harness tiene que funcionar igual sin MLflow instalado. Si
falta el paquete, o se desactiva con KAGWORLD_MLFLOW=0, todo esto degrada a
no-ops silenciosos. Nada de `import mlflow` en el resto del codigo.

Uso::

    with Run("ladder-v1", params={"steps": 3000}) as run:
        run.log_metrics({"loss": 0.1}, step=100)
        run.log_curve("opp_money_res", valores)     # una serie por horizonte
        run.log_artifact("runs/wm.pt")

Por defecto guarda en `data/mlflow.db` (sin servidor). Para verlo:

    .venv312/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_EXPERIMENT = "kaggriculture-world-model"
# El backend de ficheros ('./mlruns') esta en modo mantenimiento y MLflow lanza
# excepcion al usarlo. SQLite ademas encaja con el resto del proyecto, que ya
# lleva su propia BD.
DEFAULT_URI = "sqlite:///data/mlflow.db"
DEFAULT_ARTIFACTS = "./data/mlartifacts"


def available() -> bool:
    if os.environ.get("KAGWORLD_MLFLOW", "1") == "0":
        return False
    try:
        import mlflow  # noqa: F401
        return True
    except Exception:
        return False


def _flatten(d: dict, prefix: str = "") -> dict:
    """MLflow no admite valores anidados; se aplanan con puntos."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v[:8]))
        else:
            out[key] = v
    return out


class Run:
    """Contexto de una corrida. Es un no-op completo si MLflow no esta."""

    def __init__(self, name: str, params: dict | None = None,
                 experiment: str = DEFAULT_EXPERIMENT,
                 tracking_uri: str | None = None, enabled: bool | None = None):
        self.name = name
        self.enabled = available() if enabled is None else (enabled and available())
        self._params = params or {}
        self._experiment = experiment
        self._uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
        self._mlflow = None
        self._run = None

    def __enter__(self) -> "Run":
        if not self.enabled:
            return self
        # Que el tracking falle NUNCA debe tumbar un entrenamiento de 20
        # minutos. Cualquier problema aqui degrada a no-op y se avisa.
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            os.makedirs("data", exist_ok=True)
            os.makedirs(DEFAULT_ARTIFACTS, exist_ok=True)
            import mlflow
            mlflow.set_tracking_uri(self._uri)
            try:
                mlflow.set_experiment(self._experiment)
            except Exception:
                mlflow.create_experiment(self._experiment,
                                         artifact_location=DEFAULT_ARTIFACTS)
                mlflow.set_experiment(self._experiment)
            self._mlflow = mlflow
            self._run = mlflow.start_run(run_name=self.name)
            if self._params:
                self.log_params(self._params)
            print(f"[mlflow] corrida '{self.name}' en {self._uri} "
                  f"(experimento {self._experiment})")
        except Exception as e:
            print(f"[mlflow] desactivado: {type(e).__name__}: {e}")
            self._mlflow = None
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._mlflow is not None:
            if exc_type is not None:
                self._mlflow.set_tag("estado", f"fallo: {exc_type.__name__}")
            self._mlflow.end_run()
        return False

    # -- api ----------------------------------------------------------------
    def log_params(self, params: dict) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_params(_flatten(params))
        except Exception as e:
            print(f"[mlflow] no se pudieron registrar params: {e}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self._mlflow is None:
            return
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and v == v}   # descarta NaN
        if clean:
            try:
                self._mlflow.log_metrics(clean, step=step)
            except Exception as e:
                print(f"[mlflow] no se pudieron registrar metricas: {e}")

    def log_curve(self, name: str, values, offset: int = 1) -> None:
        """Serie indexada (p.ej. error por horizonte), una metrica por punto.

        MLflow la dibuja como curva, que es justo como hay que leer el error de
        rollout: lo que importa es la forma, no el valor final.
        """
        if self._mlflow is None:
            return
        for i, v in enumerate(values):
            self.log_metrics({name: v}, step=i + offset)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if self._mlflow is None or not os.path.exists(path):
            return
        try:
            self._mlflow.log_artifact(path, artifact_path)
        except Exception as e:
            print(f"[mlflow] no se pudo subir {path}: {e}")

    def log_text(self, text: str, filename: str) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_text(text, filename)
        except Exception as e:
            print(f"[mlflow] no se pudo subir {filename}: {e}")

    def set_tags(self, tags: dict) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
        except Exception as e:
            print(f"[mlflow] no se pudieron poner tags: {e}")


def log_horizon(run: Run, metrics: dict, prefix: str = "") -> None:
    """Registra una evaluacion a horizonte completa: curvas + resumen final."""
    keys = [k for k in ("me_sym", "me_res", "opp_sym", "opp_res", "inv_sym", "inv_res",
                        "money_sym", "money_res") if k in metrics]
    for k in keys:
        run.log_curve(f"{prefix}{k}", metrics[k])
    H = metrics.get("horizon", 0)
    final = {}
    for a, b, label in (("me_sym", "me_res", "mi_dinero"),
                        ("opp_sym", "opp_res", "rival"),
                        ("inv_sym", "inv_res", "inventario"),
                        ("money_sym", "money_res", "dinero")):
        if a in metrics and b in metrics:
            sym, res = metrics[a][-1], metrics[b][-1]
            final[f"{prefix}h{H}_{label}_simb"] = sym
            final[f"{prefix}h{H}_{label}_res"] = res
            final[f"{prefix}h{H}_{label}_mejora_pct"] = 100 * (1 - res / max(1e-9, sym))
    run.log_metrics(final)
