"""Que un instrumento NO ESTE CIEGO, y que no haya dos bucles de juego.

El 2026-09-24 el mismo fallo costo el dia tres veces, y siempre con la misma
forma: DOS DEFINICIONES del mismo bucle que divergen.

  1. la vara troceaba el micro a mano en vez de `_split_micro`
  2. la busqueda pasaba `zeros(1, N_HIST)` y no pasaba `_destinations`
     -- el MISMO episodio en el MISMO tablero: 25.335 $ contra 76.607
  3. la vara no leia `ck["delta_rampa"]`, asi que comparo una rampa contra su
     base y devolvio `+0 $ +- 0` con sd 0

El tercero es el peligroso: **un instrumento ciego dice "no hay efecto" con
exactamente las mismas palabras que un nulo verdadero**. Un test de espejo
-un checkpoint contra si mismo da 0- lo habria PASADO. Por eso el canario no
es un espejo: es un par cuya diferencia conocemos, y el instrumento tiene que
recuperarla con el signo correcto.
"""
import ast
import os
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quien tiene permiso para emitir el macro por su cuenta. Cada nombre de esta
# lista es una definicion MAS del mismo bucle, o sea una ocasion mas de
# divergir. La lista deberia encoger, nunca crecer: lo correcto es que todos
# llamen a un unico `jugar()`.
PERMITIDOS = {
    "submit_kagsym/main.py",     # el agente REAL: es la referencia
    "kagsym/policy.py",          # la politica del entrenamiento
    "kagsym/rampa.py",           # la definicion unica del desplazamiento
    "kagsym/outer.py",
    "kagsym/search.py",
    "tools/paired_yardstick.py", # la vara pareada
    "tools/evalua.py",
    "tools/cem_diales.py",
    "tools/matriz.py",
    "tools/banda.py",
    "tools/diales_vivos.py",
    "tools/verifica_interruptores.py",
    "tools/por_que_no_fresas.py",
    "tools/cotejo_submission.py",
}


def _ficheros():
    for sub in ("tools", "kagsym", "submit_kagsym"):
        d = os.path.join(RAIZ, sub)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.endswith(".py"):
                yield f"{sub}/{n}", os.path.join(d, n)


def test_no_aparecen_bucles_de_juego_nuevos():
    """Un fichero nuevo que emita el macro es una definicion mas que mantener.

    No prohibe escribirlo: obliga a declararlo aqui, que es donde alguien
    mirara cuando dos medidas discrepen. Se recorre el AST y no el texto
    porque tres auditorias por inspeccion dijeron "no queda nada" y las tres
    fallaron.
    """
    intrusos = []
    for rel, ruta in _ficheros():
        if rel in PERMITIDOS:
            continue
        try:
            arbol = ast.parse(open(ruta, encoding="utf-8").read())
        except SyntaxError:
            continue
        for nodo in ast.walk(arbol):
            # `out["macro_mu"]` en cualquiera de sus formas
            if isinstance(nodo, ast.Subscript):
                sl = nodo.slice
                if isinstance(sl, ast.Constant) and sl.value == "macro_mu":
                    intrusos.append(rel)
                    break
    assert not intrusos, (
        "estos ficheros emiten el macro y no estan declarados en PERMITIDOS: "
        + ", ".join(sorted(set(intrusos)))
        + ". Cada uno es otra definicion del bucle de juego, y de ahi han "
          "salido los tres fallos del 2026-09-24. Declaralo o -mejor- haz que "
          "llame al bucle unico."
    )


def test_el_historico_no_se_pasa_a_ceros():
    """`zeros(..., N_HIST)` es la firma del fallo que mutila al agente.

    El entrenamiento (`environment.encode` -> `Hf[i]=rival_flow`) y el agente
    que se sube lo rellenan los dos. Quien lo ponga a ceros mide otra cosa.
    `paired_yardstick` es la excepcion declarada: tiene `KAG_HIST=0` a
    proposito para poder medir el fallo viejo en semillas pareadas.
    """
    # Excepciones DECLARADAS, con su razon. Reservar un array de ceros para
    # rellenarlo acto seguido es legitimo; pasarselo a la red sin rellenar no.
    LEGITIMOS = {
        # `Hf[i] = O.rival_flow(o[0])` en la linea siguiente
        "kagsym/environment.py",
        # `KAG_HIST=0` existe a proposito para medir el fallo viejo pareado
        "tools/paired_yardstick.py",
    }
    culpables = []
    for rel, ruta in _ficheros():
        if rel in LEGITIMOS:
            continue
        txt = open(ruta, encoding="utf-8").read()
        for linea in txt.splitlines():
            if "N_HIST" in linea and "zeros" in linea and not linea.strip().startswith("#"):
                culpables.append(f"{rel}: {linea.strip()[:70]}")
    assert not culpables, (
        "historico a ceros, que ya costo una noche: " + " | ".join(culpables))
