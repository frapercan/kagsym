"""Kaggriculture: ejecutor exacto + politicas aprendidas.

El corte es el del proyecto, aplicado al espacio de acciones:

    exacto/   el motor y todo lo que tiene algoritmo exacto y cabe en 1 s:
              legalidad, aritmetica entera, asignacion humgara, liquidacion.
    redes/    lo unico que depende de juicio de valor o del rival:
              el valor de cada tarea (micro) y los objetivos del dia (macro).

Lo que NO esta aqui vive en `kagworld/` y son resultados negativos medidos:
el world model neuronal (acierta el nivel del rival, no el temporizado) y PPO
sobre acciones primitivas (convergio por debajo de no jugar). Ver README.md.
"""
