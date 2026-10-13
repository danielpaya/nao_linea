"""Genera la escena del NAO para entrenar en GPU (MJX): solo los pies chocan con el piso y el solver es mas liviano.

Uso:  python escena_mjx.py nao_assets/scene.xml nao_assets/scene_mjx.xml
Por que:
  - En la escena normal pueden chocar 17 piezas entre si (brazos, muslos, cabeza...). La GPU calcula TODOS los
    pares posibles en cada paso aunque no se toquen, y eso la vuelve lenta y llena la memoria.
  - Para aprender a caminar solo importa el contacto pies-piso. Si otra parte toca el piso, el robot ya se cayo
    y el episodio termina (eso se detecta por inclinacion, no por contacto).
  - El solver de contactos no necesita 100 iteraciones: con pocos contactos converge en 4-8.
"""
import sys
import mujoco

entrada = sys.argv[1] if len(sys.argv) > 1 else "nao_assets/scene.xml"
salida = sys.argv[2] if len(sys.argv) > 2 else "nao_assets/scene_mjx.xml"
QUEDAN = {"floor", "left_foot", "right_foot"}

spec = mujoco.MjSpec.from_file(entrada)
apagadas = 0
for g in spec.geoms:
    if g.name not in QUEDAN and (g.contype or g.conaffinity):
        g.contype, g.conaffinity = 0, 0
        apagadas += 1
spec.option.iterations = 4         # antes 100
spec.option.ls_iterations = 8      # antes 50
m = spec.compile()
open(salida, "w").write(spec.to_xml())
print("Geoms de choque apagadas: %d. Quedan: %s" % (apagadas, sorted(QUEDAN)))
print("Escena para MJX escrita en", salida)
