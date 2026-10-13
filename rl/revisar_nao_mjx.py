"""Revisa que el modelo del NAO V6 funcione en MJX (fisica en GPU) y mide la velocidad."""
import sys, time
import jax, jax.numpy as jp
import mujoco
from mujoco import mjx

ruta = sys.argv[1] if len(sys.argv) > 1 else "nao_assets/scene.xml"
n_robots = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

m = mujoco.MjModel.from_xml_path(ruta)
print("Dispositivo:", jax.devices())
print("Cuerpos %d, articulaciones %d, motores %d, geoms %d" % (m.nbody, m.njnt, m.nu, m.ngeom))
choques = [m.geom(i).name for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
print("Geoms que pueden chocar (%d):" % len(choques), choques)

# 1) mismo resultado en MuJoCo normal (CPU) y en MJX: 1 s de pie con la postura 'home'
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mx = mjx.put_model(m)
dx = mjx.put_data(m, d)
paso = jax.jit(mjx.step)
for _ in range(500):
    dx = paso(mx, dx)
    mujoco.mj_step(m, d)
print("Altura del torso tras 1 s  ->  CPU: %.4f m   MJX: %.4f m" % (d.qpos[2], float(dx.qpos[2])))

# 2) velocidad con muchos robots en paralelo (vmap = la misma funcion sobre un lote)
lote = jax.vmap(lambda d_: mjx.step(mx, d_))
lote = jax.jit(lote)
d0 = mjx.put_data(m, d)
dl = jax.tree_util.tree_map(lambda x: jp.repeat(x[None], n_robots, axis=0), d0)
dl = lote(dl); jax.block_until_ready(dl.qpos)          # la primera llamada compila
t = time.time(); N = 100
for _ in range(N):
    dl = lote(dl)
jax.block_until_ready(dl.qpos)
s = time.time() - t
print("%d robots x %d pasos de fisica en %.2f s  ->  %.0f pasos de fisica/s" % (n_robots, N, s, n_robots * N / s))
