<div align="center">

# 🧠 nao_linea · rama `rl-nao-mjx`

### Caminata por RL para un NAO V6, entrenada desde cero en una GPU de casa

*Copa NAO CCM MX 2026 · Carrera Individual · carril de 5 m × 70 cm*

![MJX](https://img.shields.io/badge/simulador-MuJoCo%20MJX-1f6feb)
![Brax](https://img.shields.io/badge/RL-Brax%20PPO-8250df)
![NAO](https://img.shields.io/badge/robot-NAO%20V6-34a853)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%208%20GB-76b900)
![estado](https://img.shields.io/badge/estado-entrenando%20v02%2Fv03-fbbc04)

</div>

---

## 📌 Resumen

| | |
|:--|:--|
| **Qué es** | Una política PPO propia que hace caminar al NAO V6 recto y rápido, entrenada desde cero |
| **Dónde** | WSL2 (Ubuntu 24.04) · RTX 5060 8 GB · JAX 0.6.2 + Brax 0.14.2 + MuJoCo Playground |
| **Escala** | 4096 NAOs simulados a la vez · ~15 600 pasos/s · 60 M de pasos ≈ 56 min |
| **Mejor resultado hasta ahora** | **v01b:** 5 m en **16.4 s**, desvío lateral de **12 cm**, 83 % de metas en condiciones no vistas |
| **Registro completo** | [`bitacora.txt`](bitacora.txt): cada decisión, error y fallo (base para el paper) |

---

## 🛠️ Cómo se montó

1. **WSL2 + Ubuntu 24.04** con el driver de Windows (617.14, CUDA 12) visible dentro de Linux.
2. **MuJoCo Playground** (DeepMind) clonado e instalado con `uv`. Se fija `jax[cuda12]==0.6.2`, la
   versión de su `uv.lock`: con JAX 0.11 se rompe `device_put_replicated` de Brax.
3. **Prueba del stack** con los ejemplos CartPole y G1 del Playground, antes de tocar el NAO.
4. **Modelo del NAO V6** (el mismo de B-Human / PenaltyStack) revisado en MJX con `rl/revisar_nao_mjx.py`.
5. **Escena de entrenamiento propia** (`rl/escena_mjx.py`): solo choques pie–piso.
6. **Entorno `NaoCarrera`** (`rl/nao_carrera.py`) y generaciones que cambian **una sola cosa** cada vez.

> Reglas del entorno: siempre `~/nao_rl/mujoco_playground/.venv/bin/python`, nunca `uv sync` otra vez
> (borra las librerías de NVIDIA) y `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`.

---

## 🤖 Diseño del entorno

### Simulación

| Parámetro | Valor |
|:--|:--|
| Paso de control | **24 ms** (2 ciclos de 12 ms del NAO), 12 subpasos de física de 2 ms |
| Episodio | 1000 pasos (24 s) · comando nuevo cada ~10 s |
| Integrador / solver | `implicitfast` · Newton, 4 iteraciones, 8 de búsqueda de línea |
| Contacto | cono piramidal · fricción 1.0 / 0.005 / 0.0001 · solref (0.02, 1) · solimp (0.9, 0.95, 0.001) |
| Motores | posición, kp 40, kv 0.5, torque ±5 N·m |
| Robot | 5.36 kg · pie de 15 × 7.5 × 3 cm · HipYawPitch acoplada por una restricción de igualdad |

### Acción

**13 motores**: las dos piernas (HipYawPitch compartida) y los hombros, que ayudan al equilibrio.
La política da un desplazamiento alrededor de la postura inicial.

### Observación (actor)

Solo lo que da el NAO real; **nada de velocidad lineal ni posición global**.

| Señal | Tamaño |
|:--|:--:|
| Giroscopio | 3 |
| Gravedad proyectada (inclinación) | 3 |
| Ángulos de motores (respecto a la postura base) | 13 |
| Velocidades de motores | 13 |
| Acción anterior | 13 |
| Comando (vx, giro) | 2 |
| Fase de la marcha | 2 |
| **Error de rumbo** (desde v01b) | 1 |

- Total: 49 valores (50 con el rumbo), **× 3 pasos de historia**, es decir 147 → 150. Giroscopio,
  inclinación, ángulos y velocidades llevan ruido de sensor.
- El **error de rumbo** es el rumbo deseado (integra el giro pedido) menos el rumbo estimado (integra el
  giroscopio con una deriva de 0.005 rad/s), recortado a ±0.5 rad. Así la política aprende a corregir
  hacia una referencia que luego puede venir de otra fuente.

### Crítico (privilegiado)

Ve además las señales sin ruido y lo que el robot real no puede medir: velocidad lineal del torso,
altura del torso y de cada pie, tiempo en el aire, fricción del piso, empujón y retraso actual. Red del actor 128×3 y del crítico 256×3.

### Recompensa

| Término | Escala | Para qué |
|:--|:--:|:--|
| `seguir_vel` | + | Ir a la velocidad pedida y sin moverse de lado (σ = 0.025) |
| `seguir_giro` | + | Girar lo pedido (σ = 0.1) |
| `seguir_rumbo` | +1.0 | Mantener el rumbo de salida (σ = 0.03 rad², ~10°), desde v01b |
| `tiempo_aire` | + | Pasos con fase de vuelo |
| `fase_pies` | + | Que el pie suba según una curva de 3 cm ⚠️ calibrada para el G1 |
| `vel_z`, `giro_xy`, `orientacion` | − | Torso estable y derecho |
| `pies_resbalan` | − | Que el pie apoyado no patine |
| `pies_juntos` | − | Separación mínima de 8 cm (el V6 tiene ~3 cm entre piernas y se pisa) |
| `postura_cadera` | −0.5 | Castigo a girar o abrir la cadera, desde v01c |
| `cambio_accion`, `energia`, `limites` | − | Movimientos suaves y dentro de rango |
| `quieto` | − | No quedarse parado si se pide avanzar |
| `caida` | −100 | Termina el episodio |

**Caída:** torso por debajo de 20 cm o inclinado demasiado.

### Robustez (domain randomization)

| Perturbación | Rango | Desde |
|:--|:--|:--:|
| Fricción (piso **y** pies, porque MuJoCo usa el máximo de los dos) | 0.3 – 1.2 | v02 |
| Masa de cada pieza | ±10 % | v02 |
| kp de cada motor | ×0.8 – 1.2 | v02 |
| Un motor de pierna al azar "caliente" (menos torque máximo) | 50 – 100 % | v03 |
| Retraso de la acción | 0 – 2 ciclos (0 – 48 ms) | v03 |
| Empujones al torso | 0.05 – 0.3 m/s cada 3 – 6 s | v04 |

---

## 🧭 Decisiones clave

| Decisión | Por qué |
|:--|:--|
| Escena de entrenamiento solo con choques pie–piso | Con los 17 geoms: 9 908 pasos de física/s y sin memoria; solo pies: **360 952/s (36×)** con la misma dinámica de caminata |
| Velocidad objetivo hasta 0.3 m/s | La meta es ganar; 0.3 m/s es lo que anda la caminata tipo B-Human |
| Los hombros entran en la acción | Los brazos ayudan al equilibrio |
| Actor–crítico asimétrico | El crítico aprende mejor con información privilegiada; el actor queda desplegable en el NAO |
| Ruido en generaciones aparte | Si se mete todo junto y falla, no se sabe qué lo rompió |
| Política cada 24 ms | Múltiplo del ciclo de 12 ms del NAO |
| Guardar checkpoint en cada evaluación | Si algo falla de noche no se pierde la corrida |
| Evaluación con condiciones vistas / no vistas | Para medir generalización y no solo lo que se entrenó |

---

## 📈 Generaciones

Cada una cambia **una sola cosa** respecto a la anterior (`GENERACIONES` en [`rl/entrenar.py`](rl/entrenar.py)).

| Gen | Cambio | Hipótesis | Estado |
|:--|:--|:--|:--|
| v01 | Base, mundo ideal | Aprende a caminar a la velocidad pedida | ✅ 60 M |
| **v01b** | + error de rumbo en la observación y `seguir_rumbo` | Deja de girar al arrancar | ✅ 60 M |
| v01c | + `postura_cadera` −0.5 | Deja de torcer las piernas | ✅ 60 M |
| v02 | + fricción, masas y kp al azar | Generaliza a pisos y motores distintos | 🔄 noche 1 (120 M) |
| v03 | + motor débil y retraso de 0–2 ciclos | Aguanta un motor caliente y la latencia | 🔄 noche 1 (120 M) |
| v04 | + empujones | Se recupera de choques | ⏳ |

### Resultados (11 condiciones × 8 carreras de 5 m, sin perturbaciones de entrenamiento)

| | v01 | **v01b** | v01c |
|:--|:--:|:--:|:--:|
| Tiempo en 5 m (nominal) | 19.1 s | **16.4 s** | 17.2 s |
| Desvío lateral | 233 cm | **12 cm** | 60 cm |
| Rumbo máximo | 45.7° | **3.6°** | 14.7° |
| Metas en condiciones no vistas | 23/80 (29 %) | 66/80 (83 %) | **71/80 (89 %)** |

**Condiciones:** nominal · fricción 0.5 / 0.2 / 1.5 · kp al 70 % · un motor al 70 / 40 % · retraso de
1 / 3 ciclos · empujón de 0.25 / 0.45 m/s. Cada una se marca como vista o no vista según la generación.

- **Ninguna aguanta 3 ciclos (72 ms) de retraso**: es lo que ataca v03.
- **v02 (vista previa, semilla 9):** deja de caerse hacia los 17 M pasos; mejor recompensa 93.3 a 44.6 M.
  A 44 M: `seguir_vel` 1437/1500 y `seguir_rumbo` 991/1000. Pero `tiempo_aire` ≈ 0: **casi no levanta los
  pies**. El checkpoint de 44 M está en `rl/politicas/nao_v02_s9_44M/`.

---

## ❌ Lo que no funcionó (y cómo se arregló)

| Problema | Causa | Solución |
|:--|:--|:--|
| `ALREADY_EXISTS` y luego `Aborted` al importar JAX | Plugins de CUDA 12 y 13 a la vez | Desinstalar el plugin cuda13 |
| JAX caía a CPU (no encontraba cuDNN) | `uv sync` borró las librerías de NVIDIA | `uv pip install jax[cuda12]==0.6.2` y no volver a correr `uv sync` |
| `device_put_replicated` roto | JAX 0.11.2 incompatible con Brax 0.14.2 | Fijar la versión del `uv.lock` |
| Escena completa lenta y sin memoria | 17 geoms con choques entre todos | Escena solo pies–piso |
| Scripts pegados con heredoc salían truncados | Límite de la terminal | Copiar los scripts desde el repo |
| v01 giraba ~40° al arrancar y torcía la pierna | No tenía ninguna referencia de rumbo; la cadera estaba libre | v01b (rumbo) y v01c (castigo de cadera). Se descartó *reward hacking*: `seguir_vel` = 914 |
| `rsync` a `/mnt/c` fallaba (chgrp, mkstemp, tiempos) | El sistema de archivos de Windows no acepta esos atributos | `cp -r`, o enlaces simbólicos de `~/nao_rl/runs` a la carpeta de Windows |
| El log de la cola nocturna llega con retraso | stdout de Python con búfer | Pendiente: `PYTHONUNBUFFERED=1` en `cola_noche.sh` |
| La GPU marca 0 % los primeros minutos | Compilación de XLA (3.5 – 6 min por corrida) | Es normal; se ve en `estado.txt` y en el log |
| v02 casi no levanta los pies | `fase_pies` calibrado para el G1, no para el NAO | Pendiente (posible v01d) |

---

## 🌗 Flujo de día y de noche

| | Día | Noche |
|:--|:--|:--|
| Duración | Ciclos de ~1 h (60 M pasos) | Cola de ~8 h |
| Qué se hace | Una generación, video, evaluación y decidir | Varias generaciones y semillas seguidas |
| Cómo | `entrenar.py` en tmux | `cola_noche.sh noches/noche_NN.txt` en tmux |
| Al terminar | `evaluar.py` y `ver_politica.py` | `resumen_manana.py` evalúa lo que falte y escribe `runs/resumen_<fecha>.md` |

La cola sigue aunque una corrida falle, y el resultado de cada corrida queda en `estado.txt`
(corriendo / terminado).

---

## 📁 Archivos

| Ruta | Qué es |
|:--|:--|
| `rl/nao_carrera.py` | Entorno `NaoCarrera` (MJX): reset, step, observación, recompensas, caída y `aleatorizar()` |
| `rl/escena_mjx.py` | Escena de entrenamiento con choques solo pie–piso |
| `rl/revisar_nao_mjx.py` | Revisión del modelo del NAO en MJX (velocidad, contactos, memoria) |
| `rl/entrenar.py` | Entrenamiento PPO por generación; guarda config, progreso y checkpoints |
| `rl/evaluar.py` | Evaluación de robustez en la GPU (vmap + scan) → `evaluacion.csv` |
| `rl/ver_politica.py` | 5 carreras de 5 m con video y rumbo a los 2 s |
| `rl/cola_noche.sh` | Cola nocturna de entrenamientos |
| `rl/resumen_manana.py` | Evalúa lo pendiente y escribe el resumen de la noche |
| `rl/noches/` | Listas de corridas de cada noche |
| `rl/runs/` | Solo los archivos livianos de cada corrida: `config.json`, `progreso.csv`, `evaluacion.csv` y resúmenes |
| `rl/politicas/` | Políticas elegidas: `nao_v01b_s0`, `nao_v01c_s0`, `nao_v02_s9_44M` |
| [`bitacora.txt`](bitacora.txt) | Bitácora de todo el proceso |

---

## ▶️ Cómo reproducirlo

```bash
# en WSL2, con MuJoCo Playground instalado (bitacora.txt, entradas 2-4)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
PY=~/nao_rl/mujoco_playground/.venv/bin/python
cd ~/nao_rl

$PY revisar_nao_mjx.py                                   # revisar el modelo en MJX
$PY entrenar.py nao_v01b --pasos 60000000 --semilla 0    # una generacion (~1 h)
$PY entrenar.py nao_v02 --prueba                         # prueba corta del pipeline
$PY evaluar.py runs/nao_v01b_s0_<fecha>                  # 11 condiciones x 8 carreras
$PY ver_politica.py runs/nao_v01b_s0_<fecha>             # video de 5 carreras
bash cola_noche.sh noches/noche_01.txt                   # cola nocturna + resumen
```

---

## 🔜 Siguientes pasos

1. Revisar la noche 1 (v01b s1, v01c s1, v02, v03) y evaluar el checkpoint de 44 M de v02.
2. Arreglar que no levante los pies: recalibrar `fase_pies` / `altura_paso` para el NAO (posible v01d).
3. v04 con empujones.
4. Subir la velocidad pedida por encima de 0.3 m/s.
5. Reemplazar el error de rumbo integrado del giroscopio por una medición externa con ruido, retraso y 10 Hz.
6. Exportar la política a numpy para la inferencia a bordo (Python 2.7, sin torch).
