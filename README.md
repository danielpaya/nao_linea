<div align="center">

# 🧠 nao_linea · rama `rl-nao-mjx`

### Caminata por RL para un NAO V6, entrenada desde cero en una GPU de casa

*Copa NAO CCM MX 2026 · Carrera Individual · carril de 5 m × 70 cm*

![MJX](https://img.shields.io/badge/simulador-MuJoCo%20MJX-1f6feb)
![Brax](https://img.shields.io/badge/RL-Brax%20PPO-8250df)
![NAO](https://img.shields.io/badge/robot-NAO%20V6-34a853)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%208%20GB-76b900)
![estado](https://img.shields.io/badge/estado-v04%20robusta%20a%200.3%20m%2Fs-34a853)

</div>

---

## 📌 Resumen

| | |
|:--|:--|
| **Qué es** | Una política PPO propia que hace caminar al NAO V6 recto y rápido, entrenada desde cero |
| **Dónde** | WSL2 (Ubuntu 24.04) · RTX 5060 8 GB · JAX 0.6.2 + Brax 0.14.2 + MuJoCo Playground |
| **Escala** | 4096 NAOs simulados a la vez · ~15 600 pasos/s · 60 M de pasos ≈ 56 min |
| **Mejor resultado hasta ahora** | **v04:** 5 m en **17.1 s**, desvío lateral de **10 cm**, **76/80** metas fuera del nominal y la única línea que aguanta **72 ms de retraso** |
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
**15 con codos** (`codos=True`, desde v01e): + `LElbowRoll` y `RElbowRoll`.
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

- Total: 49 valores (50 con el rumbo; 56 con codos), **× 3 pasos de historia**. Giroscopio,
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
| `fase_pies` | + | Que el pie suba según una curva de 3 cm. Tolerancia `sigma_pies` 0.01 (del G1) → **0.0005 desde v01d** |
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
| Fricción (piso **y** pies, porque MuJoCo usa el máximo de los dos) | 0.3 – 1.2 | v02d, v03m, v04 |
| Masa de cada pieza | ±10 % | v02d, v03m, v04 |
| kp de cada motor | ×0.8 – 1.2 | v02d, v03m, v04 |
| Retraso de la acción | 0 – 2 ciclos (0 – 48 ms) | v02r y todas las v03 / v04 |
| Empujones al torso | 0.05 – 0.3 m/s cada 3 – 6 s | v03e, v04 |
| Un motor de pierna al azar "caliente" (menos torque máximo) | 50 – 100 % | implementado, sin usar todavía |

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

Cada una cambia **una sola cosa** respecto a su base (`GENERACIONES` en [`rl/entrenar.py`](rl/entrenar.py)).
Todas a 60 M pasos (~1 h) y semilla 0, salvo donde se indica.

```
v01 ─ v01b (rumbo) ─ v01c (cadera) ─ v01d (pies) ─┬─ v01e (codos)
                                                  ├─ v02d (mundo al azar)
                                                  └─ v02r (retraso) ─┬─ v03c (codos)
                                                                     ├─ v03m (mundo)
                                                                     ├─ v03e (empujones)
                                                                     └─ v04 (las tres, 120 M)
```

| Gen | Cambio sobre su base | Hipótesis | ¿Se cumplió? |
|:--|:--|:--|:--|
| v01 | Base, mundo ideal | Camina a la velocidad pedida | Sí, pero gira ~40° al arrancar |
| v01b | + error de rumbo y `seguir_rumbo` | Deja de girar al arrancar | **Sí** (233 → 12 cm) |
| v01c | + castigo a la cadera | Deja de torcer las piernas | A medias: más robusta, menos recta |
| **v01d** | + tolerancia de `fase_pies` 0.01 → 0.0005 | Levanta los pies | **Sí**: `tiempo_aire` ×2, patina menos |
| v01e | v01d + codos | Va más recta | **Sí** (desvío 20 → 10 cm), pero cae con empujones fuertes |
| v02d | v01d + fricción, masa y kp al azar | Generaliza mejor | Poco: rumbo 8° → 5° |
| **v02r** | v01d + retraso 0–48 ms | Aguanta la latencia | **Sí**: 72 ms de retraso 0/8 → **8/8** |
| v03c | v02r + codos | Suma lo recto de v01e | **No**: se cae con motores blandos |
| v03m | v02r + mundo al azar | Menos desvío en piso resbaloso | A medias: la más rápida (16.6 s) |
| v03e | v02r + empujones | Aguanta golpes fuertes | Poco: 1/8 → 2/8 |
| **v04** | v02r + codos + mundo + empujones, 120 M | Junta lo mejor | **Sí**: la mejor en casi todo |

> v04 rompe a propósito la regla de un cambio: sus tres cambios se midieron por separado en v03c/v03m/v03e.
> Además entrenó el doble (120 M), así que **todavía no se sabe** si es mejor por combinar o por entrenar más.
> Lo responde la noche 3 (v04 con 60 M y con semillas 1 y 2).

### Resultados (11 condiciones × 8 carreras de 5 m, sin visión)

| | v01 | v01b | v01d | v01e | v02r | **v04** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Tiempo en 5 m (nominal) | 19.1 s | **16.4 s** | 17.1 s | 17.2 s | 17.2 s | 17.1 s |
| Desvío lateral (nominal) | 233 cm | 12 cm | 20 cm | **10 cm** | 12 cm | **10 cm** |
| Rumbo máximo (nominal) | 45.7° | 3.6° | 7.9° | 4.0° | 5.4° | **3.3°** |
| Retraso de 72 ms | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | **8/8** |
| Empujón de 0.45 m/s | 0/8 | 2/8 | **6/8** | 2/8 | 1/8 | 4/8 |
| Metas fuera del nominal (de 80) | 23 | 66 | 69 | 65 | 73 | **76** |

**Condiciones:** nominal · fricción 0.5 / 0.2 / 1.5 · kp al 70 % · un motor al 70 / 40 % · retraso de
1 / 3 ciclos · empujón de 0.25 / 0.45 m/s. Detalle de cada una en `rl/runs/<corrida>/evaluacion.csv` y en
los resúmenes `rl/runs/resumen_*.md`.

### Hallazgos del día 2 (13 de octubre)

1. **El premio de los pies estaba mal calibrado.** Con la tolerancia del G1, arrastrar el pie daba el 97 % del
   premio. Con 0.0005 da el 61 %, y el robot pasó a levantar los pies el doble y a patinar menos.
2. **Entrenar con retraso es lo que más vale para el robot real.** Con 0–48 ms al azar, la política aguanta
   72 ms sin haberlos visto, no pierde velocidad con 24 ms y además va más recta.
3. **Los codos ayudan o perjudican según la base.** Sobre v01d corrigieron el giro hacia un lado (v01e); sobre
   v02r hicieron caer al robot con motores blandos (v03c). Solos no son una mejora segura.
4. **La combinación (v04) funciona mejor que cada cambio por separado**, aunque falta separar el efecto de
   entrenar el doble.
5. **La recompensa del entrenamiento no predice lo que importa en la carrera.** v01e no mejoraba en el
   entrenamiento y en la evaluación iba la mitad de desviada: el desvío lateral no está en la recompensa.
   Siempre hay que evaluar.
6. **El tiempo no ha bajado de ~17 s** porque el comando máximo sigue en 0.3 m/s. Eso es la etapa siguiente.
7. **La debilidad que queda:** empujones fuertes (máximo 4/8) y sensibilidad a motores blandos.

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
| El log de la cola nocturna llega con retraso | stdout de Python con búfer | `PYTHONUNBUFFERED=1` en `cola_noche.sh` |
| La GPU marca 0 % los primeros minutos | Compilación de XLA (3.5 – 6 min por corrida) | Es normal; se ve en `estado.txt` y en el log |
| Casi no levantaba los pies | `fase_pies` con la tolerancia del G1: arrastrar daba el 97 % del premio | v01d: tolerancia 0.0005 |
| La noche 1 se cortó a las 00:50 | WSL se apagó solo (sin `.wslconfig`); tmux murió con él | Cola reanudable + `vigilante.sh` desde una tarea de Windows |
| La v04 de la noche 2 arrancó con la configuración vieja | Dos generaciones con el mismo nombre en un `dict`: gana la última | La vieja se renombró a `nao_v04_antigua`; se revisan nombres repetidos antes de enviar |
| v03c y v03m quedaron sin evaluar | `--parar` borra la hora de inicio de la cola y el resumen solo mira las últimas horas | `resumen_manana.py --horas 6` (mejora pendiente) |
| Un error de suma en la bitácora (v01d 77/80) | Error del asistente | Corregido con una entrada nueva (son 69/80) |

---

## 🌗 Flujo de día y de noche

| | Día | Noche |
|:--|:--|:--|
| Duración | Ciclos de ~1 h (60 M pasos) | Cola de ~8 h |
| Qué se hace | Una generación, video, evaluación y decidir | Varias generaciones y semillas seguidas |
| Cómo | `entrenar.py` en tmux | `cola_noche.sh noches/noche_NN.txt` en tmux |
| Al terminar | `evaluar.py` y `ver_politica.py` | `resumen_manana.py` evalúa lo que falte y escribe `runs/resumen_<fecha>.md` |

- La cola sigue aunque una corrida falle, y el resultado de cada corrida queda en `estado.txt`
  (corriendo / terminado / cortado).
- **Reanudable:** si se relanza la misma lista, salta las corridas ya terminadas.
- **Vigilante:** una tarea de Windows mantiene vivo `rl/vigilante.sh`, que relanza la cola si WSL o la cola se
  mueren. Para detenerla a propósito: `bash cola_noche.sh --parar`.

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
| `rl/cola_noche.sh` | Cola de entrenamientos (reanudable, `--parar`) |
| `rl/vigilante.sh` | Relanza la cola si se muere; lo arranca el Programador de tareas de Windows |
| `rl/logs/` | Logs de cada cola y del vigilante |
| `rl/resumen_manana.py` | Evalúa lo pendiente y escribe el resumen de la noche |
| `rl/noches/` | Listas de corridas de cada noche |
| `rl/runs/` | Solo los archivos livianos de cada corrida: `config.json`, `progreso.csv`, `evaluacion.csv` y resúmenes |
| `rl/politicas/` | Políticas elegidas: `nao_v01b_s0`, `nao_v01c_s0`, `nao_v01d_s0`, `nao_v01e_s0`, `nao_v02r_s0`, **`nao_v04_s0`** y `nao_v02_s9_44M` |
| [`bitacora.txt`](bitacora.txt) | Bitácora de todo el proceso |

---

## ▶️ Cómo reproducirlo

```bash
# en WSL2, con MuJoCo Playground instalado (bitacora.txt, entradas 2-4)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
PY=~/nao_rl/mujoco_playground/.venv/bin/python
cd ~/nao_rl

$PY revisar_nao_mjx.py                                   # revisar el modelo en MJX
$PY entrenar.py nao_v04 --prueba                         # prueba corta del pipeline (~5 min)
$PY entrenar.py nao_v04 --pasos 120000000 --semilla 0    # la mejor generacion (~1 h 40)
$PY evaluar.py runs/nao_v04_s0_<fecha>                   # 11 condiciones x 8 carreras (~12 min)
$PY ver_politica.py runs/nao_v04_s0_<fecha>              # video de 5 carreras
bash cola_noche.sh noches/noche_02.txt                   # cola + evaluacion + resumen
bash cola_noche.sh --parar                               # detenerla sin que el vigilante la relance
```

---

## 🔜 Siguientes pasos

1. **Noche 3:** v04 con 60 M y con semillas 1 y 2 (¿combinar o entrenar más?, ¿fue suerte?).
2. **Etapa de velocidad** sobre v04, para bajar de 17 s hacia ~11 s:
   - motores realistas (límite de velocidad por motor y caída de torque al girar rápido);
   - arrancar cada etapa desde una política ya entrenada;
   - subir el comando 0.35 → 0.4 → 0.5 m/s.
3. **Visión simulada:** reemplazar el error de rumbo del giroscopio por d y ψ con el ruido medido de la cámara
   (σ 1.6 cm / 1.2°), a 10 Hz y con retraso.
4. Empujones más fuertes en el entrenamiento (la debilidad que queda).
5. Exportar la política a numpy para la inferencia a bordo (Python 2.7, sin torch).
