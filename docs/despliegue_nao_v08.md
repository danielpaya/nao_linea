# Despliegue de la política `nao_v08` en el NAO real (contrato y plan)

Fecha: 2026-10-29. Carrera Individual: 3-nov-2026. Documento para quien implemente el módulo C++ en
`SabanaHerons2026` (rama `carril-vision`). Todo lo de abajo sale de `rl/nao_carrera.py` y `rl/entrenar.py` de este repo
(`danielpaya/nao_linea`, rama `rl-nao-mjx`) y del documento `DESPLIEGUE_POLITICAS_MOVIMIENTO_NAO.txt`.

## 0. Decisión de diseño: no usar ONNX Runtime en el robot

El módulo `CopaRLWalkEngine` de Camilo (`CamiloGomezB/nao-v6-rl-race`, `bhuman/Src/Modules/MotionControl/CopaRLWalkEngine/`
y `bhuman/Src/Tools/CopaRace/CopaRace.h`) ya corrió en el NAO real: fase propia del MotionEngine, interpolación a home,
paro por inclinación / temperatura / robot levantado, y la red evaluada en C++ puro con pesos leídos de un `.cfg`.
Se reutiliza ese esqueleto. Ventajas frente a ONNX: sin riesgo de versión (ORT 1.10, opset), sin hilos propios de ORT dentro
del hilo Motion de prioridad RT, y la paridad se prueba con un JSON.

Lo que cambia respecto a Camilo: la **observación** (otra), la **activación** (swish, no elu), la **acción** (15 motores,
no 13), el **normalizador** (media y desviación, sin clip) y la **historia de 3 pasos**.

Archivos de este repo para el módulo:

| Archivo | Para qué |
|---|---|
| `rl/exportar.py` | JAX → `politica_np.npz` (ya existía) |
| `rl/a_cfg.py` | `politica_np.npz` → `<gen>.cfg` (pesos) + `<gen>_golden.json` (casos de paridad) |
| `rl/a_onnx.py` | `politica_np.npz` → `politica.onnx` (plan B; no se usa si va el `.cfg`) |

Comandos (desde `rl/`, con cualquier Python que tenga numpy):

```
python exportar.py runs/nao_v08_s0_XXXX            # con el Python de Playground (JAX); deja politicas/nao_v08_s0/
python a_cfg.py politicas/nao_v08_s0/politica_np.npz ../deploy/nao_v08
```

Copiar `nao_v08.cfg` a `Config/NeuralNets/NaoRL/nao_v08.cfg` del repo Sabana (~0.9 MB).

## 1. Red

- MLP del actor: 4 capas densas `[obs → 128 → 128 → 128 → 2·15]`, activación **swish** `x / (1 + exp(-x))` en las 3 ocultas, sin
  activación en la salida.
- Normalización de la entrada: `x = (obs - media) / desviacion`, **sin clip** (vectores de 126 valores en v08).
- Acción determinista: `a = tanh(salida[0:15])` (la primera mitad); la segunda mitad (log-std) se ignora.
- Orden del vector `a` (15): `LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll, RHipRoll, RHipPitch,
  RKneePitch, RAnklePitch, RAnkleRoll, LShoulderPitch, RShoulderPitch, LElbowRoll, RElbowRoll`.
- Formato de capa en el `.cfg`: `filas` = salidas, `columnas` = entradas, `pesos` por filas (= W de JAX traspuesta),
  como en `CopaRaceLayerConfig` de Camilo.
- **Paridad**: `nao_v08_golden.json` trae 64 observaciones crudas y la acción esperada. El C++ debe reproducirla con
  diferencia máxima < 1e-4 (el exportador ya comprobó 7.6e-7 de `.cfg` contra Python con v07c).

## 2. Tiempos

- La política actúa cada **24 ms** = cada 2 ciclos de 12 ms del hilo Motion (`ciclo++ % 2 == 0`); entre decisiones se
  mantiene el último objetivo. Igual que `ciclosPorAccion` de Camilo.
- Entrenada con retraso de acción de 0 a 3 pasos (0–72 ms) en `nao_v08`.
- Coste: ~42 000 multiplicaciones por decisión. Medir en el Atom (peor caso) y registrar en la bitácora.

## 3. Observación del actor (`nao_v08`)

`obs` tiene **126 valores = 3 pasos × 42**. El bloque del paso actual va **primero** (índices 0..41); los bloques anteriores
se desplazan: al llegar una decisión nueva, `historia = [actual, h_anterior_0, h_anterior_1]` (el más viejo se descarta).
Al empezar la marcha la historia está en **ceros** (no copias del primer valor): `h1 = h2 = 0` hasta llenarse.

Bloque de 42 (orden exacto; v08 **sin** velocidades de articulación):

| Índices | Qué | Unidades / cómo calcularlo en el robot |
|---|---|---|
| 0–2 | `gyro` x,y,z | rad/s en el marco del torso (x adelante, y izquierda, z arriba). `InertialData.gyro`. |
| 3–5 | `arriba` | vector unitario "arriba del mundo" en el marco del torso = fila 3 de la matriz torso→mundo = `R^T · (0,0,1)`. De pie y derecho: `(0, 0, 1)`. Sale de `InertialData.orientation3D` (Camilo: `gravedadProyectada` = `-arriba`). |
| 6–20 | `ang` (15) | `ángulo medido − postura` en rad, motores en el orden de la acción (arriba). `JointSensorData.angles`. |
| 21–35 | `acc` (15) | **acción anterior** `a` de la decisión previa (en [-1, 1], antes de escalar). Cero en la primera decisión. |
| 36–37 | `cmd` | `(vx_cmd, wz_cmd)`. En carrera: `vx_cmd = 0.5` (máximo entrenado), `wz_cmd = 0`. |
| 38–39 | `fase` | `(cos φ, sin φ)` con `φ = 2π · f · t`, `t` = segundos desde que arrancó la marcha, `f = 1.8` Hz (en el entrenamiento f ∈ [1.6, 2.0] al azar por episodio). |
| 40–41 | `vis` | `(psi, d)` de la cámara, recortados a ±0.5 rad y ±0.5 m. Ver signos en la sección 5. |

Postura base `postura` (rad), la misma que `home`: `LHipYawPitch 0, LHipRoll 0, LHipPitch −0.4, LKneePitch 0.8,
LAnklePitch −0.4, LAnkleRoll 0, RHipRoll 0, RHipPitch −0.4, RKneePitch 0.8, RAnklePitch −0.4, RAnkleRoll 0,
LShoulderPitch 1.57, RShoulderPitch 1.57, LElbowRoll −0.2, RElbowRoll 0.2`. (Coincide con `poseHome` de Camilo.) Valores
exactos en el `.cfg` (`postura`).

`etiquetas` completas (nombre de cada una de las 126 entradas) están en `meta['etiquetas']` del `.npz` y en el golden.

### Qué hace la cámara en el entrenamiento (para imitarla)

Cuadros a ~10 Hz (cada 4 pasos de 24 ms), 1 cuadro de retraso, 5 % de cuadros perdidos (se mantiene el último), ruido
σ_d = 1.6 cm y σ_psi = 1.2°. **No hay apagones largos** en v08 (los de v07a/b sí; v08 no): si la cámara pierde el carril
por más de ~0.5 s hay que decidir qué hacer (mantener el último valor, o el paro). Ver riesgos.

## 4. Acción → articulaciones

```
objetivo[i] = clip(postura[i] + a[i] * 0.3,  lo[i],  hi[i])      i = 0..14   (action_scale = 0.3 rad)
RHipYawPitch = LHipYawPitch                                      (un solo motor físico)
```

`lo`/`hi` son los límites del modelo (en el `.cfg`/golden); en el robot usar además los de `jointLimits.cfg` con margen.
Las 11 articulaciones restantes (cabeza, hombro-roll, codo-yaw, muñecas, manos) se quedan en `home`:
`HeadYaw 0, HeadPitch 0, ShoulderRoll ±0.2, ElbowYaw ∓1.57, WristYaw 0, Hand 0` (como `CopaRace::poseHome`; la cabeza la
puede seguir llevando `HeadMotionEngine` si la cámara la necesita inclinada).

Stiffness: Camilo usa 100 en todas (contrato v1) y su política caminó en el real. En el modelo de entrenamiento los
actuadores son de posición con kp aleatorio (±20 %) y torque limitado por la curva par–velocidad (`motores_reales`).

## 5. Signos y convenciones que HAY QUE verificar en el robot (suspendido)

1. **`arriba` / `gyro`**: con el robot derecho `arriba ≈ (0,0,1)`. `arriba` es la fila 3 de la matriz torso→mundo
   (`R^T·e_z`). Si el torso se inclina **hacia adelante** (nariz abajo), su eje x apunta hacia abajo y `arriba.x` se vuelve
   **negativo**; inclinado hacia atrás, positivo. Con la regla de la mano derecha (x adelante, y izquierda, z arriba), caer
   hacia adelante es girar alrededor de +y: `gyro.y > 0`. Comprobar las dos lecturas con el robot suspendido antes de
   caminar; si alguno sale al revés, el módulo debe invertirlo (la política nunca vio otro signo).
2. **`psi` y `d`**: en el entrenamiento `d > 0` = el robot está a la **izquierda** del centro, y `psi = rumbo_deseado −
   rumbo_real` (positivo = el carril apunta a la izquierda del robot, hay que girar a la izquierda). La ley actual de
   `CarrilBehavior` es `giro = −(kD·d + kPsi·psi)`: con `giro > 0` = girar a la izquierda, eso implica que el `psi` de
   `CarrilPercept` tiene el signo **contrario** (psi = rumbo del robot respecto al carril). Si es así, `psi_política =
   −psi_percept`. Comprobar con una prueba de mano: girar el robot a la derecha y mirar `psi_política` (debe ser > 0).
3. **Orden de articulaciones**: B-Human usa 26 entradas (el enum de `Joints.h`), la política 15 motores; mapear por
   nombre (como `articulacionesAccion` de Camilo).
4. **`acc` de la primera decisión**: ceros.

## 6. Fase `NaoRLWalkPhase` (qué copiar de Camilo y qué añadir)

Estados como `CopaRLWalkPhase`: `interpolando` (stand de B-Human → home, `interpolationMs`, coseno), `enHome` (espera
`settleMs`), `caminando`. Al terminar, la fase entrega el stand de B-Human.

Seguridad **obligatoria** (el MotionEngine no la da: filtro NaN solo en Debug, sin límite de velocidad, sin corte térmico):

- Filtro NaN/Inf en la acción y en los ángulos leídos → repetir la última acción válida; N fallos seguidos (p. ej. 3) →
  paro.
- Límite de cambio por ciclo de 12 ms por articulación: `|Δ| ≤ vel_max · 0.012` con `vel_max` del `.cfg`/golden
  (4.53 rad/s caderas-roll/yaw y tobillo-roll, 6.96 rad/s cadera/rodilla/tobillo-pitch, 7.46 hombro, 6.47 codo).
- Margen interior a `jointLimits.cfg` (p. ej. 0.05 rad).
- Paro enclavado por: inclinación del torso (> 0.6 rad, como Camilo), robot levantado (`FallDownState::pickedUp`),
  temperatura de un motor de pierna (umbral a medir; Camilo usa un parámetro), corriente alta sostenida.
- Pérdida de cámara > T s → decisión explícita (mantener rumbo con el último `psi`, o paro suave).
- Tras el paro: stand de B-Human (`StandGenerator`), como Camilo. Caídas: las atrapa `FallEngine`.
- Botón del pecho = paro manual (regla de la carrera); el paro por táctil de la cabeza sigue vigente.
- Watchdog de tiempo: medir el tiempo de la decisión; si excede el presupuesto, repetir la última acción.

Odometría: no hay modelo; estimar con `vx_cmd` (como Camilo) para que `CarrilBehavior` detenga la marcha a 5 m, y mejor
añadir un paro por tiempo como respaldo.

## 7. Pruebas por etapas (no saltarse ninguna)

| Etapa | Qué | Criterio |
|---|---|---|
| E0 | Compilar para PC y Nao; cargar el `.cfg`; correr el golden JSON | diferencia < 1e-4 en los 64 casos |
| E1 | SimRobot: la fase corre y no explota | solo valida cableado: SimRobot ignora la stiffness y simula `rHipYawPitch` independiente |
| E2 | Robot **suspendido**, sin tocar el suelo | signos de la sección 5; ningún motor sale de rango; temperaturas |
| E3 | Robot con **arnés**, `vx_cmd` bajo (0.2) | camina 5 m sin colgar del arnés |
| E4 | Suelo, `vx_cmd = 0.3` y 0.4 | recto; caídas; temperatura de motores |
| E5 | Suelo, `vx_cmd = 0.5` | tiempo en 5 m; comparar con B-Human (~40 s a 0.8) |
| E6 | Carril real con la cámara | desvío, el paro por inclinación y el paro manual |

Batería > 50 %, temperaturas de motores < 50 °C antes de cada prueba. El plan B (B-Human + visión de carril) queda como
está y es el que se despliega si la política no pasa E4.

## 8. Riesgos conocidos

- **Retraso real**: B-Human asume ~36 ms (3 ciclos); Camilo midió 48–60 ms. `nao_v08` se entrenó con 0–72 ms.
- **Servo real ≠ modelo**: kp ±20 %, torque con curva par-velocidad, sin juego mecánico ni calentamiento.
- **Sin velocidad de articulaciones** en v08: la política infiere la velocidad de la historia de 3 ángulos. Es lo que se
  puede medir en el robot; v07c y anteriores **no** sirven en el real tal cual (piden esa velocidad).
- **Sin apagones de cámara** en v08 (decisión de tiempo). Si la visión pierde el carril, la política recibe el último valor
  y no sabe que es viejo.
- **Una sola semilla** entrenada, sin evaluación real previa. La regla del repo se cumple: nada va al robot sin paro por
  táctil de la cabeza, paro por inclinación, y primeras pruebas suspendido/arnés.
- **Cambia varias cosas a la vez** respecto a v07c; si se ve peor en simulación, se hace ablación (bitácora [61]).
