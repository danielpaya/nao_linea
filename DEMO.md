# Demo: caminata RL + visión de carril (Copa NAO CCM MX 2026, Carrera)

El NAO V6 simulado en MuJoCo camina con la política PPO del repositorio `CamiloGomezB/NAO` sobre una réplica de la pista oficial (reglamento 3.2.4):

- Carril de 70 cm con líneas negras de 1.9 cm.
- Duela de madera.
- Franja negra de salida y franja roja de meta a 5 m.

La cámara inferior simulada del robot ve las líneas y el módulo `carril/` estima su posición.

La demo compara dos corridas:

| Corrida | Qué hace | Resultado (semillas 1000–1002) |
|---|---|---|
| **1. Solo RL** | Camina "lo más recto posible", sin ver el carril | **Se sale del carril** a 1.9, 2.2 y 2.8 m |
| **2. RL + visión** | La cámara estima el desvío y se corrige el rumbo | **Llega a la meta** las 3 veces, en 24.3 s, con desvío máximo de 4.7–6.0 cm |

**Cómo se corrige sin reentrenar la política.** La política aprendió a anular cualquier giro que detecta en su giroscopio. La capa de visión le suma una pequeña señal falsa de giro a esa observación, y la política la "compensa" girando en sentido contrario, hacia el centro del carril. Es un truco para la demo y no tiene por qué funcionar en el robot real. La versión final debe reentrenar la política con un comando de giro explícito.

## Instalación (Windows, Linux o macOS)

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
cd demo_carril
uv sync                                   # mismas versiones que el repo de Camilo
uv run python demo/build_scene.py         # genera la pista (ya viene generada; solo si la cambian)
```

## Ejecutar

**Modo interactivo (recomendado):** `uv run python demo/run_demo.py --live` abre el visor 3D y la ventana *Lo que ve el NAO* con un panel: deslizadores de posición inicial (corrido y girado), casilla *Visión de carril* y botones **Correr / Reiniciar** y **Detener**. Se pueden repetir corridas sin cerrar nada.

```bash
uv run python demo/run_demo.py --live --inicio-d 12 --inicio-psi 8   # colocado chueco: mide, se centra y corre
uv run python demo/run_demo.py --live --inicio-d 12 --inicio-psi 8 --sin-vision   # igual, sin visión: se sale
uv run python demo/run_demo.py --live                 # visor 3D en vivo, con visión
uv run python demo/run_demo.py --live --sin-vision    # visor 3D en vivo, solo RL (se sale)
uv run python demo/run_demo.py --video demo.mp4       # graba las dos corridas lado a lado (~3-5 min)
```

- **macOS:** usar `uv run mjpython demo/run_demo.py --live`.
- **Linux sin pantalla:** anteponer `MUJOCO_GL=egl` (u `osmesa`).
- **Otras pruebas:** `--seed` cambia la corrida y `--pose-noise` el error de la pose de la cámara (en grados, 1.0 por defecto).

## Archivos

- `demo/build_scene.py`: genera `assets/nao/scene_carril.xml` (pista) y `assets/nao/nao_cam.xml` (NAO V6 con la cámara inferior de B-Human: 39.7° hacia abajo, 54.7° × 42.5°, 320×240).
- `demo/run_demo.py`: simulación en lazo cerrado y composición del video.
- `carril/`: módulo de visión de carril (el mismo que correría dentro del NAO).
- `src/`, `checkpoints/`, `assets/`: copiados del repositorio de Camilo (política entregada V2, entrenada en el modelo V5 y evaluada aquí sobre el V6).

## Límites de la demo (decirlo en la reunión)

- Es simulación: la cámara es perfecta salvo el error de pose añadido (1°), y la duela está generada por computador.
- El truco del giroscopio funciona en simulación. En el robot real lo correcto es reentrenar la política con un comando de giro.
- La política se entrenó en el modelo V5 sin aleatorización, así que todavía no está lista para el NAO real (retardo y ruido de sensores).
