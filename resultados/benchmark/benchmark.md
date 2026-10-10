# Benchmark de caminatas en el carril de 5 m

Generado por `demo/caminatas/benchmark.py` el 2026-10-10 00:04 (commit caa9533, DESKTOP-HDLPDPA, MuJoCo 3.14.0).

- **RL**: politica del equipo (`checkpoints/best_model.zip`), entrenada con friccion **mu = 1.0 sin aleatorizar**.
- **naoqi_like**: pasos planeados + control preview del ZMP (esquema de ALMotion), 0.12 m/s pedidos.
- **bhuman_like**: esquema Walk2014 (pasos de 250 ms, FSR + giroscopio), 0.30 m/s pedidos.
- **Vision**: simulada con el error medido (sigma_d 1.6 cm, sigma_psi 1.2 deg, 10 Hz).
- Salida con error de colocacion aleatorio (sigma 2 cm y 2 deg), el mismo para las 3 caminatas.

| Caminata | Vision | mu | Metas | Caidas | Salidas | Tiempo a meta [s] | Vel. media [m/s] | Desvio max [cm] | Rumbo max [deg] | Esfuerzo [N²m²s/m] |
|---|---|---|---|---|---|---|---|---|---|---|
| bhuman | no | 0.6 | 0/1 | 0 | 1 | — | 0.294 | 25.4 | 11.7 | 46 |
| bhuman | no | 1.0 | 1/1 | 0 | 0 | 16.2 ± 0.0 | 0.307 | 20.6 | 7.9 | 45 |
| bhuman | si | 0.6 | 1/1 | 0 | 0 | 16.9 ± 0.0 | 0.296 | 3.2 | 9.1 | 45 |
| bhuman | si | 1.0 | 1/1 | 0 | 0 | 16.2 ± 0.0 | 0.306 | 3.1 | 6.0 | 45 |
| naoqi | no | 0.6 | 1/1 | 0 | 0 | 47.8 ± 0.0 | 0.104 | 8.0 | 4.8 | 116 |
| naoqi | no | 1.0 | 0/1 | 0 | 1 | — | 0.100 | 25.9 | 5.8 | 121 |
| naoqi | si | 0.6 | 1/1 | 0 | 0 | 47.8 ± 0.0 | 0.104 | 2.6 | 6.5 | 116 |
| naoqi | si | 1.0 | 1/1 | 0 | 0 | 49.8 ± 0.0 | 0.100 | 2.5 | 5.8 | 120 |
| rl | no | 0.6 | 1/1 | 0 | 0 | 24.8 ± 0.0 | 0.202 | 18.8 | 7.2 | 116 |
| rl | no | 1.0 | 1/1 | 0 | 0 | 24.8 ± 0.0 | 0.201 | 20.9 | 4.3 | 110 |
| rl | si | 0.6 | 1/1 | 0 | 0 | 24.3 ± 0.0 | 0.205 | 6.9 | 7.5 | 118 |
| rl | si | 1.0 | 1/1 | 0 | 0 | 24.5 ± 0.0 | 0.204 | 6.9 | 8.5 | 116 |
