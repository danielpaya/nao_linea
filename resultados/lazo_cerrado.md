# Lazo cerrado en MuJoCo: politica RL actual con y sin vision de carril

Generado por `camara/pruebas/evaluar_lazo_cerrado.py` (2026-10-09 20:59, commit `4ca136d`, maquina `vm`).

- Politica: `checkpoints/best_model.zip (sprint inicial, V2 brazos)`, sin reentrenar.
- Pista: 5 m x 70 cm (`assets/nao/scene_carril.xml`). Salida de carril: el borde exterior de un pie cruza el borde interior de la linea (el desvio maximo de las corridas que se salen queda en ~26 cm).
- Vision: camara inferior simulada (QVGA, 54.7 x 42.5 grados), error de pose de la camara de 1 grado, odometria con 10 % de error.
- Correccion de rumbo: sesgo en la observacion del giroscopio (truco de demostracion, ver docs/fase_p_camara.md).

| | Llegan a la meta | Desvio max. (peor) | Desvio max. (medio) | Tiempo medio en 5 m |
|---|---|---|---|---|
| Solo RL | **3 / 18** | 27.0 cm | 24.9 cm | 24.8 s |
| RL + vision | **18 / 18** | 14.5 cm | 9.0 cm | 24.2 s |

## Detalle

| Inicio d [cm] | Inicio rumbo [grados] | Semilla | Solo RL | RL + vision |
|---|---|---|---|---|
| -10 | -8 | 1000 | salio del carril a 2.36 m (26 cm) | meta 24.3 s (10 cm) |
| -10 | -8 | 1001 | salio del carril a 2.72 m (26 cm) | meta 24.3 s (10 cm) |
| -10 | +0 | 1000 | meta 24.8 s (13 cm) | meta 24.2 s (10 cm) |
| -10 | +0 | 1001 | meta 24.8 s (22 cm) | meta 24.2 s (10 cm) |
| -10 | +8 | 1000 | salio del carril a 1.54 m (26 cm) | meta 24.3 s (9 cm) |
| -10 | +8 | 1001 | salio del carril a 1.52 m (27 cm) | meta 24.2 s (10 cm) |
| +0 | -8 | 1000 | salio del carril a 3.33 m (25 cm) | meta 24.3 s (5 cm) |
| +0 | -8 | 1001 | salio del carril a 3.69 m (26 cm) | meta 24.2 s (6 cm) |
| +0 | +0 | 1000 | meta 24.8 s (23 cm) | meta 24.3 s (5 cm) |
| +0 | +0 | 1001 | salio del carril a 3.09 m (26 cm) | meta 24.3 s (6 cm) |
| +0 | +8 | 1000 | salio del carril a 1.06 m (26 cm) | meta 24.3 s (5 cm) |
| +0 | +8 | 1001 | salio del carril a 1.05 m (26 cm) | meta 24.3 s (5 cm) |
| +10 | -8 | 1000 | salio del carril a 4.17 m (26 cm) | meta 24.3 s (11 cm) |
| +10 | -8 | 1001 | salio del carril a 4.53 m (25 cm) | meta 24.2 s (10 cm) |
| +10 | +0 | 1000 | salio del carril a 2.00 m (26 cm) | meta 24.3 s (11 cm) |
| +10 | +0 | 1001 | salio del carril a 1.57 m (26 cm) | meta 24.2 s (10 cm) |
| +10 | +8 | 1000 | salio del carril a 0.65 m (27 cm) | meta 24.2 s (14 cm) |
| +10 | +8 | 1001 | salio del carril a 0.65 m (27 cm) | meta 24.2 s (14 cm) |
