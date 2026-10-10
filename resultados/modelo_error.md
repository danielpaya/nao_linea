# Modelo de error de la vision de carril (para el paso 71)

Generado por `camara/pruebas/modelo_error.py` con 150 imagenes sinteticas por fila (d +-22 cm, psi +-20 deg, balanceo de la camara +-4 deg, madera clara y oscura). *Malas*: error > 6 cm o > 6 deg.

| Condicion | Error de pose | Sesgo d | sigma d | p95 d | Sesgo psi | sigma psi | p95 psi | Perdidas | Malas | Una sola linea |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.0 deg | -0.06 cm | 0.46 cm | 1.06 cm | +0.00 deg | 0.04 deg | 0.09 deg | 0.0 % | 0.0 % | 41 % |
| base | 0.5 deg | +0.03 cm | 0.67 cm | 1.32 cm | -0.08 deg | 0.39 deg | 0.74 deg | 0.0 % | 0.0 % | 41 % |
| base | 1.0 deg | +0.12 cm | 1.08 cm | 2.28 cm | -0.16 deg | 0.78 deg | 1.50 deg | 0.0 % | 0.0 % | 41 % |
| base | 1.5 deg | +0.21 cm | 1.55 cm | 3.17 cm | -0.24 deg | 1.18 deg | 2.28 deg | 0.0 % | 0.0 % | 41 % |
| base | 2.0 deg | +0.31 cm | 2.03 cm | 4.07 cm | -0.35 deg | 1.62 deg | 3.03 deg | 0.0 % | 0.7 % | 43 % |
| base | 3.0 deg | +0.54 cm | 2.98 cm | 5.74 cm | -0.51 deg | 2.65 deg | 5.79 deg | 0.0 % | 8.7 % | 49 % |
| desenfoque de movimiento | 1.5 deg | +0.22 cm | 1.55 cm | 3.07 cm | -0.24 deg | 1.18 deg | 2.29 deg | 0.0 % | 0.0 % | 41 % |
| poca luz (x0.35 + ruido) | 1.5 deg | -0.08 cm | 1.60 cm | 3.28 cm | +0.04 deg | 1.08 deg | 2.22 deg | 0.0 % | 0.0 % | 41 % |
| sobreexpuesta (x1.7) | 1.5 deg | +0.21 cm | 1.55 cm | 3.17 cm | -0.24 deg | 1.18 deg | 2.28 deg | 0.0 % | 0.0 % | 41 % |
| reflejos en la duela | 1.5 deg | +0.13 cm | 1.65 cm | 3.09 cm | -0.03 deg | 1.29 deg | 2.61 deg | 0.0 % | 0.0 % | 43 % |
| robot vecino tapando | 1.5 deg | +0.67 cm | 4.82 cm | 3.96 cm | -0.56 deg | 4.14 deg | 2.73 deg | 0.0 % | 2.0 % | 50 % |
| cinta gastada (35%) | 1.5 deg | +0.01 cm | 1.55 cm | 2.97 cm | -0.05 deg | 1.15 deg | 2.13 deg | 0.0 % | 0.0 % | 33 % |

## Valores propuestos para el paso 71 (a recalibrar con la pista real, paso 70)

| Parametro | Valor inicial | De donde sale |
|---|---|---|
| Frecuencia | 10 Hz, valor mantenido entre fotos | Camara inferior de PenaltyStack (QVGA, RGB, 10 FPS) |
| sigma de d (desplazamiento lateral) | 1.6 cm | Fila base con 1.5 deg de error de pose |
| sigma de psi (angulo respecto a las lineas) | 1.2 deg | Idem |
| Retardo | 100-200 ms | Exposicion y transferencia + 10-110 ms de procesamiento a bordo (a medir) |
| Perdidas | 5-10 % por foto (repetir el ultimo valor y bajar el indicador de valido) | Mas que lo medido en sintetico (0.0 %): la pista real es peor |
| Mediciones malas | 2-5 % con error uniforme de hasta +-15 cm | Robot vecino tapando, reflejos (tabla de arriba) |
| Indicador de valido | 0 si no hay medicion nueva en 300 ms | Contrato v2, seccion 5.1 de docs/plan_rl_carrera.md |
