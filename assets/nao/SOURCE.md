# Origen y licencia del modelo NAO

## Fuente

El modelo MuJoCo (MJCF) de este directorio se deriva de:

- **Repositorio:** [nico-bohlinger/one_policy_to_run_them_all](https://github.com/nico-bohlinger/one_policy_to_run_them_all)
- **Archivo:** `one_policy_to_run_them_all/environments/nao_v5/data/nao.xml`
- **Commit:** `7c1819444028c529b0a964fe3f32878333b1310a` (2025-05-15)
- **Licencia:** MIT, Copyright (c) 2024 Nico Bohlinger
- **Referencia:** N. Bohlinger et al., *One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion*, CoRL 2024.

Los parámetros cinemáticos y dinámicos (posiciones de articulaciones, ejes, rangos, masas e inercias) provienen a su vez de la descripción URDF oficial de Aldebaran/SoftBank Robotics, [ros-naoqi/nao_robot](https://github.com/ros-naoqi/nao_robot) (BSD-3-Clause).

## Modificaciones respecto al original

1. **Sin mallas 3D.** Todas las geometrías de malla (`.stl`) se reemplazaron por primitivas (cápsulas, cajas, esferas). Las mallas originales parecen derivar de [ros-naoqi/nao_meshes](https://github.com/ros-naoqi/nao_meshes), que tiene licencia CC BY-NC-ND 4.0 con redistribución restringida, por lo que **no se incluyen** en este repositorio. Las primitivas también aceleran la simulación de colisiones.
2. **Sin dedos.** Se eliminaron las 20 articulaciones pasivas de dedos y mano (masa despreciable, ~2e-6 kg cada una), irrelevantes para la locomoción.
3. **HipYawPitch acoplado.** En el NAO real, `LHipYawPitch` y `RHipYawPitch` los mueve un único motor. Se añade una restricción de igualdad (`RHipYawPitch = LHipYawPitch`) y solo `LHipYawPitch` tiene actuador (21 actuadores en total).
4. **Convención de ejes de Aldebaran.** El original invertía el eje (y el rango) de varias articulaciones: HipRoll, AnkleRoll, ShoulderRoll, ElbowYaw y todas las de pitch del lado derecho. Se restauró la convención oficial (la misma de NAOqi), de modo que un mismo ángulo significa lo mismo en simulación y en el robot real.
5. **Muñecas fijas.** `WristYaw` (sin actuador en el original) se eliminó; la mano es un cuerpo rígido que conserva su masa e inercia.
6. **Autocolisiones.** Brazos, torso/cabeza y piernas chocan entre sí (grupos de colisión por `contype`/`conaffinity`), de modo que ninguna parte puede atravesar a otra, por ejemplo la mano no puede atravesar el muslo ni un pie al otro. Solo se excluye el par bíceps-antebrazo, que se tocan en el codo.
7. **Actuadores de posición.** Los motores de torque se reemplazaron por servos de posición de MuJoCo (PD: `kp=30`, `kv=0.5`, igual que el controlador del trabajo original), conservando los mismos límites de torque.

Masa total: **5.305 kg**, idéntica al original.

## Versión del robot: V5 vs V6

El modelo original es de un NAO V5 (H25, masa total 5.305 kg). El NAO V6 conserva la misma estructura mecánica, dimensiones de eslabones y articulaciones. Sus cambios principales son de hardware interno (procesador, batería, sensores) y un ligero aumento de masa. Para este sprint de locomoción en simulación, el modelo V5 se considera una aproximación válida del V6.
