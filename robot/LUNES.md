# 🤖 Prueba de la visión en el NAO real: guía paso a paso

**Objetivo del día:** comprobar que la cámara del NAO detecta las líneas del carril y mide bien su
posición (`d`, qué tan corrido está del centro) y su rumbo (`psi`, qué tan girado está). Primero **sin
caminar**. Caminar es opcional, al final.

---

## 🧰 Qué llevar

| Cosa | Para qué |
|---|---|
| NAO con **batería > 50 %** y su cargador | — |
| **Cable Ethernet** (y adaptador USB-Ethernet si la laptop no tiene puerto) | Conectarse al robot sin WiFi |
| Laptop con este repo y `uv sync` hecho | Copiar archivos y revisar fotos |
| **Cinta negra de 19 mm** (la de aislar sirve) | Las líneas del carril (1.9 cm) |
| Cinta métrica o regla larga, y transportador | Medir dónde está el robot de verdad |
| Cartulina roja (opcional) | Probar la detección de la meta |
| Piso de madera o lo más parecido a la duela | — |

**Pista réplica:** dos líneas paralelas de cinta negra separadas **70 cm entre bordes interiores**, de
2 a 3 m de largo. Marca con un lápiz la línea central, que es la referencia para medir.

---

## 1. Conectarse al robot

1. Conecta el cable Ethernet entre la laptop y la cabeza del NAO.
2. Prende el NAO y espera a que termine de arrancar.
3. **Presiona una vez el botón del pecho:** el robot dice su IP en voz alta.
4. En PowerShell:
   ```
   ping <IP_DEL_NAO>
   ssh nao@<IP_DEL_NAO>          # contraseña por defecto: nao
   ```

## 2. Copiar los archivos al robot

Desde la carpeta del repo en la laptop (`C:\demo_carril`), en PowerShell:

```
ssh nao@<IP_DEL_NAO> "mkdir -p nao_linea"
scp -r carril robot nao@<IP_DEL_NAO>:/home/nao/nao_linea/
```

## 3. Diagnóstico (2 minutos, el robot no se mueve)

```
ssh nao@<IP_DEL_NAO>
cd nao_linea/robot
python diagnostico.py
```

Revisa que salga **[OK]** en:
- **Python, NAOqi y batería.** Anota si dice que hay numpy o no. Sin numpy también funciona, solo más
  lento.
- **Pose de la cámara.** Con el robot de pie debe quedar en z ≈ 0.45 m. Si sale baja, el robot está
  sentado.
- **Cámara.** Guarda una foto `.ppm` en `robot/logs/`.
- **Tiempo de la visión.** Debe quedar **por debajo de 100 ms por imagen**. Anota el número.
- **Botones.** Cuando lo pida, toca la cabeza adelante y luego el botón del pecho.

## 4. Prueba en seco: mover el robot a mano

**⚠️ El robot se pone de pie al arrancar. Mantén las manos cerca.**

```
python race_nao.py --dry-run
```

Coloca al robot de pie entre las líneas y muévelo despacio con las manos, sin levantarlo. La terminal
muestra cada medio segundo algo así:

```
ESPERA   vision d= +6.0 cm psi= -4.0 deg (2 lineas) | filtro d= +6.0 cm ...
```

**Qué comprobar:**

| Si haces esto | Debe pasar esto |
|---|---|
| Lo corres hacia **su izquierda** | `d` se vuelve **positivo** |
| Lo corres hacia **su derecha** | `d` se vuelve **negativo** |
| Lo giras hacia **su izquierda** | `psi` se vuelve **positivo** |
| Lo dejas centrado y derecho | `d ≈ 0` y `psi ≈ 0` |

Para terminar: **botón del pecho 0.3 s** o `Ctrl+C`.

## 5. Medición formal (lo más importante del día)

```
python race_nao.py --medir
```

1. Coloca al robot en una posición conocida.
2. **Mide con la cinta métrica:**
   - `d`: distancia entre el punto medio de los dos pies y la línea central (izquierda = +).
   - `psi`: ángulo de los pies respecto a las líneas.
3. **Toca la cabeza (adelante).** El robot promedia 2 s y muestra el resultado. Se guarda una fila
   numerada en `robot/logs/medicion_*.csv`.
4. **Anota en papel el número y la posición real**, para compararlas después.
5. Repite. Para terminar: botón del pecho.

**Posiciones sugeridas** (15 mediciones):

| | psi = 0° | psi = +10° | psi = −10° |
|---|---|---|---|
| d = 0 cm | ✓ | ✓ | ✓ |
| d = +10 cm | ✓ | ✓ | ✓ |
| d = −10 cm | ✓ | ✓ | ✓ |
| d = +20 cm | ✓ | | |
| d = −20 cm | ✓ | | |
| Una sola línea a la vista (muy corrido) | ✓ | | |
| Con luz distinta (prender o apagar una lámpara) | ✓ | | |

**Meta:** error menor a **3 cm** en `d` y menor a **3°** en `psi`.

## 6. Grabar fotos para revisar en la laptop

```
python race_nao.py --dry-run --save-frames 2
```

Mueve el robot por la pista unos 30 s y termina con el botón del pecho. Después, en la laptop:

```
scp -r nao@<IP_DEL_NAO>:/home/nao/nao_linea/robot/logs .\logs_robot
uv run python robot/replay.py .\logs_robot\frames_XXXXXXXX_XXXXXX --out revision
```

Abre las imágenes de `revision/`: **los puntos celestes deben caer sobre la cinta negra.** Copia
también los `medicion_*.csv` y tus notas en papel.

## 7. (Opcional) Caminar con la caminata de NAOqi

**⚠️ Solo si los pasos 4 y 5 salieron bien. Alguien camina al lado, listo para sostener al robot.**

```
python race_nao.py --vx 0.3
```

1. Toca la cabeza adelante 0.25 s para arrancar.
2. Debe caminar despacio corrigiendo hacia el centro.
3. Si no lo hace, **botón del pecho** para detener.

Hay un paro automático si se inclina demasiado. Prueba primero desde el centro y después colocándolo
chueco.

---

## 🔧 Problemas frecuentes

| Problema | Qué probar |
|---|---|
| `SIN LINEAS` todo el tiempo | Revisar que la cinta se vea en la foto del diagnóstico. Si es muy brillante o la luz rebota: `--min-contrast 20` o `--dark-ratio 0.7` |
| Cinta blanca o clara sobre piso oscuro | `--polaridad clara` |
| `d` siempre corrido unos cm, o crece raro con la distancia | El campo de visión puede ser otro: probar `--hfov 60.97 --vfov 47.64` (valores del NAO V5) |
| Los signos salen al revés | Anotarlo y avisar: no se corrige a mano |
| Muy lento (> 100 ms por imagen) | `--fps 5`, o revisar si hay numpy |
| Error al suscribir la cámara | Reiniciar NAOqi (`nao restart`) o el robot |
| El robot mueve la cabeza solo | Lo hace la "vida autónoma"; el script la desactiva, pero si persiste se apaga desde Choregraphe |

## 🛡️ Seguridad

- Nunca dejes al robot de pie sin nadie al lado.
- **Botón del pecho 0.3 s = detener.** Pruébalo antes de empezar.
- Trabaja lejos de bordes de mesa y escaleras.
- Si se calienta o la batería baja de 30 %, descansa.
