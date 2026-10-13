#!/usr/bin/env bash
# Cola nocturna: corre una lista de entrenamientos uno tras otro y al final evalua todo lo nuevo.
#
# Uso (desde ~/nao_rl, idealmente dentro de tmux para que siga aunque cierres la ventana):
#     tmux new -s noche
#     bash cola_noche.sh noches/noche_01.txt
#     (Ctrl+B y luego D para salir sin detenerla; en la manana: tmux attach -t noche)
#
# La lista tiene una corrida por linea con los argumentos de entrenar.py (las lineas con # se ignoran).
# Si una corrida falla, la cola sigue con la siguiente. Todo queda en logs/noche_<fecha>.log.
cd "$(dirname "$0")" || exit 1
PY="${PY:-$HOME/nao_rl/mujoco_playground/.venv/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
LISTA="$1"
if [ ! -f "$LISTA" ]; then echo "Uso: bash cola_noche.sh noches/noche_XX.txt"; exit 1; fi
mkdir -p logs
LOG="logs/noche_$(date +%Y%m%d_%H%M%S).log"
HORAS_INICIO=$(date +%s)
echo "=== $(date '+%Y-%m-%d %H:%M') cola $LISTA" | tee -a "$LOG"
N=0
while IFS= read -r linea || [ -n "$linea" ]; do
  linea="${linea//$'\r'/}"            # por si el archivo se edito en Windows
  linea="${linea%%#*}"                # quitar comentarios
  linea="$(echo "$linea" | xargs)"    # quitar espacios
  [ -z "$linea" ] && continue
  N=$((N + 1))
  echo "=== $(date '+%H:%M') [$N] entrenar.py $linea" | tee -a "$LOG"
  $PY entrenar.py $linea < /dev/null 2>&1 | grep --line-buffered -v "dot_search_space" | tee -a "$LOG"
  echo "=== $(date '+%H:%M') [$N] termino con codigo ${PIPESTATUS[0]}" | tee -a "$LOG"
done < "$LISTA"
HORAS=$(( ( $(date +%s) - HORAS_INICIO ) / 3600 + 1 ))
echo "=== $(date '+%H:%M') evaluando lo entrenado en las ultimas $HORAS h" | tee -a "$LOG"
$PY resumen_manana.py --horas "$HORAS" < /dev/null 2>&1 | grep --line-buffered -v "dot_search_space" | tee -a "$LOG"
echo "=== $(date '+%H:%M') cola terminada" | tee -a "$LOG"
