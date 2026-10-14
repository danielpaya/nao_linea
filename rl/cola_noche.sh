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
# Si la cola se corta (p. ej. WSL se apaga), se relanza con la misma lista: las corridas que ya tienen una
# carpeta en runs/ con estado 'terminado' (misma generacion, semilla y pasos) se saltan.
#
# Cola activa: al arrancar se anota en cola_activa.txt (lista + hora de inicio) y al terminar se borra. Mientras ese
# archivo exista, vigilante.sh relanza la cola si se muere. Para detenerla a proposito:  bash cola_noche.sh --parar
cd "$(dirname "$0")" || exit 1
PY="${PY:-$HOME/nao_rl/mujoco_playground/.venv/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export PYTHONUNBUFFERED=1             # que el log se escriba al momento y no al final

# ya_terminada "<argumentos de entrenar.py>": exito si ya hay una corrida terminada igual
ya_terminada() {
  $PY - "$1" <<'PYEOF'
import glob, json, os, sys
args = sys.argv[1].split()
gen, semilla, pasos = args[0], "0", None
for i, a in enumerate(args):
    if a == "--semilla": semilla = args[i + 1]
    if a == "--pasos": pasos = int(args[i + 1])
for d in glob.glob("runs/%s_s%s_*" % (gen, semilla)):
    try:
        if open(os.path.join(d, "estado.txt")).read().strip() != "terminado":
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        if pasos is None or int(cfg["ppo"]["num_timesteps"]) == pasos:
            print(d); sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}
LISTA="$1"
if [ "$LISTA" = "--parar" ]; then
  rm -f cola_activa.txt
  for pid in $(pgrep -f "cola_noche.sh|entrenar.py|resumen_manana.py"); do
    [ "$pid" != "$$" ] && kill "$pid" 2> /dev/null
  done
  echo "Cola detenida y desactivada (el vigilante ya no la relanza)."; exit 0
fi
if [ ! -f "$LISTA" ]; then echo "Uso: bash cola_noche.sh noches/noche_XX.txt   |   bash cola_noche.sh --parar"; exit 1; fi
mkdir -p logs
LOG="logs/noche_$(date +%Y%m%d_%H%M%S).log"
# hora de inicio de la lista: si es un relanzamiento de la misma lista, se conserva la original (para el resumen)
HORAS_INICIO=$(date +%s)
if [ -f cola_activa.txt ] && [ "$(cut -d' ' -f1 cola_activa.txt)" = "$LISTA" ]; then
  HORAS_INICIO=$(cut -d' ' -f2 cola_activa.txt)
  echo "=== $(date '+%Y-%m-%d %H:%M') RELANZAMIENTO de $LISTA (empezo $(date -d @"$HORAS_INICIO" '+%Y-%m-%d %H:%M'))" | tee -a "$LOG"
fi
echo "$LISTA $HORAS_INICIO" > cola_activa.txt
echo "=== $(date '+%Y-%m-%d %H:%M') cola $LISTA" | tee -a "$LOG"
# corridas que quedaron 'corriendo' sin ningun entrenamiento vivo: se cortaron -> se marcan 'cortado'
if ! pgrep -f "entrenar.py" > /dev/null; then
  for e in runs/*/estado.txt; do
    if [ "$(tr -d '[:space:]' < "$e")" = "corriendo" ]; then
      echo "cortado" > "$e"; echo "=== $(date '+%H:%M') marcada como cortada: $(dirname "$e")" | tee -a "$LOG"
    fi
  done
fi
N=0
while IFS= read -r linea || [ -n "$linea" ]; do
  linea="${linea//$'\r'/}"            # por si el archivo se edito en Windows
  linea="${linea%%#*}"                # quitar comentarios
  linea="$(echo "$linea" | xargs)"    # quitar espacios
  [ -z "$linea" ] && continue
  N=$((N + 1))
  if hecha=$(ya_terminada "$linea"); then
    echo "=== $(date '+%H:%M') [$N] se salta (ya terminada en $hecha): $linea" | tee -a "$LOG"
    continue
  fi
  echo "=== $(date '+%H:%M') [$N] entrenar.py $linea" | tee -a "$LOG"
  $PY entrenar.py $linea < /dev/null 2>&1 | grep --line-buffered -v "dot_search_space" | tee -a "$LOG"
  echo "=== $(date '+%H:%M') [$N] termino con codigo ${PIPESTATUS[0]}" | tee -a "$LOG"
done < "$LISTA"
HORAS=$(( ( $(date +%s) - HORAS_INICIO ) / 3600 + 1 ))
echo "=== $(date '+%H:%M') evaluando lo entrenado en las ultimas $HORAS h" | tee -a "$LOG"
$PY resumen_manana.py --horas "$HORAS" < /dev/null 2>&1 | grep --line-buffered -v "dot_search_space" | tee -a "$LOG"
rm -f cola_activa.txt
echo "=== $(date '+%H:%M') cola terminada" | tee -a "$LOG"
