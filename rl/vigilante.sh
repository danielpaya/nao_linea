#!/usr/bin/env bash
# Vigilante: mantiene viva la cola de entrenamientos aunque WSL se apague o la cola se muera.
#
# Lo arranca el Programador de tareas de Windows (al iniciar sesion y cada 10 min, sin duplicarse). Mientras este
# script corre, WSL no se apaga solo, porque hay un cliente (wsl.exe) conectado.
# Cada 2 min: si existe cola_activa.txt (hay una cola que no ha terminado) y no esta corriendo ni la cola ni ningun
# entrenamiento/evaluacion/video, relanza la cola en tmux (sesion "cola"). La cola salta lo ya terminado.
# Todo queda en logs/vigilante.log. Para que no relance:  bash cola_noche.sh --parar
cd "$(dirname "$0")" || exit 1           # ~/nao_rl (no resolver el enlace: ahi estan nao_assets y la venv)
mkdir -p logs
LOGV="logs/vigilante.log"
exec 9> /tmp/nao_vigilante.lock
flock -n 9 || exit 0                       # ya hay un vigilante corriendo
echo "$(date '+%Y-%m-%d %H:%M:%S') vigilante iniciado (pid $$)" >> "$LOGV"
while true; do
  if [ -f cola_activa.txt ]; then
    if ! pgrep -f "cola_noche.sh|entrenar.py|evaluar.py|ver_politica.py|resumen_manana.py" > /dev/null; then
      LISTA=$(cut -d' ' -f1 cola_activa.txt)
      echo "$(date '+%Y-%m-%d %H:%M:%S') la cola $LISTA no esta corriendo -> se relanza en tmux 'cola'" >> "$LOGV"
      tmux kill-session -t cola 2> /dev/null
      tmux new-session -d -s cola -c "$(pwd)" "bash cola_noche.sh $LISTA"
    fi
  fi
  sleep "${VIGILANTE_PAUSA:-120}"
done
