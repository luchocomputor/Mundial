#!/bin/bash
cd "$(dirname "$0")"

# 1. Tuer toute instance précédente — par NOM et par PORT (SIGKILL si récalcitrant)
pkill -f "streamlit run dashboard.py" 2>/dev/null
lsof -ti tcp:8501 2>/dev/null | xargs kill -9 2>/dev/null

# 2. Attendre que le port 8501 soit RÉELLEMENT libéré (sinon « Address already in use »)
for _ in $(seq 1 12); do
  lsof -ti tcp:8501 >/dev/null 2>&1 || break
  sleep 0.5
done

LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
echo ""
echo "  Sur le Mac        : http://localhost:8501"
[ -n "$LAN_IP" ] && echo "  Sur le téléphone  : http://$LAN_IP:8501  (même WiFi)"
echo "  (ferme cette fenêtre ou Ctrl-C pour arrêter le dashboard)"
echo ""

# 3. Ouvrir le navigateur DÈS que le serveur répond (poll en arrière-plan)
( for _ in $(seq 1 40); do
    curl -s -o /dev/null --max-time 1 http://localhost:8501 && { open http://localhost:8501; break; }
    sleep 0.5
  done ) &

# 4. Lancer le dashboard EN AVANT-PLAN (la fenêtre = serveur visible/contrôlable, pas
#    d'orphelin qui squatte le port plus tard). headless = on gère l'ouverture nous-mêmes.
exec streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
