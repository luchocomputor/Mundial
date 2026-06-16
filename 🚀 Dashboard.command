#!/bin/bash
cd "$(dirname "$0")"

# Tuer une instance déjà en cours
pkill -f "streamlit run dashboard.py" 2>/dev/null
sleep 0.5

# IP locale (pour ouvrir depuis le téléphone sur le même WiFi)
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)

# Lancer le dashboard (0.0.0.0 = accessible depuis le téléphone)
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 &

# Attendre que le serveur soit prêt puis ouvrir le navigateur
sleep 2
open http://localhost:8501

if [ -n "$LAN_IP" ]; then
  echo ""
  echo "  Sur le Mac        : http://localhost:8501"
  echo "  Sur le téléphone  : http://$LAN_IP:8501  (même WiFi)"
  echo ""
fi
