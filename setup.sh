#!/usr/bin/env bash
set -e

echo ">>> Virtuelle Umgebung wird erstellt..."
python3 -m venv project

echo ">>> Aktivieren..."
source project/bin/activate

echo ">>> pip aktualisieren..."
pip install --upgrade pip

echo ">>> Dependencies installieren..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo ">>> Fertig!"
echo "Aktivieren mit: source project/bin/activate"
