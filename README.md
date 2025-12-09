# Mario Reinforcement Learning – Setup (Minimal)

Dieses Projekt nutzt **stable-retro** und verschiedene RL-Algorithmen, um SuperMarioBros3-Nes-v0 zu trainieren und zu testen.

---

## Setup (WSL2 Ubuntu 22.04)

wsl --install -d Ubuntu-22.04

### 1. Systempakete installieren

sudo apt update
sudo apt-get install -y python3 python3-pip python3-venv python-is-python3 python3-opengl git zlib1g-dev libopenmpi-dev ffmpeg cmake

### 2. Repository klonen
cd ~
git clone git@github.com:Bugi-Shi/bachelor-mario3.git bachelor
cd bachelor

### 3. stable-retro klonen
git clone https://github.com/Farama-Foundation/stable-retro.git

### 4. Setup-Skript ausführen (automatisiert)

Das Projekt enthält ein Skript setup.sh, das:
- eine virtuelle Umgebung erstellt
- Pakete installiert
- stable-retro als Editable installiert

**Skript ausführen:**
chmod +x setup.sh
./setup.sh

**Danach kann die Umgebung jederzeit wieder aktiviert werden:**
source project/bin/activate

### 5. ROM hinzufügen und umwandeln (manuell)
rom.nes -> stable-retro/retro/data/stable/SuperMarioBros3-Nes-v0 verschieben

cat rom.sha
sha1sum rom.nes

-> müssen gleich sein

### 6. Projekt starten
python main.py

### 7. Projektstruktur
bachelor/
│── stable-retro/        # Emulator backend (lokaler Checkout)
│── project/             # Python virtual environment
│── src/                 # Trainings-Skripte (PPO / NEAT) und Hilfs-Skripte
│── .vscode/             # VS Code workspace settings & extension recommendations
│── main.py              # Einstiegspunkt
│── jump_test.py         # Beispielmodul für Mario-Test
│── setup.sh             # Automatisches Installationsskript (venv + Abhängigkeiten)
│── requirements.txt     # Laufzeit-Dependencies
│── requirements-dev.txt # Developer-Tools: black, isort, ruff
│── .ruff.toml           # Ruff-Konfiguration (Formatierung/Linting)
└── README.md
