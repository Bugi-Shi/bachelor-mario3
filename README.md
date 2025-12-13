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
- optional `stable-retro` als Editable installiert (falls `./stable-retro/` existiert)

**Skript ausführen:**
chmod +x setup.sh
./setup.sh

**Danach kann die Umgebung jederzeit wieder aktiviert werden:**
source project/bin/activate

---

## Setup (Allgemein / auf jedem Rechner)

Minimaler Ablauf für andere Personen:

1) Repo klonen
2) Python **3.8** installieren (inkl. venv-Modul)
3) `./setup.sh` ausführen (erstellt `project/` venv und installiert Dependencies)
4) Starten mit `./project/bin/python main.py`

Wichtig:
- `requirements.txt` = Laufzeit-Dependencies (Top-Level)
- `requirements-lock.txt` = *Freeze/Lock* (exakte Versionen). `setup.sh` nutzt standardmäßig das Lock-File.

Wenn `pip install` bei jemandem fehlschlägt, fehlen meist System-Libs (OpenGL/SDL/ffmpeg). Unter Ubuntu/Debian ist diese Liste oft ausreichend:

```bash
sudo apt update
sudo apt-get install -y \
	python3.8 python3.8-venv python3-pip \
	python3-opengl ffmpeg cmake \
	zlib1g-dev \
	libgl1 libglib2.0-0
```

### 5. ROM hinzufügen und umwandeln (manuell)
rom.nes -> stable-retro/retro/data/stable/SuperMarioBros3-Nes-v0 verschieben

cat rom.sha
sha1sum rom.nes

-> müssen gleich sein

### 6. Projekt starten

#### Training ausführen

Am einfachsten startest du das Training über den Einstiegspunkt:

```bash
# (empfohlen) venv-python direkt nutzen
./project/bin/python main.py

# alternativ (wenn venv aktiv ist)
python main.py
```

Beim Start wird automatisch ein neuer Run-Ordner unter `outputs/runs/<YYYY-MM-DD_HH-MM-SS>/` angelegt. Dort landen u.a.:

- `tb/` – TensorBoard Event-Files (für Metriken/Scalars)
- `stats/episode_stats.csv` – Episoden-Statistiken (z.B. Reward, Len, x/hpos)
- `deaths/` – Death-Logs als `.jsonl`
- `deaths_overlay.png` – Overlay für diesen Run

Zusätzlich wird run-übergreifend geschrieben:

- `outputs/allDeath.jsonl` – globale Death-Historie (JSONL)
- `outputs/all_deaths_overlay.png` – Overlay aus allen Runs

#### TensorBoard / „Tensorflow Dateien“ (Event-Files) auslesen

TensorBoard liest die Event-Files aus dem `tb/`-Ordner. Du kannst entweder einen einzelnen Run anzeigen oder alle Runs zusammen.

**1) Einen einzelnen Run anzeigen**

```bash
# Beispiel: einen konkreten Run öffnen
./project/bin/tensorboard --logdir outputs/runs/2025-12-13_12-34-56/tb --port 6006
```

**2) Alle Runs zusammen anzeigen**

```bash
./project/bin/tensorboard --logdir outputs/runs --port 6006
```

Danach im Browser öffnen:

- `http://localhost:6006`

Wenn du in WSL/Remote arbeitest und von außen zugreifen willst, nutze zusätzlich:

```bash
./project/bin/tensorboard --logdir outputs/runs --port 6006 --bind_all
```

### 7. Projektstruktur
bachelor/
│── .vscode/             # VS Code workspace settings & extension recommendations
│── assets/              # Bilder/Level-Assets (z.B. Overlay-Base)
│── env/                 # Environment-Factory (z.B. MB3_env.py)
│── wrapper/             # Gym/Retro-Wrapper (Preprocessing, Death-Logger, etc.)
│── utils/               # Callbacks, Run-Management, Plot/Overlay, Aggregation
│── outputs/             # Trainingsartefakte (Runs, TensorBoard, CSV, Death-Logs)
│── retro_custom/        # Custom Retro-Data/Configs (falls genutzt)
│── project/             # Python virtual environment (venv)
│── main.py              # Einstiegspunkt (Training starten)
│── ppo.py               # PPO-Training (Stable-Baselines3)
│── ppo_super_mario_bros3.zip # gespeichertes Modell (Beispiel/Checkpoint)
│── setup.sh             # Automatisches Installationsskript (venv + Abhängigkeiten)
│── requirements.txt     # Laufzeit-Dependencies
│── requirements-dev.txt # Developer-Tools (Format/Lint)
│── requirements-lock.txt# Gepinnte Versionen (z.B. tensorboard)
│── .ruff.toml           # Ruff-Konfiguration (Formatierung/Linting)
│── .gitignore
└── README.md
