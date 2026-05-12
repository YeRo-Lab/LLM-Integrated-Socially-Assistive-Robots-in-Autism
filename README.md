# 🧠 SAR LLM MVP: Interaction Tasks with Constrained LLMs

## Overview

This project implements a **simulated socially assistive robot (SAR) interaction system** designed to explore how large language models (LLMs) can support **adaptive, safe, and structured human–robot interaction**.

The system models interactions with a humanoid robot (e.g., Pepper) using:

- **LLM-based reasoning and generation**
- **Constraint-driven response control**
- **Simulated perception (ASR) and user behavior**
- **Multi-turn interaction loops**
- **Evaluation pipelines for experimental results**

The primary experimental task is **adaptive storytelling**, which integrates:

- Turn-taking
- Emotion recognition
- Social reasoning (next-action choice)
- Co-creative narrative interaction

---

# 🚀 What This Project Demonstrates

This MVP enables controlled evaluation of three system configurations:

| Condition | Description |
|----------|------------|
| `scripted` | Fully rule-based interaction |
| `unconstrained_llm` | Generative LLM with minimal constraints |
| `constrained_llm` | Generative LLM with safety + structure constraints |

The system measures:

- Latency (Table 1)
- Interaction quality (Table 2)
- Architectural reliability (Table 3)
- Comparative performance across conditions (Table 4)

---

# 🏗️ System Architecture

Each session simulates a full interaction loop:

Task Config → Scenario Generator → ASR Simulation → LLM Reasoning + Generation → Constraint Validation → Robot Response → Logging + Metrics

---

## Core Components

### 1. Task Layer (`config/tasks/`)
Defines structured interaction tasks.

### 2. Simulation Layer (`simulation/`)
Generates realistic interaction variability.

### 3. Perception Layer (`perception/`)
Simulates speech-to-text, confidence, and engagement.

### 4. Reasoning Layer (`reasoning/`)
Handles LLM calls, policy decisions, and constraint validation.

### 5. Interaction Layer (`interaction/`)
Controls session flow and storytelling logic.

### 6. Analytics Layer (`analytics/`)
Logs and computes metrics.

---

# 📁 Project Structure

```
pepper_llm_mvp/
│
├── config/
│   ├── app.yaml
│   ├── llm.yaml
│   └── tasks/
│       ├── emotion_recognition.yaml
│       └── adaptive_storytelling.yaml
│
├── src/pepper_llm_mvp/
│   ├── interaction/
│   ├── reasoning/
│   ├── perception/
│   ├── simulation/
│   ├── analytics/
│   ├── robot/
│   └── evaluation/
│
├── scripts/
│   ├── run_session.py
│   ├── run_ablation.py
│   ├── compute_table1.py
│   ├── compute_table2.py
│   ├── compute_table3.py
│   └── compute_table4.py
│
├── data/
│   └── raw/sessions/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup Instructions

## Clone the repository
```
git clone https://github.com/johnwaugh1/pepper_llm_mvp.git
cd pepper_llm_mvp
```

## Create environment
```
python -m venv venv
venv\Scripts\activate
```

## Install dependencies
```
pip install -r requirements.txt
```

## Set API key
Create `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

---

# ▶️ Running the System

Run a session:
```
python scripts/run_session.py --task adaptive_storytelling --condition constrained_llm
```

Outputs are saved to:
```
data/raw/sessions/
```

---

# 🔬 Running the Ablation Study

Clear sessions:
```
Remove-Item data/raw/sessions/*.json
```

Run:
```
python scripts/run_ablation.py --task adaptive_storytelling --n_sessions 20
```

Compute tables:
```
python scripts/compute_table1.py
python scripts/compute_table2.py
python scripts/compute_table3.py
python scripts/compute_table4.py
```

---

# 📊 Tables

- Table 1: Latency
- Table 2: Interaction Quality
- Table 3: Architecture Reliability
- Table 4: Ablation Study

---

# 🧩 Adaptive Storytelling

Example:

Robot → User → Robot → User → Robot

Maintains context, builds narrative, and evaluates social reasoning.

---

# 🛡️ Constraints

- Length limits
- Safe vocabulary
- Structured output
- Fallback handling

---

# ⚠️ Notes

- Fully simulated environment
- LLM responses are real (if enabled)
- Not intended for clinical claims

---

# 🚧 Future Work

- Real robot integration
- Live perception
- Longitudinal evaluation

---

# ✅ Quick Start

```
pip install -r requirements.txt
set OPENAI_API_KEY=your_key
python scripts/run_session.py --task adaptive_storytelling --condition constrained_llm
```
