
# [`AlphaPoisson`](https://alpha-poisson.vercel.app/)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat\&logo=fastapi\&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)


**Custom AI chess engines trained with self-play and heuristics — compete online or explore the models.**

---

## Preview

<img width="1710" height="986" alt="alphapoisson_cover" src="https://github.com/user-attachments/assets/2148fb4c-a6bf-4e86-a8d9-47d4934d829a" />

---

## Features

* Multiple AI engines with different difficulty levels and evaluation strategies
* Deterministic and stochastic variants for varied gameplay
* Modular design for tweaking evaluation weights or search depth
* Web interface for live gameplay
* Clear structure for students, hobbyists, and researchers

---

## Tech Stack

* **Frontend:** Typescript, React, Next.js
* **Backend:** Python (self-play engines, evaluation heuristics)
* **Search Techniques:** Minimax, alpha-beta pruning, iterative deepening
* **Training:** Self-play loops, handcrafted evaluation functions, iterative tuning
* **Deployment:** Docker and Docker Compose for easy setup

---

## Local Setup with Docker

No Python, Node.js, pip, or npm required. Docker Desktop is sufficient.

```bash
git clone https://github.com/maxtmiller/AlphaPoisson.git
cd AlphaPoisson
docker compose build
docker compose up
```

Open your browser at: `http://localhost:3000`

### Without Docker

```bash
pip install -r requirements.txt && python serve.py   # backend on :5058
cd devtools && npm install && BACKEND_URL=http://localhost:5058 npm run dev
```

Training and quantization (`quantize.py`, `benchmark.py`) also need PyTorch: `pip install -r requirements-train.txt`.

---

## Deployment

Both halves deploy to Vercel as two projects from this repo:

* **Backend:** root directory `.`, framework preset "Other". `api/index.py` serves the FastAPI app as a Python function with the INT8 models; `excludeFiles` in `vercel.json` keeps the frontend and `.pth` checkpoints out of the bundle (a root `.vercelignore` would also strip the frontend project).
* **Frontend:** root directory `devtools`, with `BACKEND_URL` (or `NEXT_PUBLIC_BACKEND_URL`) set to the backend project's URL.

---

## Next Steps

* Experiment with new evaluation functions and heuristics
* Add engine rating system for ranking difficulty levels
* Explore reinforcement learning for adaptive AI
