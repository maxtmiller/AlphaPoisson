
# [`AlphaPoisson`](https://alpha-poisson.vercel.app/)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat\&logo=docker\&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat\&logo=javascript\&logoColor=black)

**Custom AI chess engines trained with self-play and heuristics — compete online or explore the models.**

---

## Demo

<img width="1710" height="986" alt="alphapoisson_cover" src="https://github.com/user-attachments/assets/2148fb4c-a6bf-4e86-a8d9-47d4934d829a" />

---

## Features

* Multiple AI engines with different difficulty levels and evaluation strategies
* Deterministic and stochastic variants for varied gameplay
* Modular design for tweaking evaluation weights or search depth
* Web interface for live gameplay
* Clear structure for students, hobbyists, and researchers

---

## Technical Overview

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python (self-play engines, evaluation heuristics)
* **Search Techniques:** Minimax, alpha-beta pruning, iterative deepening
* **Training:** Self-play loops, handcrafted evaluation functions, iterative tuning
* **Deployment:** Docker and Docker Compose for easy setup

---

## Local Setup with Docker

No Python, Node.js, pip, or npm required. Docker Desktop is sufficient.

```bash
git clone <repository-url>
cd AlphaPoisson
docker compose build
docker compose up
```

Open your browser at: `http://localhost:3000`

---

## Play Online

* **Frontend:** [https://alpha-poisson.vercel.app/](https://alpha-poisson.vercel.app/)
* **Backend (spin up before playing):** [https://alphapoisson.onrender.com/](https://alphapoisson.onrender.com/)

---

## Next Steps

* Experiment with new evaluation functions and heuristics
* Add engine rating system for ranking difficulty levels
* Explore reinforcement learning for adaptive AI
