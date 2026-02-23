# **AlphaPoisson**

AlphaPoisson is a collection of self-trained AI chess engines developed during **ChessHacks 2025** by our team *Échec et mat du poisson*. Our goal was to explore how far lightweight, custom-trained models can go using a mix of self-play, handcrafted evaluation heuristics, and iterative refinement.

---

## Overview

AlphaPoisson features multiple chess engines trained under different configurations, each with its own style and strengths. You can play against them directly on our website or explore the source code to see how each engine was built.

Key ideas behind the project include:

- **Self-play training loops**
- **Custom evaluation functions**
- **Iterative parameter tuning**
- **Search techniques (minimax, alpha-beta pruning, etc.)**
- **Lightweight model experimentation within hackathon constraints**

---

## Features

- Multiple AI chess engines with different difficulty levels  
- Deterministic and stochastic evaluation variants  
- Modular design — easy to tweak evaluation weights or search depth  
- Web interface for live gameplay  
- Clear engine structure for students, hobbyists, and researchers  

---

## Play Locally with Docker

Ensure you have Docker installed and running. Docker Desktop includes Docker Compose.

You do **not** need Python, Node.js, pip, or npm installed locally.

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AlphaPoisson
2. Build the Docker images:
   ```bash
     docker compose build
3. Start the application:
   ```bash
     docker compose up
4. Open the application in your browser:
   ```bash
     http://localhost:3000
   
---

## Play Online

Play against the AlphaPoisson engines in your browser:

👉 https://alpha-poisson.vercel.app/

(Spin up the backend before playing, by accessing this URL): https://alphapoisson.onrender.com/

---
