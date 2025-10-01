![](images/Tic.png)


# Tic-Tac-Toe Decision Making with MLP via Reinforcement Learning

This project implements a reinforcement learning agent using a Multi-Layer Perceptron (MLP) as the policy model in C++.
The agent learns the optimal strategy for Tic-Tac-Toe through self-play.


## 🔍 Project Overview
- Train an agent using reinforcement learning techniques.
- The model is a simple MLP, taking the current board state as input and outputting values or probabilities for each move.
- During training, the agent plays against random decisions, learning from win/loss feedback without relying on predefined rules.



## 🎮 Training & Gameplay
<p align="center">
  <img src="images/game.png" alt="游戏演示" width="600">
</p>


## 🗂️  Project Structure
```
├── gomoku/                   
│   ├── gomoku.cpp          # Tic-Tac-Toe game logic
│   └── gomoku.h           
├── model/ 
│   ├── mlp.cpp             # Multi-Layer Perceptron model
│   └── mlp.h 
├── images/                 # Project images  
├── train.cpp               # Model training
├── README.md               # Project documentation
└── Makefile                # Build script
```



## 🚀  Quick Start

```bash
# Build the project using make
make
```



📧[choucisan@gmail.com]
