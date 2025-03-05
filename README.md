<div align="center">
    <h1>🚕 Q-Learning Taxi Agent for OpenAI Gym</h1>
</div>

A reinforcement learning agent trained with **Q-learning** to solve OpenAI Gym's `Taxi-v3` environment. The taxi learns to pick up passengers and drop them off at target locations efficiently.

---

## 🚀 Key Features
- **Q-table-based learning**: State-action values stored in a lookup table.
- **Epsilon-greedy exploration**: Balances exploration and exploitation during training.
- **Hyperparameter tuning**: Adjustable learning rate (`alpha`), discount factor (`gamma`), and epsilon decay.
- **Rendered visualization**: Watch the trained agent navigate the environment.

---

## 📦 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/q-learning-taxi.git
   cd q-learning-taxi
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training the Agent
Run the training and evaluation script:
```bash
python taxi_q_learning.py
```

### Hyperparameters (modify in `taxi_q_learning.py`)
```python
num_episodes = 10000   # Total training episodes
alpha = 0.9            # Learning rate
gamma = 0.95           # Discount factor
epsilon_decay = 0.9995 # Exploration decay rate
```

---

## 📊 Results
After training, the agent:
- Completes the task in **~20-30 steps**.
- Maximizes cumulative reward (average +8 to +20 per episode).
- Avoids illegal moves and optimizes paths over time.

![Taxi Agent Demo](demo.gif) <!-- Replace with your own GIF/screenshot or delete -->

---

## 🤖 Code Structure
```plaintext
q-learning-taxi/
├── taxi_q_learning.py    # Main training/evaluation script
├── README.md             # Project documentation
├── requirements.txt      # Dependency list
└── .gitignore            # Excludes unnecessary files
```

---

## 📝 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments
- OpenAI Gym for the [`Taxi-v3` environment](https://gym.openai.com/envs/Taxi-v3/).
- Q-learning fundamentals (Bellman equation, epsilon-greedy strategy).