# Neuroevolution Lab: Solving CartPole with Evolutionary Strategies 🧠

A high-interactive web application designed to demonstrate the power of **Neuroevolution (Evolutionary Strategies)** in solving the classic **CartPole-v1** control problem from OpenAI Gymnasium.

---

## 👥 Contributors
- **Grishma Makwana** (23BIT108)
- **Khushi Panchotiya** (23BIT112)

---

## 📽️ Preview & Features
- **Live Simulation**: Watch the agent learn to balance the pole in real-time.
- **Dynamic Training**: Adjust Population Size, Mutation Strength, and Learning Rate on the fly.
- **Real-time Analytics**: Track the learning progress with live-updated reward graphs.
- **Clean UI**: Built with a sleek, card-based interface using Streamlit and custom CSS.

## 🛠️ Tech Stack
- **Python**: Core logic.
- **Streamlit**: Web interface.
- **Gymnasium**: Environment simulation.
- **NumPy**: Matrix operations for neuroevolution.
- **Matplotlib**: Real-time graphing.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Grishma-Makwana/cartPole.git
   cd cartPole
   ```
2. Install dependencies:
   ```bash
   pip install streamlit gymnasium numpy matplotlib
   ```

### Running the App
```bash
streamlit run app.py
```

## 🧪 How it Works?
This project implements **Evolutionary Strategies (ES)**, a black-box optimization algorithm inspired by natural evolution.
1. **Selection**: A population of neural network weights is generated.
2. **Mutation**: Each agent is tested with random noise added to the base weights.
3. **Reward Calculation**: Agents are evaluated based on how long they can balance the pole.
4. **Update**: The base weights are updated in the direction of the highest-performing mutations.

---

## 📁 Repository Structure
- `app.py`: Main Streamlit application and neuroevolution logic.
- `static/`: Contains `style.css` for the UI layout.
- `templates/`: HTML templates for the dashboard header and detailed explanations.

---
