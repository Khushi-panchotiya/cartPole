import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Neuroevolution Lab",
    page_icon="🧠",
    layout="wide"
)

# ---------- LOAD CSS ----------
def load_css():
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------- LOAD HEADER ----------
with open("templates/header.html", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)


# ---------- SIDEBAR ----------
st.sidebar.header("Training Settings")

pop_size = st.sidebar.slider("Population Size", 10, 100, 40)
sigma = st.sidebar.slider("Mutation Strength", 0.01, 0.5, 0.1)
lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
hidden_size = st.sidebar.slider("Hidden Layer Size", 4, 32, 8)


# ---------- AGENT ----------
class Agent:
    def __init__(self):
        self.layers = [4, hidden_size, 1]
        self.param_count = sum(
            self.layers[i] * self.layers[i+1]
            for i in range(len(self.layers)-1)
        )

    def predict(self, weights, state):
        idx = 0
        curr = state

        for i in range(len(self.layers)-1):
            size = self.layers[i] * self.layers[i+1]
            W = weights[idx:idx+size].reshape(
                self.layers[i],
                self.layers[i+1]
            )
            curr = np.tanh(np.dot(curr, W))
            idx += size

        return 1 if curr[0] > 0 else 0


# ---------- LAYOUT ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Simulation")
    video = st.empty()
    status = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Learning Progress")
    graph = st.empty()
    metric = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------- TRAIN ----------
if st.button("🚀 Start Training"):

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    agent = Agent()
    weights = np.random.randn(agent.param_count)

    reward_history = []

    for gen in range(50):

        noise = np.random.randn(pop_size, agent.param_count)
        rewards = []

        for i in range(pop_size):

            test_weights = weights + sigma * noise[i]
            state, _ = env.reset()
            total_reward = 0

            for _ in range(500):

                action = agent.predict(test_weights, state)
                state, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)

        rewards = np.array(rewards)

        normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        weights += (lr / (pop_size * sigma)) * np.dot(
            noise.T,
            normalized
        )

        best_reward = int(np.max(rewards))
        reward_history.append(rewards.mean())

        # Render
        state, _ = env.reset()

        for _ in range(150):

            action = agent.predict(weights, state)
            state, _, terminated, truncated, _ = env.step(action)

            frame = env.render()
            video.image(frame, channels="RGB")

            time.sleep(0.02)

            if terminated or truncated:
                break

        # Graph
        fig, ax = plt.subplots()
        ax.plot(reward_history)
        ax.set_title("Learning Curve")
        ax.grid()

        graph.pyplot(fig)

        metric.metric("Best Reward", best_reward)
        status.text(f"Generation {gen+1}/50")

    env.close()
    st.success("Training Complete 🎉")

# ---------- LOAD EXPLANATION ----------
with open("templates/explanation.html", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

