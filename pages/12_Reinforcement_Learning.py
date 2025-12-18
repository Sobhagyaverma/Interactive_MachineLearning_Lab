import streamlit as st
import numpy as np
import time
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Q-Learning - ML Lab",
    page_icon="🎮",
    layout="wide"
)

utils.navbar()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .hero-container {
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f2937;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        font-style: italic;
    }
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #ffffff;
        border-radius: 8px;
        color: #4b5563;
        font-weight: 600;
        font-size: 16px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white !important;
        border-color: #ff4b4b;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    .grid-cell {
        width: 60px;
        height: 60px;
        font-size: 2rem;
        text-align: center;
        line-height: 60px;
        border: 2px solid #ddd;
        display: inline-block;
        margin: 2px;
        border-radius: 8px;
    }
    .cell-empty { background-color: #f9fafb; }
    .cell-agent { background-color: #dbeafe; }
    .cell-treasure { background-color: #d1fae5; }
    .cell-bomb { background-color: #fee2e2; }
    .cell-wall { background-color: #374151; }
</style>
""", unsafe_allow_html=True)

# --- Q-LEARNING ENVIRONMENT ---
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]
        self.treasure_pos = [size-1, size-1]
        self.bomb_pos = [size-1, 0]
        
    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)
    
    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        old_pos = self.agent_pos.copy()
        
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # Right
            self.agent_pos[1] += 1
        
        # Calculate reward
        if self.agent_pos == self.treasure_pos:
            reward = 100  # Cookie! 🍪
            done = True
        elif self.agent_pos == self.bomb_pos:
            reward = -100  # Slap! 👋
            done = True
        else:
            reward = -1  # Small penalty for each step
            done = False
        
        return tuple(self.agent_pos), reward, done

# --- Q-LEARNING AGENT ---
class QLearningAgent:
    def __init__(self, grid_size=5, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):
        self.grid_size = grid_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-Table: [row, col, action] -> Q-value
        self.q_table = np.zeros((grid_size, grid_size, 4))
        
    def get_action(self, state):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit
    
    def update(self, state, action, reward, next_state, done):
        # Q-Learning Formula
        current_q = self.q_table[state[0], state[1], action]
        
        if done:
            target_q = reward
        else:
            max_future_q = np.max(self.q_table[next_state[0], next_state[1]])
            target_q = reward + self.gamma * max_future_q
        
        # Update Q-value
        self.q_table[state[0], state[1], action] += self.lr * (target_q - current_q)
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- HELPER: VISUALIZE GRID ---
def render_grid(env, agent_pos):
    grid_html = "<div style='font-family: monospace;'>"
    
    for i in range(env.size):
        grid_html += "<div>"
        for j in range(env.size):
            if [i, j] == agent_pos:
                cell_class = "cell-agent"
                emoji = "🤖"
            elif [i, j] == env.treasure_pos:
                cell_class = "cell-treasure"
                emoji = "💎"
            elif [i, j] == env.bomb_pos:
                cell_class = "cell-bomb"
                emoji = "💣"
            else:
                cell_class = "cell-empty"
                emoji = "⬜"
            
            grid_html += f"<div class='grid-cell {cell_class}'>{emoji}</div>"
        grid_html += "</div>"
    
    grid_html += "</div>"
    return grid_html

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎮 Reinforcement Learning (Q-Learning)</div>
    <div class="hero-subtitle">"Figure it out yourself. Cookie for wins 🍪, slaps for losses 👋."</div>
</div>
""", unsafe_allow_html=True)

# --- INITIALIZE ---
if 'env' not in st.session_state:
    st.session_state['env'] = GridWorld(size=5)
    st.session_state['agent'] = QLearningAgent(grid_size=5)
    st.session_state['episode'] = 0
    st.session_state['total_reward'] = 0

env = st.session_state['env']
agent = st.session_state['agent']

# --- TABS ---
tab1, tab2 = st.tabs(["🎮 GridWorld Playground", "📚 Theory & Math"])

with tab1:
    st.info("""
    **🎯 The Mission:** Help the robot 🤖 learn to reach the treasure 💎 while avoiding the bomb 💣.
    - **Treasure (💎)**: +100 reward (Cookie! 🍪)
    - **Bomb (💣)**: -100 reward (Slap! 👋)
    - **Each step**: -1 reward (encourages speed)
    """)
    
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.subheader("⚙️ Hyperparameters")
        
        learning_rate = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, help="How much to update Q-values")
        discount_factor = st.slider("Discount Factor (γ)", 0.0, 1.0, 0.9, help="How much to value future rewards")
        epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, float(agent.epsilon), help="Probability of random action")
        
        agent.lr = learning_rate
        agent.gamma = discount_factor
        agent.epsilon = epsilon
        
        st.write("---")
        
        episodes = st.slider("Training Episodes", 10, 500, 100)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            rewards_history = []
            
            for ep in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                steps = 0
                max_steps = 50
                
                while not done and steps < max_steps:
                    action = agent.get_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    
                    episode_reward += reward
                    state = next_state
                    steps += 1
                
                agent.decay_epsilon()
                rewards_history.append(episode_reward)
                
                if ep % (episodes // 10) == 0:
                    progress_bar.progress((ep + 1) / episodes)
                    avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
                    status_text.text(f"Episode {ep}/{episodes} | Avg Reward: {avg_reward:.1f} | ε: {agent.epsilon:.3f}")
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"Training Complete! Final Average Reward: {np.mean(rewards_history[-50:]):.1f}")
            st.session_state['episode'] = episodes
        
        if st.button("🔄 Reset Agent", use_container_width=True):
            st.session_state['agent'] = QLearningAgent(grid_size=5)
            st.rerun()
    
    with col_viz:
        st.subheader("🗺️ GridWorld")
        
        # Show current grid
        st.markdown(render_grid(env, env.agent_pos), unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("🧠 Q-Table (Agent's Brain)")
        
        st.write("**How to read:** Each cell shows the best action: ⬆️ ⬇️ ⬅️ ➡️")
        
        # Visualize Q-Table as a grid with arrows
        q_grid_html = "<div style='font-family: monospace; font-size: 0.8rem;'>"
        actions_emoji = ["⬆️", "⬇️", "⬅️", "➡️"]
        
        for i in range(env.size):
            q_grid_html += "<div>"
            for j in range(env.size):
                q_values = agent.q_table[i, j]
                best_action = np.argmax(q_values)
                max_q = q_values[best_action]
                
                # Color based on Q-value
                if max_q > 50:
                    bg_color = "#d1fae5"  # Green for high value
                elif max_q < -50:
                    bg_color = "#fee2e2"  # Red for low value
                else:
                    bg_color = "#f9fafb"
                
                q_grid_html += f"""
                <div style='width: 70px; height: 70px; border: 2px solid #ddd; display: inline-block; 
                            margin: 2px; text-align: center; background-color: {bg_color}; border-radius: 8px;
                            padding: 5px;'>
                    <div style='font-size: 1.5rem;'>{actions_emoji[best_action]}</div>
                    <div style='font-size: 0.7rem; color: #666;'>Q:{max_q:.1f}</div>
                </div>
                """
            q_grid_html += "</div>"
        
        q_grid_html += "</div>"
        st.markdown(q_grid_html, unsafe_allow_html=True)

with tab2:
    st.markdown("## 🧠 What is Reinforcement Learning?")
    
    st.info("""
    Imagine teaching a dog a trick. You don't explain calculus to the dog. You just say:
    - **Do the trick correctly → Get a treat 🍖**
    - **Do it wrong → No treat (or worse, punishment)**
    
    The dog **learns by trial and error**. That's Reinforcement Learning!
    """)
    
    st.divider()
    
    st.markdown("### 1. The Core Concept")
    
    col_rl1, col_rl2 = st.columns(2)
    
    with col_rl1:
        st.write("**The Loop:**")
        st.code("""
1. Agent observes State
2. Agent takes Action
3. Environment gives Reward
4. Agent learns from Feedback
5. Repeat
        """)
    
    with col_rl2:
        st.write("**The Goal:**")
        st.write("""
        Maximize **Total Reward** over time.
        
        Not just immediate reward, but **future rewards** too!
        
        This is why the robot doesn't rush to the bomb (immediate -100) 
        but instead takes the long path to the treasure (+100).
        """)
    
    st.divider()
    
    st.markdown("### 2. The Q-Table: The Cheat Sheet")
    
    st.write("""
    The **Q-Table** is the agent's brain. For every (**State**, **Action**) pair, it stores a **Q-Value**.
    
    **Q-Value** = "How good is this action in this state?"
    """)
    
    st.latex(r"Q(state, action) \rightarrow \text{Expected Future Reward}")
    
    st.write("""
    **Example:**
    - Q(Top-Left, →) = 85.3 → "Moving right from top-left is good!"
    - Q(Bottom-Left, ↑) = -95.2 → "Moving up from bottom-left is bad (leads to bomb)"
    """)
    
    st.divider()
    
    st.markdown("### 3. The Q-Learning Update Formula")
    
    st.write("This is the magic equation that makes the agent learn:")
    
    st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max Q(s', a') - Q(s, a)]")
    
    st.write("**Breaking it down:**")
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown("""
        - **s**: Current state
        - **a**: Action taken
        - **r**: Reward received
        - **s'**: Next state
        - **α (alpha)**: Learning rate (0-1)
        """)
    
    with col_q2:
        st.markdown("""
        - **γ (gamma)**: Discount factor (0-1)
        - **max Q(s', a')**: Best Q-value in next state
        
        **The Formula Says:**
        "Update my guess based on the difference between reality and expectation"
        """)
    
    st.divider()
    
    st.markdown("### 4. Exploration vs Exploitation")
    
    st.write("**The Dilemma:** Should the agent:")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("""
        **🎲 Explore (ε-greedy)**
        - Try random actions
        - Discover new strategies
        - Might find shortcuts
        - **Risk:** Wastes time on bad moves
        """)
    
    with col_ex2:
        st.markdown("""
        **🎯 Exploit**
        - Use best known action
        - Play it safe
        - Maximize current knowledge
        - **Risk:** Miss better strategies
        """)
    
    st.write("""
    **Solution: ε (Epsilon) Greedy**
    - Start with ε = 1.0 (100% exploration)
    - Gradually decay to ε = 0.01 (1% exploration)
    - This way, the agent explores early, then exploits later
    """)
    
    st.divider()
    
    st.markdown("### 5. Key Concepts Summary")
    
    st.info("""
    **State**: Where the agent is (position in grid)
    
    **Action**: What the agent can do (up, down, left, right)
    
    **Reward**: Feedback from environment (+100, -100, -1)
    
    **Policy**: The agent's strategy (Q-table tells us best action per state)
    
    **Episode**: One full game from start to treasure/bomb
    
    **Discount Factor (γ)**: 
    - γ = 0 → Only care about immediate reward
    - γ = 1 → Care equally about all future rewards
    - γ = 0.9 → Balance (typical choice)
    
    **Learning Rate (α)**:
    - α = 0 → Never learn
    - α = 1 → Forget everything except latest experience
    - α = 0.1 → Gradual learning (typical choice)
    """)
