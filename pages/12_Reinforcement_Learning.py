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
    .hero-container { padding: 1rem 0; margin-bottom: 2rem; }
    .hero-title { font-size: 2.5rem; font-weight: 800; color: #1f2937; }
    .hero-subtitle { font-size: 1.1rem; color: #6c757d; font-style: italic; }
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
</style>
""", unsafe_allow_html=True)

# --- Q-LEARNING ENVIRONMENT ---
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.trap_pos = [self.size-1, 0]
        return tuple(self.agent_pos)
    
    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        row, col = self.agent_pos
        
        if action == 0 and row > 0:  # Up
            row -= 1
        elif action == 1 and row < self.size - 1:  # Down
            row += 1
        elif action == 2 and col > 0:  # Left
            col -= 1
        elif action == 3 and col < self.size - 1:  # Right
            col += 1
        
        self.agent_pos = [row, col]
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 100  # Big reward for goal!
            done = True
        elif self.agent_pos == self.trap_pos:
            reward = -100  # Big penalty for trap!
            done = True
        else:
            reward = -1  # Small penalty for each step (encourages efficiency)
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
            return np.random.randint(0, 4)  # Explore (random)
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit (best known)
    
    def update(self, state, action, reward, next_state, done):
        # Q-Learning Formula: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
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

# --- VISUALIZATION HELPER ---
def render_grid(env, show_path=None):
    """Render the grid with emojis"""
    grid_html = '<div style="display: inline-block;">'
    
    for i in range(env.size):
        grid_html += '<div style="display: flex;">'
        for j in range(env.size):
            # Determine cell content and color
            if [i, j] == env.agent_pos:
                emoji = "🤖"
                bg_color = "#60a5fa"  # Blue
            elif [i, j] == env.goal_pos:
                emoji = "💎"
                bg_color = "#34d399"  # Green
            elif [i, j] == env.trap_pos:
                emoji = "💣"
                bg_color = "#f87171"  # Red
            elif show_path and (i, j) in show_path:
                emoji = "👣"
                bg_color = "#fcd34d"  # Yellow
            else:
                emoji = "⬜"
                bg_color = "#f3f4f6"  # Gray
            
            grid_html += f'''
                <div style="width: 60px; height: 60px; background-color: {bg_color}; 
                            border: 2px solid #e5e7eb; border-radius: 8px; 
                            display: flex; align-items: center; justify-content: center;
                            font-size: 2rem; margin: 2px;">
                    {emoji}
                </div>
            '''
        grid_html += '</div>'
    
    grid_html += '</div>'
    return grid_html

def render_q_table(agent, env):
    """Render Q-table as a grid showing best actions"""
    actions_emoji = ["⬆️", "⬇️", "⬅️", "➡️"]
    
    q_grid_html = '<div style="display: inline-block; margin-top: 1rem;">'
    
    for i in range(agent.grid_size):
        q_grid_html += '<div style="display: flex;">'
        for j in range(agent.grid_size):
            q_values = agent.q_table[i, j]
            best_action = np.argmax(q_values)
            max_q = q_values[best_action]
            
            # Color based on Q-value
            if max_q > 50:
                bg_color = "#d1fae5"  # Green
            elif max_q < -50:
                bg_color = "#fee2e2"  # Red
            else:
                bg_color = "#f9fafb"  # Gray
            
            q_grid_html += f'''
                <div style="width: 70px; height: 70px; background-color: {bg_color}; 
                            border: 2px solid #ddd; border-radius: 8px; 
                            display: flex; flex-direction: column; align-items: center; 
                            justify-content: center; margin: 2px; padding: 5px;">
                    <div style="font-size: 1.8rem;">{actions_emoji[best_action]}</div>
                    <div style="font-size: 0.7rem; color: #666;">Q:{max_q:.1f}</div>
                </div>
            '''
        q_grid_html += '</div>'
    
    q_grid_html += '</div>'
    return q_grid_html

# --- INITIALIZE SESSION STATE ---
if 'env' not in st.session_state:
    st.session_state['env'] = GridWorld(size=5)
    st.session_state['agent'] = QLearningAgent(grid_size=5)
    st.session_state['episode_count'] = 0
    st.session_state['training_rewards'] = []

env = st.session_state['env']
agent = st.session_state['agent']

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎮 Reinforcement Learning (Q-Learning)</div>
    <div class="hero-subtitle">"The robot learns to navigate by trial and error: 🍪 for wins, 👋 for losses."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🎮 Interactive Playground", "📚 Theory & Math"])

with tab1:
    st.info("""
    **🎯 Mission:** Watch the robot 🤖 learn to find the treasure 💎 while avoiding the trap 💣!
    - **💎 Treasure (Goal)**: +100 reward - The robot wins!
    - **💣 Trap**: -100 reward - The robot loses!
    - **Each step**: -1 reward (teaches the robot to be efficient)
    """)
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("1. The Environment")
        
        # Grid visualization
        grid_placeholder = st.empty()
        grid_placeholder.markdown(render_grid(env), unsafe_allow_html=True)
        
        st.write("---")
        
        # Controls
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("👟 Single Step", use_container_width=True):
                state = tuple(env.agent_pos)
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                if done:
                    if reward > 0:
                        st.toast("🎉 Found Treasure! +100", icon="💎")
                    else:
                        st.toast("💥 Hit Trap! -100", icon="💣")
                    time.sleep(1)
                    env.reset()
                    st.session_state['episode_count'] += 1
                
                st.rerun()
        
        with col_btn2:
            if st.button("🎬 Watch Episode", use_container_width=True):
                state = env.reset()
                path = [list(state)]
                step_count = 0
                max_steps = 50
                episode_reward = 0
                
                status_placeholder = st.empty()
                
                while step_count < max_steps:
                    action = agent.get_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    
                    episode_reward += reward
                    path.append(list(env.agent_pos))
                    
                    # Show animation
                    grid_placeholder.markdown(render_grid(env, path), unsafe_allow_html=True)
                    status_placeholder.write(f"Step {step_count + 1} | Reward: {reward:+.0f} | Total: {episode_reward:+.0f}")
                    time.sleep(0.3)
                    
                    if done:
                        if reward > 0:
                            st.success(f"✅ Episode Complete! Total Reward: {episode_reward:+.0f} | Steps: {step_count + 1}")
                        else:
                            st.error(f"❌ Hit Trap! Total Reward: {episode_reward:+.0f} | Steps: {step_count + 1}")
                        break
                    
                    state = next_state
                    step_count += 1
                
                agent.decay_epsilon()
                st.session_state['episode_count'] += 1
                st.session_state['training_rewards'].append(episode_reward)
                time.sleep(1)
                env.reset()
                st.rerun()
        
        with col_btn3:
            if st.button("⚡ Train 50x", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for ep in range(50):
                    state = env.reset()
                    episode_reward = 0
                    step_count = 0
                    max_steps = 50
                    
                    while step_count < max_steps:
                        action = agent.get_action(state)
                        next_state, reward, done = env.step(action)
                        agent.update(state, action, reward, next_state, done)
                        
                        episode_reward += reward
                        state = next_state
                        step_count += 1
                        
                        if done:
                            break
                    
                    agent.decay_epsilon()
                    st.session_state['training_rewards'].append(episode_reward)
                    
                    progress_bar.progress((ep + 1) / 50)
                    status_text.text(f"Episode {ep + 1}/50 | Reward: {episode_reward:+.0f} | ε: {agent.epsilon:.3f}")
                
                st.session_state['episode_count'] += 50
                env.reset()
                st.success("Training Complete!")
                st.rerun()
    
    with col_right:
        st.subheader("2. The Robot's Brain (Q-Table)")
        
        st.write("**What is this?** The Q-Table is the robot's memory. Each cell shows:")
        st.write("- **Arrow**: Best action to take from that position")
        st.write("- **Q-value**: How good that position is")
        
        # Q-table visualization
        qtable_placeholder = st.empty()
        qtable_placeholder.markdown(render_q_table(agent, env), unsafe_allow_html=True)
        
        st.write("---")
        
        # Stats
        st.metric("Episodes Trained", st.session_state['episode_count'])
        st.metric("Exploration Rate (ε)", f"{agent.epsilon:.3f}", 
                 help="Probability of taking random action vs best known action")
        
        if st.session_state['training_rewards']:
            recent_avg = np.mean(st.session_state['training_rewards'][-10:])
            st.metric("Avg Reward (Last 10)", f"{recent_avg:+.1f}")
        
        if st.button("🔄 Reset Everything", use_container_width=True):
            st.session_state['env'] = GridWorld(size=5)
            st.session_state['agent'] = QLearningAgent(grid_size=5)
            st.session_state['episode_count'] = 0
            st.session_state['training_rewards'] = []
            st.rerun()

with tab2:
    st.markdown("## 🧠 What is Reinforcement Learning?")
    
    st.info("""
    Imagine teaching a puppy where to find its food bowl. You don't give it a map. 
    You let it explore the house. When it finds the bowl → **Treat!** 🍖  
    When it goes near the trash → **No!** 👎
    
    Eventually, the puppy learns the optimal path. That's **Reinforcement Learning**!
    """)
    
    st.divider()
    
    st.markdown("### 1. The Core Loop")
    
    col_loop1, col_loop2 = st.columns([1, 1])
    
    with col_loop1:
        st.write("**The Process:**")
        st.code("""
1. Robot observes STATE
   (Where am I?)

2. Robot takes ACTION  
   (Up, Down, Left, Right)

3. Environment gives REWARD
   (+100, -100, or -1)

4. Robot learns from experience
   (Updates Q-Table)

5. Repeat until done
        """, language="")
    
    with col_loop2:
        st.write("**The Goal:**")
        st.write("""
        Maximize **cumulative reward** over time.
        
        Not just immediate reward, but **all future rewards** too!
        
        This is why the robot learns to avoid the trap even though 
        it might be tempting to explore there initially.
        """)
    
    st.divider()
    
    st.markdown("### 2. The Q-Table Explained")
    
    st.write("""
    The **Q-Table** is like the robot's cheat sheet. For every position and action, 
    it remembers: *"How good is it to do this action here?"*
    """)
    
    st.latex(r"Q(state, action) = \text{Expected Total Future Reward}")
    
    st.write("**Example Values:**")
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.success("Q((0,0), RIGHT) = +85.3")
        st.caption("✅ Moving right from top-left is GREAT! (leads toward goal)")
    
    with col_ex2:
        st.error("Q((4,1), UP) = -92.1")
        st.caption("❌ Moving up from bottom row is TERRIBLE! (leads to trap)")
    
    st.divider()
    
    st.markdown("### 3. The Learning Formula (Bellman Equation)")
    
    st.write("This is the **magic equation** that makes learning happen:")
    
    st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]")
    
    st.write("**Breaking it down:**")
    
    col_formula1, col_formula2 = st.columns(2)
    
    with col_formula1:
        st.markdown("""
        **Variables:**
        - **s**: Current state (position)
        - **a**: Action taken
        - **r**: Immediate reward received
        - **s'**: New state after action
        - **α (alpha)**: Learning rate (0-1)
        - **γ (gamma)**: Discount factor (0-1)
        """)
    
    with col_formula2:
        st.markdown("""
        **What it means:**
        
        "Update my guess by comparing:
        - What I **expected** to get
        - What I **actually** got (reward + best future)"
        
        If reality is better → increase Q-value  
        If reality is worse → decrease Q-value
        """)
    
    st.divider()
    
    st.markdown("### 4. Exploration vs Exploitation (ε-greedy)")
    
    st.write("**The Dilemma:** Should the robot...")
    
    col_expl1, col_expl2 = st.columns(2)
    
    with col_expl1:
        st.markdown("""
        **🎲 EXPLORE (Random Actions)**
        - Try new paths
        - Might discover shortcuts
        - Learn about the world
        
        **Risk:** Waste time on bad moves
        """)
    
    with col_expl2:
        st.markdown("""
        **🎯 EXPLOIT (Best Known Action)**
        - Use current knowledge
        - Play it safe
        - Maximize rewards
        
        **Risk:** Never find better strategies
        """)
    
    st.info("""
    **Solution: ε (Epsilon) Greedy**
    
    - Start with ε = 1.0 (100% exploration - robot is clueless)
    - Gradually decay to ε = 0.01 (1% exploration - robot is expert)
    
    This way: **Explore early, exploit later!**
    """)
    
    st.divider()
    
    st.markdown("### 5. Hyperparameters Explained")
    
    col_hyper1, col_hyper2 = st.columns(2)
    
    with col_hyper1:
        st.markdown("""
        **Learning Rate (α)**
        - α = 0 → Never learn (stuck)
        - α = 1 → Only remember latest experience (unstable)
        - α = 0.1 → **Typical choice** (gradual learning)
        """)
        
        st.markdown("""
        **Discount Factor (γ)**
        - γ = 0 → Only care about immediate reward (greedy)
        - γ = 1 → Value all future rewards equally
        - γ = 0.9 → **Typical choice** (balanced)
        """)
    
    with col_hyper2:
        st.markdown("""
        **Exploration Rate (ε)**
        - Starts high (explore unknown)
        - Decays over time
        - Minimum value prevents getting stuck
        """)
        
        st.markdown("""
        **Why -1 for each step?**
        - Encourages shortest path
        - Without it, robot might wander aimlessly
        - Makes reaching goal quickly more valuable
        """)