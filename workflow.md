# System Workflow: Battery Health-Aware Scheduling in IoT/M2M Wireless Sensor Networks Using Double Deep Q-Networks

---

## III. System Architecture and Workflow

This section presents the end-to-end architecture of the proposed battery health-aware scheduling framework. The system operates as a centralized reinforcement learning (RL) controller that observes the joint state of all $N$ sensor nodes in the network, selects a per-node sleep/awake action vector, and updates its policy via off-policy temporal-difference learning. The four principal subsystems — environment and state aggregation, agent architecture and action selection, physics-grounded reward computation, and experience replay with the training loop — are described in detail below and correspond directly to the workflow illustrated in Fig. 1.

---

### A. Environment and State Aggregation

#### Network Topology and Initialization

The simulated IoT/M2M network consists of $N = 50$ battery-powered sensor nodes deployed uniformly at random within a two-dimensional arena of dimensions $500 \times 500$ m$^2$. A single sink node is fixed at the geometric center $(250, 250)$ m and acts as the data aggregation point. At the start of each episode, node positions are re-sampled from a uniform spatial distribution, and all battery states are initialized to full charge ($\text{SoC}_i = E_{\max} = 100$ normalized units, $\text{SoH}_i = 1.0$) with no prior activity history.

#### Per-Node State Features and Global Observation Vector

Although Fig. 1 depicts three separate "State Collection" blocks, this visual representation conveys the parallel and distributed nature of state sensing across all $N$ nodes simultaneously. In the centralized formulation, these readings are concatenated into a single flat global observation vector $\mathbf{s}_t \in \mathbb{R}^{6N}$ that is presented to the agent at each discrete timestep $t$. For node $i$, the six-dimensional local feature vector $\phi_i(t)$ is defined as:

$$
\phi_i(t) = \bigl[\widetilde{\text{SoC}}_i,\; \text{SoH}_i,\; a_{i,t-1},\; \tilde{d}_i,\; \rho_i(t),\; c_i(t)\bigr]
$$

where the components are:

- $\widetilde{\text{SoC}}_i = \text{SoC}_i / E_{\max} \in [0, 1]$: normalized State of Charge.
- $\text{SoH}_i \in [0, 1]$: State of Health, a monotonically non-increasing measure of long-term battery capacity.
- $a_{i,t-1} \in \{0, 1\}$: the action (SLEEP/AWAKE) executed by node $i$ in the previous timestep.
- $\tilde{d}_i \in [0, 1]$: Euclidean distance from node $i$ to the sink, normalized by the arena diagonal $\sqrt{500^2 + 500^2}$.
- $\rho_i(t) \in [0, 1]$: exponential moving average (EMA) of recent activity, updated as $\rho_i(t) = 0.9\,\rho_i(t-1) + 0.1\,a_{i,t}$, serving as a proxy for duty-cycle fairness.
- $c_i(t) \in \{0, 1\}$: binary charging flag, set to 1 when node $i$ is actively recuperating energy from an external source.

The global state vector is the row-major concatenation of all per-node features:

$$
\mathbf{s}_t = \bigl[\phi_1(t),\, \phi_2(t),\, \ldots,\, \phi_N(t)\bigr] \in \mathbb{R}^{6N}
$$

For $N = 50$, this yields a 300-dimensional input to the neural network. The observation space is bounded element-wise to $[0, 1]$, ensuring numerical stability during gradient-based optimization.

#### Battery Model: State of Charge and State of Health Dynamics

Each node $i$ is associated with an instance of the `BatteryModel` class, which tracks two coupled physical quantities — SoC and SoH — under a combined cycle-degradation and calendar-aging model.

**Discharge dynamics.** When node $i$ executes an AWAKE action, the energy draw per timestep is distance-weighted to account for the increased transmission power required by nodes farther from the sink:

$$
e_i^{\text{awake}} = E_{\text{awake}} \cdot \bigl(1 + 0.1\,\tilde{d}_i\bigr)
$$

where $E_{\text{awake}} = 1.0$ (normalized units). When the node executes SLEEP, it incurs only a leakage drain:

$$
e_i^{\text{sleep}} = E_{\text{sleep}} = 0.01
$$

The SoC update after drawing $e_i$ units is:

$$
\text{SoC}_i(t+1) = \max\!\bigl(0,\; \text{SoC}_i(t) - e_i\bigr)
$$

**SoH degradation.** At each discharge event, the Depth of Discharge (DoD) for node $i$ is computed as:

$$
\text{DoD}_i = \frac{|\text{SoC}_i(t) - \text{SoC}_i(t+1)|}{E_{\max}}
$$

SoH is then updated under a combined cycle-based and calendar-aging model:

$$
\text{SoH}_i(t+1) = \text{SoH}_i(t) - k_{\text{cycle}} \cdot \text{DoD}_i^{\,\alpha} - \delta_{\text{cal}}
$$

where $k_{\text{cycle}} = 5 \times 10^{-5}$ is the cycle degradation rate constant, $\alpha = 1.2$ is the DoD exponent (penalizing deep discharges super-linearly), and $\delta_{\text{cal}} = 5 \times 10^{-7}$ is the calendar fade applied at every timestep regardless of activity. SoH is clipped to $[0, 1]$ and is strictly non-recovering — it represents irreversible electrochemical degradation.

A node is declared dead and removed from scheduling consideration when either of the following conditions holds:

$$
\text{SoC}_i \leq 0.0001 \cdot E_{\max} \quad \text{or} \quad \text{SoH}_i \leq 0.05
$$

An episode terminates when the fraction of dead nodes exceeds the death threshold $\theta = 0.3$, i.e., $|\{i : \text{dead}(i)\}| > \theta N$, or when the step counter reaches $T_{\max} = 1000$.

#### Charging State Machine

To model energy harvesting or opportunistic charging in M2M deployments, the environment implements a hysteretic charging state machine. At each timestep, prior to executing the agent's scheduled action, the environment evaluates the following transition rules for every live node $i$:

- **Entry condition:** If $\widetilde{\text{SoC}}_i < \varphi_{\text{entry}} = 0.2$, the node enters the charging state ($c_i \leftarrow 1$).
- **Exit condition:** If $\widetilde{\text{SoC}}_i \geq \varphi_{\text{exit}} = 0.95$, the node exits the charging state ($c_i \leftarrow 0$).

While in the charging state, the node is forcibly overridden to SLEEP regardless of the agent's requested action, and its SoC is incremented by:

$$
\text{SoC}_i(t+1) = \min\!\bigl(E_{\max},\; \text{SoC}_i(t) + \eta_c \cdot E_{\max}\bigr)
$$

where $\eta_c = 0.05$ is the per-step charging rate expressed as a fraction of $E_{\max}$. Calendar aging ($\delta_{\text{cal}}$) continues to apply during charging, reflecting real-world electrolyte and thermal stress during the charge phase. The charging flag $c_i(t)$ is exposed in the observation vector (feature index 5), enabling the agent to learn that a node in the charging state is temporarily unavailable for transmission scheduling.

#### Cooperative Wake-Up Mechanism

To prevent coverage holes caused by the simultaneous low-battery condition of spatially proximate nodes, the environment enforces a cooperative wake-up rule. This rule executes after the charging override but before the physics update, operating on the agent's effective (post-override) action vector $\mathbf{a}_t^{\text{eff}}$. Specifically, for each node $i$ that satisfies all of the following:

1. $a_{i,t}^{\text{eff}} = 1$ (currently scheduled AWAKE),
2. $\widetilde{\text{SoC}}_i \leq \omega = 0.5$ (operating near low-battery threshold), and
3. $\text{dead}(i) = \text{False}$,

the nearest node $j^*$ satisfying $a_{j,t}^{\text{eff}} = 0$, $c_j = 0$, and $\text{dead}(j) = \text{False}$ is identified by:

$$
j^* = \arg\min_{j \neq i} \|\mathbf{p}_i - \mathbf{p}_j\|_2
$$

and its action is overridden to AWAKE: $a_{j^*,t}^{\text{eff}} \leftarrow 1$. The set of cooperatively woken node identifiers is recorded in the step's `info` dictionary under `cooperative_wakes` for analysis and logging. This mechanism is executed prior to any energy draw, ensuring that coverage continuity is maintained even when the agent's learned policy has not yet adapted to emergent low-SoC conditions. The charging flag $c_i$ appearing in the observation space allows the agent to eventually internalize this mechanism and pre-emptively schedule backup nodes before the cooperative rule fires.

---

### B. DDQN Agent Architecture and Action Selection

#### Q-Network Architecture

The agent maintains two deep neural networks — an online (policy) network $Q_{\theta}$ and a target network $Q_{\theta^-}$ — with identical feedforward architectures. The input layer accepts the global state vector $\mathbf{s}_t \in \mathbb{R}^{6N}$ and the output layer produces $2N$ scalar Q-values. For $N = 50$, the full architecture is:

$$
\mathbb{R}^{300} \xrightarrow{\text{FC}_{512}} \text{ReLU} \xrightarrow{\text{FC}_{256}} \text{ReLU} \xrightarrow{\text{FC}_{100}} \mathbb{R}^{2N}
$$

The output is reshaped to a matrix $\mathbf{Q}(s) \in \mathbb{R}^{N \times 2}$, where $\mathbf{Q}(s)[i, a]$ represents the expected cumulative discounted return for node $i$ executing action $a \in \{0, 1\}$ given global state $\mathbf{s}$. Both networks are instantiated with the same random initialization; the target network weights $\theta^-$ are subsequently updated from $\theta$ only at periodic synchronization intervals (every 500 gradient steps), providing the temporal separation that stabilizes the bootstrapped TD target.

#### Epsilon-Greedy Exploration

During training, the agent follows an annealed epsilon-greedy policy. The exploration rate at learn-step $t$ is computed as:

$$
\varepsilon(t) = \varepsilon_{\text{end}} + (\varepsilon_{\text{start}} - \varepsilon_{\text{end}}) \cdot \max\!\left(0,\; 1 - \frac{t}{\varepsilon_{\text{decay}}}\right)
$$

with $\varepsilon_{\text{start}} = 1.0$, $\varepsilon_{\text{end}} = 0.05$, and $\varepsilon_{\text{decay}} = 50{,}000$ steps. With probability $\varepsilon(t)$, a uniformly random action vector is sampled; otherwise, the greedy action is selected per node:

$$
a_{i,t} = \arg\max_{a \in \{0,1\}} Q_\theta(\mathbf{s}_t)[i,\, a], \quad \forall\, i \in \{1, \ldots, N\}
$$

During evaluation, $\varepsilon$ is set to zero (fully greedy), and the agent operates deterministically.

#### Decoupled DDQN Target vs. Standard DQN Baseline

A central contribution of the proposed framework is the use of the Double DQN formulation to address the positive bias introduced by the $\max$ operator in standard Q-learning. In the standard DQN (used as an ablation baseline), the bootstrapped target for a transition $(s, a, r, s', d)$ is:

$$
y^{\text{DQN}} = r + \gamma \cdot \max_{a'} Q_{\theta^-}(s')[a'] \cdot (1 - d)
$$

This formulation conflates action selection and action evaluation within the same (target) network, leading to systematic overestimation of Q-values in stochastic environments. The DDQN decouples these two operations: the online network $Q_\theta$ nominates the best action, and the target network $Q_{\theta^-}$ evaluates its value:

$$
a^* = \arg\max_{a'} Q_\theta(s')[a'] \qquad \text{(online network selects)}
$$

$$
y^{\text{DDQN}} = r + \gamma \cdot Q_{\theta^-}(s')[a^*] \cdot (1 - d) \qquad \text{(target network evaluates)}
$$

Since the online and target networks diverge between synchronization events, $a^*$ and $Q_{\theta^-}(\cdot)[a^*]$ are produced by networks with different parameterizations, substantially reducing overestimation bias. In the multi-node WSN setting, where up to $N = 50$ concurrent Q-values must be estimated under a non-stationary joint reward signal, this decoupling is particularly important for stable convergence. The per-node Q-values are averaged across nodes before computing the scalar mean-squared error (MSE) loss:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}}\!\left[\left(\overline{Q}_\theta(s, a) - y^{\text{DDQN}}\right)^2\right]
$$

where $\overline{Q}_\theta(s, a) = \frac{1}{N}\sum_{i=1}^{N} Q_\theta(s)[i,\, a_i]$ is the mean selected Q-value across all nodes.

---

### C. Physics-Grounded Reward Computation

The scalar reward $R_t$ returned to the agent at each timestep is a weighted linear combination of four domain-specific components, each normalized to a bounded range to prevent any single objective from dominating gradient updates:

$$
R_t = w_{\text{cov}}\, r_{\text{cov}} + w_{\text{eng}}\, r_{\text{eng}} + w_{\text{soh}}\, r_{\text{soh}} + w_{\text{bal}}\, r_{\text{bal}}
$$

with weights $(w_{\text{cov}}, w_{\text{eng}}, w_{\text{soh}}, w_{\text{bal}}) = (10.0,\, 5.0,\, 1.0,\, 2.0)$.

**Coverage reward** $r_{\text{cov}}$. A $20 \times 20$ uniform grid of sample points is overlaid on the arena. A grid point $\mathbf{g}$ is considered covered if at least one live, AWAKE node $i$ satisfies $\|\mathbf{p}_i - \mathbf{g}\|_2 \leq R_{\text{sense}} = 100$ m. The coverage fraction is:

$$
r_{\text{cov}} = \frac{\left|\left\{\mathbf{g} : \exists\, i\ \text{alive},\; a_{i,t}^{\text{eff}} = 1,\; \|\mathbf{p}_i - \mathbf{g}\|_2 \leq R_{\text{sense}}\right\}\right|}{400} \in [0, 1]
$$

**Energy efficiency reward** $r_{\text{eng}}$. To penalize energy drain preferentially on nodes with low residual charge — since depleting an already-depleted node is costlier in terms of network lifetime — the energy term is charge-weighted:

$$
r_{\text{eng}} = -\,\text{clip}\!\left(\frac{\sum_{i=1}^{N} e_i \cdot (1 - \widetilde{\text{SoC}}_i)}{N \cdot E_{\text{awake}} \cdot 2},\; 0,\; 1\right) \in [-1, 0]
$$

Nodes with lower normalized SoC contribute disproportionately to the penalty, incentivizing the agent to rotate scheduling load toward nodes with higher residual charge.

**Battery health reward** $r_{\text{soh}}$. To discourage deep-discharge cycling that accelerates irreversible SoH degradation:

$$
r_{\text{soh}} = \text{clip}\!\bigl(\overline{\text{SoH}} - 0.99,\; -1,\; 1\bigr) \in [-1, 1]
$$

where $\overline{\text{SoH}} = \frac{1}{N}\sum_{i=1}^{N} \text{SoH}_i$. This term is near-zero when average health is high and becomes strongly negative as the network ages.

**Load-balance reward** $r_{\text{bal}}$. To promote equitable energy distribution across nodes, thereby extending aggregate network lifetime:

$$
r_{\text{bal}} = \text{clip}\!\left(-\,\sigma\!\left(\left\{\widetilde{\text{SoC}}_i\right\}_{i=1}^{N}\right),\; -1,\; 0\right) \in [-1, 0]
$$

where $\sigma(\cdot)$ denotes the standard deviation of the normalized SoC distribution. A uniform charge distribution yields $r_{\text{bal}} = 0$; high disparity yields a strongly negative value.

**Terminal penalty.** If the death threshold $\theta = 0.3$ is breached (more than 30% of nodes dead), the episode terminates immediately and an additional scalar penalty of $-10$ is subtracted from $R_t$ to strongly discourage policies that allow catastrophic node failure.

---

### D. Experience Replay and Training Loop

#### Replay Buffer

Transitions $(s_t, \mathbf{a}_t, R_t, s_{t+1}, d_t)$ are stored in a circular first-in-first-out (FIFO) replay buffer $\mathcal{D}$ with capacity $|\mathcal{D}_{\max}| = 200{,}000$ transitions. Uniform random sampling from $\mathcal{D}$ decorrelates consecutive experience tuples, mitigating the non-stationarity of the on-policy data distribution. The agent's `learn_step()` function is a no-op until $|\mathcal{D}| \geq |\mathcal{D}_{\min}| = 500$, ensuring sufficient diversity in the initial mini-batches before gradient updates begin.

#### Per-Timestep Training Step

At each environment step $t$ within each training episode, the following sequence is executed:

1. Construct global observation $\mathbf{s}_t = [\phi_1(t), \ldots, \phi_N(t)]$ via `_get_obs()`.
2. Agent selects action $\mathbf{a}_t \leftarrow \varepsilon\text{-greedy}(Q_\theta, \mathbf{s}_t)$.
3. Environment applies charging overrides and cooperative wake-up to produce effective action $\mathbf{a}_t^{\text{eff}}$.
4. Physics step: update SoC, SoH, `recent_activity`, and `charging` flags for all nodes.
5. Compute coverage $r_{\text{cov}}$, energy draw $r_{\text{eng}}$, health $r_{\text{soh}}$, and balance $r_{\text{bal}}$; assemble scalar reward $R_t$.
6. Construct next observation $\mathbf{s}_{t+1}$; check termination condition to set $d_t$.
7. Store $(\mathbf{s}_t, \mathbf{a}_t, R_t, \mathbf{s}_{t+1}, d_t)$ in $\mathcal{D}$.
8. If $|\mathcal{D}| \geq |\mathcal{D}_{\min}|$, sample mini-batch of size $B = 64$ from $\mathcal{D}$ and perform one gradient step.

#### Gradient Update

For each sampled mini-batch, the DDQN target is computed under `torch.no_grad()` to prevent gradients from flowing into the target network:

$$
y_k = R_k + \gamma \cdot Q_{\theta^-}(s_k')\!\left[\arg\max_{a'} Q_\theta(s_k')[a']\right] \cdot (1 - d_k)
$$

The MSE loss is computed between the mean online Q-value and the target:

$$
\mathcal{L}(\theta) = \frac{1}{B}\sum_{k=1}^{B} \left(\overline{Q}_\theta(s_k, \mathbf{a}_k) - y_k\right)^2
$$

Gradients are computed via backpropagation, clipped to a maximum $\ell_2$-norm of $10.0$ to prevent gradient explosion in deep multi-objective reward landscapes, and the online network weights are updated using the Adam optimizer with learning rate $\eta = 10^{-4}$ and discount factor $\gamma = 0.99$:

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta), \quad \|\nabla_\theta \mathcal{L}(\theta)\|_2 \leq 10.0
$$

The target network is synchronized with the online network every $\tau = 500$ gradient steps via a hard copy: $\theta^- \leftarrow \theta$.

#### Full Training Algorithm

The complete per-episode training procedure is formalized as **Algorithm 1**.

---

**Algorithm 1: Battery Health-Aware DDQN Training for WSN Scheduling**

**Input:** Number of episodes $E$, nodes $N$, max steps per episode $T_{\max}$, discount $\gamma$, learning rate $\eta$, replay capacity $|\mathcal{D}_{\max}|$, min replay size $|\mathcal{D}_{\min}|$, target update interval $\tau$, weights $(w_{\text{cov}}, w_{\text{eng}}, w_{\text{soh}}, w_{\text{bal}})$

**Output:** Trained policy network $Q_\theta$; per-episode series for coverage, SoH, alive fraction, energy consumption, throughput

1. Initialize $Q_\theta$, $Q_{\theta^-}$ with shared random weights; $\theta^- \leftarrow \theta$
2. Initialize replay buffer $\mathcal{D} \leftarrow \emptyset$; learn step counter $t \leftarrow 0$
3. **for** episode $= 1, \ldots, E$ **do**
4. $\quad$ Reset environment: randomize node positions; reset all SoC $= E_{\max}$, SoH $= 1.0$
5. $\quad$ Observe $\mathbf{s}_0 \leftarrow [\phi_1(0), \ldots, \phi_N(0)]$; done $\leftarrow$ False
6. $\quad$ **while** not done **do**
7. $\quad\quad$ Sample $\mathbf{a} \sim \varepsilon(t)\text{-greedy}(Q_\theta, \mathbf{s})$
8. $\quad\quad$ Apply charging overrides: $\forall i$ with $\widetilde{\text{SoC}}_i < 0.2$: $a_i^{\text{eff}} \leftarrow 0$, recover SoC
9. $\quad\quad$ Apply cooperative wake-up: $\forall i$ with $a_i^{\text{eff}} = 1$ and $\widetilde{\text{SoC}}_i \leq 0.5$: wake nearest idle neighbor
10. $\quad\quad$ Discharge each node by $e_i^{\text{awake}}$ or $e_i^{\text{sleep}}$; update SoC and SoH
11. $\quad\quad$ Compute $r_{\text{cov}}, r_{\text{eng}}, r_{\text{soh}}, r_{\text{bal}}$; assemble $R = w_{\text{cov}} r_{\text{cov}} + w_{\text{eng}} r_{\text{eng}} + w_{\text{soh}} r_{\text{soh}} + w_{\text{bal}} r_{\text{bal}}$
12. $\quad\quad$ If dead fraction $> \theta$: $R \leftarrow R - 10$; done $\leftarrow$ True
13. $\quad\quad$ Observe $\mathbf{s}' \leftarrow [\phi_1, \ldots, \phi_N]$; push $(\mathbf{s}, \mathbf{a}, R, \mathbf{s}', \text{done})$ to $\mathcal{D}$
14. $\quad\quad$ **if** $|\mathcal{D}| \geq |\mathcal{D}_{\min}|$ **then**
15. $\quad\quad\quad$ Sample mini-batch $\{(s_k, \mathbf{a}_k, R_k, s_k', d_k)\}_{k=1}^{B}$ from $\mathcal{D}$
16. $\quad\quad\quad$ Compute DDQN targets: $y_k = R_k + \gamma (1 - d_k) \cdot Q_{\theta^-}(s_k')[\arg\max_{a'} Q_\theta(s_k')[a']]$
17. $\quad\quad\quad$ Update $\theta$ via Adam: minimize $\mathcal{L}(\theta)$; clip $\|\nabla\|_2 \leq 10.0$; $t \leftarrow t + 1$
18. $\quad\quad\quad$ **if** $t \bmod \tau = 0$: $\theta^- \leftarrow \theta$
19. $\quad\quad$ $\mathbf{s} \leftarrow \mathbf{s}'$
20. $\quad$ **end while**
21. $\quad$ Record episode metrics: mean coverage, mean SoH, final alive fraction, $\Delta\overline{\text{SoC}}$ (energy consumed), throughput $= \text{coverage} \times \text{alive fraction}$
22. **end for**
23. **return** $Q_\theta$, episode series

---

#### Network Lifetime Metric

Beyond per-episode reward, the primary long-horizon metric reported is the *network lifetime* $L$, defined as the first episode index at which the alive fraction drops below $(1 - \theta) = 0.70$:

$$
L = \min\bigl\{e : \text{alive\_fraction}(e) < 1 - \theta\bigr\}
$$

If the alive fraction never falls below this threshold across all $E$ training episodes, $L$ is reported as $E$. This metric directly captures the ability of the learned scheduling policy to extend the operational lifespan of the WSN under realistic battery degradation dynamics.

#### Throughput Proxy Metric

Per-episode throughput is computed as the product of mean spatial coverage and the episode-final alive fraction:

$$
\Gamma = \overline{r}_{\text{cov}} \times \text{alive\_fraction}_{\text{final}}
$$

This joint metric penalizes policies that achieve high coverage at the cost of accelerated node death, incentivizing balanced scheduling strategies that are aligned with the multi-objective reward structure described in Section III-C.

---

## Fig. 1 — System Workflow Diagram (draw.io Plantuml)

Copy the block below and paste it into [draw.io](https://app.diagrams.net/) via **Extras → Edit Diagram** to render the full workflow figure.

@startuml
' Layout Settings for A4 Width
left to right direction
skinparam componentStyle rectangle
skinparam shadowing false
skinparam packageStyle rectangle
skinparam nodesep 20
skinparam ranksep 40

' Styling for readability
skinparam rectangle {
    BackgroundColor White
    BorderColor Black
}

title Battery Health-Aware DDQN Scheduling — System Workflow (Horizontal Layout)

' =========================
' ENVIRONMENT & STATE
' =========================
package "1. Environment State" {
    rectangle "Node Cluster\n(N=50)\nSoC, SoH, Features" as NODES
    rectangle "State Vector\ns_t ∈ R³⁰⁰" as OBS
    NODES --> OBS : _get_obs()
}

' =========================
' AGENT LOGIC
' =========================
package "2. DDQN Agent" {
    rectangle "Online Q-Net\n(300→512→256→100)" as Q
    rectangle "ε-Greedy\nPolicy" as EPS
    rectangle "Target Q-Net\n(θ⁻ Sync)" as QT
    
    OBS --> Q
    Q --> EPS
}

' =========================
' ACTION & ENVIRONMENT STEP
' =========================
rectangle "Action a_t\n{0,1}⁵⁰" as ACT
EPS --> ACT : select

package "3. Physics Engine (WSNEnv)" {
    rectangle "Charging & Wake-up\nLogic" as PHYS
    rectangle "Energy & SoH\nDecay Models" as MODEL
    PHYS -> MODEL
}

ACT --> PHYS

' =========================
' REWARDS & TRANSITION
' =========================
package "4. Metrics" {
    rectangle "Reward (R_t)\n[Cov, Eng, SoH, Bal]" as RT
    rectangle "Next State\ns_{t+1}" as NEXT
}

MODEL --> RT
MODEL --> NEXT

' =========================
' LEARNING ENGINE
' =========================
package "5. Training Pipeline" {
    database "Replay Buffer\n(200k samples)" as RB
    rectangle "Adam Optimizer\n(Batch size 64)" as OPT
}

RT --> RB
NEXT --> RB
RB --> OPT
OPT .up.> Q : Update θ
OPT .up.> QT : Sync every τ

' Feedback Loop
NEXT .[#blue,bold].> OBS : Loop to next t

@enduml
