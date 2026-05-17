# Major Problems Still Remaining

## 1. Placeholders Still Present

This is the first thing to fix.

### Remaining placeholders
- Page 7 still contains:
  > "[PLACEHOLDER: Insert analysis of per-step energy expenditure...]"

- Page 9 still contains:
  > "[PLACEHOLDER: Leave empty for now — to be filled with hyperparameter sensitivity results.]"

### Required Fix
Fill these subsections with real results or remove them entirely.

Do not leave an empty **Hyperparameter Sensitivity** subsection unless the experiment has actually been performed.

---

## 2. Evaluation Protocol Is Not Acceptable for a Strong Journal

The paper still evaluates the policy using the moving average of the final 10 training episodes and explicitly states that there are no separate evaluation runs.

This is weak for reinforcement learning and networking research.

### Required Evaluation Protocol
- Train the model
- Freeze the trained policy
- Run deterministic test episodes with:
  - \( \epsilon = 0 \)
- Use unseen random topologies
- Report:
  - Mean
  - Standard deviation
  - Preferably 95% confidence intervals

### Minimum Acceptable Change
Use:
- 5 independent seeds
- 20–50 deterministic test episodes per seed

If compute is limited:
- 5 seeds × 20 test episodes

This is still substantially better than final-10 training averages.

---

## 3. Network Lifetime Is Defined Incorrectly

The manuscript currently defines network lifetime as the episode index at which alive fraction drops below:

\[
1 - \delta
\]

This is incorrect.

Network lifetime should be measured **within an episode**, using:
- Timesteps
- Rounds
- Seconds
- Simulated operating hours

### Current Problematic Definition
> "Network Lifetime (episodes): 500 (max)"

This only indicates training duration and says nothing about operational survival time.

### Recommended Definition

\[
T_{life} = \min \{ t : dead\_fraction(t) > \delta \}
\]

### Required Reporting
Report:
- Mean lifetime in timesteps
- Standard deviation
- Maximum lifetime if no failure occurs before \( T_{max} \)

---

## 4. No Real DDQN Advantage Demonstrated

### Reported Results

| Metric | DQN | DDQN-WSN | Difference |
|---|---|---|---|
| Reward | 8297.79 | 8305.82 | +0.10% |
| Coverage | 96.45% | 96.36% | -0.09 pp |
| SoH | 0.99968 | 0.99968 | Parity |
| Lifetime | 500 | 500 | Parity |

These results do **not** demonstrate statistically significant superiority of DDQN over DQN.

### Recommended Claim
Instead of claiming a performance advantage, write:

> "DDQN-WSN achieves performance comparable to DQN under the current 500-episode setting, while providing a more robust target-estimation mechanism for factorized per-node scheduling."

---

## 5. Constraint-Layer Ablation Needs Quantitative Results

The subsection **Constraint-Layer Analysis** currently provides only narrative discussion.

Claims currently mentioned:
- Constraint-only coverage around 50–55%
- Unconstrained DDQN around 12% failure rate
- Full DDQN-WSN outperforming both

These require quantitative support.

### Recommended Table Structure

| Variant | Coverage (%) | Failure Rate (%) | Mean Reward | Mean SoH | Lifetime |
|---|---|---|---|---|---|
| Constraint-only random |  |  |  |  |  |
| Unconstrained DDQN |  |  |  |  |  |
| Full DDQN-WSN |  |  |  |  |  |

This ablation may become stronger evidence than the DQN-vs-DDQN comparison.

---

## 6. Battery Model Became More Complex but Less Clear

The manuscript added:
- Temperature terms
- Chemistry terms
- Activation energy
- C-rate terms

However, the paper later states:
- Nominal temperature
- Uniform chemistry
- Standard discharge rates

Therefore the added complexity is not actually used in the simulation.

### Problem
The manuscript appears to claim a richer battery model than is actually implemented.

### Informal Wording to Remove
Current wording:
> "so we've gotten rid of it"

This is inappropriate for journal writing.

### Recommended Replacement
> "In the present simulation, all nodes are assumed to operate under nominal isothermal conditions with homogeneous cell chemistry and standard low-rate discharge. Therefore, the chemistry, temperature, and C-rate correction terms are set to unity, reducing the cycle-aging model to the baseline DoD-dependent degradation term used in the experiments."

### Parameters That Must Be Defined
The manuscript must define:
- \( \kappa_{chem} \)
- \( E_a \)
- \( R \)
- \( T_i \)
- \( T_{ref} \)
- \( f(C\text{-rate}) \)
- \( E_{a,cal} \)
- \( \beta \)

---

## 7. Charging Model Is Physically Under-Explained

The manuscript allows nodes to recover:

\[
0.05E_{max}
\]

per step during charging states.

However, the energy source is never explained.

### Questions Left Unanswered
- Rechargeable WSN?
- Energy-harvesting WSN?
- Wireless rechargeable sensor network?
- Mobile charger?
- Solar source?

### Required Clarification
Add a statement such as:

> "We model a rechargeable WSN in which nodes can enter a low-power charging state and recover energy from an assumed local replenishment source."

If the network is not rechargeable, remove the charging model entirely.

---

## 8. Related Work Section Is Still Weak

### Current Problems
Although newer references [10]–[15] were added, the discussion still mainly focuses on references [1]–[9].

Foundational reinforcement learning references are still missing or insufficiently discussed:
- DQN
- Double DQN
- Factorized or branching action architectures
- Invalid action handling
- Battery aging / SoH modeling

### Formatting Issue
The related-work table uses:
> "Ours 2026"

Replace with:
> "Proposed"

unless publication year is finalized.

---

## 9. Figure Quality and Consistency Problems

### Figure 1 Issues
Figure 1 contains:
- "traffic load"
- "channel state"
- "Transmit Decision"

These do not align with the implemented action space:
- SLEEP
- AWAKE

### Recommended State Vector
Input state should explicitly include:
- SoC
- SoH
- Previous executed action
- Distance to sink
- \( x, y \)
- Local awake-neighbor density
- EMA activity
- Charging flag

### Recommended Pipeline
- Factorized DDQN
- Raw action \( a_t \)
- Constraint layer \( T(s_t, a_t) \)
- Executed action \( \tilde{a}_t \)
- Environment
- Reward
- Replay buffer

### Figure 2 Issue
The y-axis label is unclear.

Suggested alternatives:
- "Mean energy consumption per episode"
- "Mean SoC-weighted energy cost"

depending on the actual metric plotted.

---

## 10. Equations and Definitions Need Tightening

### CRM Denominator Issue
Coverage Redundancy Metric denominator becomes zero when no grid point is covered.

### Required Fix
Define:

\[
\Gamma(t) = 0
\]

when no grid point is covered.

Alternatively, use a small \( \epsilon \) in the denominator.

---

### Local Awake-Neighbor Density

Current normalization by \( N - 1 \) may distort values in sparse networks.

### Recommended Definition

\[
n_i^{awake} =
\frac{|\{j \in \mathcal{N}_i : \tilde{a}_j^t = 1, \neg dead(j)\}|}
{|\mathcal{N}_i| + \epsilon}
\]

where:

\[
\mathcal{N}_i = \{j \neq i : ||p_i - p_j||_2 \le r_s\}
\]

This provides a true local density.

---

### Constraint-Layer Edge Cases

The implementation must explicitly define behavior when:
- No eligible neighbor exists
- Multiple low-SoC nodes select the same neighbor
- The nearest neighbor is also close to charging threshold

Otherwise the implementation remains ambiguous.

---

# Language and Formatting Issues

Problems identified:
- Page 2:
  > "where W = H = 500mthis normalized distance..."

- Page 3:
  > "Exponentially reduce State of Health"

- Page 3:
  > "we've gotten rid of it"

- Page 4:
  > "four design choices" but five are listed

- Page 7:
  Placeholders still visible

- Page 9:
  > "Fortunately, there are certain necessary and desirable..."

- Page 10:
  Incomplete references and broken quotation formatting

These reduce reviewer confidence.

---

# Recommended Roadmap

## Priority 1 — Remove All Placeholders
Remove or fill:
- Energy analysis placeholder
- Hyperparameter sensitivity placeholder

---

## Priority 2 — Fix Evaluation Protocol

Replace final-10-training-episode reporting with deterministic held-out evaluation.

### Minimum Acceptable
- 5 seeds
- 20 test episodes per seed
- Report mean ± standard deviation

### Better Protocol
- 10 seeds
- 50–100 test episodes per seed
- Report 95% confidence intervals

---

## Priority 3 — Fix Lifetime Metric

Report lifetime in timesteps rather than training episodes.

---

## Priority 4 — Add Real Ablation Table

Minimum recommended variants:
- DQN
- DDQN-WSN
- Constraint-only
- Unconstrained DDQN
- DDQN without SoH reward
- DDQN without cooperative wake-up

---

## Priority 5 — Soften Overclaims

Do not claim strong empirical superiority over DQN.

Focus instead on:
- Factorization
- Constraint-aware execution
- Stability-motivated DDQN formulation

---

## Priority 6 — Clean Battery Section

Either:
- Simplify battery equations

or

- Fully define all parameters

Do not imply high-fidelity electrochemical modeling unless experimentally validated.
