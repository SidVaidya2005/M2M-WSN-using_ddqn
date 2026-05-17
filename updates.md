# Methodology Revision Notes

---

## 1. Stop describing the action as a full joint 2^50 action solved by standard DDQN

This is the most important fix.

Right now the paper says the action is a 50-bit binary vector, but the network outputs per-node binary Q-values and averages them. That is not standard DDQN over the full joint action space. Standard DQN and Double DQN are defined for finite discrete action sets, and large multi-discrete problems usually need explicit factorization or branching.

### Minimum Fix

Do not change the network much. Just reframe the method as one of these:

- Factorized DDQN
- Branching DDQN-style per-node binary decision model
- Shared DDQN with per-node action branches

### What to Write

Instead of:

> "The action space is a binary vector $\mathbf{a}_t \in \{0,1\}^N$, and DDQN learns the optimal joint action."

Write:

> "To avoid the combinatorial intractability of the full $2^N$ joint action space, we adopt a factorized per-node action formulation with shared parameters across nodes. The network outputs two Q-values per node, corresponding to SLEEP and AWAKE, and the final scheduling vector is formed by combining the per-node greedy decisions. Thus, the proposed method is a factorized DDQN scheduler rather than a flat-action DDQN over the full joint action space."

That one change removes a major theoretical objection.

---

## 2. Add executed-action notation because your environment overrides the policy

Your current environment forces low-SoC nodes into charging/sleep and may force neighbors awake through cooperative wake-up. That means the executed action is not always the same as the agent-selected action. If you do not state this clearly, reviewers can say the environment is doing the scheduling, not the agent.

### Minimum Fix

Keep the charging and cooperative rule if you want. But define:

- $\mathbf{a}_t$: raw action proposed by the agent
- $\tilde{\mathbf{a}}_t$: executed action after environment feasibility rules

And say clearly that replay stores $\tilde{\mathbf{a}}_t$, not $\mathbf{a}_t$.

### Add This to Methodology

> **Executed action.** Let $\mathbf{a}_t \in \{0,1\}^N$ denote the action proposed by the DDQN agent. Because charging and cooperative wake-up constraints may modify infeasible or unsafe decisions, the environment applies a deterministic transformation $\tilde{\mathbf{a}}_t = \mathcal{T}(s_t, \mathbf{a}_t)$. The transition and reward are generated using $\tilde{\mathbf{a}}_t$, and the replay buffer stores $(s_t, \tilde{\mathbf{a}}_t, r_t, s_{t+1}, d_t)$.

This is a small change in writing, but it fixes a serious methodological gap. Invalid-action handling is a known issue in RL, so this correction makes your paper much safer.

---

## 3. Replace "override" language with "constraint-aware scheduling"

This is mostly presentation, but it matters. Right now "override" sounds like the environment is correcting the agent after the fact. That weakens the RL contribution.

### Minimum Fix

Describe charging and cooperative wake-up as:

- safety/feasibility constraints
- domain-informed transition rules
- constraint-aware post-processing

### Better Wording

Instead of:

> "The raw action may be overridden by two environment-level mechanisms."

Write:

> "The raw per-node decisions are passed through a constraint-aware execution layer that enforces charging feasibility and coverage continuity rules before the physical state transition is applied."

This sounds much more rigorous and much less like hidden hand-tuning.

---

## 4. Slightly improve the state representation, but do not redesign everything

Your current six features are:

1. SoC
2. SoH
3. previous action
4. distance to sink
5. activity ratio
6. charging flag

That is too weak for a coverage-driven problem, because coverage depends on spatial relations, not only sink distance. An RL task should observe enough information to support the transition and reward structure.

### Minimum Fix

Do not redesign into a GNN if you want minimal change. Just add 2 to 3 extra features per node:

- normalized $x_i$
- normalized $y_i$
- local neighbor count or local coverage redundancy score

That is enough to show reviewers that the agent sees actual spatial information.

### Best Minimal Version

Add:

- $x_i / W$
- $y_i / H$
- $n_i^{\text{awake}}$: number of awake neighbors within sensing range, normalized

Then change the paper from **6-feature state** to **9-feature state**.

### Why This Helps

It fixes the reviewer question:

> "How can the agent optimize area coverage without seeing geometry?"

And it does that without changing the whole architecture.

---

## 5. Downgrade the battery-model claim unless you calibrate it

Your paper currently presents the battery model as *physics-grounded* or *physics-accurate*, but the actual model is still a surrogate:

- simple SoC drain
- power-law cycle degradation
- constant calendar decay

Battery aging in the literature depends on temperature, C-rate, SOC window, and chemistry. Public battery-aging studies and NREL reduced-order life models are much richer than this.

### Minimum Fix

Do not change the equations yet. Just change the wording from:

- "physics-accurate battery model"
- "physics-grounded battery model"

to:

- surrogate battery degradation model
- empirical SoC/SoH-aware battery model

That one wording correction will save you from a lot of reviewer pushback.

---

## If You Change Only One Equation Block, Change This Part

You should rewrite the methodology logic like this:

### Current Core Problem

The paper sounds like:

- full joint action
- standard DDQN
- environment overrides actions

### Minimum Corrected Formulation

It should sound like:

- factorized per-node binary scheduling
- shared DDQN backbone
- constraint-aware executed action
- centralized observation
- empirical SoC/SoH-aware transition model

That is enough to make the method technically coherent without rewriting the whole algorithm.

---

## The Smallest Defensible Revised Methodology Structure

You can keep your Methodology section almost the same, but change the subsection titles and wording to this:

**IV. Proposed Methodology: Constraint-Aware Factorized DDQN for WSN Scheduling**

- **A. Centralized state representation** — Keep current text, but add node coordinates and local neighborhood activity.
- **B. Factorized multi-binary action formulation** — Say the full joint action is combinatorial, so you use a parameter-shared per-node decision model.
- **C. Constraint-aware action execution** — Introduce executed action $\tilde{\mathbf{a}}_t$ and explain charging and cooperative continuity rules.
- **D. Reward design** — Keep mostly as is.
- **E. DDQN learning architecture** — Keep the same MLP and training recipe, but call it factorized/shared DDQN, not plain joint-action DDQN.

---

## What You Should NOT Do If You Want Minimal Changes

Do not do these if your goal is minimal revision:

- do not move to PPO now
- do not move to MARL now
- do not change the whole network into GNN now
- do not add a very complex battery electrochemistry model now

Those would be stronger long-term improvements, but they are not minimum changes.

---

## My Honest "Minimum Acceptable" Methodology Package

If you want the smallest set of methodology changes that still improves acceptance potential, do these four things:

1. Reframe DDQN as factorized/branching per-node DDQN, not flat joint-action DDQN.
2. Introduce executed-action notation and state that replay stores executed action after constraints.
3. Add node coordinates plus one local coverage feature to the state.
4. Rename the battery model as empirical/surrogate, not physics-accurate.

That is the minimum I would accept before submission.

---

## Very Important Warning

Even after these minimum methodology fixes, high acceptance is still unlikely unless you also improve:

- multi-seed evaluation
- stronger baselines
- statistical reporting
- removal of placeholders
- honest claim wording

So the right way to think about this is:

- **minimum methodology fix** = makes the paper defensible
- **minimum full-paper fix** = methodology fix + stronger experiments + rewritten claims

---

## My Practical Recommendation

If you want the best tradeoff between small change and higher acceptance chance, do this:

Keep DDQN, keep the network, keep the reward, keep the environment. Only change:

- the method description
- the action semantics
- the state features slightly
- the battery-model wording
