# Empirical Test: A Real Economic Network and an Out-of-Sample Prediction Benchmark

The next step is to turn the proposed graph/agent framework into an empirical model that can fail. The objective is not to demonstrate that an economic network can be represented as a graph—the existing literature already establishes that. The objective is to determine whether the additional structure proposed here—dynamic topology, heterogeneous agents, asymmetric flows, network position, partial observability, and non-equilibrium dynamics—produces measurably better out-of-sample predictions than conventional econometric and network models.

The cleanest first test is a global production network constructed from the OECD Inter-Country Input-Output (ICIO) tables. The OECD ICIO database explicitly maps production, consumption, investment, and international intermediate-goods and services flows by country and economic activity. The current 2025 edition covers 80 economies plus a rest-of-world aggregate, 50 industries, and 1995–2022. This makes it unusually well suited to the proposed framework: the nodes can be interpreted as country-industry agents, while intermediate-input flows become directed weighted transaction edges.

The empirical question will be:

> **Does the dynamic graph/agent model predict subsequent country-industry output changes, particularly large systemic contractions, better out of sample than conventional macroeconometric and network-production baselines?**

The first high-value test will use the COVID-19 production shock as a genuinely held-out episode. The model will be trained without observations from 2020 and will generate predictions using information available before the test period. The 2020 realization will then be evaluated without refitting the model.

## 1. Constructing the economic network

Let

[
V_t={(c,s)},
]

where (c) indexes countries and (s) indexes industries. Each node therefore represents one country-industry production unit.

For every year (t), define a directed weighted graph

[
G_t=(V,E_t,W_t).
]

An edge

[
i=(c,s)\rightarrow j=(c',s')
]

represents an intermediate production flow from industry (s) in country (c) to industry (s') in country (c'). Its weight is the monetary value of the intermediate input.

This is a direct empirical instantiation of the proposed transaction subgraph (G_T): the framework defines transactions as material interactions with corresponding counterflows, and defines the economic graph as a time-indexed directed hypergraph of interacting agents and objects.

For the first implementation we should deliberately **not** use the full hypergraph. We can start with its weighted directed graph projection. That gives us a falsifiable baseline implementation rather than an enormous model in which every theoretical component is simultaneously estimated.

For each node (i), the observed state vector will contain:

[
S_{i,t} =
[
Y_{i,t},
VA_{i,t},
L_{i,t},
K_{i,t},
X_{i,t},
M_{i,t},
C_{i,t},
d^{in}*{i,t},
d^{out}*{i,t},
B_{i,t},
E_{i,t}
],
]

where (Y) is gross output, (VA) value added, (L) labor input, (K) capital or capital-related measures where available, (X) exports, (M) imports, (C) consumption/final-demand exposure, and the remaining variables describe the node's graph position and network exposure.

The OECD's harmonized input-output data provide industrial production relationships, while the ICIO framework provides the international production network.

The graph should be normalized in at least two ways:

1. **Raw-value graph:** edge weights are monetary intermediate flows.
2. **Input-share graph:** each supplier's flow is normalized by the recipient industry's total intermediate expenditure.

The second representation is particularly important because it prevents large economies from dominating the graph solely because of their scale.

## 2. Turning the proposed agent model into an estimable model

The proposed framework defines an agent as a state vector, capabilities, and an autonomous policy:

[
Agent_i(t)=
\langle S_i(t),C_i(t),\pi_i\rangle.
]

It also explicitly models the economic system as a dynamic directed hypergraph and treats macroeconomic observables as projections of that underlying graph.
For this empirical exercise, we operationalize that architecture as a **dynamic graph transition model**:

[
h_{i,t+1}
=========

F_\theta
\left(
h_{i,t},
\sum_{j\in N(i)}
\phi_\theta(h_{j,t},h_{i,t},w_{ji,t}),
z_t
\right),
]

where:

* (h_{i,t}) is the latent state of country-industry agent (i);
* (w_{ji,t}) is the observed economic-flow weight;
* (\phi_\theta) is a learned relational message-passing function;
* (z_t) contains common macroeconomic conditions;
* (F_\theta) produces the next-period node state.

The prediction target will initially be one-year-ahead real output growth:

[
\hat g_{i,t+1}
==============

f_\theta(G_t,S_t,z_t)_i.
]

A second target will be a tail-risk indicator:

[
D_{i,t+1}
=========

1[g_{i,t+1}<q_{0.10}],
]

where (q_{0.10}) is the training-sample tenth percentile of node-level output growth.

This gives us two tests:

* **continuous forecasting:** how accurately does the model predict output growth?
* **tail-event forecasting:** how accurately does it identify unusually severe contractions?

The latter is particularly important because the theoretical framework claims that network structure generates non-equilibrium tail behavior rather than merely explaining average fluctuations.

## 3. What is genuinely new in the proposed model?

We should not call a generic GNN a test of the theory. A GNN could outperform a regression simply because it has more flexible function approximation.

The proposed model therefore needs identifiable structural components.

The first component is **network position**. The framework explicitly treats degree, betweenness, and eigenvector centrality as economically meaningful structural variables.

The second is **asymmetric flow propagation**. The model explicitly proposes that wages, rents, interest, and other flows need not be symmetric and that preferential attachment can reinforce high-degree hubs.

The third is **partial observability**. The theoretical model distinguishes the observed graph (\hat G_t) from the underlying graph (G_t^*), with latent and shadow components.

The fourth is **heterogeneous agent response**. The proposed POMDP formulation allows agents to possess different policies, including adaptive, bounded-rational, imitative, rule-based, leader-directed, and noise-trader strategies.

The empirical model therefore should have two versions:

### Model A — Graph-only model

[
\hat y_{t+1}=f_\theta(G_t,X_t).
]

This tests whether network information itself adds predictive power.

### Model B — Graph/agent model

[
\hat y_{t+1}
============

f_\theta(G_t,X_t,B_t,\Pi_t),
]

where (B_t) represents inferred node beliefs/state uncertainty and (\Pi_t) represents heterogeneous response policies.

The difference between A and B is critical. If Model B does not outperform Model A, the additional agent machinery has not earned its complexity.

## 4. Conventional econometric benchmarks

The comparison must include models that economists would regard as serious alternatives, not weak straw men.

### Benchmark 1: Autoregressive panel model

[
g_{i,t+1}
=========

\alpha_i+\delta_t+
\rho g_{i,t}
+
\beta'X_{i,t}
+
\epsilon_{i,t+1}.
]

Country-industry fixed effects and year effects control for persistent heterogeneity and global shocks.

### Benchmark 2: Dynamic factor model

A small number of latent global and regional factors will be estimated from the historical output-growth panel. Each node's forecast is then generated from its factor exposures and own history.

This provides a strong conventional macroeconomic benchmark because it allows substantial cross-sectional dependence without explicitly using the production graph.

### Benchmark 3: VAR / panel-VAR

A lower-dimensional aggregate version will model country or industry output growth jointly using lagged macroeconomic variables.

This tests whether the proposed graph representation actually adds information beyond conventional dynamic macro relationships.

## 5. Network benchmarks

We also need to defeat simpler network explanations.

### Benchmark 4: Leontief network model

The standard input-output multiplier is

[
x_t=(I-A_t)^{-1}f_t,
]

where (A_t) is the input coefficient matrix and (f_t) final demand.

This is arguably the most important benchmark because it uses exactly the same production network but does not introduce learned nonlinear dynamics or autonomous agents.

The OECD itself provides Leontief inverse matrices as part of its harmonized input-output database.

### Benchmark 5: Centrality regression

A deliberately simple network model will use:

[
g_{i,t+1}
=========

\alpha_i+\delta_t+
\beta_1 d_i^{in}
+\beta_2d_i^{out}
+\beta_3 C_i^{between}
+\beta_4 C_i^{eigen}
+\gamma'X_{i,t}
+\epsilon_{i,t+1}.
]

If this performs as well as the full graph/agent model, the elaborate architecture is unnecessary.

### Benchmark 6: Network autoregression

Finally:

[
g_{t+1}
=======

\alpha+\rho g_t+\lambda W_tg_t+\beta X_t+\epsilon_{t+1}.
]

This tests whether simple spatial/network spillovers explain the phenomenon without nonlinear message passing.

## 6. The out-of-sample experiment

The primary experiment will use a strict temporal split.

Training:

[
1995\text{--}2017
]

Validation:

[
2018\text{--}2019
]

Held-out test:

[
2020
]

Post-test robustness:

[
2021\text{--}2022.
]

The model will never see 2020 observations during parameter estimation or hyperparameter selection.

This matters because the central claim is predictive rather than descriptive. A model that can reproduce the historical graph after observing the outcome has demonstrated almost nothing about forecasting.

The 2020 episode is particularly useful because the production network experienced a large, geographically heterogeneous disruption. OECD supply-use data explicitly document the use of input-output relationships to examine supply-chain shocks during the COVID-19 period.

The primary prediction is therefore not simply:

> “Did GDP fall in 2020?”

That would be trivial.

Instead, the question is:

> **Given the network observed before the shock, which country-industries should experience the largest contractions, and how accurately can the model rank them?**

This transforms the experiment into a cross-sectional test of network propagation.

## 7. A stronger conditional experiment

The most convincing version conditions on the magnitude of the aggregate shock.

Let

[
\Delta Y_{i,2020}
]

denote the realized output change.

We predict

[
E[\Delta Y_{i,2020}\mid G_{2019},X_{2019},Z_{2020}],
]

where (Z_{2020}) contains common shock information available at the forecasting date.

The model is therefore not asked to predict that an unprecedented pandemic will occur. Instead, every model receives the same aggregate shock information and must determine how that shock propagates through the economic network.

The relevant scientific test becomes:

[
\text{Does network structure improve the allocation of the shock across nodes?}
]

This is much harder to dismiss as either clairvoyance or hindsight.

## 8. Evaluation metrics

We will report at least five metrics.

### Continuous prediction

[
RMSE=
\sqrt{
\frac{1}{N}
\sum_i
(\hat g_i-g_i)^2
}
]

and

[
MAE=
\frac{1}{N}
\sum_i|\hat g_i-g_i|.
]

### Ranking

Because systemic-risk applications often care more about identifying vulnerable nodes than predicting an exact percentage, we will report Spearman rank correlation between predicted and realized contractions.

### Tail classification

For the severe-contraction indicator (D_i), report:

* AUROC;
* AUPRC;
* precision among the top 10% predicted-risk nodes;
* recall among the actual bottom 10%.

### Calibration

Predicted probabilities of severe contraction will be compared with realized frequencies.

### Economic value

Finally, construct a simple intervention rule:

[
I_i=1
\quad\text{if}\quad
P(D_i=1)>c.
]

The cost of missed severe contractions and unnecessary interventions can then be varied to determine whether the model produces economically useful rankings rather than merely statistically significant improvements.

## 9. The decisive comparison

The central table should ultimately look like this:

| Model                  |  RMSE |   MAE | Spearman ρ | AUROC | AUPRC | Top-10% recall |
| ---------------------- | ----: | ----: | ---------: | ----: | ----: | -------------: |
| Panel AR               |     — |     — |          — |     — |     — |              — |
| Dynamic factor         |     — |     — |          — |     — |     — |              — |
| Panel-VAR              |     — |     — |          — |     — |     — |              — |
| Leontief               |     — |     — |          — |     — |     — |              — |
| Centrality regression  |     — |     — |          — |     — |     — |              — |
| Network autoregression |     — |     — |          — |     — |     — |              — |
| Graph-only model       |     — |     — |          — |     — |     — |              — |
| **Graph/agent model**  | **—** | **—** |      **—** | **—** | **—** |          **—** |

No conclusion about superiority should be made until this table exists.

## 10. Ablation tests

The proposed theory makes several claims that can be independently tested.

We will therefore remove one component at a time:

1. **No graph:** node-level temporal model only.
2. **No edge weights:** topology only.
3. **No direction:** symmetrized network.
4. **No centrality/state variables:** raw message passing only.
5. **No heterogeneity:** shared agent policy.
6. **No temporal state:** static graph.
7. **No nonlinear propagation:** linear network diffusion.
8. **No partial observability:** complete-information model.

The theoretical framework explicitly proposes graph diffusion, multiplicative returns, asymmetric flow drift, and preferential-attachment dynamics as mechanisms that generate departures from simple diffusion.

If removing these components does not materially reduce predictive performance, they are not empirically necessary.

That result would be scientifically useful even if the full model wins: it would tell us which pieces of the proposed theory actually matter.

## 11. What would count as success?

I would define success before running the experiment.

The graph/agent model passes the primary test if, on the untouched 2020 test set:

[
RMSE_{GA}<RMSE_{best}
]

and simultaneously improves tail-risk discrimination, with the improvement surviving multiple temporal splits and ablations.

A particularly strong result would be:

[
\text{Graph/Agent}

>

\text{Graph-only}

>

\text{Network baseline}

>

\text{Econometric baseline}.
]

But a more modest result could still validate part of the theory:

[
\text{Graph-only}

>

\text{Econometric baseline},
]

while

[
\text{Graph/Agent}
\approx
\text{Graph-only}.
]

That would support the claim that network topology matters while providing little evidence that the proposed agent/POMDP machinery adds predictive value.

Conversely, if

[
\text{Graph/Agent}
\leq
\text{Leontief/network baselines},
]

then the central empirical claim fails, regardless of how compelling the theoretical construction appears.

This is exactly the kind of falsification test the original framework currently lacks. The document already proposes prediction-observation divergence,

[
\Delta(t)=|Y_{\mathrm{predicted}}(t)-Y_{\mathrm{measured}}(t)|,
]

and says persistent divergence should trigger revision of the model, measurement assumptions, or structural regime. The empirical experiment above turns that principle into a conventional out-of-sample forecasting protocol.

## 12. Why this network is the right first experiment

The OECD ICIO network has three advantages.

First, it is genuinely economic rather than a synthetic simulation. It contains observed production and intermediate-flow relationships across countries and industries.

Second, it naturally maps onto the proposed transaction graph: intermediate production flows are directed material interactions, and node-level output and value added are observable state variables.

Third, it has enough temporal depth to permit genuine rolling out-of-sample evaluation. The current ICIO release covers 1995–2022, while the OECD's previous-year-price ICIO extends the real-production-network framework through 2024, although the most recent years include projections.

Most importantly, this experiment connects the proposed framework to an established empirical literature rather than treating it as an isolated theory. Existing work has already shown that input-output linkages can generate substantial macroeconomic tail risk from heterogeneous microeconomic shocks. The proposed contribution therefore has to be stronger than “networks matter.” It must demonstrate that its particular dynamic graph/agent representation extracts predictive information that those existing network models leave unused.

The empirical program should therefore begin with a deliberately constrained claim:

> **A dynamic country-industry production graph contains information about the cross-sectional propagation of aggregate economic shocks, and the proposed graph/agent model extracts that information better out of sample than established econometric, input-output, and network baselines.**

Everything else in the larger world model should be treated as a subsequent hypothesis, not as an established consequence of the formalism.
