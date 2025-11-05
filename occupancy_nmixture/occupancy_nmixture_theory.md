# Occupancy and N-mixture Models: Mathematical Foundations

## 1. Occupancy Modeling

### Basic Occupancy Model

**Concept**: Occupancy models estimate the probability that a site (household) is "occupied" (consumes a protein type) while accounting for imperfect detection.

**Mathematical Framework**:

Let $z_i \in \{0,1\}$ be the true occupancy state (latent) for household $i$:
- $z_i = 1$: household consumes the protein type
- $z_i = 0$: household does not consume the protein type

The occupancy probability is modeled as:
$$\psi_i = P(z_i = 1) = \text{logit}^{-1}(X_i^T \beta)$$

where:
- $X_i$ is the covariate vector for household $i$
- $\beta$ are regression coefficients
- $\text{logit}^{-1}(x) = \frac{e^x}{1+e^x}$ is the inverse logit (sigmoid) function

**Observation Model**: 
Given that a household is occupied, we observe consumption with probability $p$ (detection probability):
$$P(y_i = 1 | z_i = 1) = p$$
$$P(y_i = 0 | z_i = 1) = 1-p$$
$$P(y_i = 0 | z_i = 0) = 1$$

**Likelihood**:
$$P(y_i) = \begin{cases}
\psi_i p + (1-\psi_i) \cdot 0 = \psi_i p & \text{if } y_i = 1 \\
\psi_i (1-p) + (1-\psi_i) \cdot 1 = 1 - \psi_i p & \text{if } y_i = 0
\end{cases}$$

For weekly data, we can simplify by assuming perfect detection ($p=1$) if consumption is recorded, leading to:
$$y_i \sim \text{Bernoulli}(\psi_i)$$

---

## 2. Dynamic Occupancy Model

**Concept**: Extends basic occupancy to multiple time periods, modeling how occupancy changes over time through colonization (start consuming) and extinction (stop consuming) probabilities.

**Mathematical Framework**:

For household $i$ at time $t$:
- **True occupancy state**: $z_{i,t} \in \{0,1\}$ (latent)
- **Colonization probability**: $\gamma_{i,t} = P(z_{i,t} = 1 | z_{i,t-1} = 0)$
- **Extinction probability**: $\epsilon_{i,t} = P(z_{i,t} = 0 | z_{i,t-1} = 1)$
- **Persistence probability**: $\phi_{i,t} = P(z_{i,t} = 1 | z_{i,t-1} = 1) = 1 - \epsilon_{i,t}$

**State Transition Model**:

First time period ($t=1$):
$$z_{i,1} \sim \text{Bernoulli}(\psi_{i,1})$$
$$\psi_{i,1} = \text{logit}^{-1}(X_{i,1}^T \beta)$$

Subsequent periods ($t > 1$):
- If $z_{i,t-1} = 0$: $z_{i,t} \sim \text{Bernoulli}(\gamma_{i,t})$
- If $z_{i,t-1} = 1$: $z_{i,t} \sim \text{Bernoulli}(\phi_{i,t})$

Or equivalently:
$$z_{i,t} | z_{i,t-1} = 0 \sim \text{Bernoulli}(\gamma_{i,t})$$
$$z_{i,t} | z_{i,t-1} = 1 \sim \text{Bernoulli}(1 - \epsilon_{i,t})$$

**Colonization and Extinction Probabilities**:
$$\gamma_{i,t} = \text{logit}^{-1}(W_{i,t}^T \alpha_\gamma)$$
$$\epsilon_{i,t} = \text{logit}^{-1}(W_{i,t}^T \alpha_\epsilon)$$

where $W_{i,t}$ are covariates (can include effects from other protein types).

**Observation Model**:
$$y_{i,t} | z_{i,t} \sim \begin{cases}
\text{Bernoulli}(p) & \text{if } z_{i,t} = 1 \\
\text{Bernoulli}(0) & \text{if } z_{i,t} = 0
\end{cases}$$

**Joint Likelihood**:
For a time series $y_{i,1:T}$:
$$P(y_{i,1:T}) = \sum_{z_{i,1:T}} P(y_{i,1:T} | z_{i,1:T}) P(z_{i,1:T})$$

where:
$$P(z_{i,1:T}) = P(z_{i,1}) \prod_{t=2}^T P(z_{i,t} | z_{i,t-1})$$

**Key Insight**: This model separates:
1. **Ecological process**: Occupancy dynamics (colonization/extinction)
2. **Observation process**: Detection given occupancy

---

## 3. N-mixture Model

**Concept**: Extends occupancy models to count/continuous data, modeling both occurrence (occupancy) and abundance (how much).

**Mathematical Framework**:

**State Process**: The true abundance $N_i$ (latent) follows:
$$N_i \sim \text{Poisson}(\lambda_i)$$

or for continuous mass data:
$$N_i \sim \text{Gamma}(\alpha_i, \beta_i) \quad \text{or} \quad \log(N_i) \sim \mathcal{N}(\mu_i, \sigma^2)$$

where:
$$\lambda_i = \exp(X_i^T \beta) \quad \text{(for Poisson)}$$
$$\mu_i = X_i^T \beta \quad \text{(for log-normal)}$$

**Observation Process**: Observed counts/mass $y_i$ given true abundance:
$$y_i | N_i \sim \begin{cases}
\text{Binomial}(N_i, p) & \text{for counts} \\
\mathcal{N}(\log(N_i + \epsilon), \sigma_{\text{obs}}^2) & \text{for continuous}
\end{cases}$$

**For Continuous Mass Data** (as in your case):

We can model the observed mass directly with a two-part model:

1. **Presence/Absence component**: 
$$z_i \sim \text{Bernoulli}(\psi_i)$$
$$\psi_i = \text{logit}^{-1}(X_i^T \beta_1)$$

2. **Abundance component** (conditional on presence):
$$\log(y_i) | z_i = 1 \sim \mathcal{N}(\mu_i, \sigma^2)$$
$$\mu_i = X_i^T \beta_2$$

**Full Likelihood**:
$$P(y_i) = \begin{cases}
(1-\psi_i) & \text{if } y_i = 0 \\
\psi_i \cdot f_{\mathcal{N}}(\log(y_i); \mu_i, \sigma^2) & \text{if } y_i > 0
\end{cases}$$

Or, more simply for continuous data where zeros are meaningful:
$$\log(y_i + \epsilon) \sim \mathcal{N}(\mu_i, \sigma^2)$$

where $\mu_i = X_i^T \beta$ includes:
- Fixed effects (covariates from your previous model)
- Effects from other protein types (actual masses)

**Key Distinction from Occupancy**:
- **Occupancy**: Binary (0/1) - "Did they consume it?"
- **N-mixture**: Count/Continuous - "How much did they consume?"

---

## 4. Model Comparison

| Aspect | Occupancy | Dynamic Occupancy | N-mixture |
|--------|-----------|-------------------|-----------|
| **Outcome** | Binary (0/1) | Binary (0/1) | Count/Continuous |
| **Time structure** | Single period | Multiple periods | Single/Multiple |
| **Key parameters** | $\psi$ (occupancy) | $\gamma$, $\epsilon$ (colonization/extinction) | $\lambda$ (abundance) |
| **State transitions** | None | Markov chain | None (or Markov for dynamics) |
| **Detection** | Yes ($p$) | Yes ($p$) | Implicit in abundance |

---

## 5. For Your Application

### Dynamic Occupancy Model for Protein Consumption

**Target**: Binary indicator of whether household $i$ consumed protein type $j$ in week $t$

**Predictors**:
- Fixed effects: season, village, sex, edu, ame, count.hunters, record.days, non.hunt.income, festivity
- **Binary indicators** of other 3 protein types (Fish, Domestic meat, Invertebrate)

**Model Structure**:
$$\psi_{i,t,1} = \text{logit}^{-1}(\beta_0 + X_{i,t}^T \beta_{\text{fixed}} + \beta_{\text{fish}} \cdot I_{\text{fish}} + \beta_{\text{domestic}} \cdot I_{\text{domestic}} + \beta_{\text{invertebrate}} \cdot I_{\text{invertebrate}})$$

$$\gamma_{i,t} = \text{logit}^{-1}(\alpha_{\gamma,0} + W_{i,t}^T \alpha_\gamma)$$
$$\epsilon_{i,t} = \text{logit}^{-1}(\alpha_{\epsilon,0} + W_{i,t}^T \alpha_\epsilon)$$

### N-mixture Model for Protein Consumption

**Target**: Actual mass consumed (continuous)

**Predictors**:
- Fixed effects: same as above
- **Actual masses** of other 3 protein types (continuous)

**Model Structure**:
$$\log(y_{i,t} + \epsilon) \sim \mathcal{N}(\mu_{i,t}, \sigma^2)$$

$$\mu_{i,t} = \beta_0 + X_{i,t}^T \beta_{\text{fixed}} + \beta_{\text{fish}} \cdot \log(\text{mass}_{\text{fish}} + \epsilon) + \beta_{\text{domestic}} \cdot \log(\text{mass}_{\text{domestic}} + \epsilon) + \beta_{\text{invertebrate}} \cdot \log(\text{mass}_{\text{invertebrate}} + \epsilon)$$

**Interpretation**:
- Positive coefficients → complements (consume together)
- Negative coefficients → substitutes (consume one instead of others)


