# Occupancy and N-Mixture Model Interpretation Guide

## Occupancy Model Interpretation

### What the Occupancy Model Measures

The **Dynamic Occupancy Model** predicts **binary consumption** (presence/absence):
- **Outcome**: Whether a household consumes Wild meat on a given day (1 = yes, 0 = no)
- **Process**: Models the probability of consumption based on:
  - **Initial occupancy (ψ)**: Probability of consuming Wild meat at the first observation
  - **Colonization (γ)**: Probability of starting to consume Wild meat if not consuming before
  - **Extinction (ε)**: Probability of stopping consumption if currently consuming
  - **Persistence (φ = 1 - ε)**: Probability of continuing consumption

### Coefficient Interpretation

Coefficients are on the **logit (log-odds) scale**. They indicate how other proteins affect the probability of Wild meat consumption.

#### Key Coefficients from Your Results:

1. **fish_binary**: -1.6460 (95% HDI: [-2.2240, -1.1120])
   - **Meaning**: When a household consumes Fish on a given day, the log-odds of Wild meat consumption decreases by 1.646
   - **Effect**: Negative and significant (HDI doesn't include zero)
   - **Interpretation**: Fish consumption **reduces** the probability of Wild meat consumption

2. **domestic_binary**: -0.8040 (95% HDI: [-1.4160, -0.2200])
   - **Meaning**: When a household consumes Domestic meat, the log-odds of Wild meat consumption decreases by 0.804
   - **Effect**: Negative and significant
   - **Interpretation**: Domestic meat consumption **reduces** the probability of Wild meat consumption

3. **invertebrate_binary**: -0.5780 (95% HDI: [-1.0320, -0.0920])
   - **Meaning**: When a household consumes Invertebrates, the log-odds of Wild meat consumption decreases by 0.578
   - **Effect**: Negative and significant
   - **Interpretation**: Invertebrate consumption **reduces** the probability of Wild meat consumption

### Converting Log-Odds to Probabilities

To interpret the practical effect, you can convert log-odds to probability changes:

**Formula**: Probability = exp(logit) / (1 + exp(logit))

**Example for fish_binary (coefficient = -1.646)**:
- Without Fish consumption: Baseline probability = 0.5 (50%)
- With Fish consumption: logit decreases by 1.646
  - New logit = baseline_logit - 1.646
  - If baseline logit = 0 (p = 0.5), new logit = -1.646
  - New probability ≈ exp(-1.646) / (1 + exp(-1.646)) ≈ 0.16 (16%)
  - **Interpretation**: Fish consumption reduces Wild meat consumption probability from 50% to ~16%

### Magnitude of Effects

**Order of effect strength** (from strongest to weakest):
1. **Fish** (-1.646): Strongest negative effect
2. **Domestic meat** (-0.804): Moderate negative effect
3. **Invertebrates** (-0.578): Weakest (but still significant) negative effect

All three show **substitution effects**: consuming other proteins reduces the likelihood of Wild meat consumption.

---

## N-Mixture Model Interpretation

### What the N-Mixture Model Measures

The **N-Mixture Model** predicts **consumption mass** (how much Wild meat is consumed):
- **Outcome**: Log-transformed mass of Wild meat consumed (continuous)
- **Process**: Models the expected consumption mass based on covariates

### Coefficient Interpretation

Coefficients represent **change in log mass** of Wild meat consumption for a one-unit increase in the predictor.

#### Key Coefficients from Your Results:

1. **log_fish_mass**: -5.0940 (95% HDI: [-5.1270, -5.0600])
   - **Meaning**: A 1-unit increase in log Fish mass consumed is associated with a **decrease** of 5.094 units in log Wild meat mass
   - **Effect**: Strong negative effect (very precise - narrow HDI)
   - **Interpretation**: Higher Fish consumption is associated with **much lower** Wild meat consumption

2. **log_domestic_mass**: -3.5130 (95% HDI: [-3.5430, -3.4810])
   - **Meaning**: A 1-unit increase in log Domestic meat mass is associated with a decrease of 3.513 units in log Wild meat mass
   - **Effect**: Moderate negative effect
   - **Interpretation**: Higher Domestic meat consumption is associated with lower Wild meat consumption

3. **log_invertebrate_mass**: -2.6340 (95% HDI: [-2.6650, -2.6030])
   - **Meaning**: A 1-unit increase in log Invertebrate mass is associated with a decrease of 2.634 units in log Wild meat mass
   - **Effect**: Weakest but still significant negative effect
   - **Interpretation**: Higher Invertebrate consumption is associated with lower Wild meat consumption

### Converting Log-Scale to Actual Mass

**Example for fish_mass**:
- On log scale: coefficient = -5.094 means a 1-unit increase in log(Fish mass) decreases log(Wild meat mass) by 5.094
- On original scale: This represents a **multiplicative effect**
  - If Fish consumption doubles (log increases by ~0.693), Wild meat consumption decreases by a factor of exp(-5.094 × 0.693) ≈ exp(-3.53) ≈ 0.029
  - **Interpretation**: Doubling Fish consumption is associated with Wild meat consumption being reduced to ~3% of original

---

## Key Findings Summary

### Both Models Show Consistent Results:

1. **Substitution Effects**: All three other proteins reduce Wild meat consumption
2. **Effect Strength**: Fish > Domestic meat > Invertebrates (in both models)
3. **Statistical Significance**: All effects are significant (95% HDI excludes zero)

### Practical Interpretation:

- **Fish consumption** has the strongest negative relationship with Wild meat consumption
  - Occupancy: Reduces probability of Wild meat consumption
  - N-mixture: Associated with much lower Wild meat mass when consumed

- **Domestic meat consumption** has moderate negative relationship
  - Occupancy: Reduces probability of Wild meat consumption
  - N-mixture: Associated with lower Wild meat mass

- **Invertebrate consumption** has weakest (but still significant) negative relationship
  - Occupancy: Reduces probability of Wild meat consumption
  - N-mixture: Associated with lower Wild meat mass

### Biological/Economic Interpretation:

These results suggest:
- **Protein substitution**: Households substitute other proteins for Wild meat
- **Fish as strong substitute**: Fish consumption particularly reduces Wild meat consumption
- **Dietary complementarity is unlikely**: If proteins were complementary, coefficients would be positive

---

## How to Read the Plots

### Coefficient Comparison Plot
- **Y-axis**: Coefficient value (log-odds for occupancy, log-scale for n-mixture)
- **X-axis**: Protein type (Fish, Domestic, Invertebrate)
- **Bars**: Show mean coefficient value
- **Error bars**: 95% credible interval (HDI)
- **Zero line**: If error bar doesn't cross zero, effect is significant
- **Negative bars**: Negative effect (reduces Wild meat consumption)
- **Bar height**: Indicates effect magnitude

### Interpretation Checklist

For each coefficient:
1. ✅ Is it significant? (HDI doesn't include zero)
2. ✅ What's the direction? (Positive = increases, Negative = decreases)
3. ✅ What's the magnitude? (Larger absolute value = stronger effect)
4. ✅ What's the uncertainty? (Narrow HDI = precise estimate, Wide HDI = uncertain)

---

## Comparison: Occupancy vs N-Mixture Models

### Occupancy Model (Binary)
- **Questions**: "Does protein X affect whether households consume Wild meat?"
- **Answer**: Yes - consuming other proteins reduces the **probability** of Wild meat consumption

### N-Mixture Model (Continuous)
- **Questions**: "Does protein X affect how much Wild meat households consume?"
- **Answer**: Yes - higher consumption of other proteins is associated with **lower mass** of Wild meat consumption

### Combined Interpretation

Both models tell the same story:
- Other proteins **reduce both the likelihood AND the amount** of Wild meat consumption
- This suggests **substitution** rather than **complementation** in protein consumption patterns

