# ConversationGuardian Target Boundary Examples — Gate 1

This document specifies the exact boundary evaluation behavior across all eight canonical cases using both formal notation and synthetic conversation scenarios.

---

## Boundary Decision Matrix

| Scenario # | Filtered Dialogue Length ($L$) | Attack Position ($p_{\text{att}}$) | Checkpoint ($n_{\text{obs}}$) | Window Range | Eligibility Status | Reason Code | Target Label | Rule Rationale |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---|
| **Case 1** | $6$ | None | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `NEGATIVE_EVENT` | `0` | All 3 future messages observed, 0 attacks. |
| **Case 2** | $5$ | None | $3$ | $[3, 4, 5]$ | `EXCLUDED` | `CENSORED_INCOMPLETE_WINDOW` | `None` | Only 2 future messages observed ($L=5 < 6$); peaceful termination is censored. |
| **Case 3** | $4$ | None | $5$ | $[5, 6, 7]$ | `EXCLUDED` | `INSUFFICIENT_OBSERVED_PREFIX` | `None` | Thread length ($4$) is shorter than the observation checkpoint ($5$). |
| **Case 4** | $6$ | $1$ | $3$ | $[3, 4, 5]$ | `EXCLUDED` | `ATTACK_ALREADY_OBSERVED` | `None` | Attack occurred at position 1, inside observed input $[0, 1, 2]$. |
| **Case 5** | $6$ | $3$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs exactly at the first future message ($p_{\text{att}} = n_{\text{obs}}$). |
| **Case 6** | $7$ | $5$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs at the boundary position of the 3-message window ($p_{\text{att}} = n_{\text{obs}} + 2$). |
| **Case 7** | $8$ | $6$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `NEGATIVE_EVENT` | `0` | Attack occurs outside window ($p_{\text{att}} = 6 > 5$); window $[3, 4, 5]$ is complete and clean. |
| **Case 8** | $4$ | $3$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs at position 3, followed by immediate thread termination. Positive event holds despite incomplete window. |

---

## Detailed Case Walkthroughs

### Case 1: Fully Observed Negative Event
- **Setup:** A conversation between two editors discussing citations spans 6 turns ($0, 1, 2, 3, 4, 5$). No personal attacks occur anywhere.
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Observed input: comments $0, 1, 2$.
  - Forecast window: comments $3, 4, 5$.
  - Future messages available: $6 - 3 = 3$ (positions $3, 4, 5$).
  - Attacks in window: $0$.
  - Result: **ELIGIBLE**, `NEGATIVE_EVENT`, label = **0**.

### Case 2: Right-Censored Peaceful Dialogue
- **Setup:** A short conversation concludes politely in 5 turns ($0, 1, 2, 3, 4$). No personal attacks occur.
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Observed input: comments $0, 1, 2$.
  - Forecast window requires 3 future turns ($3, 4, 5$).
  - Future messages available: only $2$ turns ($3, 4$).
  - Attacks in window: $0$.
  - Result: **EXCLUDED**, `CENSORED_INCOMPLETE_WINDOW`, label = **None**.
  - *Academic Rationale:* Calling this an ordinary negative would artificially inflate model negative specificity by assuming a hypothetical 6th turn would never derail.

### Case 3: Insufficient Observation Prefix
- **Setup:** A conversation consists of 4 turns ($0, 1, 2, 3$).
- **Evaluation at $n_{\text{obs}} = 5$:**
  - Observation requirement: 5 turns ($0, 1, 2, 3, 4$).
  - Available turns: 4.
  - Result: **EXCLUDED**, `INSUFFICIENT_OBSERVED_PREFIX`, label = **None**.

### Case 4: Attack Already Observed (Post-Derailment)
- **Setup:** An attack is launched early at turn 1. The thread continues for 5 more turns ($L = 6$).
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Observed input: turns $0, 1, 2$. Turn 1 is an attack.
  - Result: **EXCLUDED**, `ATTACK_ALREADY_OBSERVED`, label = **None**.
  - *Academic Rationale:* Forecasting whether an attack will occur when an attack has already occurred in plain view is a trivial detection task, not an early warning.

### Case 5: Immediate Next-Turn Derailment
- **Setup:** Turns $0, 1, 2$ are civil. At turn 3, an editor insults another contributor. The thread terminates at turn 6.
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Observed input: turns $0, 1, 2$.
  - Forecast window: turns $3, 4, 5$. Turn 3 contains an annotated attack.
  - Result: **ELIGIBLE**, `POSITIVE_EVENT`, label = **1**.

### Case 6: Window-Boundary Derailment
- **Setup:** Turns $0, 1, 2$ are civil. Turns $3$ and $4$ show escalating tension without personal attacks. Turn $5$ erupts into a personal attack.
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Forecast window: turns $3, 4, 5$.
  - Turn 5 is inside the 3-turn horizon ($p_{\text{att}} = 3 + 3 - 1 = 5$).
  - Result: **ELIGIBLE**, `POSITIVE_EVENT`, label = **1**.

### Case 7: Delayed Derailment Outside Window
- **Setup:** An 8-turn conversation ($0 \dots 7$). Turns $0 \dots 5$ are free of attacks. At turn 6, an attack occurs.
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Forecast window: turns $3, 4, 5$.
  - Turns $3, 4, 5$ are fully observed and contain 0 attacks.
  - Attack at turn 6 is outside the $H=3$ window.
  - Result: **ELIGIBLE**, `NEGATIVE_EVENT`, label = **0**.
  - *Academic Rationale:* The model is tasked with predicting whether an attack happens *within the next 3 turns*. Since it does not occur within those 3 turns, the negative forecast is correct for that horizon.

### Case 8: Positive Event with Premature Termination
- **Setup:** Turns $0, 1, 2$ are observed. Turn 3 is a vitriolic attack, and the thread is immediately locked or abandoned ($L = 4$).
- **Evaluation at $n_{\text{obs}} = 3$:**
  - Forecast window: turn 3 is observed and positive.
  - Even though turns 4 and 5 do not exist, a positive event has already manifested within the window.
  - Result: **ELIGIBLE**, `POSITIVE_EVENT`, label = **1**.
