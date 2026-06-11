# M3 sweep — thesis checks across overlap x seed

Grid from PLAN.md M3. Check 1: debate consensus > best single agent (Aph. #10). Check 2: moderated final agreement rate > fixed-kappa (Ch. 7). Check 3: wrong-claim repeat rate decays (Aph. #11).

| overlap | seed | acc A | acc B | debate | gap | agree (mod) | agree (fixed) | wrong-repeat | C1 | C2 | C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 7 | 0.786 | 0.791 | 0.835 | +0.043 | 0.895 | 0.871 | 1.00 -> 0.62 | PASS | PASS | PASS |
| 0.3 | 13 | 0.825 | 0.813 | 0.862 | +0.037 | 0.927 | 0.864 | 1.00 -> 0.58 | PASS | PASS | PASS |
| 0.3 | 21 | 0.825 | 0.796 | 0.858 | +0.033 | 0.905 | 0.853 | 1.00 -> 0.60 | PASS | PASS | PASS |
| 0.45 | 7 | 0.674 | 0.654 | 0.730 | +0.057 | 0.866 | 0.788 | 1.00 -> 0.59 | PASS | PASS | PASS |
| 0.45 | 13 | 0.707 | 0.697 | 0.771 | +0.063 | 0.902 | 0.818 | 1.00 -> 0.45 | PASS | PASS | PASS |
| 0.45 | 21 | 0.680 | 0.692 | 0.765 | +0.073 | 0.859 | 0.783 | 1.00 -> 0.57 | PASS | PASS | PASS |
| 0.6 | 7 | 0.501 | 0.496 | 0.604 | +0.104 | 0.822 | 0.672 | 1.00 -> 0.47 | PASS | PASS | PASS |
| 0.6 | 13 | 0.572 | 0.552 | 0.647 | +0.076 | 0.881 | 0.712 | 1.00 -> 0.48 | PASS | PASS | PASS |
| 0.6 | 21 | 0.513 | 0.525 | 0.612 | +0.087 | 0.853 | 0.701 | 1.00 -> 0.49 | PASS | PASS | PASS |

**Totals:** check 1: 9/9 · check 2: 9/9 · check 3: 9/9

**Gap vs overlap** (mean debate gap per overlap level):

- overlap 0.3: mean gap +0.038 (n=3)
- overlap 0.45: mean gap +0.064 (n=3)
- overlap 0.6: mean gap +0.089 (n=3)

Expectation from PLAN.md M3: CONFIRMED — the collaboration gap grows with task overlap.
