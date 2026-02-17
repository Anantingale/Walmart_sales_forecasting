# Code Review Notes

## Scope
Reviewed the implementation in:
- `Part_B.ipynb` (Walmart forecasting workflow)
- `Part_A_(2).ipynb` (consumer store function)

## Findings

### 1) Incorrect linear-regression cost formula (high)
In `Part_B.ipynb`, the cost is computed as:

```python
total_cost = (1/2*m) * cost_sum
```

Because of operator precedence, this is interpreted as `(m/2) * cost_sum`, not `(1/(2*m)) * cost_sum`.
This scales the loss by `m^2` relative to the standard MSE-style objective and makes logged cost values misleading during optimization.

**Recommendation:**
Use `total_cost = (1 / (2 * m)) * cost_sum`.

---

### 2) Test-set leakage/overwriting between models (high)
`Part_B.ipynb` reuses variable names (`x_train, x_test, y_train, y_test`) for linear regression, then overwrites `y_test` during Random Forest train/test split (`X_train, X_test, y_train, y_test`).
Later, linear regression metrics are calculated against the *overwritten* `y_test`, which no longer corresponds to `x_test`.

This can produce invalid or shape-mismatched evaluation for the linear model.

**Recommendation:**
Use separate names per model, e.g. `y_test_lr` and `y_test_rf`, and compute each model’s metrics against its own held-out labels.

---

### 3) Inconsistent variable naming can raise `UnboundLocalError` (medium)
In `Part_A_(2).ipynb`, `top_product_name` is initialized but `top_product` is assigned and returned.
If no item exceeds `highest_contribution` (e.g., empty input), `top_product` is never defined before return.

**Recommendation:**
Use one variable consistently (`top_product_name`) and return it. Also handle empty input explicitly.

