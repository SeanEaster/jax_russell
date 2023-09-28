# Usage

To use jax_russell in a project

```
import jax_russell
```

For example, this computes the value of an example option given in _The Complete Guide to Option Pricing Formulas_:

```
from jax import numpy as jnp
import jax_russell as jr
haug_start_price = jnp.array([100.0])
haug_volatility = jnp.array([0.3])
haug_time_to_expiration = jnp.array([0.5])
haug_is_call = jnp.array([0.0])
haug_strike = jnp.array([95.0])
haug_risk_free_rate = jnp.array([0.08])
haug_cost_of_carry = jnp.array([0.08])
haug_inputs = (
    haug_start_price,
    haug_volatility,
    haug_time_to_expiration,
    haug_risk_free_rate,
    haug_is_call,
    haug_strike,
)
tree = jr.StockOptionCRRTree(5, "american")
print(tree(*haug_inputs))
```


Run this, and you should see `Array([4.9192114], dtype=float32)`. Try calling `tree.first_order(*haug_inputs)` to see the example's first-order greeks:
```
>>> tree.first_order(*haug_inputs)
Array([[ -0.37834674,  26.823406  ,   5.7209864 , -14.537726  ]], dtype=float32)
```

These values correspond to the order of the inputs, i.e. respectfully delta, vega or kappa, theta and rho.
Vega, rho report the change resulting from a full-unit change in volatility, interest rate; divide these by 100 and they'll report the value per 1% change.
Likewise, the returned theta value is reported per year. Typically, theta is reported per trading day, so the value given by `first_order()` will require adjustment to match what you see in trading platforms.

In the example above, five is an impractically small number of steps.
As of this writing (Sept. 2023), [Interactive Brokers reports implied volatility using a 100-step tree](https://www.interactivebrokers.com/en/general/education/pdfnotes/PDF-OIR.php).
Because `__call__()`, `first_order()` and `second_order()` are just-in-time compiled, steps values of this magnitude can make for a long first call to these methods. 
Subsequent calls are speedy.