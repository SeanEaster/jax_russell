"""Black-Scholes-Merton classes."""

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
import typeguard
from jax.scipy.stats.norm import cdf

from jax_russell.base import ValuationModel


class GeneralizedBlackScholesMerten(ValuationModel):
    """Generalized Black-Scholes-Merten for European options."""

    @partial(jax.jit, static_argnums=0)
    @typeguard.typechecked
    def value(
        self,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "*#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        """Calculate values for option contracts.

        Returns:
            jnp.array: contract values
        """
        sign = 2.0 * is_call - 1.0

        d1 = (
            jnp.log(start_price / strike)
            + (cost_of_carry + jnp.power(volatility, jnp.array(2.0)) / 2.0) * time_to_expiration
        ) / (volatility * jnp.power(time_to_expiration, jnp.array(0.5)))

        d2 = d1 - volatility * jnp.power(time_to_expiration, jnp.array(0.5))

        return sign * start_price * jnp.exp((cost_of_carry - risk_free_rate) * time_to_expiration) * cdf(
            sign * d1
        ) - sign * strike * jnp.exp(-risk_free_rate * time_to_expiration) * cdf(sign * d2)
