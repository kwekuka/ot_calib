import jax
import jax.numpy as jnp
from typing import Literal
from ott.geometry import costs
from ott.solvers.nn import neuraldual

class L2Dual(neuraldual.W2NeuralDual):
    def __init__(self, **kwargs):
        super().__init__(
        dim_data=kwargs['dim_data'],
        neural_f=kwargs['neural_f'],
        neural_g=kwargs['neural_g'],
        optimizer_f=kwargs['optimizer_f'],
        optimizer_g=kwargs['optimizer_g'],
        num_train_iters=kwargs['num_train_iters']
    )

    def get_step_fn(
            self, train: bool, to_optimize: Literal["f", "g", "parallel", "both"]
    ):
        """Create a parallel training and evaluation function."""

        def loss_fn(params_f, params_g, f_value, g_value, g_gradient, batch):
            """Loss function for both potentials."""
            # get two distributions
            source, target = batch["source"], batch["target"]

            init_source_hat = g_gradient(params_g)(target)

            def g_value_partial(y: jnp.ndarray) -> jnp.ndarray:
                """Lazy way of evaluating g if f's computation needs it."""
                return g_value(params_g)(y)

            def f_grad_value(x: jnp.ndarray) -> jnp.ndarray:
                return self.neural_f.potential_gradient_fn(params_f)(x)

            f_value_partial = f_value(params_f, g_value_partial)
            if self.conjugate_solver is not None:
                finetune_source_hat = lambda y, x_init: self.conjugate_solver.solve(
                    f_value_partial, y, x_init=x_init
                ).grad
                finetune_source_hat = jax.vmap(finetune_source_hat)
                source_hat_detach = jax.lax.stop_gradient(
                    finetune_source_hat(target, init_source_hat)
                )
            else:
                source_hat_detach = init_source_hat

            batch_dot = jax.vmap(jnp.dot)

            f_source = f_value_partial(source)
            f_star_target = batch_dot(source_hat_detach,
                                      target) - f_value_partial(source_hat_detach)
            dual_source = f_source.mean()
            dual_target = f_star_target.mean()

            f_grad = f_grad_value(source)
            l2_loss = costs.PNormP(p=2).pairwise(f_grad, target)
            dual_loss = dual_source + dual_target + l2_loss

            if self.amortization_loss == "regression":
                amor_loss = ((init_source_hat - source_hat_detach) ** 2).mean()
            elif self.amortization_loss == "objective":
                f_value_parameters_detached = f_value(
                    jax.lax.stop_gradient(params_f), g_value_partial
                )
                amor_loss = (
                        f_value_parameters_detached(init_source_hat) -
                        batch_dot(init_source_hat, target)
                ).mean()
            else:
                raise ValueError("Amortization loss has been misspecified.")

            if to_optimize == "both":
                loss = dual_loss + amor_loss + l2_loss
            elif to_optimize == "f":
                loss = dual_loss + l2_loss
            elif to_optimize == "g":
                loss = amor_loss
            else:
                raise ValueError(
                    f"Optimization target {to_optimize} has been misspecified."
                )

            if self.pos_weights:
                # Penalize the weights of both networks, even though one
                # of them will be exactly clipped.
                # Having both here is necessary in case this is being called with
                # the potentials reversed with the back_and_forth.
                loss += self.beta * self._penalize_weights_icnn(params_f) + \
                        self.beta * self._penalize_weights_icnn(params_g)

            # compute Wasserstein-2 distance
            C = jnp.mean(jnp.sum(source ** 2, axis=-1)) + \
                jnp.mean(jnp.sum(target ** 2, axis=-1))
            W2_dist = C - 2. * (f_source.mean() + f_star_target.mean())

            return loss, (dual_loss, amor_loss, W2_dist)

        @jax.jit
        def step_fn(state_f, state_g, batch):
            """Step function of either training or validation."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
            if train:
                # compute loss and gradients
                (loss, (loss_f, loss_g, W2_dist)), (grads_f, grads_g) = grad_fn(
                    state_f.params,
                    state_g.params,
                    state_f.potential_value_fn,
                    state_g.potential_value_fn,
                    state_g.potential_gradient_fn,
                    batch,
                )
                # update state
                if to_optimize == "both":
                    return (
                        state_f.apply_gradients(grads=grads_f),
                        state_g.apply_gradients(grads=grads_g), loss, loss_f, loss_g,
                        W2_dist
                    )
                if to_optimize == "f":
                    return state_f.apply_gradients(grads=grads_f), loss_f, W2_dist
                if to_optimize == "g":
                    return state_g.apply_gradients(grads=grads_g), loss_g, W2_dist
                raise ValueError("Optimization target has been misspecified.")

            # compute loss and gradients
            (loss, (loss_f, loss_g, W2_dist)), _ = grad_fn(
                state_f.params,
                state_g.params,
                state_f.potential_value_fn,
                state_g.potential_value_fn,
                state_g.potential_gradient_fn,
                batch,
            )

            # do not update state
            if to_optimize == "both":
                return loss_f, loss_g, W2_dist
            if to_optimize == "f":
                return loss_f, W2_dist
            if to_optimize == "g":
                return loss_g, W2_dist
            raise ValueError("Optimization target has been misspecified.")

        return step_fn