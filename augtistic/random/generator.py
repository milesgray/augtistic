from tensorflow.python.ops import variables
from tensorflow.python.ops import stateful_random_ops

class RandomGenerator(stateful_random_ops.Generator):
    """A subclass that allows creation inside distribution strategies.
    This is a temporary solution to allow creating tf.random.Generator inside
    distribution strategies. It will be removed when proper API is in place.
    All replicas will have the same RNG state and generate the same random
    numbers.
    """

    # TODO(b/157995497): Temporarily use primary variable handle inside cross
    # replica context.
    @property
    def state(self):
        """The internal state of the RNG."""
        state_var = self._state_var
        try:
            _ = getattr(state_var, 'handle')
            return state_var
        except ValueError:
            return state_var.values[0]

    def _create_variable(self, *args, **kwargs):
        # This function does the same thing as the base class's namesake, except
        # that it skips the distribution-strategy check. When we are inside a
        # distribution-strategy scope, variables.Variable will pick a proper
        # variable class (e.g. MirroredVariable).
        return variables.Variable(*args, **kwargs)


def get(seed=None):
    if seed:
        return RandomGenerator.from_seed(seed)
    else:
        return RandomGenerator.from_non_deterministic_state()