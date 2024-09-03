import jax
import jax.numpy as jnp


def expected_maximum(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Returns the expected value of the maximum of n random variables following the given distribution. 
    Assumes that the realizations are sorted in non descending order.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n: int.

    Returns:
        jnp.ndarray of shape ().
    """
    def f(p, pair):
        r_i, p_i = pair 
        p_new = p + p_i
        w = r_i*(p_new**n - p**n)
        return p_new, w

    _, ws = jax.lax.scan(f, 0, (realizations, probabilities))
    return jnp.sum(ws)


def pos_test_probabilities(probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Returns an array where array[i] is the probability that the box contains a value that is at least realization i.
    Assumes that the corresponding realizations are sorted in non descending order.

    Args:
        probabilities (jnp.ndarray): Array of shape (m,).

    Returns:
        jnp.ndarray of shape (m,)
    """
    return jnp.cumsum(probabilities[::-1])[::-1]


def conditional_expected_values(realizations: jnp.ndarray, probabilities: jnp.ndarray, pos_test_prbs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns two arrays where array1[i] is the condititional expected value if the box is tested positive for realization i.
    The second holds the equivalent for negative tests.
    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        pos_test_prbs (jnp.ndarray): Array of shape (m,).

    Returns:
        jnp.ndarray of shape (m,)
    """
    prod = realizations*probabilities
    cond_exp_vs_pos = jnp.cumsum(prod[::-1])[::-1]/pos_test_prbs
    cond_exp_vs_neg = jnp.cumsum(prod)[:-1]/(1-pos_test_prbs[1:])
    return cond_exp_vs_pos, jnp.insert(cond_exp_vs_neg, 0, 0)  # To match the length, we insert a 0.


def expected_value_optimal_strategy_online(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Returns the expected value achieved by the optimal strategy in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n: int.
    
    Returns:
        jnp.ndarray of shape ().
    """
    ev = jnp.sum(realizations*probabilities)
    
    pos_test_prbs = pos_test_probabilities(probabilities)
    cond_exp_value, _ = conditional_expected_values(realizations, probabilities, pos_test_prbs)

    def f(ev_remaining, _):
        j = jnp.min(jnp.where(realizations >= ev_remaining, jnp.arange(realizations.shape[0]), realizations.shape[0]))  # This is the smallest index whose corresponding realization is at least ev_remaining.
        p = pos_test_prbs[j]
        return p*cond_exp_value[j] + (1-p)*ev_remaining, None

    ev_total, _ = jax.lax.scan(f, ev, None, length=n-1)
    return ev_total


def competitive_ratio_online(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Returns the competitive ratio achieved by the optimal strategy in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n (int).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return expected_value_optimal_strategy_online(realizations, probabilities, n)/expected_maximum(realizations, probabilities, n)


def normalize_box(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sorts realizations by size and enforces non-negativity by applying the exponential function to the
    realizations and probabilities.
    Normalizes the probabilities so they sum up to 1.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
    
    Returns:
        tuple [(jnp.ndarray): Array of shape (m,), (jnp.ndarray): Array of shape (m,)]

    """
    sorted_indices = jnp.argsort(realizations)
    realizations_exp = jnp.exp(realizations)
    probabilities_exp = jnp.exp(probabilities)
    return realizations_exp[sorted_indices], probabilities_exp[sorted_indices]/jnp.sum(probabilities_exp)


def competitive_ratio_online_normalized(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Applies the normalization function to the inputs and then 
    calulates the competitive ratio achieved by the optimal strategy in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n (int).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return competitive_ratio_online(*normalize_box(realizations, probabilities), n)


def expected_value_optimal_strategy_semi_online(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Returns the expected value achieved by the optimal strategy in the semi-online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n: int.
    
    Returns:
        jnp.ndarray of shape ().
    """
    pos_test_prbs = pos_test_probabilities(probabilities)
    neg_test_prbs = 1 - pos_test_prbs
    v_pair = conditional_expected_values(realizations, probabilities, pos_test_prbs) # Initialize v_pos and v_neg with the conditional expected values of X.

    def f(v_pair, _):
        v_pos, v_neg = v_pair  # v_pos[j] contains the expected value we will get in total if boxes i+1, ... n are untested and e* = E[X | X >= r_j]. Where e* is the largest cond. expected value of all tested boxes.
        v_max_pos = jax.numpy.max(pos_test_prbs*v_pos + (neg_test_prbs)*v_pos[..., None], axis=1)
        v_max_neg = jax.numpy.max(pos_test_prbs*v_pos + (neg_test_prbs)*v_neg[..., None], axis=1)
        return (v_max_pos, v_max_neg), None
    
    v_pair, _ = jax.lax.scan(f, v_pair, None, length=n-1)
    v_pos, v_neg = v_pair
    return jax.numpy.max(pos_test_prbs*v_pos + neg_test_prbs*v_neg)


def competitive_ratio_semi_online(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Returns the competitive ratio achieved by the optimal strategy in the semi-online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n (int).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return expected_value_optimal_strategy_semi_online(realizations, probabilities, n)/expected_maximum(realizations, probabilities, n)


def competitive_ratio_semi_online_normalized(realizations: jnp.ndarray, probabilities: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Applies the normalization function to the inputs and then 
    calulates the competitive ratio achieved by the optimal strategy in the semi-online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (m,).
        probabilities (jnp.ndarray): Array of shape (m,).
        n (int).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return competitive_ratio_semi_online(*normalize_box(realizations, probabilities), n)

