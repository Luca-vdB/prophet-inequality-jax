import jax
import jax.numpy as jnp


def expected_maximum(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the expected value of the maximum of n random variables. All random variables have m possible outcomes. 
    Assumes that the realizations are sorted in non descending order.

    Args:
        realizations (jnp.ndarray): Array of shape (n,m).
        probabilities (jnp.ndarray): Array of shape (n, m).

    Returns:
        jnp.ndarray of shape ().
    """
    def f(s, r):
        prob_r_is_max = (jnp.prod(jnp.sum(jnp.where(realizations <= r, probabilities, 0), axis=1)) - jnp.prod(jnp.sum(jnp.where(realizations < r, probabilities, 0), axis=1)))
        return s + r*prob_r_is_max, None
    
    s, _ = jax.lax.scan(f, 0, jnp.unique(realizations, size=realizations.size, fill_value=-1))
    return s


def get_pos_test_probabilities(probabilities: jnp.ndarray) -> jnp.ndarray:
    return jnp.cumsum(probabilities[::-1])[::-1]

def get_cond_exp_values(realizations: jnp.ndarray, probabilities: jnp.ndarray, pos_test_prbs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    prod = realizations*probabilities
    cond_exp_vs_pos = jnp.cumsum(prod[::-1])[::-1]/pos_test_prbs
    cond_exp_vs_neg = jnp.cumsum(prod)[:-1]/(1-pos_test_prbs[1:])
    return cond_exp_vs_pos, jnp.insert(cond_exp_vs_neg, 0, 0)

def expected_value_optimal_strategy_online(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the expected value achieved by the optimal strategy in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
    
    Returns:
        jnp.ndarray of shape ().
    """
    ev_last_box = jnp.sum(realizations[-1]*probabilities[-1])
    def f(ev_remaining, pair):
        r_i, p_i = pair
        pos_test_prbs = get_pos_test_probabilities(p_i)
        cond_exp_values, _ = get_cond_exp_values(r_i, p_i, pos_test_prbs)
        j = jnp.min(jnp.where(r_i >= ev_remaining, jnp.arange(r_i.shape[0]), r_i.shape[0]))  # This is the smallest index whose corresponding realization is at least ev_reamining.
        p = jnp.sum(jnp.where(r_i >= ev_remaining, p_i, 0)) # The probability that X_i is at least ev_remaining
        return p*cond_exp_values[j] + (1-p)*ev_remaining, None 

    ev_total, _ = jax.lax.scan(f, ev_last_box, (realizations[::-1][1:], probabilities[::-1][1:]))
    return ev_total


def competitive_ratio_online(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the competitive ratio achieved by the optimal strategy in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return expected_value_optimal_strategy_online(realizations, probabilities)/expected_maximum(realizations, probabilities)


def normalize_boxes(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sorts realizations by size and enforces non-negativity by applying the exponential function to the
    realizations and probabilities.
    Normalizes the probabilities so they sum up to 1.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
    
    Returns:
        tuple [(jnp.ndarray): Array of shape (n, m), (jnp.ndarray): Array of shape (n, m)]

    """
    sorted_indices = jnp.argsort(realizations, axis=1)

    realizations_exp = jnp.exp(realizations)
    probabilities_exp = jnp.exp(probabilities)

    probabilities_normalized = probabilities_exp/jnp.sum(probabilities_exp, axis=1)[..., None]

    return jnp.take_along_axis(realizations_exp, sorted_indices, axis=1), jnp.take_along_axis(probabilities_normalized, sorted_indices, axis=1)


def competitive_ratio_online_normalized(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the normalization function to the inputs and then 
    calulates the competitive ratio achieved by the optimal strategy in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return competitive_ratio_online(*normalize_boxes(realizations, probabilities))


def expected_value_optimal_order(realizations: jnp.ndarray, probabilities: jnp.ndarray, permutations: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the expected value achieved by the optimal strategy on the optimal order in the online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
        permutations (jnp.ndarray): Array of shape (n, k).
    
    Returns:
        jnp.ndarray of shape ().
    """
    def g(perm):
        return expected_value_optimal_strategy_online(realizations[perm], probabilities[perm])
    """
    def f(perms):
        return jax.vmap(g)(perms)
    
    permutations = permutations.reshape(jax.local_device_count(), -1, realizations.shape[0])
    """
    evs = jax.vmap(g)(permutations)
    return jnp.max(evs)  # .reshape(-1)


def competitive_ratio_order_selection(realizations: jnp.ndarray, probabilities: jnp.ndarray, permutations: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the normalization function to the inputs and then 
    calulates the competitive ratio achieved by the optimal strategy in the order selection online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
        permutations (jnp.ndarray): Array of shape (n,).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return expected_value_optimal_order(realizations, probabilities, permutations)/expected_maximum(realizations[permutations[0]], probabilities[permutations[0]])


def competitive_ratio_order_selection_normalized(realizations: jnp.ndarray, probabilities: jnp.ndarray, permutations: jnp.ndarray) -> jnp.ndarray:
    return competitive_ratio_order_selection(*normalize_boxes(realizations, probabilities), permutations)


def expected_value_prophet_secretary(realizations: jnp.ndarray, probabilities: jnp.ndarray, permutations: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the expected value achieved by the optimal strategy in the online prophet secretary setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
        permutations (jnp.ndarray): Array of shape (n,)
    
    Returns:
        jnp.ndarray of shape ().
    """
    
    def g(perm):
        return expected_value_optimal_strategy_online(realizations[perm], probabilities[perm])
    """
    def f(perms):
        return jax.vmap(g)(perms)
    
    permutations = permutations.reshape(jax.local_device_count(), -1, realizations.shape[0])
    """
    evs = jax.vmap(g)(permutations)
    return jnp.mean(evs)  # .reshape(-1)


def competitive_ratio_prophet_secretary(realizations: jnp.ndarray, probabilities: jnp.ndarray, permutations: jnp.ndarray) -> jnp.ndarray:
    return expected_value_prophet_secretary(realizations, probabilities, permutations)/expected_maximum(realizations[permutations[0]], probabilities[permutations[0]])


def competitive_ratio_prophet_secretary_normalized(realizations: jnp.ndarray, probabilities: jnp.ndarray, permutations: jnp.ndarray) -> jnp.ndarray:
    return competitive_ratio_prophet_secretary(*normalize_boxes(realizations, probabilities), permutations)

def pos_test_probabilities(probabilities: jnp.ndarray) -> jnp.ndarray:
    return jnp.cumsum(probabilities[:, ::-1], axis=1)[:, ::-1]


def cond_exp_values(realizations: jnp.ndarray, probabilities: jnp.ndarray, pos_test_prbs: jnp.ndarray) -> jnp.ndarray:
    prod = realizations*probabilities
    cond_exp_vs_pos = jnp.cumsum(prod[:, ::-1], axis=1)[:, ::-1]/pos_test_prbs
    cond_exp_vs_neg = jnp.cumsum(prod, axis=1)[:, :-1]/(1-pos_test_prbs[:, 1:])
    return cond_exp_vs_pos, jnp.insert(jnp.nan_to_num(cond_exp_vs_neg), 0, 0, axis=1)


def expected_value_optimal_strategy_semi_online(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the expected value achieved by the optimal strategy in the semi-online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
        n: int.
    
    Returns:
        jnp.ndarray of shape ().
    """
    pos_test_prbs = pos_test_probabilities(probabilities)
    neg_test_prbs = 1 - pos_test_prbs
    v_pos, v_neg = cond_exp_values(realizations, probabilities, pos_test_prbs)

    n = realizations.shape[0]
    """
    for k in range(n-1, 0, -1):  # We cycle through all boxes, starting at the last one.
        v_pos_sum = pos_test_prbs[k]*jnp.maximum(v_pos[k], v_pos[:k][..., None]) \
            +  neg_test_prbs[k]*jnp.maximum(v_neg[k], v_pos[:k][..., None])
        v_neg_sum = pos_test_prbs[k]*jnp.maximum(v_pos[k], v_neg[:k][..., None]) \
            + neg_test_prbs[k]*jnp.maximum(v_neg[k], v_neg[:k][..., None])

        v_pos = jnp.max(v_pos_sum, axis=2)
        v_neg = jnp.max(v_neg_sum, axis=2)
    """
    
    def f(carry, prbs_and_k):
        v_pos, v_neg, = carry
        pos_test_prbs, neg_test_prbs, k = prbs_and_k

        v_pos_sum = pos_test_prbs*jnp.maximum(v_pos[k], v_pos[..., None]) \
            +  neg_test_prbs*jnp.maximum(v_neg[k], v_pos[..., None])
        v_neg_sum = pos_test_prbs*jnp.maximum(v_pos[k], v_neg[..., None]) \
            + neg_test_prbs*jnp.maximum(v_neg[k], v_neg[..., None])

        v_pos = jnp.max(v_pos_sum, axis=2)
        v_neg = jnp.max(v_neg_sum, axis=2)

        v_pos = jnp.where(jnp.arange(n)[:, None] > k+1, 0, v_pos)
        v_neg = jnp.where(jnp.arange(n)[:, None] > k+1, 0, v_neg)

        return (v_pos, v_neg), None
    
    v_pair, _ = jax.lax.scan(f, (v_pos, v_neg), (pos_test_prbs[1:][::-1], neg_test_prbs[1:][::-1], jnp.arange(n-1, 0, -1)))
    v_pos, v_neg = v_pair
    
    return jnp.max(pos_test_prbs[0]*v_pos[0] + neg_test_prbs[0]*v_neg[0])


def competitive_ratio_semi_online(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the competitive ratio achieved by the optimal strategy in the semi-online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return expected_value_optimal_strategy_semi_online(realizations, probabilities)/expected_maximum(realizations, probabilities)


def competitive_ratio_semi_online_normalized(realizations: jnp.ndarray, probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the normalization function to the inputs and then 
    calulates the competitive ratio achieved by the optimal strategy in the semi-online setting.

    Args:
        realizations (jnp.ndarray): Array of shape (n, m).
        probabilities (jnp.ndarray): Array of shape (n, m).
        n (int).
    
    Returns:
        jnp.ndarray of shape ().
    """
    return competitive_ratio_semi_online(*normalize_boxes(realizations, probabilities))