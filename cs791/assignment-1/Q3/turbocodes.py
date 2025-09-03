import json
import math


def normalize(array):
    total = sum(array)
    return [x / total for x in array] if total > 0 else array


generator1 = 0o5
generator2 = 0o7


def generate_trellis(generator1, generator2, history_length=3):
    # generator 1 is for giving feedback
    # generator 2 is for calculating parity bit
    """
    Build trellis for the RSC encoder implementation
    State representation: integer from 0 to (2**history_length - 1)
     (left = oldest bit, right = newest bit).

    Return format  dict: state -> { input_bit -> (next_state, parity_bit) }
    """
    # Convert generators to binary strings reversed like in your encoder
    bin_g1 = bin(generator1)[2:][::-1].zfill(history_length)
    bin_g2 = bin(generator2)[2:][::-1].zfill(history_length)

    def tuple_to_int(t):
        val = 0
        for b in t:
            val = (val << 1) | int(b)
        return val

    def int_to_tuple(x):
        s = format(x, "0{}b".format(history_length))
        return tuple(int(ch) for ch in s)

    trellis = {}
    max_state = 2**history_length
    for state_int in range(max_state):
        state = int_to_tuple(state_int)  # tuple of bits (len = history_length)
        trellis[state_int] = {}
        for u in (0, 1):
            #   shift left that is remove left bit, move every other bit one step left and add the input bit to the right
            temp = state[1:] + (u,)

            # feedback = dot(temp, g2) % 2
            feedback = 0
            for bit, g in zip(temp, bin_g2):
                feedback ^= bit & int(g)

            # replace last element with feedback
            history_after_feedback = temp[:-1] + (feedback,)

            # compute parity using generator1 after feedback
            parity = 0
            for bit, g in zip(history_after_feedback, bin_g1):
                parity ^= bit & int(g)

            next_state_int = tuple_to_int(history_after_feedback)

            trellis[state_int][u] = (next_state_int, parity)
    return trellis


def viterbi(
    noisy_output, probability_matrix, generator1=0o5, generator2=0o7, history_length=3
):
    """
    Viterbi decoder : uses sysout + par1
    """
    # Build trellis
    trellis = generate_trellis(generator1, generator2, history_length=history_length)
    n_states = 2**history_length

    # Split observations: take systematic and parity1
    T = len(noisy_output) // 3
    sys_obs = noisy_output[0::3]  # observed symbols for systematic bits
    par1_obs = noisy_output[1::3]  # observed symbols for parity from encoder1

    # probabilities: probability_matrix[b][obs] = P(obs | bit=b)
    P = probability_matrix

    # small epsilon to avoid log(0)
    eps = 1e-300
    neglog = lambda p: -math.log(p + eps)

    # Initialize path metrics: set all to +inf except start state 0
    INF = 1e300
    path_metric_prev = [INF] * n_states
    path_metric_prev[0] = 0.0  # assume encoder starts in state 0

    # For backtracking
    prev_state = [[-1] * n_states for _ in range(T)]
    prev_bit = [[-1] * n_states for _ in range(T)]

    # Viterbi forward recursion
    for t in range(T):
        obs_s = sys_obs[t]
        obs_p = par1_obs[t]

        # initialize next metrics to +inf
        path_metric_next = [INF] * n_states

        for s in range(n_states):
            if path_metric_prev[s] >= INF:
                continue  # unreachable state

            current_metric = path_metric_prev[s]

            # for input 0 and 1, compute transition
            for u in (0, 1):
                next_s, parity = trellis[s][u]

                # likelihood: P(sys_obs | u) * P(par_obs | parity)
                p_sys = P[u][obs_s] if u < len(P) and obs_s < len(P[u]) else 0.0
                p_par = (
                    P[parity][obs_p]
                    if parity < len(P) and obs_p < len(P[parity])
                    else 0.0
                )

                # branch metric = -log( p_sys * p_par )
                branch_cost = neglog(p_sys) + neglog(p_par)

                candidate_metric = current_metric + branch_cost

                # update if better
                if candidate_metric < path_metric_next[next_s]:
                    path_metric_next[next_s] = candidate_metric
                    prev_state[t][next_s] = s
                    prev_bit[t][next_s] = u

        # move to next time
        path_metric_prev = path_metric_next

    # Termination: find the state with smallest metric at time T-1
    best_final_state = min(range(n_states), key=lambda s: path_metric_prev[s])

    # Backtrack to extract input bits
    decoded_bits = [0] * T
    cur_state = best_final_state
    for t in range(T - 1, -1, -1):
        b = prev_bit[t][cur_state]
        ps = prev_state[t][cur_state]
        if b == -1 or ps == -1:
            # Something went wrong (unreachable) — fallback to zeros for remaining
            decoded_bits[: t + 1] = [0] * (t + 1)
            break
        decoded_bits[t] = b
        cur_state = ps

    return "".join(str(b) for b in decoded_bits)


def bcjr(
    noisy_output,
    probability_matrix,
    app_probability=None,
    generator1=0o5,
    generator2=0o7,
    history_length=3,
):
    """ """
    # Build trellis
    trellis = generate_trellis(generator1, generator2, history_length=history_length)
    n_states = 2**history_length
    T = len(noisy_output) // 3
    sys_obs = noisy_output[0::3]
    par1_obs = noisy_output[1::3]
    P = probability_matrix  # P[b][obs] = P(obs | transmitted bit = b)

    # prepare transitions: for each state s, for input u in {0,1}, get (next_s, parity)
    transitions = []  # list of (s, u, next_s, parity)
    incoming = [
        [] for _ in range(n_states)
    ]  # incoming[next_s] = list of indices into transitions
    for s in range(n_states):
        for u in (0, 1):
            next_s, parity = trellis[s][u]
            idx = len(transitions)
            transitions.append((s, u, next_s, parity))
            incoming[next_s].append(idx)

    # optional apriori probabilities per bit
    if app_probability is None:
        # uniform prior: P(u=0)=P(u=1)=0.5 for every t
        apriori = [[0.5] * T, [0.5] * T]  # apriori[bit][t]
    else:
        apriori = app_probability  # expected shape (2, T)

    gamma = [[0.0] * len(transitions) for _ in range(T)]
    for t in range(T):
        o_s = sys_obs[t]
        o_p = par1_obs[t]
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            p_sys = P[u][o_s] if (u < len(P) and o_s < len(P[u])) else 0.0
            p_par = (
                P[parity][o_p] if (parity < len(P) and o_p < len(P[parity])) else 0.0
            )
            prior_u = apriori[u][t] if t < len(apriori[0]) else apriori[u][-1]
            gamma_val = p_sys * p_par * prior_u
            gamma[t][tr_idx] = gamma_val

    # Forward recursion alpha
    alpha = [[0.0] * n_states for _ in range(T + 1)]
    alpha[0][0] = 1.0
    scales = [1.0] * T  # scaling factors for each time step
    for t in range(1, T + 1):
        for s2 in range(n_states):
            alpha[t][s2] = 0.0
        # for each transition, add contribution alpha[t-1][s] * gamma[t-1][tr]
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            val = alpha[t - 1][s] * gamma[t - 1][tr_idx]
            if val:
                alpha[t][next_s] += val
        # scaling
        scale = sum(alpha[t])
        if scale == 0.0:
            scale = 1.0
        scales[t - 1] = scale
        for s2 in range(n_states):
            alpha[t][s2] /= scale

    # Backward recursion (beta)
    beta = [[0.0] * n_states for _ in range(T + 1)]
    for s in range(n_states):
        beta[T][s] = 1.0  # unscaled
    # iterate backward
    for t in range(T - 1, -1, -1):
        for s in range(n_states):
            total = 0.0

            for tr_idx, (ss, u, next_s, parity) in enumerate(transitions):
                if ss != s:
                    continue
                total += gamma[t][tr_idx] * beta[t + 1][next_s]
            beta[t][s] = total
        scale = scales[t] if t < len(scales) else 1.0
        if scale == 0.0:
            scale = 1.0
        for s in range(n_states):
            beta[t][s] /= scale

    decoded_bits = [0] * T
    for t in range(T):
        app_num = [0.0, 0.0]
        app_den = 0.0
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            contrib = alpha[t][s] * gamma[t][tr_idx] * beta[t + 1][next_s]
            app_num[u] += contrib
            app_den += contrib
        # avoid divide by zero
        if app_den == 0.0:
            # fallback to apriori
            p0 = apriori[0][t]
            p1 = apriori[1][t]
            decoded_bits[t] = 1 if p1 >= p0 else 0
        else:
            p1 = app_num[1] / app_den
            decoded_bits[t] = 1 if p1 >= 0.5 else 0

    return "".join(str(b) for b in decoded_bits)


def bcjr_soft(
    noisy_output,
    probability_matrix,
    app_probability=None,
    generator1=0o5,
    generator2=0o7,
    history_length=3,
):
    """
    BCJR that returns APP probabilities for each bit (soft output).
    Returns: list of tuples [(P0, P1), ...] where P0 = P(u=0|y), P1 = P(u=1|y).
    """
    trellis = generate_trellis(generator1, generator2, history_length=history_length)
    n_states = 2**history_length
    T = len(noisy_output) // 3
    sys_obs = noisy_output[0::3]
    par_obs = noisy_output[1::3]
    P = probability_matrix

    transitions = []
    for s in range(n_states):
        for u in (0, 1):
            next_s, parity = trellis[s][u]
            transitions.append((s, u, next_s, parity))

    if app_probability is None:
        apriori = [[0.5] * T, [0.5] * T]
    else:
        apriori = app_probability

    # gamma
    gamma = [[0.0] * len(transitions) for _ in range(T)]
    for t in range(T):
        o_s = sys_obs[t]
        o_p = par_obs[t]
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            p_sys = P[u][o_s] if (u < len(P) and o_s < len(P[u])) else 0.0
            p_par = (
                P[parity][o_p] if (parity < len(P) and o_p < len(P[parity])) else 0.0
            )
            prior_u = apriori[u][t]
            gamma[t][tr_idx] = p_sys * p_par * prior_u

    # alpha
    alpha = [[0.0] * n_states for _ in range(T + 1)]
    alpha[0][0] = 1.0
    scales = [1.0] * T
    for t in range(1, T + 1):
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            alpha[t][next_s] += alpha[t - 1][s] * gamma[t - 1][tr_idx]
        scale = sum(alpha[t])
        if scale == 0.0:
            scale = 1.0
        scales[t - 1] = scale
        for s in range(n_states):
            alpha[t][s] /= scale

    # beta
    beta = [[0.0] * n_states for _ in range(T + 1)]
    for s in range(n_states):
        beta[T][s] = 1.0
    for t in range(T - 1, -1, -1):
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            beta[t][s] += gamma[t][tr_idx] * beta[t + 1][next_s]
        scale = scales[t]
        if scale == 0.0:
            scale = 1.0
        for s in range(n_states):
            beta[t][s] /= scale

    # APPs
    app_probs = []
    for t in range(T):
        num = [0.0, 0.0]
        den = 0.0
        for tr_idx, (s, u, next_s, parity) in enumerate(transitions):
            contrib = alpha[t][s] * gamma[t][tr_idx] * beta[t + 1][next_s]
            num[u] += contrib
            den += contrib
        if den == 0.0:
            app_probs.append((0.5, 0.5))
        else:
            p0 = num[0] / den
            p1 = num[1] / den
            app_probs.append((p0, p1))
    return app_probs


def turbo_decode(
    noisy_output,
    probability_matrix,
    permutation,
    n_iter=2,
    generator1=0o5,
    generator2=0o7,
    history_length=3,
):
    """
    Turbo decoding using two BCJR decoders exchanging info.
    """
    T = len(noisy_output) // 3

    # Split data
    sys_obs = noisy_output[0::3]
    par1_obs = noisy_output[1::3]
    par2_obs = noisy_output[2::3]

    noisy1 = []
    for t in range(T):
        noisy1.extend([sys_obs[t], par1_obs[t], 0])  # dummy third symbol
    noisy2 = []
    for t in range(T):
        noisy2.extend([sys_obs[permutation[t]], par2_obs[t], 0])

    # Initialize a priori = uniform
    apriori1 = [[0.5] * T, [0.5] * T]
    apriori2 = [[0.5] * T, [0.5] * T]

    for _ in range(n_iter):
        app1 = bcjr_soft(
            noisy1, probability_matrix, apriori1, generator1, generator2, history_length
        )
        extr1 = [
            (max(1e-12, p0 / apriori1[0][t]), max(1e-12, p1 / apriori1[1][t]))
            for t, (p0, p1) in enumerate(app1)
        ]
        # normalize
        extr1 = [normalize(x) for x in extr1]

        inter_extr = [extr1[permutation[t]] for t in range(T)]
        apriori2 = [[x[0] for x in inter_extr], [x[1] for x in inter_extr]]

        app2 = bcjr_soft(
            noisy2, probability_matrix, apriori2, generator1, generator2, history_length
        )
        extr2 = [
            (max(1e-12, p0 / apriori2[0][t]), max(1e-12, p1 / apriori2[1][t]))
            for t, (p0, p1) in enumerate(app2)
        ]
        extr2 = [normalize(x) for x in extr2]

        deinter_extr = [None] * T
        for t in range(T):
            deinter_extr[permutation[t]] = extr2[t]
        apriori1 = [[x[0] for x in deinter_extr], [x[1] for x in deinter_extr]]

    final_app = app1
    decoded_bits = [1 if p1 >= 0.5 else 0 for (p0, p1) in final_app]
    return "".join(str(b) for b in decoded_bits)


class Inference:
    def __init__(self, testcase):
        ### NOTE that you will not have access to the original bitstring in the actual evaluation
        self.noisy_output = testcase["noisy_output"]
        self.length = len(self.noisy_output) // 3
        self.probability_matrix = testcase["probability_matrix"]
        self.probability_matrix = [
            normalize(self.probability_matrix),
            normalize(self.probability_matrix)[::-1],
        ]
        self.permutation = testcase["permutation"]

    def get_viterbi_output(self):
        return viterbi(self.noisy_output, self.probability_matrix)

    def get_bcjr_output(self):
        return bcjr(self.noisy_output, self.probability_matrix, app_probability=None)

    def get_turbocode_output(self):
        return turbo_decode(
            self.noisy_output, self.probability_matrix, self.permutation
        )


if __name__ == "__main__":
    with open("./turbocodes_testcases.json", "r") as f:
        data = json.load(f)

    testcases = data["testcases"]

    results = []

    for i, testcase in enumerate(testcases, start=1):
        infer = Inference(testcase)

        viterbi_out = infer.get_viterbi_output()
        bcjr_out = infer.get_bcjr_output()
        turbo_out = infer.get_turbocode_output()
        results.append(
            {
                "testcase": i,
                "viterbi_errors": len(
                    [
                        j
                        for j in range(len(viterbi_out))
                        if viterbi_out[j] != testcase["bitstring"][j]
                    ]
                ),
                "bcjr_errors": len(
                    [
                        j
                        for j in range(len(bcjr_out))
                        if bcjr_out[j] != testcase["bitstring"][j]
                    ]
                ),
                "turbo_errors": len(
                    [
                        j
                        for j in range(len(turbo_out))
                        if turbo_out[j] != testcase["bitstring"][j]
                    ]
                ),
            }
        )

    with open("turbocodes_results.json", "w") as f:
        json.dump(results, f, indent=4)

