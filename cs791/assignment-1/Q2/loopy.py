import heapq
import itertools
import json
import math
from copy import deepcopy

# Helpers
def normalize(vec, normalizing_constant):
    s = sum(vec)
    if s > 0:
        return [v / s for v in vec]

    # assume uniform distribution if sum is 0
    return [1.0 / normalizing_constant for _ in vec]


def elementwise_product(a, b):
    assert len(a) == len(b)
    return [a[i] * b[i] for i in range(len(a))]


def base_k_index(digits_without_obs, obs_val, K):
    # digits_without_obs: list/tuple length F, each in [0..K-1]
    idx = 0
    for d in digits_without_obs:
        idx = idx * K + int(d)
    return idx * K + int(obs_val)


class Inference:
    def __init__(self, data):
        self.M = data["Factors_Count"]
        self.K = data["State_Count"]
        self.T = data["Number of Observations"]

        self.max_iters = 10

        self.observation_sequence = data["Observation Sequence"]
        self.transition_potentials = data["Transition Potentials"]
        self.transition_potentials = list(self.transition_potentials.values())
        self.transition_potentials_sum = [
            [
                sum(self.transition_potentials[i][j : j + self.K])
                for j in range(0, len(self.transition_potentials[i]), self.K)
            ]
            for i in range(len(self.transition_potentials))
        ]
        self.transition_potentials = [
            [
                potential / self.transition_potentials_sum[i][j // self.K]
                for j, potential in enumerate(self.transition_potentials[i])
            ]
            for i in range(len(self.transition_potentials))
        ]
        self.state_factor_potentials = data["State_Factor_Potentials"]
        self.state_factor_potentials_sum = [
            sum(self.state_factor_potentials[i : i + self.K])
            for i in range(0, len(self.state_factor_potentials), self.K)
        ]
        self.state_factor_potentials = [
            potential / self.state_factor_potentials_sum[i // self.K]
            for i, potential in enumerate(self.state_factor_potentials)
        ]

        self.variables_count = self.M * self.T
        self.messages_from_factors_to_variables = [
            [[1] * self.K for _ in range(self.M)] for _ in range(self.T)
        ]
        self.messages_from_transitions_to_variables = [
            [[[1] * self.K] * 2 for factor_num in range(self.M)]
            for n in range(self.T - 1)
        ]
        self.messages_from_variables_to_factors = [
            [[1] * self.K for _ in range(self.M)] for _ in range(self.T)
        ]
        self.messages_from_variables_to_transitions = [
            [[[1] * self.K] * 2 for factor_num in range(self.M)]
            for n in range(self.T - 1)
        ]

        # Generate all possible joint states and precompute emission and transition probabilities
        self.joint_states = list(itertools.product(range(self.K), repeat=self.M))
        self.S = len(self.joint_states)  # = K^M

        # Emission per joint state and observation value using state_factor_potentials
        # emission[joint_state_idx][obs_val] = P(observation = obs_val | joint hidden state)
        self.joint_emission = [[0.0 for _ in range(self.K)] for _ in range(self.S)]
        for idx, joint in enumerate(self.joint_states):
            base_idx = 0
            for digit in joint:
                base_idx = base_idx * self.K + int(digit)
            for obs_val in range(self.K):
                full_idx = base_idx * self.K + obs_val
                self.joint_emission[idx][obs_val] = self.state_factor_potentials[
                    full_idx
                ]

        # Transition between joint states as product over factors
        # transition[prev_joint_idx][curr_joint_idx] = P(current joint state | previous joint state)
        self.joint_transition = [[0.0 for _ in range(self.S)] for _ in range(self.S)]
        for i_prev, prev in enumerate(self.joint_states):
            for i_curr, curr in enumerate(self.joint_states):
                prod = 1.0
                for f in range(self.M):
                    prod *= self.transition_potentials[f][prev[f] * self.K + curr[f]]
                self.joint_transition[i_prev][i_curr] = prod

    def compute_marginals(self):
        # Make local copies for messages
        messages_f2v = deepcopy(self.messages_from_factors_to_variables)
        messages_v2f = deepcopy(self.messages_from_variables_to_factors)
        messages_tr2v = deepcopy(self.messages_from_transitions_to_variables)
        messages_v2tr = deepcopy(self.messages_from_variables_to_transitions)
        trans = deepcopy(self.transition_potentials)

        for _ in range(self.max_iters):
            # 1) Update transition factor -> variable messages
            new_tr2v = [
                [[[0.0] * self.K, [0.0] * self.K] for _ in range(self.M)]
                for _ in range(self.T - 1)
            ]
            for t in range(self.T - 1):
                for f in range(self.M):
                    # Forward pass
                    # Right to variable at time t+1: use msg from left variable
                    left_msg = messages_v2tr[t][f][0]
                    factor_T = [0.0] * (self.K * self.K)
                    for a in range(self.K):
                        row_base = a * self.K
                        for b in range(self.K):
                            factor_T[b * self.K + a] = trans[f][row_base + b]
                    new_tr2v[t][f][1] = self.loopy_belief_propagate_to_factor(
                        prev_beliefs=None,
                        factor_beliefs=factor_T,
                        prev_messages=left_msg,
                        prev_messages_from_factor=None,
                    )

                    # Backward pass
                    # Left to variable at time t: use message from right variable to the transition
                    right_msg = messages_v2tr[t][f][1]
                    new_tr2v[t][f][0] = self.loopy_belief_propagate_to_factor(
                        prev_beliefs=None,
                        factor_beliefs=trans[f],
                        prev_messages=right_msg,
                        prev_messages_from_factor=None,
                    )
            messages_tr2v = new_tr2v

            # 2) Update observation factor -> variable messages per time slice using current variable -> factor messages
            new_f2v = [[[0.0] * self.K for _ in range(self.M)] for _ in range(self.T)]
            for t in range(self.T):
                obs_val = int(self.observation_sequence[t])
                # Precompute product over all variables of m_{V->F}(s_v) for each joint assignment
                out_per_factor = [[0.0] * self.K for _ in range(self.M)]
                total_joint = self.K**self.M
                for joint_idx in range(total_joint):
                    # decode base-K digits for F variables
                    rem = joint_idx
                    digits = [0] * self.M
                    for d in range(self.M - 1, -1, -1):
                        digits[d] = rem % self.K
                        rem //= self.K
                    idx = base_k_index(digits, obs_val, self.K)
                    pot = self.state_factor_potentials[idx]
                    if pot == 0.0:
                        continue
                    # For each variable f, accumulate product of messages from all other variables except f
                    for f in range(self.M):
                        prod_others = pot
                        for g in range(self.M):
                            if g == f:
                                continue
                            prod_others *= messages_v2f[t][g][digits[g]]
                            if prod_others == 0.0:
                                break
                        if prod_others != 0.0:
                            out_per_factor[f][digits[f]] += prod_others

                # normalize and store
                for f in range(self.M):
                    new_f2v[t][f] = normalize(out_per_factor[f], self.K)
            messages_f2v = new_f2v

            # 3) Update variable -> observation/transition messages and compute beliefs
            for t in range(self.T):
                for f in range(self.M):
                    # Incoming from observation factor
                    in_obs = messages_f2v[t][f]
                    # Incoming from left and right transition factors (if exist)
                    in_left = [1.0] * self.K
                    in_right = [1.0] * self.K
                    if t > 0:
                        # from transition at (t-1) to V_{t,f} (right side of that transition)
                        in_left = messages_tr2v[t - 1][f][1]
                    if t < self.T - 1:
                        # from transition at t to V_{t,f} (left side of that transition)
                        in_right = messages_tr2v[t][f][0]

                    # Belief proportional to product of incoming messages (unnormalized)
                    belief_unnorm = elementwise_product(
                        in_obs, elementwise_product(in_left, in_right)
                    )
                    belief_vec = normalize(belief_unnorm, self.K)

                    # m_{V->Obs}(x): exclude obs incoming
                    messages_v2f[t][f] = self.loopy_belief_propagate_to_variable(
                        prev_beliefs=belief_unnorm,
                        prev_messages=in_obs,
                        prev_messages_from_variables=None,
                        clique_members=None,
                    )

                    # Messages to transitions exclude the target transition
                    # To right transition at edge t (between t and t+1)
                    if t < self.T - 1:
                        messages_v2tr[t][f][0] = (
                            self.loopy_belief_propagate_to_variable(
                                prev_beliefs=belief_unnorm,
                                prev_messages=in_right,
                                prev_messages_from_variables=None,
                                clique_members=None,
                            )
                        )
                    # To left transition at edge t-1 (between t-1 and t)
                    if t - 1 >= 0:
                        messages_v2tr[t - 1][f][1] = (
                            self.loopy_belief_propagate_to_variable(
                                prev_beliefs=belief_unnorm,
                                prev_messages=in_left,
                                prev_messages_from_variables=None,
                                clique_members=None,
                            )
                        )

        # After iterations, compute final beliefs and return as marginals ordered by time then factor
        variable_beliefs = []
        for t in range(self.T):
            for f in range(self.M):
                in_obs = messages_f2v[t][f]
                in_left = [1.0] * self.K
                in_right = [1.0] * self.K
                if t - 1 >= 0 and self.T >= 2:
                    in_left = messages_tr2v[t - 1][f][1]
                if t < self.T - 1 and self.T >= 2:
                    in_right = messages_tr2v[t][f][0]
                belief_vec = [
                    in_obs[i] * in_left[i] * in_right[i] for i in range(self.K)
                ]
                variable_beliefs.append(normalize(belief_vec, self.K))

        self.variable_beliefs = variable_beliefs
        return self.variable_beliefs

    def get_z_value(self):
        # Forward pass: alpha[t][s] = P(o_1..o_t, s_t = s)
        alpha_prev = [0.0 for _ in range(self.S)]
        obs_0 = int(self.observation_sequence[0])
        for s in range(self.S):
            # Consider the initial probability as uniform
            alpha_prev[s] = 1.0 / self.S * self.joint_emission[s][obs_0]
        for t in range(1, self.T):
            obs_t = int(self.observation_sequence[t])
            alpha_curr = [0.0 for _ in range(self.S)]
            for curr in range(self.S):
                em = self.joint_emission[curr][obs_t]
                if em == 0.0:
                    continue
                ssum = 0.0
                for prev in range(self.S):
                    tp = self.joint_transition[prev][curr]
                    if tp != 0.0 and alpha_prev[prev] != 0.0:
                        ssum += alpha_prev[prev] * tp
                alpha_curr[curr] = em * ssum
            alpha_prev = alpha_curr

        return (self.K**self.M) * sum(alpha_prev)

    def compute_top_k(self, k=2):
        # Forward pass: alpha[t][s] = P(o_1..o_t, s_t = s)
        obs_0 = int(self.observation_sequence[0])
        init_prob = 1.0 / self.S
        alpha_prev = [[] for _ in range(self.S)]
        for s in range(self.S):
            prob = init_prob * self.joint_emission[s][obs_0]
            if prob > 0.0:
                alpha_prev[s].append((prob, [s]))

        for t in range(1, self.T):
            obs_t = int(self.observation_sequence[t])
            alpha_curr = [[] for _ in range(self.S)]
            for curr in range(self.S):
                em = self.joint_emission[curr][obs_t]
                if em == 0.0:
                    continue
                heap = []  # min-heap of (prob, path)
                for prev in range(self.S):
                    tp = self.joint_transition[prev][curr]
                    if tp == 0.0 or not alpha_prev[prev]:
                        continue
                    for prev_prob, prev_path in alpha_prev[prev]:
                        cand_prob = prev_prob * tp * em
                        if cand_prob == 0.0:
                            continue
                        if len(heap) < k:
                            heapq.heappush(heap, (cand_prob, prev_path + [curr]))
                        else:
                            if cand_prob > heap[0][0]:
                                heapq.heapreplace(heap, (cand_prob, prev_path + [curr]))
                if heap:
                    lst = sorted(heap, key=lambda x: -x[0])
                    alpha_curr[curr] = lst
            alpha_prev = alpha_curr

        # Collect top-k among all ending states
        all_candidates = []
        for s in range(self.S):
            for prob, path in alpha_prev[s]:
                all_candidates.append((prob, path))
        all_candidates.sort(key=lambda x: -x[0])
        topk = all_candidates[:k]

        # Normalize
        Z = self.get_z_value()
        output = []
        for prob, path in topk:
            assignment = []
            for s_idx in path:
                joint = self.joint_states[s_idx]
                for f in range(self.M):
                    assignment.append(int(joint[f]))
            normalized_prob = prob / Z
            output.append(
                {
                    "assignment": assignment,
                    "probability": (self.K**self.M) * normalized_prob,
                }
            )
        return output

    def loopy_belief_propagate_to_variable(
        self,
        prev_beliefs,
        prev_messages,
        prev_messages_from_variables,
        clique_members,
        # composition_func, # assumed to be element-wise product
        # marginalization_func, # assumed to be sum
        # inverse_composition_func, # assumed to be division
    ):
        # compute message from a variable to a clique by excluding the clique's incoming message
        out = []
        for i in range(len(prev_beliefs)):
            denom = prev_messages[i]
            if denom == 0:
                out.append(0.0)
            else:
                out.append(prev_beliefs[i] / denom)
        return normalize(out, len(prev_beliefs))

    def loopy_belief_propagate_to_factor(
        self,
        prev_beliefs,
        factor_beliefs,
        prev_messages,
        prev_messages_from_factor,
        # composition_func, # assumed to be element-wise product
        # inverse_composition_func, # assumed to be division
    ):
        K = math.sqrt(len(factor_beliefs))
        if math.floor(K) == K:
            K = int(K)

            # message to the left variable = sum over right variables
            out = [0.0] * K
            for x_left in range(K):
                ssum = 0.0
                row_base = x_left * K
                for x_right in range(K):
                    ssum += factor_beliefs[row_base + x_right] * prev_messages[x_right]
                out[x_left] = ssum
            return normalize(out, K)

        if prev_beliefs:
            K = len(prev_beliefs)
            return [1.0 / K for _ in range(K)]

        return []

    def factor_in_fhmm(
        self,
        prev_beliefs,
        prev_messages,
        step_size,
        new_messages,
        # composition_func,
        # inverse_composition_func,
    ):
        # new = (1 - step_size) * old + step_size * proposed
        out = [0.0] * len(prev_beliefs)
        for i in range(len(prev_beliefs)):
            out[i] = (1.0 - step_size) * prev_beliefs[i] + step_size * new_messages[i]

        return normalize(out, len(prev_beliefs))


########################################################################

# Do not change anything below this line

########################################################################


class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, "r") as file:
            self.data = json.load(file)

    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]["Input"])
            marginals = inference.compute_marginals()
            z_value = inference.get_z_value()
            top_k_assignments = inference.compute_top_k()
            output.append(
                {
                    "Marginals": marginals,
                    "Top_k_assignments": top_k_assignments,
                    "Z_value": z_value,
                }
            )
        self.output = output

    def write_output(self, file_name):
        with open(file_name, "w") as file:
            json.dump(self.output, file, indent=4)


if __name__ == "__main__":
    evaluator = Get_Input_and_Check_Output("TestCases.json")
    evaluator.get_output()
    evaluator.write_output("Sample_Testcase_Output.json")
