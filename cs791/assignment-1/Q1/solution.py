import json
import itertools
import heapq


class Inference:
    def __init__(self, data):
        self.factors_count = data["Factors_Count"]
        self.states_count = data["State_Count"]
        self.num_observations = data["Number of Observations"]
        self.observation_sequence = data["Observation Sequence"]
        self.transition_potentials = data["Transition Potentials"]
        self.transition_potentials = list(self.transition_potentials.values())
        self.transition_potentials_sum = [
            [
                sum(self.transition_potentials[i][j : j + self.states_count])
                for j in range(0, len(self.transition_potentials[i]), self.states_count)
            ]
            for i in range(len(self.transition_potentials))
        ]
        self.transition_potentials = [
            [
                potential / self.transition_potentials_sum[i][j // self.states_count]
                for j, potential in enumerate(self.transition_potentials[i])
            ]
            for i in range(len(self.transition_potentials))
        ]
        self.state_factor_potentials = data["State_Factor_Potentials"]
        self.state_factor_potentials_sum = [
            sum(self.state_factor_potentials[i : i + self.states_count])
            for i in range(0, len(self.state_factor_potentials), self.states_count)
        ]
        self.state_factor_potentials = [
            potential / self.state_factor_potentials_sum[i // self.states_count]
            for i, potential in enumerate(self.state_factor_potentials)
        ]
        self.k = data["K"]

        self.M = self.factors_count
        self.K = self.states_count
        self.T = self.num_observations

        self.joint_states = []
        if self.M == 0:
            self.joint_states = [tuple()]
        else:
            self.joint_states = list(itertools.product(range(self.K), repeat=self.M))

        self.S = len(self.joint_states)  # = K^M

        self.num_joint_states = len(self.joint_states)
        self.joint_state_to_index = {s: idx for idx, s in enumerate(self.joint_states)}

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

        self.joint_transition = [[0.0 for _ in range(self.S)] for _ in range(self.S)]
        for i_prev, prev in enumerate(self.joint_states):
            for i_curr, curr in enumerate(self.joint_states):
                prod = 1.0
                for f in range(self.M):
                    prod *= self.transition_potentials[f][prev[f] * self.K + curr[f]]
                self.joint_transition[i_prev][i_curr] = prod

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

    def compute_marginals(self):
        if self.T == 0:
            return []

        S = self.num_joint_states
        KpowM = self.S

        # Forward pass
        alpha = [[0.0 for _ in range(S)] for _ in range(self.T)]
        o0 = int(self.observation_sequence[0])
        init_prob = 1.0 / KpowM
        for s in range(S):
            alpha[0][s] = init_prob * self.joint_emission[s][o0]
        for t in range(1, self.T):
            ot = int(self.observation_sequence[t])
            for curr in range(S):
                em = self.joint_emission[curr][ot]
                if em == 0.0:
                    alpha[t][curr] = 0.0
                    continue
                ssum = 0.0
                for prev in range(S):
                    tp = self.joint_transition[prev][curr]
                    if tp != 0.0 and alpha[t - 1][prev] != 0.0:
                        ssum += alpha[t - 1][prev] * tp
                alpha[t][curr] = em * ssum

        # Backward pass
        beta = [[0.0 for _ in range(S)] for _ in range(self.T)]
        for s in range(S):
            beta[self.T - 1][s] = 1.0
        for t in range(self.T - 2, -1, -1):
            ot1 = int(self.observation_sequence[t + 1])
            for prev in range(S):
                ssum = 0.0
                for curr in range(S):
                    tp = self.joint_transition[prev][curr]
                    if tp == 0.0:
                        continue
                    em = self.joint_emission[curr][ot1]
                    b = beta[t + 1][curr]
                    if em == 0.0 or b == 0.0:
                        continue
                    ssum += tp * em * b
                beta[t][prev] = ssum

        p_obs = sum(alpha[self.T - 1])
        if p_obs == 0:
            out = []
            for _ in range(self.S):
                out.append([0.0 for _ in range(self.K)])
            return out

        # marginals: gamma = alpha[t][s] * beta[t][s] / p_obs
        result = []
        for t in range(self.T):
            per_factor = [[0.0 for _ in range(self.K)] for _ in range(self.M)]
            for s_idx, joint in enumerate(self.joint_states):
                gamma = (alpha[t][s_idx] * beta[t][s_idx]) / p_obs
                if gamma == 0.0:
                    continue
                for f_idx in range(self.M):
                    val = joint[f_idx]
                    per_factor[f_idx][val] += gamma
            for f in range(self.M):
                total = sum(per_factor[f])
                if total > 0:
                    per_factor[f] = [x / total for x in per_factor[f]]
                result.append(per_factor[f])
        return result

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
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
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
