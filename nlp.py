# =========================
# HMM model definition
# =========================

states = ["PRN", "MD", "V", "N", "DET"]

start_prob = {
    "PRN": 0.9,
    "MD": 0.0,
    "V": 0.0,
    "N": 0.1,
    "DET": 0.8
}

transition_prob = {
    "PRN": {"PRN": 0.1, "MD": 0.6, "V": 0.3, "N": 0.0, "DET": 0.0},
    "MD":  {"PRN": 0.0, "MD": 0.1, "V": 0.8, "N": 0.1, "DET": 0.1},
    "V":   {"PRN": 0.2, "MD": 0.1, "V": 0.2, "N": 0.5, "DET": 0.0},
    "N":   {"PRN": 0.1, "MD": 0.4, "V": 0.5, "N": 0.0, "DET": 0.0},
    "DET": {"PRN": 0.01,"MD": 0.0, "V": 0.03,"N": 0.9, "DET": 0.0},
}

emission_prob = {
    "the": {"PRN": 1.0, "DET": 1.0},
    "dog": {"PRN": 0.1, "N": 0.7},
    "can": {"MD": 0.7, "V": 0.1, "N": 0.2},
    "run": {"V": 0.6, "N": 0.4}
}
def viterbi(words):
    V = [{}]
    path = {}

    # Initialization
    for s in states:
        emit = emission_prob.get(words[0], {}).get(s, 0)
        V[0][s] = start_prob.get(s, 0) * emit
        path[s] = [s]

    # Recursion
    for t in range(1, len(words)):
        V.append({})
        new_path = {}

        for curr_state in states:
            emit = emission_prob.get(words[t], {}).get(curr_state, 0)
            if emit == 0:
                V[t][curr_state] = 0
                continue

            (prob, prev_state) = max(
                (
                    V[t-1][prev] *
                    transition_prob[prev].get(curr_state, 0) *
                    emit,
                    prev
                )
                for prev in states
            )

            V[t][curr_state] = prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path

    # Termination
    final_state = max(V[-1], key=V[-1].get)
    return path[final_state], V[-1][final_state]
def beam_search(words, beam_width=2):
    beams = [([], 1.0)]

    # First word
    new_beams = []
    for s in states:
        emit = emission_prob.get(words[0], {}).get(s, 0)
        if emit > 0:
            score = start_prob.get(s, 0) * emit
            new_beams.append(([s], score))

    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    # Remaining words
    for t in range(1, len(words)):
        candidates = []
        for seq, score in beams:
            prev_state = seq[-1]
            for curr_state in states:
                emit = emission_prob.get(words[t], {}).get(curr_state, 0)
                trans = transition_prob[prev_state].get(curr_state, 0)
                if emit > 0 and trans > 0:
                    new_score = score * trans * emit
                    candidates.append((seq + [curr_state], new_score))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beams[0]
sentence = ["the", "dog", "can", "run"]

tags_viterbi, prob_viterbi = viterbi(sentence)
tags_beam, prob_beam = beam_search(sentence, beam_width=2)

print("Viterbi result:")
print(list(zip(sentence, tags_viterbi)))
print("Probability:", prob_viterbi)

print("\nBeam Search result:")
print(list(zip(sentence, tags_beam)))
print("Probability:", prob_beam)
