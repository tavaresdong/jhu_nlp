import numpy as np

"""
The first order hidden markov model instantiates two simplifying assumptions:
1. the probability of a particular states depends only on its previous state
2. the probability of the output observation depends only on the state that produced
   the output
"""
class HMM(object):
    def __init__(self, nStates, transitionProb=None, initialProb=None, emissionProb=None):
        '''

        :param nStates: N, the number of states
        :param transitionProb: a N * N numpy array storing the transition probabilities between states
        :param initialProb: a vector of length N storing the initial prob.distribution over states
        :param emissionProb: dict of key(state) and value dict of key(observation) and value (probability)
        '''
        self._nStates = nStates
        self._transitionProb = transitionProb
        self._initialProb = initialProb
        self._emissionProb = emissionProb
        if self.ready():
            self.check_constraints()

    def check_constraints(self):
        # 1. check that each row of transition probability sums to one
        probs = np.sum(self._transitionProb, axis=1)
        for prob in probs:
            if abs(prob - 1.0) > 1e-3:
                raise ValueError("Illegal transition probability distribution")

        # 2. check that sum of initial probability is one
        sum_initial = sum(self._initialProb)
        if abs(sum_initial - 1.0) > 1e-3:
            raise ValueError("Illegal initial probability distribution")

    def ready(self):
        if self._transitionProb is None or \
           self._initialProb is None or\
           self._emissionProb is None:
            return False
        return True

    def inialProb(self, stateIndex):
        return self._initialProb[stateIndex]

    def transitionProb(self, oldState, newState):
        return self._transitionProb[oldState][newState]

    def emissionProb(self, stateIndex, observ):
        '''
        Get the emission probability from state to observation
        return 0.0 if the observation was not observed in this state before
        Maybe we can do smoothing here? TODO
        :param stateIndex: integer: index of the state
        :param observ:
        :return:
        '''
        observes = self._emissionProb[stateIndex]
        if observ in observes:
            return observes[observ]
        else:
            return 0.0

    def likelihood(self, observations):
        '''
        The first fundamental problem: given the model, calculate probability for
        the observations
        :return:
        '''
        if not observations:
            raise ValueError("observations must not be empty")
        if not self.ready():
            raise RuntimeError("HMM is not fed/trained with probabilities")

        init = [self.inialProb(i) * self.emissionProb(i, observations[0]) for i in range(self._nStates)]
        forward = [init]
        observation_len = len(observations)
        for t in range(1, observation_len):
            # Calculate the new forward values
            trellis = [0.0] * self._nStates
            prev = forward[-1]
            for s in range(self._nStates):
                # calculate path_sum
                path_sum = 0.0
                for ss in range(self._nStates):
                    path_sum = path_sum + prev[ss] * self.transitionProb(ss, s) * self.emissionProb(s, observations[t])
                trellis[s] = path_sum
            forward.append(trellis)

        finalprob = sum(forward[-1])
        return finalprob

    def decode(self, observations):
        '''
        Given a HMM and a sequence of observations, find the most probable hidden state sequence.
        The Viterbi algorithm
        :param observations:
        :return: best path (list of states) and its probability
        '''
        if not observations:
            raise ValueError("observations must not be empty")
        if not self.ready():
            raise RuntimeError("HMM is not fed/trained with probabilities")

        # Initialize the forward trellis and backpointers
        forward = [[self.inialProb(i) * self.emissionProb(i, observations[0]) for i in range(self._nStates)]]
        backpointers = [[0] * self._nStates]

        observation_len = len(observations)
        for t in range(1, observation_len):
            trellis = [0.0] * self._nStates
            bp = [0] * self._nStates
            prev = forward[-1]
            for s in range(self._nStates):
                aggregate = -1.0
                for ss in range(self._nStates):
                    extended_prob = prev[ss] * self.transitionProb(ss, s) * self.emissionProb(s, observations[t])
                    if extended_prob >= aggregate:
                        aggregate = extended_prob
                        bp[s] = ss
                trellis[s] = aggregate
            forward.append(trellis)
            backpointers.append(bp)

        best_path_prob = max(forward[-1])
        best_path = []
        best_path_node = forward[-1].index(best_path_prob)
        for i in range(observation_len - 1, -1, -1):
            best_path.insert(0, best_path_node)
            best_path_node = backpointers[i][best_path_node]

        return best_path, best_path_prob

def __likelihood_sanitycheck():
    nStates = 3
    #transitionProb = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    transitionProb = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    emissionProb = [{"Red":0.5, "White":0.5}, {"Red":0.4, "White":0.6}, {"Red":0.7, "White":0.3}]
    initialProb = [0.2, 0.4, 0.4]

    hmm = HMM(nStates, transitionProb, initialProb, emissionProb)
    lk1 = hmm.likelihood(["Red", "White", "Red"])

    answer = 0.13022
    print(lk1)
    assert(abs(answer - lk1) < 1e-3)

def __likelihood_illegalArgumentCheck():
    nStates = 3
    #transitionProb = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    transitionProb = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    emissionProb = [{"Red":0.5, "White":0.5}, {"Red":0.4, "White":0.6}, {"Red":0.7, "White":0.3}]
    initialProb = [0.2, 0.4, 0.8]

    try:
        hmm = HMM(nStates, transitionProb, initialProb, emissionProb)
        assert False
    except ValueError as e:
        print(e)
        assert True

def __decode_sanitycheck():
    nStates = 3
    transitionProb = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    emissionProb = [{"Red":0.5, "White":0.5}, {"Red":0.4, "White":0.6}, {"Red":0.7, "White":0.3}]
    initialProb = [0.2, 0.4, 0.4]

    hmm = HMM(nStates, transitionProb, initialProb, emissionProb)
    best_path, best_path_prob = hmm.decode(["Red", "White", "Red"])
    print(best_path)
    print(best_path_prob)
    assert(best_path[0] == 2 and best_path[1] == 2 and best_path[2] == 2)
    answer = 0.0147
    assert(abs(answer - best_path_prob) < 1e-3)

if __name__ == "__main__":
    __likelihood_sanitycheck()
    __likelihood_illegalArgumentCheck()
    __decode_sanitycheck()