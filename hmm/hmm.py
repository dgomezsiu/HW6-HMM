import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables

        T = len(input_observation_states)
        N = len(self.hidden_states)
        alpha = np.zeros((T, N))
        obs_indices = [self.observation_states_dict[obs] for obs in input_observation_states]

        # Step 2. Calculate probabilities
        # initialize alpha values
        for i in range(N):
            alpha[0, i] = self.prior_p[i] * self.emission_p[i, obs_indices[0]]

        for t in range(1, T):
            for j in range(N):
        # calculate the sum of all paths coming into state j
                alpha[t, j] = np.sum(alpha[t-1, :] * self.transition_p[:, j]) * self.emission_p[j, obs_indices[t]]
 
        # Step 3. Return final probability 
        # sum over the final alpha values to get the probability of the observation sequence
        forward_probability = np.sum(alpha[T-1, :])

        return forward_probability
 


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        T = len(decode_observation_states)
        N = len(self.hidden_states)
        viterbi_table = np.zeros((T, N))
        backpointer_table = np.zeros((T, N), dtype=int)
        
        # map observation sequence to indices
        obs_indices = [self.observation_states_dict[obs] for obs in decode_observation_states]

        # Initialize first column of viterbi table and backpointer
        for i in range(N):
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, obs_indices[0]]
            backpointer_table[0, i] = 0
        
        # Step 2. Calculate Probabilities
        for t in range(1, T):
            for j in range(N):
                (prob, state) = max(
                    (viterbi_table[t-1, i] * self.transition_p[i, j] * self.emission_p[j, obs_indices[t]], i) 
                    for i in range(N))
                viterbi_table[t, j] = prob
                backpointer_table[t, j] = state
        
        # Step 3. Traceback
        best_path_index = np.zeros(T, dtype=int)
        best_path_index[-1] = np.argmax(viterbi_table[T-1, :])
        
        for t in range(T-2, -1, -1):
            best_path_index[t] = backpointer_table[t+1, best_path_index[t+1]]
        
        # Step 4. Return best hidden state sequence
        best_hidden_state_sequence = [self.hidden_states[i] for i in best_path_index]
        
        return best_hidden_state_sequence     
        