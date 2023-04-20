import numpy as np
import pandas as pd

class MarkovChains():
    """
    Creates a nxn matrix of binary values, with each value representing if it is possible to go from state i to j. 
    For each value, fit a logistic regression model with the x_j as input and find parameters.
    """
    def __init__(self, data_dir: str) -> None:
        self.data = pd.read_csv(data_dir)

    def create_transition_matrix(self):
        """
        Creates a binary transition matrix of nxn states with 
        1 indicating transition has occurred atleast once
        0 indicated transition has never occurred.
        """

        states = self.data["FAMINC"].unique()
        sequences = self.data["CPSID"].unique()
        matrix = np.zeros(shape=(states.size, states.size))
        for s in sequences:
            seq = self.data[self.data["FAMINC"] == s]
            for i in range(0, seq.size-1):
                j = i+1
                if seq.loc[j, "FAMINC"] > seq.loc[i, "FAMINC"]:
                    matrix[seq.loc[i, "FAMINC"], seq.loc[j, "FAMINC"]] = 1

        return matrix


        


if __name__ == "__main__":
    model = MarkovChains('../../data/cps_processed.csv')
    matrix = model.create_transition_matrix()
    print(matrix)



