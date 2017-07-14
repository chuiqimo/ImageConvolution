import numpy as np
class OneConvolution:

    def __init__(self,X):
        self.X = X

    def convolution(self, W):
        N= self.X.shape[0]
        s = W.shape[0]
        m = s//2
        Z = np.zeros(N)

        for i in range(N):
            for j in range(s):
                k = i + j - m
                if k < 0 or k >= N:
                    if W[j] != 0:
                        Z[i] = 0
                        break
                Z[i] = Z[i] + W[j] * self.X[k]

        return Z
