class Sigmoid():

    def forward(self, x):
        self.x = x
        pos = np.nonzero(x>=0)
        neg = np.nonzero(x<0)
        y = np.zeros_like(x)
        y[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        y[neg] = np.exp(x[neg]) / (1.0 + np.exp(x[neg]))
        # y = 1.0 / (1.0 + np.exp(-x))
        self.y = y
        return self.y
