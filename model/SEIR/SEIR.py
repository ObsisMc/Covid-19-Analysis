import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt


class SEIR:
    def __init__(self, S0=0.9, E0=0, I0=0.1, mu=0.1, beta=0.9, gamma=0.05):
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.r0 = 1 - self.S0 - self.E0 - self.I0
        assert self.r0 >= -1e-6

    # differential equation
    def diff_eqs(self, INP, t):
        """
        The main set of equations
        """
        Y = np.zeros((4))
        V = INP
        Y[0] = - self.mu * V[0] * V[2]
        Y[1] = self.mu * V[0] * V[2] - self.beta * V[1]
        Y[2] = self.beta * V[1] - self.gamma * V[2]
        Y[3] = self.gamma * V[2]
        return Y  # For odeint

    def predict(self, t_range):
        init_value = (self.S0, self.E0, self.I0, self.r0)
        res = spi.odeint(self.diff_eqs, init_value, t_range)
        return res

    def show(self, res, figsize=(15, 10), title="SEIR"):
        # plot
        plt.figure(figsize=figsize)
        plt.plot(res[:, 0], '-g', label='Susceptibles')
        plt.plot(res[:, 1], '-y', label='Exposeds')
        plt.plot(res[:, 2], '-r', label='Infectious')
        plt.plot(res[:, 3], '-k', label='Recovereds')
        plt.legend(loc=0)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Infectious Susceptibles and Recovereds')
        plt.xlabel('Time')
        plt.show()


def main():
    sir = SEIR()
    step = 1.0
    start_day = 0
    end_days = 160.0
    t_range = np.arange(start_day, start_day + end_days, step)
    res = sir.predict(t_range)
    print(res)

    sir.show(res)


if __name__ == "__main__":
    main()
