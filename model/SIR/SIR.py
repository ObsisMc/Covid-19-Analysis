import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt


class SIR:
    def __init__(self, S0=1 - 1e-6, I0=1e-6, D0=0, beta=0.55, gamma=0.2, dead_rate=0.001):
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0
        self.I0 = I0
        self.D0 = D0
        self.dead_rate = dead_rate
        self.r0 = 1 - self.S0 - self.I0 - self.D0
        assert self.r0 >= -1e-6

    # differential equation
    def diff_eqs(self, INP, t):
        """
        The main set of equations
        """
        Y = np.zeros((4))
        V = INP
        Y[0] = - self.beta * V[0] * V[1]
        Y[1] = self.beta * V[0] * V[1] - (self.gamma + self.dead_rate) * V[1]
        Y[2] = self.gamma * V[1]
        Y[3] = self.dead_rate * V[1]
        return Y  # For odeint

    def predict(self, t_range):
        init_value = (self.S0, self.I0, self.r0, self.D0)
        res = spi.odeint(self.diff_eqs, init_value, t_range)
        return res

    def show(self, res, figsize=(15, 10), title="SIR_Model"):
        # plot
        plt.figure(figsize=figsize)
        plt.plot(res[:, 0], '-g', label='Susceptibles')
        plt.plot(res[:, 1], '-r', label='Infectious')
        plt.plot(res[:, 2], '-k', label='Recovereds')
        plt.plot(res[:, 3], color="purple", label='Death')
        plt.legend(loc=0)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Infectious Susceptibles and Recovereds')
        plt.xlabel('Time')
        plt.show()


def main():
    sir = SIR()
    step = 1.0
    start_day = 0
    end_days = 160.0
    t_range = np.arange(start_day, start_day + end_days, step)
    res = sir.predict(t_range)
    print(res)

    sir.show(res)


if __name__ == "__main__":
    main()
