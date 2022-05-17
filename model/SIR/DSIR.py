import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

try:
    from SIR import SIR
except:
    from .SIR import SIR


class DSIR(SIR):
    """
    Dynamic SIR
    """

    def __init__(self, S0=0.9, I0=0.1, contact_coef=1, beta=0.22, gamma=0.028):
        super().__init__(S0, I0, beta, gamma)
        self.contact_coef = contact_coef

    def contact_adjust(self, I0):
        # contact = 1 / (1 + np.exp((-I0 + self.contact_coef) * 5))
        contact = self.contact_coef
        if contact > 1 + 1e-7:
            raise "contact %f" % contact
        return contact

    def diff_eqs(self, INP, t):
        """
        The main set of equations
        """
        Y = np.zeros((3))
        V = INP
        valid_contact = self.beta * V[0] * V[1] * self.contact_adjust(V[1])
        Y[0] = - valid_contact
        Y[1] = valid_contact - self.gamma * V[1]
        Y[2] = self.gamma * V[1]
        return Y  # For odeint

    def show(self, res, figsize=(15, 10), title="DSIR"):
        super().show(res, figsize, title)


def main():
    sir = DSIR()
    step = 1.0
    start_day = 0
    end_days = 160.0
    t_range = np.arange(start_day, start_day + end_days, step)
    res = sir.predict(t_range)
    print(res)

    sir.show(res)


if __name__ == "__main__":
    main()
