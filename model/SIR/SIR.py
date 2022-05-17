import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

# arguments
beta = 1.4247
gamma = 0.14286
S0 = 1 - 1e-6
I0 = 1e-6
INIT = (S0, I0, 0.0)

step = 1.0
start_day = 0
end_days = 70.0


# differential equation
def diff_eqs(INP, t):
    '''The main set of equations'''
    Y = np.zeros((3))
    V = INP
    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y  # For odeint


def main():
    t_range = np.arange(start_day, start_day + end_days, step)
    res = spi.odeint(diff_eqs, INIT, t_range)
    print(res)

    # plot
    plt.figure(figsize=(20, 10))
    plt.plot(res[:, 1], '-r', label='Infectious')
    plt.plot(res[:, 0], '-g', label='Susceptibles')
    plt.plot(res[:, 2], '-k', label='Recovereds')
    plt.legend(loc=0)
    plt.title('SIR_Model.py')
    plt.xlabel('Time')
    plt.ylabel('Infectious Susceptibles and Recovereds')
    plt.xlabel('Time')
    plt.show()


if __name__ == "__main__":
    main()
