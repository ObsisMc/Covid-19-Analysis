import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


def read_data(path):
    with open(path, "r") as f:
        json_data = json.load(f)
        data = pd.DataFrame(json_data["data"])
    return data


def show_plot(data, column_name, end_day, start_day=0):
    column = data.loc[:, column_name].iloc[start_day:]
    plt.plot(column)
    plt.show()
    print("%s from %s to %s" % (column_name, data.loc[start_day, "dateId"], data.loc[end_day - 1, "dateId"]))


def init_params(city: str, data, start_day=0, pop_density_path="../../dataset/population_density.csv"):
    populations_density = pd.read_csv(pop_density_path)

    # SH as base
    base_pop = 24870895 / 15
    base_density = 2640
    all_num = base_pop * populations_density[populations_density.Province == city].iloc[0, 1] / base_density

    I0_num = data.loc[start_day, "currentConfirmedCount"]
    D0_num = data.loc[start_day, "deadCount"]
    r0_num = data.loc[start_day, "curedCount"]
    S0_num = all_num - I0_num - r0_num - D0_num
    assert S0_num > 0

    S0 = S0_num / all_num
    I0 = I0_num / all_num
    D0 = D0_num / all_num
    r0 = r0_num / all_num
    return all_num, [S0_num, I0_num, D0_num, r0_num], [S0, I0, D0, r0]


def show_predict(pred,data, all_num, end_day, start_day):

    pre_infected = pred[:, 1] * all_num
    gt_infected = data.loc[start_day:end_day - 1, "currentConfirmedCount"].reset_index(drop=True)
    plt.plot(pre_infected, '-g', label="predict")
    plt.plot(gt_infected, '-r', label="actual")
    plt.show()
    print("Average error:", eval(pre_infected, gt_infected))
    return pred


def eval(pred, test):
    diff = np.abs(pred - test)
    return np.sum(diff) / diff.shape[0]