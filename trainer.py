import numpy as np
from gaussian_process import MFGP, SFGP

def train_sfgp(name):
    # 1) load training data
    sifi = np.loadtxt("Data/" + name + "_sifi_train.csv", skiprows=1, delimiter=',')
    labels = ['mu_sf', 's^2_sf', 'L_sf', 'noise_sf']

    # 2) reshape training data from CSV to valid format
    X = data[:, 0:2].reshape(-1, 2)  # columns 1 and 2 are (x,y) points
    y = data[:, 2].reshape(-1, 1)  # column 3 is f(x,y)

    # 3) initialize model
    len_sf = 0.1
    model = SFGP(X, y, len_sf)

    # 4) train model and display results
    model.train()
    eh = np.exp(model.hyp)
    for i in range(len(labels)):
        print(labels[i] + ' : ' + str(eh[i]))

    # 5) save hyperparameters if training was a success
    valid = input("Save single-fidelity hyperparameters?")
    if valid.lower() == "y":
        np.savetxt("Data/" + name + "_sf_hyp.csv", model.hyp, delimiter=',')


def train_mfgp(name):
    # 1) load training data
    lofi = np.loadtxt("Data/" + name + "_lofi_train.csv", skiprows=1, delimiter=',')
    hifi = np.loadtxt("Data/" + name + "_hifi_train.csv", skiprows=1, delimiter=',')
    labels = ['mu_lo', 's^2_lo', 'L_lo', 'mu_hi', 's^2_hi', 'L_hi', 'rho', 'noise_lo', 'noise_hi']

    # 2) reshape training data from CSV to valid format
    X_L = lofi[:, 0:2].reshape(-1, 2)  # columns 1 and 2 are (x,y) points
    y_L = lofi[:, 2].reshape(-1, 1)  # column 3 is f(x,y)
    X_H = hifi[:, 0:2].reshape(-1, 2)
    y_H = hifi[:, 2].reshape(-1, 1)

    # 3) initialize model
    len_L = 0.2
    len_H = 0.1
    model = MFGP(X_L, y_L, X_H, y_H, len_L, len_H)

    # 4) train model and display results
    model.train()
    eh = np.exp(model.hyp)
    for i in range(len(labels)):
        print(labels[i] + ' : ' + str(eh[i]))

    # 5) save hyperparameters if training was a success
    valid = input("Save multi-fidelity hyperparameters?")
    if valid.lower() == "y":
        np.savetxt("Data/" + name + "_mf_hyp.csv", model.hyp, delimiter=',')


if __name__ == "__main__":

    np.random.seed(1234)
    name = "diag"
    train_mfgp(name)
