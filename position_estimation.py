from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer


def load_train_data():
    path = 'dataset/development.csv'
    df = pd.read_csv(path, low_memory=False)

    y_train = df[['x', 'y']]
    X_train = df.drop(['x', 'y'], axis=1)

    return X_train, y_train, df


def load_test_data():
    path = 'dataset/evaluation.csv'
    df = pd.read_csv(path, low_memory=False)

    ids = df['Id']
    X_test = df.drop('Id', axis=1)

    return ids, X_test


def save_csv(ids, preds):
    e = [[i, p] for (i, p) in zip(ids, preds)]
    se = sorted(e, key=lambda x: x[0])

    with open('outputs/output.csv', 'w') as file:
        file.write('Id,Predicted\n')

        for r in se:
            file.write(f'{int(r[0])},{r[1][0]}|{r[1][1]}\n')


# Compute the average euclidean distance
def avg_dist(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Matrices must have the same shape"

    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))


def drop_tmax_rms(x):
    return x.drop(columns=['tmax[0]',
                           'tmax[1]',
                           'tmax[2]',
                           'tmax[3]',
                           'tmax[4]',
                           'tmax[5]',
                           'tmax[6]',
                           'tmax[7]',
                           'tmax[8]',
                           'tmax[9]',
                           'tmax[10]',
                           'tmax[11]',
                           'tmax[12]',
                           'tmax[13]',
                           'tmax[14]',
                           'tmax[15]',
                           'tmax[16]',
                           'tmax[17]',
                           'rms[0]',
                           'rms[1]',
                           'rms[2]',
                           'rms[3]',
                           'rms[4]',
                           'rms[5]',
                           'rms[6]',
                           'rms[7]',
                           'rms[8]',
                           'rms[9]',
                           'rms[10]',
                           'rms[11]',
                           'rms[12]',
                           'rms[13]',
                           'rms[14]',
                           'rms[15]',
                           'rms[16]',
                           'rms[17]'
    ])


def dropnoise(x):
    return x.drop(columns=['pmax[0]',
                           'negpmax[0]',
                           'area[0]',
                           'tmax[0]',
                           'rms[0]',
                           'pmax[7]',
                           'negpmax[7]',
                           'area[7]',
                           'tmax[7]',
                           'rms[7]',
                           'pmax[12]',
                           'negpmax[12]',
                           'area[12]',
                           'tmax[12]',
                           'rms[12]',
                           'pmax[15]',
                           'negpmax[15]',
                           'area[15]',
                           'tmax[15]',
                           'rms[15]',
                           'pmax[16]',
                           'negpmax[16]',
                           'area[16]',
                           'tmax[16]',
                           'rms[16]',
                           'pmax[17]',
                           'negpmax[17]',
                           'area[17]',
                           'tmax[17]',
                           'rms[17]',
                           ])


def addrangemax(x):
    rangemax_0 = x.loc[:,'pmax[0]']-x.loc[:,'negpmax[0]']
    rangemax_1 = x.loc[:,'pmax[1]']-x.loc[:,'negpmax[1]']
    rangemax_2 = x.loc[:,'pmax[2]']-x.loc[:,'negpmax[2]']
    rangemax_3 = x.loc[:,'pmax[3]']-x.loc[:,'negpmax[3]']
    rangemax_4 = x.loc[:,'pmax[4]']-x.loc[:,'negpmax[4]']
    rangemax_5 = x.loc[:,'pmax[5]']-x.loc[:,'negpmax[5]']
    rangemax_6 = x.loc[:,'pmax[6]']-x.loc[:,'negpmax[6]']
    rangemax_7 = x.loc[:,'pmax[7]']-x.loc[:,'negpmax[7]']
    rangemax_8 = x.loc[:,'pmax[8]']-x.loc[:,'negpmax[8]']
    rangemax_9 = x.loc[:,'pmax[9]']-x.loc[:,'negpmax[9]']
    rangemax_10 = x.loc[:,'pmax[10]']-x.loc[:,'negpmax[10]']
    rangemax_11 = x.loc[:,'pmax[11]']-x.loc[:,'negpmax[11]']
    rangemax_12 = x.loc[:,'pmax[12]']-x.loc[:,'negpmax[12]']
    rangemax_13 = x.loc[:,'pmax[13]']-x.loc[:,'negpmax[13]']
    rangemax_14 = x.loc[:,'pmax[14]']-x.loc[:,'negpmax[14]']
    rangemax_15 = x.loc[:,'pmax[15]']-x.loc[:,'negpmax[15]']
    rangemax_16 = x.loc[:,'pmax[16]']-x.loc[:,'negpmax[16]']
    rangemax_17 = x.loc[:,'pmax[17]']-x.loc[:,'negpmax[17]']
    x.insert(0, 'rangemax[0]', rangemax_0)
    x.insert(1, 'rangemax[1]', rangemax_1)
    x.insert(2, 'rangemax[2]', rangemax_2)
    x.insert(3, 'rangemax[3]', rangemax_3)
    x.insert(4, 'rangemax[4]', rangemax_4)
    x.insert(5, 'rangemax[5]', rangemax_5)
    x.insert(6, 'rangemax[6]', rangemax_6)
    x.insert(7, 'rangemax[7]', rangemax_7)
    x.insert(8, 'rangemax[8]', rangemax_8)
    x.insert(9, 'rangemax[9]', rangemax_9)
    x.insert(10, 'rangemax[10]', rangemax_10)
    x.insert(11, 'rangemax[11]', rangemax_11)
    x.insert(12, 'rangemax[12]', rangemax_12)
    x.insert(13, 'rangemax[13]', rangemax_13)
    x.insert(14, 'rangemax[14]', rangemax_14)
    x.insert(15, 'rangemax[15]', rangemax_15)
    x.insert(16, 'rangemax[16]', rangemax_16)
    x.insert(17, 'rangemax[17]', rangemax_17)
    return x


def plot_pmax_vs_x():
    X, y, df = load_train_data()

    # Create a 3 by 6 grid of subplots
    fig, axs = plt.subplots(3, 6, figsize=(12, 8))

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # Loop through each subplot and customize as needed
    for i in range(18):
        area = f'pmax[{i}]'
        axs[i].scatter(df[area], df['x'], s=1)
        axs[i].set_xlim(0, 150)  # Set x-axis limits 30-75
        axs[i].set_ylim(200, 600)  # Set y-axis limits
        axs[i].set_xlabel(area)
        axs[i].set_ylabel('x')

    fig.suptitle('Relationship between pmax and x')

    # Adjust layout to prevent subplot titles from overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_pmax_vs_y():
    X, y, df = load_train_data()

    # Create a 3 by 6 grid of subplots
    fig, axs = plt.subplots(3, 6, figsize=(12, 8))

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # Loop through each subplot and customize as needed
    for i in range(18):
        area = f'pmax[{i}]'
        axs[i].scatter(df[area], df['y'], s=1)
        axs[i].set_xlim(0, 150)  # Set x-axis limits 30-75
        axs[i].set_ylim(200, 600)  # Set y-axis limits
        axs[i].set_xlabel(area)
        axs[i].set_ylabel('x')

    fig.suptitle('Relationship between pmax and y')

    # Adjust layout to prevent subplot titles from overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_tpoints():
    X, y, df = load_train_data()

    plt.scatter(y['x'], y['y'], s=1)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training point position')
    plt.show()


def do_baseline():
    X, y, _ = load_train_data()
    ids, X_eval = load_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    print(f"{avg_dist(y_test, y_pred)}")

    y_pred = rfr.predict(X_eval)
    save_csv(ids, y_pred)


def notmaxrms():
    X, y, _ = load_train_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train_n = drop_tmax_rms(X_train)
    X_test_n = drop_tmax_rms(X_test)

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    print(f"{avg_dist(y_test, y_pred)}")

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train_n, y_train)
    y_pred = rfr.predict(X_test_n)
    print(f"{avg_dist(y_test, y_pred)}")


def yesnoise():
    X, y, _ = load_train_data()
    X = drop_tmax_rms(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train_n = dropnoise(X_train)
    X_test_n = dropnoise(X_test)

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    print(f"{avg_dist(y_test, y_pred)}")

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train_n, y_train)
    y_pred = rfr.predict(X_test_n)
    print(f"{avg_dist(y_test, y_pred)}")


def yesrangemax():
    X, y, _ = load_train_data()
    X = drop_tmax_rms(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train_n = addrangemax(X_train)
    feature_names = X_train_n.columns
    X_test_n = addrangemax(X_test)

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    print(f"{avg_dist(y_test, y_pred)}")

    rfr = RandomForestRegressor(n_jobs=-1)
    rfr.fit(X_train_n, y_train)
    y_pred = rfr.predict(X_test_n)
    print(f"{avg_dist(y_test, y_pred)}")

    ifs = sorted(zip(feature_names, rfr.feature_importances_), key=lambda x: x[1], reverse=True)
    for i in ifs:
        print(i)


def grid1():
    X, y, _ = load_train_data()
    X = addrangemax(drop_tmax_rms(X))

    param_grid = {
        "n_estimators": [100],
        "max_features": ["sqrt", 10, 20, 30, 40, None],
        "max_depth": [10, 20, None],
        "n_jobs": [-1]
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='r2', verbose=3, n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


def grid2():
    X, y, _ = load_train_data()
    X = addrangemax(drop_tmax_rms(X))

    param_grid = {
        "n_estimators": [200],
        "max_features": ["sqrt", 10, 20, 30],
        "max_depth": [None],
        "n_jobs": [-1]
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='r2', verbose=3, n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


def gridnest():
    X, y, _ = load_train_data()
    X = addrangemax(drop_tmax_rms(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    n = [10, 20, 50, 100, 200, 300, 600, 1000]
    d = []
    t = []
    for nn in n:
        start = time()
        rfr = RandomForestRegressor(n_estimators=nn, max_features=10, max_depth=None, n_jobs=-1)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        end = time()
        d.append(avg_dist(y_test, y_pred))
        t.append(end - start)
        print(f"{d[-1]} ({t[-1]})")

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    axs = axs.flatten()

    axs[0].plot(n, d, 'bo-', linewidth=2, markersize=8)
    axs[0].set_xlabel('Number of estimators')
    axs[0].set_ylabel('Average distance (um)')
    axs[0].set_title('')

    axs[1].plot(n, t, 'bo-', linewidth=2, markersize=8)
    axs[1].set_xlabel('Number of estimators')
    axs[1].set_ylabel('Training time (s)')
    axs[1].set_title('')

    fig.suptitle('Average distance and training time for different number of estimators.')

    plt.tight_layout()

    plt.show()


def final():
    X, y, _ = load_train_data()
    ids, X_eval = load_test_data()

    rfr = RandomForestRegressor(n_estimators=1000, max_features=10, max_depth=None, n_jobs=-1)
    rfr.fit(X, y)
    y_pred = rfr.predict(X)

    save_csv(ids, y_pred)


def main():
    # Regressor used on the leaderboard
    final()

    # Support code used for the report
    plot_pmax_vs_x()
    plot_pmax_vs_y()
    plot_tpoints()
    do_baseline()
    notmaxrms()
    yesnoise()
    yesrangemax()
    grid1()
    grid2()
    gridnest()


if __name__ == '__main__':
    main()
