import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import numpy as np


def get_interval(weights, interval=1):
    assert len(weights.shape) == 2
    num_syn, num_days = weights.shape
    X,Y = [],[]
    for day in range(num_days - interval):
        X.append(weights[:, day])
        Y.append(weights[:, day + interval])

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y

def gen_idxs(dataset_size,num_train,num_test, **kwargs):
    perm_idxs = np.random.RandomState(seed=kwargs.get('seed', 0)).permutation(dataset_size)

    train_idxs = perm_idxs[:num_train]
    test_idxs = perm_idxs[num_train:num_train+num_test]

    return train_idxs,test_idxs

def linear_regress(X, Y,
                   fit_intercept=True,
                   num_total=None,
                   return_idxs=False,
                   **dataset_kwargs):

    if num_total is None:
        num_total = X.shape[0]

    num_train = int(np.ceil(dataset_kwargs.get('train_frac',0.50)*num_total))
    num_test = num_total - num_train

    train_idxs,test_idxs = gen_idxs(dataset_size=X.shape[0],
                                    num_train=num_train,
                                    num_test=num_test,
                                    **dataset_kwargs)

    reg = LinearRegression(fit_intercept=fit_intercept).fit(np.expand_dims(X[train_idxs],axis=-1), Y[train_idxs])
    if return_idxs:
        return reg, train_idxs, test_idxs
    else:
        return reg

def interval_regress(weights,interval,weights_2=None,fit_intercept=True,**dataset_kwargs):
    X,Y = get_interval(weights,interval=interval)
    num_syn = X.shape[0]


    if weights_2 is not None:
        X_2,Y_2 = get_interval(weights_2,interval=interval)
        num_syn = np.min([X.shape[0],X_2.shape[0]])

    reg, train_idxs, test_idxs = linear_regress(X, Y,
                                               fit_intercept=fit_intercept,
                                               num_total=num_syn,
                                               return_idxs=True,
                                               **dataset_kwargs)
    plt.plot(X[test_idxs],Y[test_idxs],'o')
    plt.plot(X[train_idxs],Y[train_idxs],'rx')

    lin = range(weights.shape[1])

    plt.plot(lin,lin*reg.coef_+reg.intercept_,'r')
    plt.title('Weights 1')
    plt.show()

    print('Weights 1 R2',reg.score(np.expand_dims(X[test_idxs],axis=-1),Y[test_idxs]))
    if weights_2 is not None:

        train_idxs_2,test_idxs_2 = gen_idxs(dataset_size=X_2.shape[0],
                                            num_train=len(train_idxs),
                                            num_test=len(test_idxs),**dataset_kwargs)

        plt.plot(X_2[test_idxs_2],Y_2[test_idxs_2],'o')
        plt.plot(X[train_idxs],Y[train_idxs],'rx')

        print('Weights 2 R2: ',reg.score(np.expand_dims(X_2[test_idxs_2],axis=-1),Y_2[test_idxs_2]))
        lin = range(weights.shape[1])

        plt.plot(lin,lin*reg.coef_+reg.intercept_,'r')
        plt.title('Weights 2')

        plt.show()

    return reg.coef_,reg.intercept_


def fit_dx_x(weights):

    num_tp = weights.shape[1]

    X,Y = [],[]
    for i in range(num_tp-1):
        X.append(weights[:,i])
        Y.append(weights[:,i+1]-weights[:,i])

    X = np.concatenate(X,axis=0)
    Y = np.concatenate(Y,axis=0)

    a,b,_,_,_ = linregress(X,Y)

    plt.plot(X,Y,'co',alpha=0.2)

    xs = np.array([0,np.max(X)])
    plt.plot(xs,xs*a+b,'r--')
    plt.show()

def run_kesten(a, b, c, d,
               num_steps,
               first_day,
               seed_val=0,
               death_threshold=0,
               death_val=0):

    assert(death_threshold >= death_val)
    if seed_val is not None:
        np.random.seed(seed_val)

    simulated_weight_trajectory = [first_day]
    num_synapses = first_day.shape[0]

    for t in range(1, num_steps+1):
        new_state = []
        for s in range(num_synapses):
            x_prev_syn = simulated_weight_trajectory[t-1][s]
            if x_prev_syn > death_threshold:
                x_curr_syn = x_prev_syn + np.random.normal(loc=(a*x_prev_syn + c), scale=np.sqrt(b*x_prev_syn + d))
                if x_curr_syn < death_threshold: # clip values below the death threshold to the death value (default: 0)
                    x_curr_syn = death_val
            else:
                # keep 0 values at 0 to represent eliminated synapses
                x_curr_syn = x_prev_syn



            new_state.append(x_curr_syn)
        new_state = np.array(new_state)
        simulated_weight_trajectory.append(new_state)
    simulated_weight_trajectory = np.array(simulated_weight_trajectory)
    return simulated_weight_trajectory
