import numpy as np

def models(index, n_obs=10000):

    if index == 1:
        X = np.random.randn(n_obs, 4)

        true = np.zeros((4, 4), dtype='int')

        return X, true

    if index == 2:
        X = np.random.randn(n_obs, 4)
        X[:,1] += X[:,2] # X2 -> X1
        X[:,3] += X[:,2] # X2 -> X3

        true = np.zeros((4, 4), dtype='int')
        true[2,1] = 1
        true[2,3] = 1
        return X, true
    
    if index == 11:
        X = np.random.randn(n_obs, 4)
        X[:,1] += X[:,2]**2 # X2 -> X1
        X[:,3] += -X[:,2]**2 # X2 -> X3

        true = np.zeros((4, 4), dtype='int')
        true[2,1] = 1
        true[2,3] = 1
        return X, true

    if index == 12:
        X = np.random.randn(n_obs, 4)
        X[:,1] *= X[:,2] # X2 -> X1 (mult)
        X[:,3] *= X[:,2] # X2 -> X3 (mult)

        true = np.zeros((4, 4), dtype='int')
        true[2,1] = 1
        true[2,3] = 1
        return X, true

    if index == 3:
        X = np.random.randn(n_obs, 4)
        X[:,2] += X[:,1] + X[:,3] # X1, X3 -> X2

        true = np.zeros((4, 4), dtype='int')
        true[1,2] = 1
        true[3,2] = 1
        return X, true

    if index == 4:
        X = np.random.randn(n_obs, 4)
        X[:,1] += X[:,0] # X0 -> X1
        X[:,2] += X[:,0] # X0 -> X2
        X[:,3] += X[:,1] + X[:,2] # X1, X2 -> X3

        true = np.zeros((4, 4), dtype='int')
        true[0,1] = 1
        true[0,2] = 1
        true[1,3] = 1
        true[2,3] = 1

        return X, true

    if index == 5:
        X = np.random.randn(n_obs, 4)
        X[:,3] += X[:,1] + X[:,2] # X1, X2 -> X3
        X[:,0] += X[:,1] + X[:,2] # X1, X2 -> X0

        true = np.zeros((4, 4), dtype='int')
        true[1,3] = 1
        true[1,0] = 1
        true[2,3] = 1
        true[2,0] = 1

        return X, true

    if index == 6:
        X = np.random.randn(n_obs, 5)
        X[:,4] += X[:,2] + X[:,3] # X2, X3 -> X4
        X[:,1] += X[:,2] + X[:,3] # X2, X3 -> X1
        X[:,0] += X[:,1] + X[:,4] # X1, X4 -> X0
        
        true = np.zeros((5, 5), dtype='int')
        true[3,4] = 1
        true[2,4] = 1
        true[2,1] = 1
        true[3,1] = 1
        true[1,0] = 1
        true[4,0] = 1

        return X, true


    if index == 7:
        X = np.random.randn(n_obs, 4)
        X[:,1] += X[:,0] # X0 -> X1
        X[:,3] += X[:,0] # X0 -> X3
        X[:,2] += 0.5*X[:,0] + 0.5*X[:,1] + 0.5*X[:,3] # X0, X1, X3 -> X2

        true = np.zeros((4, 4), dtype='int')
        true[0,1] = 1
        true[0,2] = 1
        true[0,3] = 1
        true[3,2] = 1
        true[1,2] = 1

        return X, true

    if index == 8:
        X = np.random.randn(n_obs, 5)
        X[:,3] += 2.*X[:,2] #+ X[:,4] 
        X[:,4] += 0.8*X[:,2] + 0.6*X[:,3]
        X[:,1] += 0.5*X[:,2] + 0.5*X[:,3] 
        X[:,0] += 0.5*X[:,1] + 0.6*X[:,4] 

        true = np.zeros((5, 5), dtype='int')
        true[3,4] = 1
        true[2,4] = 1
        true[2,1] = 1
        true[3,1] = 1
        true[1,0] = 1
        true[4,0] = 1
        true[2,3] = 1

        return X, true

    if index == 9:
        X = np.random.randn(n_obs, 15)
        X[:,4] += 0.5*X[:,2] + 0.4*X[:,3]
        X[:,1] += 0.5*X[:,2] + 0.4*X[:,3] 
        X[:,0] += 0.5*X[:,1] + 0.6*X[:,4] 

        true = np.zeros((15, 15), dtype='int')
        true[3,4] = 1
        true[2,4] = 1
        true[2,1] = 1
        true[3,1] = 1
        true[1,0] = 1
        true[4,0] = 1

        return X, true

    if index == 10:
        X = np.random.randn(n_obs, 5)
        # (0, 2), (1, 2), (1, 3), (4, 3)
        X[:,2] += 0.5*X[:,0] + 0.5*X[:,1] # X0, X1 -> X2
        X[:,3] += 0.5*X[:,1] + 0.5*X[:,4] # X1, X4 -> X3

        true = np.zeros((5, 5), dtype='int')
        true[0,2] = 1
        true[1,2] = 1
        true[1,3] = 1
        true[4,3] = 1

        return X, true

    if index == 13:
        X = np.random.randn(n_obs, 2)
        # X[:,1] *= X[:,2]
        # X[:,3] *= X[:,2]

        true = np.zeros((2, 2), dtype='int')
        # true[2,1] = 1
        # true[2,3] = 1
        return X, true

    if index == 14:
        X = np.random.randn(n_obs, 4)
        X[:,3] += .2*X[:,1]
        # X[:,3] *= X[:,2]

        true = np.zeros((4, 4), dtype='int')
        true[1,3] = 1
        # true[2,3] = 1
        return X, true

    if index == 15:
        X = np.random.randn(n_obs, 4)
        for t in range(1, n_obs):
            X[t,1] += 0.9*X[t-1,1]
            X[t,3] += 0.9*X[t-1,3]
        # X[:,3] *= X[:,2]

        true = np.zeros((2, 2), dtype='int')
        # true[0,1] = 1
        # true[2,3] = 1
        return X, true

    if index == 16:
        X = np.random.randn(n_obs, 4)
        # for t in range(1, n_obs):
        #     X[t,2] = (1.+10*t/float(n_obs))*np.random.randn()
        X[:,1]  = 0.2*X[:,2] # deterministic
        X[:,3] += 0.2*X[:,2] # X2 -> X3

        true = np.zeros((4, 4), dtype='int')
        true[2,1] = 1
        true[2,3] = 1
        return X, true

    if index == 17:
        X = np.random.randn(10*n_obs, 4)
        S = (X[:,1] > 0.) | (X[:,3] > 0.) # X1 \lor X3

        true = np.zeros((4, 4), dtype='int')
        X = X[S][:n_obs] 
        # Select only the obs which satisfy X1 and X3
        # Selection bias

        return X, true

    if index == 18:
        X = np.random.randn(n_obs, 4)
        L = np.random.randn(n_obs) # constant -> latent variable

        X[:,1] += 0.5*L # L -> X1
        X[:,3] += 0.6*L # L -> X3
        # Conditioning on the common cause transforms X2 to a L
        # X1 and X3 are not d-separated
        # So CMI(X1, X3 | L) > 0

        true = np.zeros((4, 4), dtype='int')

        return X, true

    if index == 19:
        X = np.random.randn(n_obs, 5)
        X[:,4] += 0.3*X[:,2] + 0.3*X[:,3]
        X[:,1] += 0.3*X[:,2] + 0.3*X[:,3] 
        X[:,0] += 0.3*X[:,1] + 0.3*X[:,4] 

        true = np.zeros((5, 5), dtype='int')
        true[3,4] = 1
        true[2,4] = 1
        true[2,1] = 1
        true[3,1] = 1
        true[1,0] = 1
        true[4,0] = 1

        return X, true

    if index == 20:
        X = np.random.randn(n_obs, 15)
        X[:,4] += 0.3*X[:,2] + 0.3*X[:,3]
        X[:,1] += 0.3*X[:,2] + 0.3*X[:,3] 
        X[:,0] += 0.3*X[:,1] + 0.3*X[:,4] 

        for i in range(5, 15):
            X[:,i] += 0.3*X[:,1] + 0.3*X[:,4] 

        true = np.zeros((15, 15), dtype='int')
        true[3,4] = 1
        true[2,4] = 1
        true[2,1] = 1
        true[3,1] = 1
        true[1,0] = 1
        true[4,0] = 1
        for i in range(5, 15):
            true[1,i] = 1
            true[4,i] = 1

        return X, true

    if index == 21:
        X = np.random.randn(n_obs, 4)
        # for t in range(1, n_obs):
        #     X[t,2] = (1.+10*t/float(n_obs))*np.random.randn()
        X[:,2] += 0.7*X[:,1]
        X[:,3] += 0.7*X[:,2] - 0.49*X[:,1]

        true = np.zeros((4, 4), dtype='int')
        true[1,2] = 1
        true[2,3] = 1
        true[1,3] = 1

        return X, true

    if index == 22:
        X = np.random.randn(n_obs, 2)
        for t in range(n_obs):
            if t % 2 == 0:
                X[t,1] += 0.6*X[t,0]
            else:
                X[t,1] += -0.6*X[t,0]

        true = np.zeros((2, 2), dtype='int')
        true[0,1] = 1

        return X, true

    if index == 23:
        X = np.random.randn(n_obs, 4)

        true = np.zeros((4, 4), dtype='int')

        return X, true

if __name__ == '__main__':

    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.independence_tests.gpdc import GPDC
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.pcmci import PCMCI
    import tigramite.data_processing as pp 
    
    np.random.seed(42)

    n_obs = 250
    verbosity = 0

    alpha_level = 0.05
    # ci_test = ParCorr()
    ci_test = GPDC()
    
    n_repetitions = 100

    # Go through [1, 2, 11, 12, 14, 15, 16, 17, 18]
    for index in [
        # 1,
        # 2,
        # 11,
        # 12,
        14, 
        # 15, 
        # 16, 
        # 17, 
        # 18,
    ]: # 
       
        print("Model = ", index)
        test_decisions = np.zeros(n_repetitions)

        for r in range(n_repetitions):
            # if r % 10 == 0:
            #     print("%d " %r)

            data, graph_true = models(index, n_obs=n_obs)

            #
            # Do conditional independence test
            #

            # Adapt the data indices depending on model
            x = np.expand_dims(data[:,1], axis=1)
            y = np.expand_dims(data[:,3], axis=1)
            z = np.expand_dims(data[:,2], axis=1)

            val, pval = ci_test.run_test_raw(x, y, z=z)

            # Test decision: Reject if pval <= alpha
            test_decisions[r] = (pval <= alpha_level)

            #
            # Causal discovery
            #

            # dataframe = pp.DataFrame(data)
            # pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test)
            # graph = pcmci.run_pcalg_non_timeseries_data()['graph'].squeeze()
            # print(graph)
            
        # Look at the model to see the existing and absent causal links
        # to evaluate true positive rate / false positive rate
        rate = (test_decisions != 0).mean()

        print("\tRate = ", rate)