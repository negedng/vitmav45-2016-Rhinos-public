import numpy as np

def train_validate_test(data, train_rate, validate_rate,test_rate):
    """
    Split the data into 3 array
    Inputs:
        data
        [training rate, 
        validate rate, 
        test rate] the sum of the 3 must be 1
    Outputs:
        [training data,
        validate data,
        test data]
    """
    if train_rate+validate_rate+test_rate!=1:
        raise TypeError('sum of the rates must be 1')
    tr_set = []
    val_set = []
    ts_set = []
    for line in data:
        #random number 0 < *tr_example* < tr_rate < *val_example < tr_rate+val_rate < test_example <sum==1
        r_num = np.random.random()
        if r_num-train_rate<0:
            tr_set.append(line)
        else:
            if r_num-train_rate-validate_rate<0:
                val_set.append(line)
            else:
                if r_num-train_rate-validate_rate-test_rate<0:
                    ts_set.append(line)
    if len(tr_set)==0 or len(val_set)==0 or len(ts_set)==0:
        print('One of the datasets has zero element. Use more sample, different rates or try again.')
    return [tr_set, val_set, ts_set]

def normalize_by_column(data, columns):
    """Normalize the data where row 1
    Inputs:
        data   [[],[],[],...]
        colums [ 0, 1, 1,...]
    Output:
        normalized data [[],[],[],...]
        colums [ 0, 1, 1,...]
        average[0.,0.,1.,...]
        std    [0.,0.,1.,...]
    """
    if np.shape(data)[-1]!=np.shape(columns)[-1]:
        error_message = 'input shape error '+str(np.shape(data)[-1])+'!='+str(np.shape(columns)[-1])
        raise TypeError(error_message)
    a = np.array(data)
    column_av = np.mean(a, axis=0)
    column_std = np.std(a,axis=0)
    
    new_matrix = (a-column_av[np.newaxis,:])/column_std[np.newaxis,:]
    
    b = np.array(columns)
    #a value where b is 0 and new_value where b is 1
    ret = (a*(1-b[np.newaxis,:]))+(new_matrix*(b[np.newaxis,:]))
    return [ret,b,column_av,column_std]
