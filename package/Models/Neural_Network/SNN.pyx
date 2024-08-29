# cython: profile=True
from libc.stdlib cimport rand
import numpy as np
cimport numpy as np



# ----------------------------------------------------------------------------------------------------------------------
# Activation functions

cdef double sigmoid(double val):
    return 1.0 / (1.0 + np.exp(-val))

cdef double relu(double val):
    return np.max((0, val))

cdef double leaky_relu(double val):
    cdef double alpha = 0.05
    if val >= 0:
        return val
    else:
        return alpha*val
# ----------------------------------------------------------------------------------------------------------------------
# Helpers:
### --- Alias np.dot to dot ---------------------------------------------------------------------------------------- ###

cdef double dot(
        np.ndarray[double] w,
        np.ndarray[double] x
):
    # TODO: Specify the return type of dot 'out' - param. for possible better performance?
    return np.dot(w, x)

### ----- Errors --------------------------------------------------------------------------------------------------- ###

cdef double mse(
        np.ndarray[double] w,
        np.ndarray[double] x
):
    # We assume that w, x are of the same length, & use np vectorize operations here.
    return np.mean((w-x)**2)

# TODO: Hinge Loss, ... many others i think - maybe backprop only applies to mse, maybe i dont need any of these?


# ----------------------------------------------------------------------------------------------------------------------

'''
How does feed-forward work?

For each datapoint in training set:
    
    Put each x_dp into the input layer of the SNN - L0, then for L1 we get L1.i = linear. comb. of x_dp, ...
    Then the activation function takes this, ...
    And so on...
    
'''

cdef np.ndarray[double, ndim=1] _ff(
        np.ndarray[double, ndim=1] x_dp,
        list layers
        # TODO: pass in activation function as an argument

):
    cdef int L = len(layers)
    cdef int i = 0
    cdef int j, layer_size
    cdef np.ndarray[double, ndim=2] layer
    cdef np.ndarray[double, ndim=1] current_vector = x_dp
    cdef np.ndarray[double, ndim=1] vector

    while i < L:
        layer_size = len(layers[i])
        vector = np.zeros(layer_size)
        layer = layers[i]
        for j in range(layer_size):
            vector[j] = sigmoid(dot(layer[j], current_vector))
        current_vector = vector
        i += 1
    return current_vector

cpdef np.ndarray[double, ndim=1] feedforward(
        np.ndarray[double, ndim=1] x_dp,
        list layers
        # TODO: pass in activation function as an argument
):
    return _ff(x_dp, layers)

cdef list _ff_all(
        np.ndarray[double, ndim=1] x_dp,
        list layers
        # TODO: pass in activation function as an argument

):
    cdef int L = len(layers)
    cdef int i = 0
    cdef int j, layer_size
    cdef np.ndarray[double, ndim=2] layer
    cdef np.ndarray[double, ndim=1] current_vector = x_dp
    cdef np.ndarray[double, ndim=1] vector
    cdef list activations = []

    while i < L:
        layer_size = len(layers[i])
        vector = np.zeros(layer_size)
        layer = layers[i]
        for j in range(layer_size):
            vector[j] = sigmoid(dot(layer[j], current_vector))
        current_vector = vector
        activations.append(current_vector)
        i += 1
    return activations

# ----------------------------------------------------------------------------------------------------------------------

'''
Backpropagation <--<--

How does this work:
    
    We will just take MSE as the cost function
    Here we get then that for the k-th data-point
    If the output layer of a k-input is f(k), and k-dp is truly labelled by y(k)
    then, 
    
    C(k) = SUM [ (f(k) - y(k))^2 ] <-  note y(k) is 'constant', want to see how C(k) varies based on f(k) - the SNN
    note that in a sense 
    f(k) = g_L( ... g_1(g_0(k)) .. ) where L is the number of hidden layers
    and g_j is a function that takes from one layer to the next
    g_0(k)_i = sig(dot(layers[0][i], k)
                        ^^^^^^^^^^^
                          weights
    if we let k_1 = g_0(k)
    then k_2 = g_1(k_1), k_3 = g_2(k_2), and so on ...
    
    let W_i be the matrix of weights of the i-th layer
    now consider g_L is exactly parameterized by the weights of the output layer
    
    and we have k_L+1 = sig( WL . k_L )
    now C(k) = sum ( [sig(WL . k_L) - y(k)]^2 ) & t.f. we get 
    dC/dWL = ... 
    
  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 

    Let's tidy up some of the notation, let the activation function be assumed to be sigmoid, and denote it by s(x). 
    Let the weight matrix of each layer i be W_i, here a layer can be any of the hidden or output layers
    
    spps. we have layers 1, ... L hidden layers, an input layer 0, and an output layer indexed by Q
    then let the 'feed-forward value' at each layer i be f_i(x) where x is the input data point
    
    So f_0(x) = x, and f_Q(x) is the output of the SNN,
    
    Consider then that f_Q(x) = s( W_Q * s( W_L * s( ...   s( W_1 * x )   ... )))
    & let C(y_predicted) be the cost function, here we take it to MSE
   
    
    
    

    
'''



cdef _bp(

):
    pass