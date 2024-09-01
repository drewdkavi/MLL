# cython: profile=True
from libc.stdlib cimport rand
import numpy as np
cimport numpy as np



# ----------------------------------------------------------------------------------------------------------------------
# Activation functions

cdef double clipped_sigmoid(double val):
    if val > 500:
        val = 500.0
    if val < -500:
        val = -500.0
    return 1.0 / (1.0 + np.exp(-val))

cdef double relu(double val):
    return max(0.0, val)

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

cdef np.ndarray[double, ndim=1] softmax(
        np.ndarray[double, ndim=1] logits
):
    cdef np.ndarray[double, ndim=1] logits_max, exp_logits, probabilities

    logits_max = np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probabilities = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    return probabilities

# def cross_entropy_loss(predictions, labels):
#     predictions = np.clip(predictions, 1e-15, 1.0 - 1e-15)
#     loss = -np.sum(labels * np.log(predictions)) / predictions.shape[0]
#     return loss
#
# def cross_entropy_gradient(predictions, labels):
#     gradient = (predictions - labels) / predictions.shape[0]
#     return gradient



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
        list layers,
        list biases
        # TODO: pass in activation function as an argument

):
    cdef int L = len(layers)
    cdef int i = 0
    cdef int j, layer_size
    cdef np.ndarray[double, ndim=2] layer
    cdef np.ndarray[double, ndim=1] current_vector = x_dp
    cdef np.ndarray[double, ndim=1] vector, logits

    while i < L - 1:
        layer_size = len(layers[i])
        vector = np.zeros(layer_size)
        layer = layers[i]
        for j in range(layer_size):
            vector[j] = relu( dot(layer[j], current_vector) + biases[i][j] )
        current_vector = vector
        i += 1

    layer_size = len(layers[L-1])
    logits = np.zeros(layer_size)
    for j in range(layer_size):
        logits[j] = dot(layers[L-1][j], current_vector) + biases[i][j]

    return softmax(logits)

cpdef np.ndarray[double, ndim=1] feedforward(
        np.ndarray[double, ndim=1] x_dp,
        list layers,
        list biases
        # TODO: pass in activation function as an argument
):
    return _ff(x_dp, layers, biases)


# cdef np.ndarray[double, ndim=1] _ff_clipped(
#         np.ndarray[double, ndim=1] x_dp,
#         list layers,
#         list biases
#         # TODO: pass in activation function as an argument
#
# ):
#     cdef int L = len(layers)
#     cdef int i = 0
#     cdef int j, layer_size
#     cdef np.ndarray[double, ndim=2] layer
#     cdef np.ndarray[double, ndim=1] current_vector = x_dp
#     cdef np.ndarray[double, ndim=1] vector
#
#     while i < L:
#         layer_size = len(layers[i])
#         vector = np.zeros(layer_size)
#         layer = layers[i]
#         for j in range(layer_size):
#             vector[j] = clipped_sigmoid( dot(layer[j], current_vector) + biases[i][j] )
#         current_vector = vector
#         i += 1
#     return current_vector
#
# cpdef np.ndarray[double, ndim=1] feedforward_clipped(
#         np.ndarray[double, ndim=1] x_dp,
#         list layers,
#         list biases
#         # TODO: pass in activation function as an argument
# ):
#     return _ff_clipped(x_dp, layers, biases)

cdef _ff_all(
        np.ndarray[double, ndim=1] x_dp,
        list layers,
        list biases
        # TODO: pass in activation function as an argument
):
    cdef int L = len(layers)
    cdef int i = 0
    cdef int j, layer_size
    cdef np.ndarray[double, ndim=2] layer
    cdef np.ndarray[double, ndim=1] current_vector = x_dp
    cdef np.ndarray[double, ndim=1] vector, logits
    cdef list activations = []

    while i < L-1:
        layer_size = len(layers[i])
        vector = np.zeros(layer_size)
        layer = layers[i]
        for j in range(layer_size):
            vector[j] = relu( dot(layer[j], current_vector) + biases[i][j] )
        current_vector = vector
        activations.append(current_vector)
        i += 1

    layer_size = len(layers[L - 1])
    logits = np.zeros(layer_size)
    for j in range(layer_size):
        logits[j] = dot(layers[L - 1][j], current_vector) + biases[i][j]

    return activations, softmax(logits)

cpdef feedforward_all(
        np.ndarray[double, ndim=1] x_dp,
        list layers,
        list biases
        # TODO: pass in activation function as an argument
):
    return _ff_all(x_dp, layers, biases)


# cdef list _ff_all_clipped(
#         np.ndarray[double, ndim=1] x_dp,
#         list layers,
#         list biases
#         # TODO: pass in activation function as an argument
# ):
#     cdef int L = len(layers)
#     cdef int i = 0
#     cdef int j, layer_size
#     cdef np.ndarray[double, ndim=2] layer
#     cdef np.ndarray[double, ndim=1] current_vector = x_dp
#     cdef np.ndarray[double, ndim=1] vector
#     cdef list activations = []
#
#     while i < L:
#         layer_size = len(layers[i])
#         vector = np.zeros(layer_size)
#         layer = layers[i]
#         for j in range(layer_size):
#             vector[j] = clipped_sigmoid( dot(layer[j], current_vector) + biases[i][j] )
#         current_vector = vector
#         activations.append(current_vector)
#         i += 1
#     return activations
#
# cpdef list feedforward_all_clipped(
#         np.ndarray[double, ndim=1] x_dp,
#         list layers,
#         list biases
#         # TODO: pass in activation function as an argument
# ):
#     return _ff_all_clipped(x_dp, layers, biases)




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
   
    Now we see that we want to adjust the weights W_Q, W_L, ..., W_1 s.t. we minimise C(f_Q(x))
    Consider:
        dC/dW_Q = 2 * ( s(Wv) - y ) * s(Wv) * (1 - s(Wv)) * vT
        where v  = f_L(x)

    
'''

cpdef _get_derivative(
        np.ndarray[double, ndim=1] w, # ff_activation_current = s(Wv)
        np.ndarray[double, ndim=1] v, # ff_activation_prev = vT
        np.ndarray[double, ndim=1] y
):
    cdef np.ndarray[double, ndim=1] temp = (2 * (w - y) * w * (1 - w))
    return np.outer(temp, v)

cpdef _bp_single(
    list weights,
    list activations,
    np.ndarray[double, ndim=1] y,
    np.ndarray[double, ndim=1] xdp,
    np.ndarray[double, ndim=1] output_activations
):
    cdef np.ndarray[double, ndim=1] layer_error
    cdef list deltas = []
    cdef list biases = []
    cdef np.ndarray[double, ndim=1] _activation_curr, _activation_prev, temp
    cdef int layer_index = len(activations)

    # Layer error is the derivative of the Loss function at each layer, w.r.t. zL, where zL are the pre-activation values

    layer_error = output_activations - y
    while layer_index > 0:
        _activation_prev = activations[layer_index-1]

        deltas.append(
            np.outer(layer_error, _activation_prev)
        )
        biases.append(
            layer_error
        )

        # print(weights[layer_index].T, layer_error)
        temp = weights[layer_index].T @ layer_error

        layer_error = temp * np.heaviside(_activation_prev, 0)
        layer_index -= 1

    # have to handle the final layer separately because of how I stupidly structured this
    deltas.append(
        np.outer(layer_error, xdp)
    )
    biases.append(
        layer_error
    )

    deltas.reverse()
    biases.reverse()

    return deltas, biases


