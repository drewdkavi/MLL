# cython: profile=True
import numpy as np
cimport numpy as np

cpdef generateRule_cy(
        np.ndarray[double, ndim=2] XS,
        np.ndarray[long, ndim=1] YS,
        long INP_DIM,
        long NUM_CATEGORIES
                   ):

    cdef int j
    cdef int i, dim_index, datapoint_index, count_left, count_right, count_left_best

    cdef int N = len(XS)

    cdef float gini_best = 2
    cdef float l_gini, r_gini, left_gini_impurity, right_gini_impurity, weighted_gini_impurity
    cdef float split_val, split_val_best
    cdef int split_dim, split_dim_best
    cdef np.ndarray[double, ndim=1] left_category_count, right_category_count
    cdef np.ndarray[double, ndim=2] XST
    cdef np.ndarray[long, ndim=1] YST
    cdef np.ndarray[double, ndim=1] x_dp
    cdef long y_dp
    cdef bint changed

    cdef double lss, rss
    cdef np.ndarray[long, ndim=1] sorted_indicies, best_ordering


    for dim_index in range(INP_DIM):

        sorted_indicies = np.argsort(XS[:, dim_index])

        XST = XS[sorted_indicies]
        YST = YS[sorted_indicies]

        left_category_count = np.zeros(NUM_CATEGORIES)
        right_category_count = np.zeros(NUM_CATEGORIES)
        lss = 0
        rss = 0
        changed = False

        for i in range(N):
            y_dp = YST[i]
            right_category_count[y_dp] += 1

        for j in range(NUM_CATEGORIES):
            rss += right_category_count[j] ** 2

        for datapoint_index in range(N - 1):

            # O(1) work in here

            x_dp, y_dp = XST[datapoint_index], YST[datapoint_index]
            split_val = x_dp[dim_index]

            count_left = datapoint_index + 1
            count_right = N - count_left

            lss += 2 * left_category_count[y_dp] + 1
            rss += -2 * right_category_count[y_dp] + 1

            left_category_count[y_dp] += 1
            right_category_count[y_dp] -= 1

            l_gini = lss / (count_left ** 2)
            r_gini = rss / (count_right ** 2)

            left_gini_impurity = 1 - l_gini
            right_gini_impurity = 1 - r_gini

            weighted_gini_impurity = (1 / N) * (count_left * left_gini_impurity + count_right * right_gini_impurity)

            if weighted_gini_impurity < gini_best:
                gini_best = weighted_gini_impurity
                count_left_best = count_left
                split_val_best = split_val
                split_dim_best = dim_index
                best_ordering = sorted_indicies
                if gini_best == 0:
                    break
        if gini_best == 0:
            break


    xs_l, xs_r = np.split(XS[best_ordering], [count_left_best])
    ys_l, ys_r = np.split(YS[best_ordering], [count_left_best])

    return xs_l, ys_l, xs_r, ys_r, split_val_best, split_dim_best

