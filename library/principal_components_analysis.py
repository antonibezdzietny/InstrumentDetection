import numpy as np
import matplotlib.pyplot as plt



def stack_cast_matrix(eigen_pair: list, shape_bound: int):
    """
    Stack in sequence horizontally (column wise) first rows from `eigen_pair`
    number of rows limited by `shape_bound`.

    Args:
        eigen_pair (list): Eigen value and eigen vector sorted by eigen value.
        shape_bound (int): Limits of shape.

    Returns:
        cast_matrix (np.ndarray): Cast PCA array
    """
    cast_matrix = np.zeros((len(eigen_pair), shape_bound))
    for i in range(shape_bound):
        cast_matrix[:,i] = eigen_pair[i][1]
    return cast_matrix


def pca_cast_matrix(X_std: np.ndarray, var_bound: float = 0.8, shape_bound: int = None, return_var: bool = False):
    """
    Function estimate PCA cast_matrix.

    Args:
        X_std (np.ndarray): Data 
        var_bound (float, optional): Variance bound in range `0...1`. Defaults to 0.8.
        shape_bound (int, optional): Shape bound (transform shape). Defaults to None.
        return_var (bool, optional): If true function return `cast_matrix`, [`var_exp`, `cum_var_exp`]. Defaults to False.

    Returns:
        cast_matrix (np.ndarray): PCA casting matrix
    """
    
    # EVD
    cov_mat = np.cov(X_std.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat) 

    sum_eigen_values = sum(eigen_values)
    var_exp = [(eigen_val/sum_eigen_values) 
                for eigen_val in sorted(eigen_values, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # Grouping 
    eigen_pair = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_vectors))]
    eigen_pair.sort(key=lambda k:k[0], reverse=True)
    
    #Cast matrix
    if shape_bound != None:
        cast_matrix = stack_cast_matrix(eigen_pair, shape_bound)
    else:
        # Wartość sumowane wariancji 
        shape_bound = np.argmax(cum_var_exp > var_bound)
        cast_matrix = stack_cast_matrix(eigen_pair, shape_bound)

    if not return_var:
        return cast_matrix
    
    return cast_matrix, [var_exp, cum_var_exp]


def plot_variance(var_exp, cum_var_exp):
    plt.bar(range(1,len(var_exp)+1), var_exp, alpha=0.5, align='center', label='Wariancja px')
    plt.step(range(1,len(cum_var_exp)+1), cum_var_exp, where='mid', label='Sumowana wariancja')
    plt.legend(loc='right')
    plt.title('Wyjaśnienie zmienności danych przez kolejne składowe główne')
    plt.xlabel('Cechy')
    plt.ylabel('Wariancja')
    plt.show()