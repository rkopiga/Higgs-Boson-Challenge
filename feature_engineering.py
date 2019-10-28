import numpy as np
import params


def feature_engineer(tX,
                     group=params.GROUP,
                     polynomial_expansion=params.FEATURE_EXPANSION,
                     degree=params.DEGREE,
                     polynomial_multiplication=params.FEATURE_MULTIPLICATION,
                     triple_multiplication=params.TRIPLE_MULTIPLICATION,
                     add_cos=params.ADD_COS,
                     add_sin=params.ADD_SIN,
                     add_tan=params.ADD_TAN,
                     add_exp=params.ADD_EXP,
                     add_log=params.ADD_LOG,
                     add_sqrt=params.ADD_SQRT,
                     add_cos2=params.ADD_COS2,
                     add_sin2=params.ADD_SIN2,
                     one_column=params.ONE_COLUMN):
    
    """
    Perform feature engineering on the data. The following feature engineering techniques are used:
        - Polynomial expansion: expanding the data matrix by enhancing the feature vector up to some power.
        - Polynomial multiplication: expanding the data matrix by multiplying features to each other up to some 
          degree 2 or 3.
        - Adding cosinus: expanding the data matrix by appending the cosinus of the feature vectors.
        - Adding sinus: expanding the data matrix by appending the sinus of the feature vectors.
        - Adding tangent: expanding the data matrix by appending the tangent of the feature vectors.
        - Adding exponenital: expanding the data matrix by appending the exponential of the features vectors.
        - Adding logarithm: expanding the data matrix by taking the logarithm of the feature vectors.
        - Adding the square root: expanding the data matrix by appending the square root of each vectors.
        - Adding cosinus square: expanding the data matrix by appending cos^2 of each vectors.
        - Adding sinus square: expanding the data matrix by appending sin^2 of each vectors.
        - Adding one column: expanding a column of 1 to the input matrix.
    
    Parameters
    -----------
    
    tX: array
        The feature matrix
    group: boolean
        Rather we organize the dataset in group or not
    polynomial_expansion: boolean
        Rather we do polynomial expansion on dataset or not
    degree: integer
        The degree of the polynomial basis 
    polynomial_multiplication: boolean
        Rather we do multiplication of each feature with each other (pair) or not
    triple_multiplication: boolean
        Rather we do multiplication of each feature with each other up to degree 3.
    add_cos: boolean
        Rather we add cosinus of the features to the dataset or not
    add_sin: boolean
        Rather we add sinus of the features to the dataset or not
    add_tan: boolean
        Rather we add tangent of the features to the dataset or not
    add_exp: boolean
        Rather we add exponential of the features to the dataset or not
    add_log: boolean
        Rather we add logarithm of the features to the dataset or not
    add_square_root: boolean
        Rather we add the square root of the feaures to the dataset or not
    add_cos2: boolean
        Rather we add the square of cosinus of the features to the dataset or not
    add_sin2: boolean
        Rather we add the square of sinus of the features to the dataset or not
    one_column: boolean
        Rather we add a column of one to the dataset
        
    Return
    -------
    
    tX: array
        The dataset on which some feature engineering has been applied.
    """
    print('\tFeature engineering...')
    print("\t\tPOLY EXPANSION: ", polynomial_expansion)
    print("\t\tPOLY MULTI: ", polynomial_multiplication)
    print("\t\tTRIPLE MULTI: ", triple_multiplication)
    print("\t\tCOS: ", add_cos)
    print("\t\tSIN: ", add_sin)
    print("\t\tTAN: ", add_tan)
    print("\t\tEXP: ", add_exp)
    print("\t\tLOG: ", add_log)
    print("\t\tSQRT: ", add_sqrt)
    print("\t\tCOS2: ", add_cos2)
    print("\t\tSIN2: ", add_sin2)
    print("\t\tONE_COLUMN: ", one_column)
    if group:
        if polynomial_expansion:
            tX = feature_expansion_grouped(tX, degree)
        if polynomial_multiplication:
            tX = feature_multiplication_grouped(tX,triple_multiplication)
        if add_cos:
            tX = add_cosinus_grouped(tX)
        if add_sin:
            tX = add_sinus_grouped(tX)
        if add_tan:
            tX = add_tangent_grouped(tX)
        if add_exp:
            tX = add_exponential_grouped(tX)
        if add_log:
            tX = add_logarithm_grouped(tX)
        if add_sqrt:
            tX = add_square_root_grouped(tX)
        if add_cos2:
            tX = add_cosinus_2_grouped(tX)
        if add_sin2:
            tX = add_sinus_2_grouped(tX)
        if one_column:
            tX = add_ones_column_grouped(tX)
    else:
        if polynomial_expansion:
            tX = feature_expansion(tX, degree)
        if polynomial_multiplication:
            tX = feature_multiplication(tX,triple_multiplication)
        if add_cos:
            tX = add_cosinus(tX)
        if add_sin:
            tX = add_sinus(tX)
        if add_exp:
            tX = add_exponential(tX)
        if add_log:
            tX = add_logarithm(tX)
        if add_sqrt:
            tX = add_square_root(tX)
        if add_cos2:
            tX = add_cosinus_2(tX)
        if add_sin2:
            tX = add_sinus_2(tX)
        if one_column:
            tX = add_ones_column(tX)
    print('\tFeature engineering ok.')
    return tX


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=2 up to j=degree. Given an 1D array x, the function 
    computes vectors x to the power up to degree.
        
    Parameters
    ----------
    x: array
        The vector that we want to raise to power up to degree
    degree: scalar
        The maximal degree
    
    Return
    -------
    a: array   
        matrix with x raised to the power from 2 to degree
    """
    a = [np.power(x, d) for d in range(2, degree+1)]
    return np.asarray(a)


def feature_expansion(tX, degree):
    """
    Feature expansion using build_poly. This function takes every features of the data and build a polynomial basis with this 
    feature, and add it to the data. 
        
    Parameters
    ----------
    tX: array
        The input matrix 
    degree: scalar
        The maximal degree up to which we will raise the feature to power of
    
    Return
    -------
    tX: array   
        data matrix with the features expanded on a polynomial basis
    """     
    for feature_index in range(tX.shape[1]):
        feature = tX[:, feature_index]
        expanded_feature = build_poly(feature, degree).T
        tX = np.hstack((tX, expanded_feature))
    return tX


def feature_expansion_grouped(tX_grouped, degree):
    """
    Feature expansion performed on list of data matrices. 
        
    Parameters
    ----------
    tX_grouped: list 
        list of data matrices. 
    degree: scalar
        maximal degree up to which we will raise the feature to power of
    
    Return
    -------
    tX_expanded: list of array   
        list of data matrices with expanded features.
    """      
    tX_expanded = []
    for i in range(len(tX_grouped)):
        tX_expanded.append(feature_expansion(tX_grouped[i], degree))
    return tX_expanded


def feature_multiplication(tX,triple_multiplication):
    """
    Feature multiplication on features. This function takes every features and multiply each other. 
        
    Parameters
    ----------
    tX: array
        The input data matrix
    
    Return
    -------
    new_tX: array   
        The data matrix to which the features multiplied to each other are appended.
    """
    
    new_tX = tX
    tX_column_size = tX.shape[1]
    for i in range(tX.shape[1]):
        col = tX[:, i].reshape(tX.shape[0], 1)
        tX_concat = np.multiply(tX[:, i:], col)
        new_tX = np.hstack((new_tX, tX_concat))
    if triple_multiplication:
        new_tX = feature_triple_multiplication(new_tX,tX_column_size)
    return new_tX


def feature_multiplication_grouped(tX_grouped, triple_multiplication):
    """
    Feature multiplication performed on list of data matrices. 
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    triple_multiplication: bool
        States whether a triple feature multiplication is needed. If the parameter is FALSE, only pairs of feature 
        will be multiplied to each other.

    
    Return
    -------
    new_tX_grouped: list of array   
        list of data matrices with expanded features.
    """     
    
    #First round
    new_tX_grouped = []
    for i in range(len(tX_grouped)):
        new_tX_grouped.append(feature_multiplication(tX_grouped[i],False))
    #Second round
    if triple_multiplication:
        for i in range(len(new_tX_grouped)):
            tX = new_tX_grouped[i]
            tX_column_size = tX_grouped[i].shape[1]
            new_tX = feature_triple_multiplication(tX,tX_column_size)
            new_tX_grouped[i] = new_tX
    
    return new_tX_grouped


def feature_triple_multiplication(tX,tX_column_size):
    """
    Feature multiplication of each other up to degree 3. For example if there are 3 features x1,x2, and x3, this function
    will output x1*x1*x1, x1*x1*x2, x1*x1*x3, ... , i.e all possible combinations of "triple multiplication".
    
    Parameters
    ----------
    tX: array
        The input data matrix
    tX_column_size: integer
        
    Return
    -------
    new_tX: array
        Matrix with the features of the input matrix,to which the multiplication up to degree of features are appended.
    """
    basic_features = tX[:,:tX_column_size]
    augmented_features= tX[:,tX_column_size:]
    new_tX = basic_features
    for feature_index in range(basic_features.shape[1]):
        feature = basic_features[:,feature_index].reshape(tX.shape[0], 1)
        if feature_index == 0:
            tX_concat = np.multiply(augmented_features[:,feature_index:], feature)
        else:
            ind = (feature_index-1)*tX_column_size + (tX_column_size - (feature_index-1))
            tX_concat = np.multiply(augmented_features[:,ind:],feature)
        new_tX = np.hstack((new_tX,tX_concat))
    new_tX = np.hstack((new_tX,augmented_features))
    return new_tX


def add_cosinus(tX):
    """
    Compute the cosinus of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    new_tX: array   
        Matrix with the cosinus of each feature appended to the input matrix.
    """
    new_tX = tX
    new_tX = np.hstack((tX, np.cos(new_tX))) 
    return new_tX


def add_cosinus_grouped(tX_grouped):
    """
    Compute add_cosinus of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        List of matrices with cosinus-expansion.
    """      
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_cosinus(tX_grouped[i]))
    return tX_grouped_new


def add_cosinus_2(tX):
    """
    Compute the square of cosinus of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    -------
    new_tX array   
        Matrix with the square of cosinus computed on of each feature appended to the input matrix.
    """    
    new_tX = tX
    new_tX = np.hstack((tX, np.cos(new_tX)**2))
    return new_tX


def add_cosinus_2_grouped(tX_grouped):
    """
    Compute add_cosinus_2 of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        list of matrices with the cosinus squared expansion.
    """    
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_cosinus_2(tX_grouped[i]))
    return tX_grouped_new


def add_sinus(tX):
    """
    Compute the sinus of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    new_tX: array   
        Matrix with the sinus of each feature appended to the input matrix.
    """    
    new_tX = tX
    new_tX = np.hstack((tX, np.sin(new_tX)))
    return new_tX


def add_sinus_grouped(tX_grouped):
    """
    Compute add_sinus of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        list of matrices with sinus-expansion.
    """     
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_sinus(tX_grouped[i]))
    return tX_grouped_new


def add_sinus_2(tX):
    """
    Compute the square of sinus of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    new_tX: array   
        Matrix with the square of sinus computed on of each feature appended to the input matrix.
    """        
    new_tX = tX
    new_tX = np.hstack((tX, np.sin(new_tX)**2))
    return new_tX


def add_sinus_2_grouped(tX_grouped):
    """
    Compute add_sinus_2 of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        list of matrices with the sinus squared expansion.
    """        
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_sinus_2(tX_grouped[i]))
    return tX_grouped_new


def add_tangent(tX):
    """
    Compute the tangent of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    new_tX: array   
        Matrix with the tangent of each feature appended to the input matrix.
    """    
    new_tX = tX
    new_tX = np.hstack((tX, np.tan(new_tX)))
    return new_tX


def add_tangent_grouped(tX_grouped):
    """
    Compute add_tangent of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        List of matrices with tangent-expansion.
    """        
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_tangent(tX_grouped[i]))
    return tX_grouped_new


def add_exponential(tX):
    """
    Compute the exponential of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    new_tX: array   
        Matrix with the exponential of each feature appended to the input matrix.
    """        
    new_tX = tX
    new_tX = np.hstack((tX, np.exp(new_tX)))
    return new_tX


def add_exponential_grouped(tX_grouped):
    """
    Compute add_exponential of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        List of matrices with exponential expansion.
    """    
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_exponential(tX_grouped[i]))
    return tX_grouped_new


def add_logarithm(tX):
    """
    Compute the logarithm of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    return_tX: array   
        Matrix with the logarithm of each feature appended to the input matrix.
    """         
    new_tX = tX.T
    minimum_by_feature = np.reshape(np.abs(np.min(new_tX, axis=1))+1, [new_tX.shape[0], 1])
    new_tX += minimum_by_feature
    logarithms = np.log(new_tX.T)
    return_tX =  np.hstack((tX, logarithms))
    return return_tX


def add_logarithm_grouped(tX_grouped):
    """
    Compute add_logarithm of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        List of matrices with logarithm expansion.
    """        
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_logarithm(tX_grouped[i]))
    return tX_grouped_new


def add_square_root(tX):
    """
    Compute the square root of each features and append them to the data matrix.  
        
    Parameters
    ----------
    tX: array
        The input data matrix. 
    
    Return
    ------
    return_tX: array
        Matrix with the square root of each feature appended to the input matrix.
    """    
    new_tX = tX.T
    minimum_by_feature = np.reshape(np.abs(np.min(new_tX, axis=1)), [new_tX.shape[0], 1])
    new_tX += minimum_by_feature
    square_roots = np.sqrt(new_tX.T)
    return_tX = np.hstack((tX, square_roots)) 
    return return_tX


def add_square_root_grouped(tX_grouped):
    """
    Compute add_square_root of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        List of matrices with square root expansion.
    """        
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_square_root(tX_grouped[i]))
    return tX_grouped_new


def add_ones_column(tX):
    """
    Add a feature vector with entries equal to 1 to the input matrix.  
        
    Parameters
    ----------
    tX: array 
        The input data matrix. 
    
    Return
    ------
    new_tX: array   
        Input matrix with one column of 1 append.
    """    
    len_tX = len(tX)
    ones = np.reshape(np.ones(len_tX), [len_tX, 1])
    new_tX = np.hstack((ones, tX))
    return new_tX


def add_ones_column_grouped(tX_grouped):
    """
    Add a feacture vector with entries equal to 1, to each matrix of a list of data matrices.  
        
    Parameters
    ----------
    tX_grouped: list 
        The list of data matrices. 
    
    Return
    ------
    tX_grouped_new: list   
        List of matrices with logarithm expansion.
    """        
    tX_grouped_new = []
    for i in range(len(tX_grouped)):
        tX_grouped_new.append(add_ones_column(tX_grouped[i]))
    return tX_grouped_new
