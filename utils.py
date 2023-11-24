import numpy as np
import tensorflow as tf
import scipy.sparse as sp

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()

    identity = sp.eye(adj.shape[0], format='coo', dtype=np.int8)

    # Convert identity to the same sparse format as adj
    identity = identity.tocoo()

    # Convert to NumPy arrays for debugging
    adj = adj.toarray()
    identity = identity.toarray()

    # Check if both matrices have the same shape before addition
    if adj.shape != identity.shape:
        raise ValueError("Inconsistent shapes for adj and identity matrices: {adj.shape} vs {identity.shape}")

    # Add self-loops using element-wise addition
    adj_ = sp.coo_matrix(adj + identity)

    rowsum = np.array(adj_.sum(1))
    rowsum += np.finfo(float).eps
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_norm = (adj_.dot(degree_mat_inv_sqrt.T)).dot(degree_mat_inv_sqrt)

    return adj_norm




def constructNet(miR_tar_matrix):
    miR_matrix = np.zeros((miR_tar_matrix.shape[0], miR_tar_matrix.shape[1]), dtype=np.int8)
    tar_matrix = np.zeros((miR_tar_matrix.shape[0], miR_tar_matrix.shape[1]), dtype=np.int8)

    mat1 = np.hstack((miR_matrix, miR_tar_matrix))
    mat2 = np.hstack((miR_tar_matrix.T, tar_matrix))
    adj = np.vstack((mat1, mat2))

    return adj

def constructHNet(miR_tar_matrix, miR_matrix, tar_matrix):
    print("miR_matrix shape:", miR_matrix.shape)
    print("miR_tar_matrix shape:", miR_tar_matrix.shape)
    print("tar_matrix shape:", tar_matrix.shape)

    columns_needed = miR_matrix.shape[1]

    if tar_matrix.shape[1] > columns_needed:
        # Truncate tar_matrix to match the number of columns in miR_matrix
        tar_matrix_truncated = tar_matrix[:, :columns_needed]

        print("tar_matrix_truncated shape:", tar_matrix_truncated.shape)

        # Now, miR_matrix and tar_matrix_truncated should have the same number of columns
        # You can proceed with vertical stacking
        mat1 = np.hstack((miR_matrix, miR_tar_matrix))
        mat2 = np.hstack((miR_tar_matrix.T, tar_matrix_truncated.T))

        # Pad or trim mat1 to have the same number of columns as mat2
        if mat1.shape[1] < mat2.shape[1]:
            mat1 = np.pad(mat1, ((0, 0), (0, mat2.shape[1] - mat1.shape[1])), mode='constant')
        elif mat1.shape[1] > mat2.shape[1]:
            mat1 = mat1[:, :mat2.shape[1]]
    else:
        # tar_matrix already has the same number of columns as miR_matrix
        # You can directly proceed with vertical stacking
        mat1 = np.hstack((miR_matrix, miR_tar_matrix))
        mat2 = np.hstack((miR_tar_matrix.T, tar_matrix))

    print("mat1 shape:", mat1.shape)
    print("mat2 shape:", mat2.shape)

    # Vertical stacking
    hnet = np.vstack((mat1, mat2))

    return hnet

def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    ls_return = []
    for i in range(0, (n - 1) * j, j):
        ls_return.append(ls[i:i + j])
    ls_return.append(ls[(n - 1) * j:])
    return ls_return
