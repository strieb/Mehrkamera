import numpy as np
import modules.homography as ho
import numpy.testing as nptest


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    ptsA = np.asarray([[0, 0], [100, 50]])
    ptsB = np.asarray([[50, 25, 1], [200, 100, 2], [200, 100, 2]])
    N = ho.findNormalizationMatrixWithSize((100, 50))

    nptest.assert_array_almost_equal(ho.project(N, ptsA), np.asarray([[-1, -0.5], [1, 0.5]]))
    nptest.assert_array_almost_equal(ho.project(N, ptsB), np.asarray([[0, 0], [1, 0.5], [1, 0.5]]))

    nptest.assert_array_almost_equal(ho.project(N, np.asarray([0, 100, 2])), [-1, 0.5])
    nptest.assert_array_almost_equal(ho.project(N, np.asarray([0, 100])), [-1, 1.5])

    svd_src = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
    svd_dst = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]])

    svd_mat = ho.findSVD(svd_src, svd_dst)
    nptest.assert_array_almost_equal(svd_mat, [[10, 0, 0], [0, 10, 0], [0, 0, 1]])
    nptest.assert_array_almost_equal(ho.project(svd_mat, svd_src), svd_dst)

    special_mat = np.asarray([[0.42, 0.22, 1], [0.1, 11, -242], [0.1, -0.2, 1.2]])
    special_dst = ho.project(special_mat, svd_src)
    nptest.assert_array_almost_equal(ho.findSVD(svd_src, special_dst), special_mat/special_mat[2, 2])

    H_inv = np.linalg.inv(special_mat)
    nptest.assert_array_almost_equal(svd_src, ho.project(H_inv, special_dst))

    nptest.assert_array_almost_equal(ho.goldStandardError(special_mat, svd_src, special_dst), 0)

    print("All tests passed!")
