#ifndef LA3DM_GP_REGRESSOR_H
#define LA3DM_GP_REGRESSOR_H

#include <Eigen/Dense>
#include <vector>

namespace la3dm {

    /*
     * @brief A Simple Gaussian Process Regressor
     * @param dim dimension of data (2, 3, etc.)
     * @param T data type (float, double, etc.)
     */
    template<int dim, typename T>
    class GPRegressor {
    public:
        /// Eigen matrix type for training and test data and kernel
        using MatrixXType = Eigen::Matrix<T, -1, dim, Eigen::RowMajor>;
        using MatrixKType = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
        using MatrixDKType = Eigen::Matrix<T, -1, 1>;
        using MatrixYType = Eigen::Matrix<T, -1, 1>;

        GPRegressor(T sf2, T ell, T noise) : sf2(sf2), ell(ell), noise(noise), trained(false) { }

        /*
         * @brief Train Gaussian Process
         * @param x input vector (3N, row major)
         * @param y target vector (N)
         */
        void train(const std::vector<T> &x, const std::vector<T> &y) {
            assert(x.size() % dim == 0 && (int) (x.size() / dim) == y.size());
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), x.size() / dim, dim);
            MatrixYType _y = Eigen::Map<const MatrixYType>(y.data(), y.size(), 1);
            train(_x, _y);
        }

        /*
         * @brief Train Gaussian Process
         * @param x input matrix (NX3)
         * @param y target matrix (NX1)
         */
        void train(const MatrixXType &x, const MatrixYType &y) {
            this->x = MatrixXType(x);
            covMaterniso3(x, x, K);
           // covSparse(x, x, K);
            K = K + noise * MatrixKType::Identity(K.rows(), K.cols());
            Eigen::LLT<MatrixKType> llt(K);
            alpha = llt.solve(y);
            L = llt.matrixL();
            trained = true;
        }

        /*
         * @brief Predict with Gaussian Process
         * @param xs input vector (3M, row major)
         * @param m predicted mean vector (M)
         * @param var predicted variance vector (M)
         */
        void predict(const std::vector<T> &xs, std::vector<T> &m, std::vector<T> &var) const {
            assert(xs.size() % dim == 0);
            MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);

            MatrixYType _m, _var;
            predict(_xs, _m, _var);

            m.resize(_m.rows());
            var.resize(_var.rows());
            for (int r = 0; r < _m.rows(); ++r) {
                m[r] = _m(r, 0);
                var[r] = _var(r, 0);
            }
        }

        /*
         * @brief Predict with Gaussian Process
         * @param xs input vector (MX3)
         * @param m predicted mean matrix (MX1)
         * @param var predicted variance matrix (MX1)
         */
        void predict(const MatrixXType &xs, MatrixYType &m, MatrixYType &var) const {
            assert(trained == true);
            MatrixKType Ks;
            covMaterniso3(x, xs, Ks);
           // covSparse(x, xs, Ks);
            m = Ks.transpose() * alpha;

            MatrixKType v = L.template triangularView<Eigen::Lower>().solve(Ks);
            MatrixDKType Kss;
            covMaterniso3(xs, xs, Kss);
            // covSparse(xs, xs, Kss);
            var = Kss - (v.transpose() * v).diagonal();
        }

    private:
        /*
         * @brief Compute Euclid distances between two vectors.
         * @param x input vector
         * @param z input vecotr
         * @return d distance matrix
         */
        void dist(const MatrixXType &x, const MatrixXType &z, MatrixKType &d) const {
            d = MatrixKType::Zero(x.rows(), z.rows());
            for (int i = 0; i < x.rows(); ++i) {
                d.row(i) = (z.rowwise() - x.row(i)).rowwise().norm();
            }
        }

        /*
         * @brief Matern3 kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         */
        void covMaterniso3(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(1.73205 / ell * x, 1.73205 / ell * z, Kxz);
            Kxz = ((1 + Kxz.array()) * exp(-Kxz.array())).matrix() * sf2;
        }

        /*
         * @brief Diagonal of Matern3 kernel.
         * @return Kxz sf2 * I
         */
        void covMaterniso3(const MatrixXType &x, const MatrixXType &z, MatrixDKType &Kxz) const {
            Kxz = MatrixDKType::Ones(x.rows()) * sf2;
        }

        /*
         * @brief Sparse kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         * @ref A sparse covariance function for exact gaussian process inference in large datasets.
         */
        void covSparse(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(x / ell, z / ell, Kxz);
            Kxz = ((2 + (Kxz * 2 * 3.1415926).array().cos()) * (1.0 - Kxz.array()) / 3.0 +
                  (Kxz * 2 * 3.1415926).array().sin() / (2 * 3.1415926)).matrix() * sf2;
            for (int i = 0; i < Kxz.rows(); ++i) {
                for (int j = 0; j < Kxz.cols(); ++j) {
                    if (Kxz(i,j) < 0.0) {
                        Kxz(i,j) = 0.0f;
                    }
                }
            }
        }

        /*
         * @brief Diagonal of sparse kernel.
         * @return Kxz sf2 * I
         */
        void covSparse(const MatrixXType &x, const MatrixXType &z, MatrixDKType &Kxz) const {
            Kxz = MatrixDKType::Ones(x.rows()) * sf2;
        }

        T sf2;    // signal variance
        T ell;    // length-scale
        T noise;  // noise variance

        MatrixXType x;   // temporary storage of training data
        MatrixKType K;
        MatrixYType alpha;
        MatrixKType L;

        bool trained;    // true if gpregressor has been trained
    };

    typedef GPRegressor<3, float> GPR3f;
}
#endif // LA3DM_GP_REGRESSOR_H
