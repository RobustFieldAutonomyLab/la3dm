#ifndef LA3DM_BGK_H
#define LA3DM_BGK_H

namespace la3dm {

	/*
     * @brief Bayesian Generalized Kernel Inference on Bernoulli distribution
     * @param dim dimension of data (2, 3, etc.)
     * @param T data type (float, double, etc.)
     * @ref Nonparametric Bayesian inference on multivariate exponential families
     */
    template<int dim, typename T>
    class BGKInference {
    public:
        /// Eigen matrix type for training and test data and kernel
        using MatrixXType = Eigen::Matrix<T, -1, dim, Eigen::RowMajor>;
        using MatrixKType = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
        using MatrixDKType = Eigen::Matrix<T, -1, 1>;
        using MatrixYType = Eigen::Matrix<T, -1, 1>;

        BGKInference(T sf2, T ell) : sf2(sf2), ell(ell), trained(false) { }

        /*
         * @brief Fit BGK Model
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
         * @brief Fit BGK Model
         * @param x input matrix (NX3)
         * @param y target matrix (NX1)
         */
        void train(const MatrixXType &x, const MatrixYType &y) {
            this->x = MatrixXType(x);
            this->y = MatrixYType(y);
            trained = true;
        }

        /*
         * @brief Predict with BGK Model
         * @param xs input vector (3M, row major)
         * @param ybar positive class kernel density estimate (\bar{y})
         * @param kbar kernel density estimate (\bar{k})
         */
        void predict(const std::vector<T> &xs, std::vector<T> &ybar, std::vector<T> &kbar) const {
            assert(xs.size() % dim == 0);
            MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);

            MatrixYType _ybar, _kbar;
            predict(_xs, _ybar, _kbar);

            ybar.resize(_ybar.rows());
            kbar.resize(_kbar.rows());
            for (int r = 0; r < _ybar.rows(); ++r) {
                ybar[r] = _ybar(r, 0);
                kbar[r] = _kbar(r, 0);
            }
        }

        /*
         * @brief Predict with nonparametric Bayesian generalized kernel inference
         * @param xs input vector (M x 3)
         * @param ybar positive class kernel density estimate (M x 1)
         * @param kbar kernel density estimate (M x 1)
         */
        void predict(const MatrixXType &xs, MatrixYType &ybar, MatrixYType &kbar) const {
            assert(trained == true);
	        MatrixKType Ks;
        	covSparse(xs, x, Ks);
        	ybar = (Ks * y).array();
        	kbar = Ks.rowwise().sum().array();
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
         * @brief Sparse kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         * @ref A sparse covariance function for exact gaussian process inference in large datasets.
         */
        void covSparse(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(x / ell, z / ell, Kxz);
            Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
                  (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * sf2;

            // Clean up for values with distance outside length scale
            // Possible because Kxz <= 0 when dist >= ell
            for (int i = 0; i < Kxz.rows(); ++i)
            {
                for (int j = 0; j < Kxz.cols(); ++j)
                    if (Kxz(i,j) < 0.0)
                        Kxz(i,j) = 0.0f;
            }
        }

        T sf2;    // signal variance
        T ell;    // length-scale

        MatrixXType x;   // temporary storage of training data
        MatrixYType y;   // temporary storage of training labels

        bool trained;    // true if bgkinference stored training data
    };

    typedef BGKInference<3, float> BGK3f;

}
#endif // LA3DM_BGK_H
