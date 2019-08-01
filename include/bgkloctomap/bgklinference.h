#ifndef LA3DM_BGKL_H
#define LA3DM_BGKL_H

namespace la3dm {

	/*
     * @brief Bayesian Generalized Kernel Inference on Bernoulli distribution
     * @param dim dimension of data (2, 3, etc.)
     * @param T data type (float, double, etc.)
     * @ref Nonparametric Bayesian inference on multivariate exponential families
     */
    template<int dim, typename T>
    class BGKLInference {
    public:
        /// Eigen matrix type for training and test data and kernel
        using MatrixXType = Eigen::Matrix<T, -1, 2*dim, Eigen::RowMajor>;
        using MatrixPType = Eigen::Matrix<T, -1, dim, Eigen::RowMajor>;
        using MatrixKType = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
        using MatrixDKType = Eigen::Matrix<T, -1, 1>;
        using MatrixYType = Eigen::Matrix<T, -1, 1>;

        float EPSILON = 0.0001;

        BGKLInference(T sf2, T ell) : sf2(sf2), ell(ell), trained(false) { }

        /*
         * @brief Fit BGK Model
         * @param x input vector (3N, row major)
         * @param y target vector (N)
         */
        void train(const std::vector<T> &x, const std::vector<T> &y) {
            assert(x.size() % (2*dim) == 0 && (int) (x.size() / (2*dim)) == y.size());
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), x.size() / (2*dim), 2*dim);
            MatrixYType _y = Eigen::Map<const MatrixYType>(y.data(), y.size(), 1);
            train(_x, _y);
        }

        /*
         * @brief Fit BGK Model
         * @param x input matrix (NX3)
         * @param y target matrix (NX1)
         */
        void train(const MatrixXType &x, const MatrixYType &y) {
            // std::cout << "training pt2" << std::endl;
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
            // std::cout << "predicting" << std::endl;
            assert(xs.size() % dim == 0);
            // std::cout << "passed assertion" << std::endl;
            MatrixPType _xs = Eigen::Map<const MatrixPType>(xs.data(), xs.size() / dim, dim);
            // std::cout << "matrix conversion successful" << std::endl;

            MatrixYType _ybar, _kbar;
            predict(_xs, _ybar, _kbar);
            // std::cout << "finished prediction" << std::endl;
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
        void predict(const MatrixPType &xs, MatrixYType &ybar, MatrixYType &kbar) const {
            // std::cout << "second prediction step" << std::endl;
            assert(trained == true);
	        MatrixKType Ks;
        	covSparseLine(xs, x, Ks);
            // std::cout << "computed covsparseline" << std::endl;
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


        // TODO: validate me
        void point_to_line_dist(const MatrixPType &x, const MatrixXType &z, MatrixKType &d) const {
            assert((x.cols() == 3) && (z.cols() == 6));
            // std::cout << "made it" << std::endl;
            d = MatrixKType::Zero(x.rows(), z.rows());
            float line_len;
            point3f p, p0, p1, v, w, line_vec, pnt_vec, nearest;
            float t;
            for (int i = 0; i < x.rows(); ++i) {
                p = point3f(x(i,0), x(i,1), x(i,2));
                for (int j = 0; j < z.rows(); ++j) {
                    p0 = point3f(z(j,0), z(j,1), z(j,2));
                    p1 = point3f(z(j,3), z(j,4), z(j,5));
                    line_vec = p1 - p0;
                    line_len = line_vec.norm();
                    pnt_vec = p - p0;
                    if (line_len < EPSILON) {
                        d(i,j) = (p-p0).norm();
                    }
                    else {
                        double c1 = pnt_vec.dot(line_vec);
                        if ( c1 <= 0) {
                            d(i,j) = (p - p0).norm();
                        }
                        double c2 = line_vec.dot(line_vec);
                        if (c2 <= c1) {
                            d(i,j) = (p - p1).norm();
                        }
                        double b = c1 / c2;
                        nearest = p0 + (line_vec*b);
                        d(i,j) = (p - nearest).norm();
                    }

                    }
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

        /*
         * @brief Sparse kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         * @ref A sparse covariance function for exact gaussian process inference in large datasets.
         */
        void covSparseLine(const MatrixPType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            point_to_line_dist(x, z, Kxz); // Check on this
            Kxz /= ell;
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

    typedef BGKLInference<3, float> BGKL3f;

}
#endif // LA3DM_BGKL_H
