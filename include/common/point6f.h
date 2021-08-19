#ifndef LA3DM_VECTOR6_H
#define LA3DM_VECTOR6_H

#include <iostream>
#include <math.h>
#include "point3f.h"

namespace la3dm {

    /*!
     * \brief This class represents a six-dimensional vector
     *
     * We use the six-dimensional vector to represent the start
     * and end points of a line segment
     */
    class Vector6 {
    public:

        /*!
         * \brief Default constructor
         */
        Vector6() { data[0] = data[1] = data[2] = data[3] = data[4] = data[5] = 0.0; }

        /*!
         * \brief Copy constructor
         *
         * @param other a vector of dimension 6
         */
        Vector6(const Vector6 &other) {
            data[0] = other(0);
            data[1] = other(1);
            data[2] = other(2);
            data[3] = other(3);
            data[4] = other(4);
            data[5] = other(5);
        }

        /*!
        * \brief 6-D vector as 2 copies of a 3-D vector
        *
        * @param point a vector of dimension 3
        */
        Vector6(const Vector3 &point) {
            data[0] = data[3] = point(0);
            data[1] = data[4] = point(1);
            data[2] = data[5] = point(2);
        }

        /*!
        * \brief 6-D vector from start and endpoint of a line
        *
        * @param start a vector of dimension 3
        * @param end a vector of dimension 3
        */
        Vector6(const Vector3 &start, const Vector3 &end) {
            data[0] = start(0);
            data[1] = start(1);
            data[2] = start(2);
            data[3] = end(0);
            data[4] = end(1);
            data[5] = end(2);
        }

        /*!
         * \brief Constructor
         *
         * Constructs a six-dimensional vector from
         * three single values x, y, z by duplication
         */
        Vector6(float x0, float y0, float z0) {
            data[0] = x0;
            data[1] = y0;
            data[2] = z0;
            data[3] = x0;
            data[4] = y0;
            data[5] = z0;
        }

        /*!
         * \brief Constructor
         *
         * Constructs a six-dimensional vector from
         * six single values
         */
        Vector6(float x0, float y0, float z0, float x1, float y1, float z1) {
            data[0] = x0;
            data[1] = y0;
            data[2] = z0;
            data[3] = x1;
            data[4] = y1;
            data[5] = z1;
        }


        /* inline Eigen3::Vector6f getVector6f() const { return Eigen3::Vector6f(data[0], data[1], data[2]) ; } */
        /* inline Eigen3::Vector4f& getVector4f() { return data; } */
        /* inline Eigen3::Vector4f getVector4f() const { return data; } */

        /*!
         * \brief Assignment operator
         *
         * @param other a vector of dimension 6
         */
        inline Vector6 &operator=(const Vector6 &other) {
            data[0] = other(0);
            data[1] = other(1);
            data[2] = other(2);
            data[3] = other(3);
            data[4] = other(4);
            data[5] = other(5);
            return *this;
        }

        /// dot product
        inline double dot(const Vector6 &other) const {
            return x0() * other.x0() + y0() * other.y0() + z0() * other.z0() + x1() * other.x1() + y1() * other.y1() + z1() * other.z1();
        }

        inline const float &operator()(unsigned int i) const {
            return data[i];
        }

        inline float &operator()(unsigned int i) {
            return data[i];
        }

        inline point3f start() {
            return point3f(x0(), y0(), z0());
        }

        inline point3f end() {
            return point3f(x1(), y1(), z1());
        }

        inline float &x0() {
            return operator()(0);
        }

        inline float &y0() {
            return operator()(1);
        }

        inline float &z0() {
            return operator()(2);
        }

        inline float &x1() {
            return operator()(3);
        }

        inline float &y1() {
            return operator()(4);
        }

        inline float &z1() {
            return operator()(5);
        }

        inline const point3f start() const {
            return point3f(x0(), y0(), z0());
        }

        inline const point3f end() const {
            return point3f(x1(), y1(), z1());
        }

        inline const float &x0() const {
            return operator()(0);
        }

        inline const float &y0() const {
            return operator()(1);
        }

        inline const float &z0() const {
            return operator()(2);
        }

        inline const float &x1() const {
            return operator()(3);
        }

        inline const float &y1() const {
            return operator()(4);
        }

        inline const float &z1() const {
            return operator()(5);
        }


        inline Vector6 operator-() const {
            Vector6 result;
            result(0) = -data[0];
            result(1) = -data[1];
            result(2) = -data[2];
            result(3) = -data[3];
            result(4) = -data[4];
            result(5) = -data[5];
            return result;
        }

        inline Vector6 operator+(const Vector6 &other) const {
            Vector6 result(*this);
            result(0) += other(0);
            result(1) += other(1);
            result(2) += other(2);
            result(3) += other(3);
            result(4) += other(4);
            result(5) += other(5);
            return result;
        }

        inline Vector6 operator*(float x) const {
            Vector6 result(*this);
            result(0) *= x;
            result(1) *= x;
            result(2) *= x;
            result(3) *= x;
            result(4) *= x;
            result(5) *= x;
            return result;
        }

        inline Vector6 operator-(const Vector6 &other) const {
            Vector6 result(*this);
            result(0) -= other(0);
            result(1) -= other(1);
            result(2) -= other(2);
            result(3) -= other(3);
            result(4) -= other(4);
            result(5) -= other(5);
            return result;
        }

        inline void operator+=(const Vector6 &other) {
            data[0] += other(0);
            data[1] += other(1);
            data[2] += other(2);
            data[3] += other(3);
            data[4] += other(4);
            data[5] += other(5);
        }

        inline void operator-=(const Vector6 &other) {
            data[0] -= other(0);
            data[1] -= other(1);
            data[2] -= other(2);
            data[3] -= other(3);
            data[4] -= other(4);
            data[5] -= other(5);
        }

        inline void operator/=(float x) {
            data[0] /= x;
            data[1] /= x;
            data[2] /= x;
            data[3] /= x;
            data[4] /= x;
            data[5] /= x;
        }

        inline void operator*=(float x) {
            data[0] *= x;
            data[1] *= x;
            data[2] *= x;
            data[3] *= x;
            data[4] *= x;
            data[5] *= x;
        }

        inline bool operator==(const Vector6 &other) const {
            for (unsigned int i = 0; i < 6; i++) {
                if (operator()(i) != other(i))
                    return false;
            }
            return true;
        }

        /// @return length of the line segment start -> end
        inline double line_length() const {
            return sqrt((start() - end()).norm_sq());
        }

        /// @return length of the vector ("L2 norm")
        inline double norm() const {
            return sqrt(norm_sq());
        }

        /// @return squared length ("L2 norm") of the vector
        inline double norm_sq() const {
            return (x0() * x0() + y0() * y0() + z0() * z0() + x1() * x1() + y1() * y1() + z1() * z1());
        }

        /// normalizes this vector, so that it has norm=1.0
        inline Vector6 &normalize() {
            double len = norm();
            if (len > 0)
                *this /= (float) len;
            return *this;
        }

        /// @return normalized vector, this one remains unchanged
        inline Vector6 normalized() const {
            Vector6 result(*this);
            result.normalize();
            return result;
        }

        // inline double angleTo(const Vector6 &other) const {
        //     double dot_prod = this->dot(other);
        //     double len1 = this->norm();
        //     double len2 = other.norm();
        //     return acos(dot_prod / (len1 * len2));
        // }


        // inline double distance(const Vector6 &other) const {
        //     double dist_x = x() - other.x();
        //     double dist_y = y() - other.y();
        //     double dist_z = z() - other.z();
        //     return sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);
        // }

        // inline double distanceXY(const Vector6 &other) const {
        //     double dist_x = x() - other.x();
        //     double dist_y = y() - other.y();
        //     return sqrt(dist_x * dist_x + dist_y * dist_y);
        // }

        // Vector6 &rotate_IP(double roll, double pitch, double yaw);

        //    void read (unsigned char * src, unsigned int size);
        std::istream &read(std::istream &s);

        std::ostream &write(std::ostream &s) const;

        std::istream &readBinary(std::istream &s);

        std::ostream &writeBinary(std::ostream &s) const;


    protected:
        float data[6];

    };

    //! user friendly output in format (x0 y0 z0 x1 y1 z1)
    std::ostream &operator<<(std::ostream &out, la3dm::Vector6 const &v);

    typedef Vector6 point6f;

}

#endif // LA3DM_VECTOR6_H
