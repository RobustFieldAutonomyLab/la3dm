#include "point6f.h"
#include <cassert>
#include <math.h>
#include <string.h>

namespace la3dm {

    // Vector3 &Vector3::rotate_IP(double roll, double pitch, double yaw) {
    //     double x, y, z;
    //     // pitch (around y)
    //     x = (*this)(0);
    //     z = (*this)(2);
    //     (*this)(0) = (float) (z * sin(pitch) + x * cos(pitch));
    //     (*this)(2) = (float) (z * cos(pitch) - x * sin(pitch));


    //     // yaw (around z)
    //     x = (*this)(0);
    //     y = (*this)(1);
    //     (*this)(0) = (float) (x * cos(yaw) - y * sin(yaw));
    //     (*this)(1) = (float) (x * sin(yaw) + y * cos(yaw));

    //     // roll (around x)
    //     y = (*this)(1);
    //     z = (*this)(2);
    //     (*this)(1) = (float) (y * cos(roll) - z * sin(roll));
    //     (*this)(2) = (float) (y * sin(roll) + z * cos(roll));

    //     return *this;
    // }

    std::istream &Vector6::read(std::istream &s) {
        int temp;
        s >> temp; // should be 6
        for (unsigned int i = 0; i < 6; i++)
            s >> operator()(i);
        return s;
    }


    std::ostream &Vector6::write(std::ostream &s) const {
        s << 6;
        for (unsigned int i = 0; i < 6; i++)
            s << " " << operator()(i);
        return s;
    }


    std::istream &Vector6::readBinary(std::istream &s) {
        int temp;
        s.read((char *) &temp, sizeof(temp));
        double val = 0;
        for (unsigned int i = 0; i < 6; i++) {
            s.read((char *) &val, sizeof(val));
            operator()(i) = (float) val;
        }
        return s;
    }


    std::ostream &Vector6::writeBinary(std::ostream &s) const {
        int temp = 6;
        s.write((char *) &temp, sizeof(temp));
        double val = 0;
        for (unsigned int i = 0; i < 6; i++) {
            val = operator()(i);
            s.write((char *) &val, sizeof(val));
        }
        return s;
    }


    std::ostream &operator<<(std::ostream &out, la3dm::Vector6 const &v) {
        return out << '(' << v.x0() << ' ' << v.y0() << ' ' << v.z0() << ' ' << v.x1() << ' ' << v.y1() << ' ' << v.z1() << ')';
    }
}