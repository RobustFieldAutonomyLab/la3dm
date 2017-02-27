#include "gpoctree_node.h"
#include "gpregressor.h"
#include <cmath>

namespace la3dm {
    /// Default static values
    float Occupancy::sf2 = 1.0f;
    float Occupancy::ell = 1.0f;
    float Occupancy::noise = 0.01f;
    float Occupancy::l = 100.f;

    float Occupancy::max_ivar = 1000.0f;
    float Occupancy::min_ivar = 0.001f;

    float Occupancy::min_known_ivar = 10.0f;
    float Occupancy::free_thresh = 0.3f;
    float Occupancy::occupied_thresh = 0.7f;

    Occupancy::Occupancy(float m, float var) : m_ivar(m / var), ivar(1.0f / var) {
        if (ivar < Occupancy::min_known_ivar)
            state = State::UNKNOWN;
        else {
            ivar = ivar > Occupancy::max_ivar ? Occupancy::max_ivar : ivar;
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }

    float Occupancy::get_prob() const {
        // logistic regression function
        return 1.0f / (1.0f + (float) exp(-l * m_ivar / Occupancy::max_ivar));
    }

    void Occupancy::update(float new_m, float new_var) {
        ivar += 1.0 / new_var - Occupancy::sf2;
        m_ivar += new_m / new_var;
        if (ivar < Occupancy::min_known_ivar)
            state = State::UNKNOWN;
        else {
            // chop variance
            ivar = ivar > Occupancy::max_ivar ? Occupancy::max_ivar : ivar;
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }

    std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc) {
        os.write((char *) &oc.m_ivar, sizeof(oc.m_ivar));
        os.write((char *) &oc.ivar, sizeof(oc.ivar));
        return os;
    }

    std::ifstream &operator>>(std::ifstream &is, Occupancy &oc) {
        float m_ivar, ivar;
        is.read((char *) &m_ivar, sizeof(m_ivar));
        is.read((char *) &ivar, sizeof(ivar));
        oc = OcTreeNode(m_ivar / ivar, 1.0f / ivar);
        return is;
    }

    std::ostream &operator<<(std::ostream &os, const Occupancy &oc) {
        return os << '(' << (oc.m_ivar / oc.ivar) << ' ' << (1.0 / oc.ivar) << ' ' << oc.get_prob() << ')';
    }
}
