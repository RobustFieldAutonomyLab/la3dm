#include "octree_node.h"
// #include "gpregressor.h"
#include <cmath>

namespace la3dm {

    /// Default static values
    float Occupancy::sf2 = 1.0f;
    float Occupancy::ell = 1.0f;
    // float Occupancy::noise = 0.01f;
    // float Occupancy::l = 100.f;

    // float Occupancy::max_ivar = 1000.0f;
    // float Occupancy::min_ivar = 0.001f;

    // float Occupancy::min_known_ivar = 10.0f;
    float Occupancy::free_thresh = 0.3f;
    float Occupancy::occupied_thresh = 0.7f;
    float Occupancy::prior_A = 0.5f;
    float Occupancy::prior_B = 0.5f;
    float Occupancy::var_thresh = 1000.0f;


    Occupancy::Occupancy(float A, float B) : m_A(Occupancy::prior_A + A), m_B(Occupancy::prior_B + B) {
        classified = false;
        float var = get_var();
        // std::cout << var << std::endl;
        if (var > Occupancy::var_thresh)
            state = State::UNKNOWN;
        else {
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }

    float Occupancy::get_prob() const {
        // logistic regression function
        // return 1.0f / (1.0f + (float) exp(-l * m_ivar / Occupancy::max_ivar));
        // return m_ivar;
        return m_A / (m_A + m_B);
    }

    void Occupancy::update(float ybar, float kbar) {
        // ivar += 1.0 / new_var - Occupancy::sf2;
        // m_ivar += new_m / new_var;
        // m_ivar = new_m;
        classified = true;
        m_A += ybar;
        if (kbar < 0.0)
        {
            std::cerr << "YBAR: " << ybar << ", KBAR: " << kbar << std::endl;
            abort();
        }
        m_B += kbar - ybar;


        float var = get_var();
        // std::cout << var << std::endl;
        if (var > Occupancy::var_thresh)
            state = State::UNKNOWN;
        else {
            float p = get_prob();
            // if (p != 0.75) std::cout << "ERR: " <<  p << std::endl;
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }

    std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc) {
        os.write((char *) &oc.m_A, sizeof(oc.m_A));
        os.write((char *) &oc.m_B, sizeof(oc.m_B));
        return os;
    }

    std::ifstream &operator>>(std::ifstream &is, Occupancy &oc) {
        float m_A, m_B;
        is.read((char *) &m_A, sizeof(m_A));
        is.read((char *) &m_B, sizeof(m_B));
        oc = OcTreeNode(m_A, m_B);
        return is;
    }

    std::ostream &operator<<(std::ostream &os, const Occupancy &oc) {
        return os << '(' << oc.m_A << ' ' << oc.m_B << ' ' << oc.get_prob() << ')';
    }
}
