#include "bgklvoctree_node.h"
#include <cmath>

namespace la3dm {

    /// Default static values
    float Occupancy::sf2 = 1.0f;
    float Occupancy::ell = 1.0f;
    float Occupancy::free_thresh = 0.3f;
    float Occupancy::occupied_thresh = 0.7f;
    float Occupancy::var_thresh = 1000.0f;
    float Occupancy::prior_A = 0.5f;
    float Occupancy::prior_B = 0.5f;
    bool Occupancy::original_size = true;
    float Occupancy::min_W = 0.1f;

    Occupancy::Occupancy(float A, float B) : m_A(Occupancy::prior_A + A), m_B(Occupancy::prior_B + B) {
        classified = false;
        float var = get_var();
        if (var > Occupancy::var_thresh)
            state = State::UNCERTAIN;
        else {
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }

    float Occupancy::get_prob() const {
        float prob, W;
        if (m_A + m_B < Occupancy::min_W){
            W = Occupancy::min_W;
        }
        else{
            W = m_A + m_B;
        }

        if(m_A > m_B){
            prob = m_A / (W-m_B) + (W-m_A-m_B)*0.5 / (W-m_B);
        }
        else{
            prob = 0.5*(W-m_B-m_A) / (W-m_A);
        }

        return prob;
    }

    float Occupancy::get_var() const { 
            float var, W;
            float prob = get_prob();

            if (m_A + m_B < Occupancy::min_W){
                W = Occupancy::min_W;
            }
            else{
                W = m_A + m_B;
            }

            var = m_A / W * pow(1-prob,2) + (W-m_A-m_B) / W * pow(0.5-prob,2) + m_B / W * pow(prob,2);
            return var; 
            
        }

    void Occupancy::update(float ybar, float kbar) {
        classified = true;
        m_A += ybar;
        m_B += kbar - ybar;

        float var = get_var();
        if (var > Occupancy::var_thresh)
            state = State::UNCERTAIN;
        else {
            float p = get_prob();
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