#ifndef LA3DM_OCCUPANCY_H
#define LA3DM_OCCUPANCY_H

#include <iostream>
#include <fstream>

namespace la3dm {

    /// Occupancy state: before pruning: FREE, OCCUPIED, UNKNOWN; after pruning: PRUNED
    enum class State : char {
        FREE, OCCUPIED, UNKNOWN, PRUNED
    };

    /*
     * @brief GP regression ouputs and occupancy state.
     *
     * Occupancy has member variables: m_ivar (m*ivar), ivar (1/var) and State.
     * This representation speeds up the updates via BCM.
     * Before using this class, set the static member variables first.
     */
    class Occupancy {
        friend std::ostream &operator<<(std::ostream &os, const Occupancy &oc);

        friend std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc);

        friend std::ifstream &operator>>(std::ifstream &is, Occupancy &oc);

        friend class BGKOctoMap;

    public:
        /*
         * @brief Constructors and destructor.
         */
        Occupancy() : m_A(Occupancy::prior_A), m_B(Occupancy::prior_B), state(State::UNKNOWN) { classified = false; }

        Occupancy(float A, float B);

        Occupancy(const Occupancy &other) : m_A(other.m_A), m_B(other.m_B), state(other.state) { }

        Occupancy &operator=(const Occupancy &other) {
            m_A = other.m_A;
            m_B = other.m_B;
            state = other.state;
            return *this;
        }

        ~Occupancy() { }

        /*
         * @brief Bayesian Committee Machine (BCM) update for Gaussian Process regression.
         * @param new_m mean resulted from GP regression
         * @param new_var variance resulted from GP regression
         */
        void update(float ybar, float kbar);

        /// Get probability of occupancy.
        float get_prob() const;

        /// Get variance of occupancy (uncertainty)
        inline float get_var() const { return (m_A * m_B) / ( (m_A + m_B) * (m_A + m_B) * (m_A + m_B + 1.0f)); }

        /*
         * @brief Get occupancy state of the node.
         * @return occupancy state (see State).
         */
        inline State get_state() const { return state; }

        /// Prune current node; set state to PRUNED.
        inline void prune() { state = State::PRUNED; }

        /// Only FREE and OCCUPIED nodes can be equal.
        inline bool operator==(const Occupancy &rhs) const {
            return this->state != State::UNKNOWN && this->state == rhs.state;
        }

        bool classified;

    private:
        // float m;  // m / var or m * ivar
        // float var;    // 1.0 / var
        float m_A;
        float m_B;
        State state;

        static float sf2;   // signal variance
        static float ell;   // length-scale
        // static float noise; // noise variance
        // static float l;     // gamma in logistic functions

        // static float max_ivar; // minimum variance
        // static float min_ivar; // maximum variance
        // static float min_known_ivar;  // maximum variance for nodes to be considered as FREE or OCCUPIED

        static float prior_A;
        static float prior_B;

        static float free_thresh;     // FREE occupancy threshold
        static float occupied_thresh; // OCCUPIED occupancy threshold
        static float var_thresh;
    };

    typedef Occupancy OcTreeNode;
}

#endif //GPOCTOMAP_OCCUPANCY_H
