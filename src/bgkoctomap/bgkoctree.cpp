#include "bgkoctree.h"

#include <cmath>

namespace la3dm {

    unsigned short OcTree::max_depth = 0;

    OcTreeHashKey node_to_hash_key(unsigned short depth, unsigned short index) {
        return (depth << 16) + index;
    }

    void hash_key_to_node(OcTreeHashKey key, unsigned short &depth, unsigned short &index) {
        depth = (unsigned short) (key >> 16);
        index = (unsigned short) (key & 0xFFFF);
    }

    OcTree::OcTree() {
        if (max_depth <= 0)
            node_arr = nullptr;
        else {
            node_arr = new OcTreeNode *[max_depth]();
            for (unsigned short i = 0; i < max_depth; ++i) {
                node_arr[i] = new OcTreeNode[(int) pow(8, i)]();
            }
        }
    }

    OcTree::~OcTree() {
        if (node_arr != nullptr) {
            for (unsigned short i = 0; i < max_depth; ++i) {
                if (node_arr[i] != nullptr) {
                    delete[] node_arr[i];
                }
            }
            delete[] node_arr;
        }
    }

    OcTree::OcTree(const OcTree &other) {
        if (other.node_arr == nullptr) {
            node_arr = nullptr;
            return;
        }

        node_arr = new OcTreeNode *[max_depth]();
        for (unsigned short i = 0; i < max_depth; ++i) {
            if (other.node_arr[i] != nullptr) {
                int n = (int) pow(8, i);
                node_arr[i] = new OcTreeNode[n]();
                std::copy(node_arr[i], node_arr[i] + n, other.node_arr[i]);
            } else
                node_arr[i] = nullptr;
        }
    }

    OcTree &OcTree::operator=(const OcTree &other) {
        OcTreeNode **local_node_arr = new OcTreeNode *[max_depth]();
        for (unsigned short i = 0; i < max_depth; ++i) {
            if (local_node_arr[i] != nullptr) {
                int n = (int) pow(8, i);
                local_node_arr[i] = new OcTreeNode[n]();
                std::copy(local_node_arr[i], local_node_arr[i] + n, other.node_arr[i]);
            } else
                local_node_arr[i] = nullptr;
        }

        node_arr = local_node_arr;
        return *this;
    }

    bool OcTree::is_leaf(unsigned short depth, unsigned short index) const {
        if (node_arr != nullptr && node_arr[depth] != nullptr && node_arr[depth][index].get_state() != State::PRUNED) {
            if (depth + 1 < max_depth) {
                if (node_arr[depth + 1] == nullptr || node_arr[depth + 1][index * 8].get_state() == State::PRUNED)
                    return true;
            } else {
                return true;
            }
        }
        return false;
    }

    bool OcTree::is_leaf(OcTreeHashKey key) const {
        unsigned short depth = 0;
        unsigned short index = 0;
        hash_key_to_node(key, depth, index);
        return is_leaf(depth, index);
    }

    bool OcTree::search(OcTreeHashKey key) const {
        unsigned short depth;
        unsigned short index;
        hash_key_to_node(key, depth, index);

        return node_arr != nullptr &&
               node_arr[depth] != nullptr &&
               node_arr[depth][index].get_state() != State::PRUNED;
    }

    bool OcTree::prune() {
        if (node_arr == nullptr)
            return false;

        bool pruned = false;
        for (unsigned short depth = max_depth - 1; depth > 0; --depth) {
            OcTreeNode *layer = node_arr[depth];
            OcTreeNode *parent_layer = node_arr[depth - 1];
            if (layer == nullptr)
                continue;

            bool empty_layer = true;
            unsigned int n = (unsigned int) pow(8, depth);
            for (unsigned short index = 0; index < n; index += 8) {
                State state = layer[index].get_state();
                if (state == State::UNKNOWN) {
                    empty_layer = false;
                    continue;
                }
                if (state ==  State::PRUNED)
                    continue;

                bool collapsible = true;
                for (unsigned short i = 1; i < 8; ++i) {
                    if (layer[index + i].get_state() != state) {
                        collapsible = false;
                        continue;
                    }
                }

                if (collapsible) {
                    parent_layer[(int) floor(index / 8)] = layer[index];
                    for (unsigned short i = 0; i < 8; ++i) {
                        layer[index + i].prune();
                    }
                    pruned = true;
                } else {
                    empty_layer = false;
                }
            }

            if (empty_layer) {
                delete[] layer;
                node_arr[depth] = nullptr;
            }
        }
        return pruned;
    }

    OcTreeNode &OcTree::operator[](OcTreeHashKey key) const {
        unsigned short depth;
        unsigned short index;
        hash_key_to_node(key, depth, index);
        return node_arr[depth][index];
    }
}