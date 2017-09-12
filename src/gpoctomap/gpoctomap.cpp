#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include "gpoctomap.h"
#include "gpregressor.h"

using std::vector;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace la3dm {

    GPOctoMap::GPOctoMap() : GPOctoMap(0.1f, 4, 1.0, 1.0, 0.01, 100, 0.001f, 1000.0f, 0.02f, 0.3f, 0.7f) { }

    GPOctoMap::GPOctoMap(float resolution, unsigned short block_depth, float sf2, float ell, float noise, float l,
                         float min_var,
                         float max_var, float max_known_var, float free_thresh, float occupied_thresh)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);

        OcTree::max_depth = block_depth;

        OcTreeNode::sf2 = sf2;
        OcTreeNode::ell = ell;
        OcTreeNode::noise = noise;
        OcTreeNode::l = l;

        OcTreeNode::min_ivar = 1.0f / max_var;
        OcTreeNode::max_ivar = 1.0f / min_var;
        OcTreeNode::min_known_ivar = 1.0f / max_known_var;
        OcTreeNode::free_thresh = free_thresh;
        OcTreeNode::occupied_thresh = occupied_thresh;
    }

    GPOctoMap::~GPOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void GPOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void GPOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        OcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void GPOctoMap::insert_training_data(const GPPointCloud &xy) {
        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, GPR3f *> gpr_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            }
            GPR3f *gpr = new GPR3f(OcTreeNode::sf2, OcTreeNode::ell, OcTreeNode::noise);
            gpr->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                gpr_arr.emplace(key, gpr);
            };
        }
#ifdef DEBUG
        Debug_Msg("GP training done");
        Debug_Msg("GP prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            Block *block;
#ifdef OPENMP
#pragma omp critical
#endif
            {
                block = search(key);
                if (block == nullptr)
//                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto gpr = gpr_arr.find(*block_it);
                if (gpr == gpr_arr.end())
                    continue;

                vector<float> m, var;
                gpr->second->predict(xs, m, var);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    node.update(m[j], var[j]);
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("GP prediction done");
#endif
        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            auto block = block_arr.find(key);
            if (block == block_arr.end())
                continue;
            block->second->prune();
        }
#ifdef DEBUG
        Debug_Msg("Pruning done");
#endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = gpr_arr.begin(); it != gpr_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void GPOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, GPR3f *> gpr_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            }
            GPR3f *gpr = new GPR3f(OcTreeNode::sf2, OcTreeNode::ell, OcTreeNode::noise);
            gpr->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                gpr_arr.emplace(key, gpr);
            };
        }
#ifdef DEBUG
        Debug_Msg("GP training done");
        Debug_Msg("GP prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto gpr = gpr_arr.find(*block_it);
                if (gpr == gpr_arr.end())
                    continue;

                vector<float> m, var;
                gpr->second->predict(xs, m, var);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    node.update(m[j], var[j]);
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("GP prediction done");
#endif
        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        /////////////////////////////////////////////////
// #ifdef OPENMP
// #pragma omp parallel for
// #endif
//         for (int i = 0; i < test_blocks.size(); ++i) {
//             BlockHashKey key = test_blocks[i];
//             auto block = block_arr.find(key);
//             if (block == block_arr.end())
//                 continue;
//             block->second->prune();
//         }
// #ifdef DEBUG
//         Debug_Msg("Pruning done");
// #endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = gpr_arr.begin(); it != gpr_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void GPOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPPointCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(it->second->get_center(), 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    void GPOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {
        PCLPointCloud sampled_hits;
        downsample(cloud, sampled_hits, ds_resolution);

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        xy.clear();
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);
            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            xy.emplace_back(p, 1.0f);

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);

            frees.push_back(PCLPointType(origin.x(), origin.y(), origin.z()));
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
                frees.push_back(PCLPointType(p->x(), p->y(), p->z()));
                frees.width++;
            }
        }

        PCLPointCloud sampled_frees;
        downsample(frees, sampled_frees, ds_resolution);

        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.emplace_back(point3f(it->x, it->y, it->z), -1);
        }
    }

    void GPOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
        if (ds_resolution < 0) {
            out = in;
            return;
        }

        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(out);

//        vector<int> indices;
//        pcl_out.is_dense = false;
//        pcl::removeNaNFromPointCloud(out, out, indices);
    }

    void GPOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution) const {
        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = free_resolution;
        while (d < l) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d += free_resolution;
        }
        if (l > free_resolution)
            frees.emplace_back(x0 + nx * (l - free_resolution), y0 + ny * (l - free_resolution), z0 + nz * (l - free_resolution));
    }

    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void GPOctoMap::bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x());
            y.push_back(it->first.y());
            z.push_back(it->first.z());
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void GPOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int GPOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         GPPointCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int GPOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int GPOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         GPPointCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, GPOctoMap::search_callback, static_cast<void *>(&out));
    }

    int GPOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, GPOctoMap::count_callback, NULL);
    }

    bool GPOctoMap::count_callback(GPPointType *p, void *arg) {
        return false;
    }

    bool GPOctoMap::search_callback(GPPointType *p, void *arg) {
        GPPointCloud *out = static_cast<GPPointCloud *>(arg);
        out->push_back(*p);
        return true;
    }


    int GPOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int GPOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         GPPointCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *GPOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    OcTreeNode GPOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
            return OcTreeNode();
        } else {
            return OcTreeNode(block->search(p));
        }
    }

    OcTreeNode GPOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
