#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include "bgklvoctomap.h"
#include "bgklvinference.h"
#include <iostream>

using std::vector;

//#define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace la3dm {

    BGKLVOctoMap::BGKLVOctoMap() : BGKLVOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        1.0, // sf2
                                        1.0, // ell
                                        0.3f, // free_thresh
                                        0.7f, // occupied_thresh
                                        1.0f, // var_thresh
                                        1.0f, // prior_A
                                        1.0f, // prior_B
                                        true, //original_size
                                        0.1f // min_W
                                    ) { }

    BGKLVOctoMap::BGKLVOctoMap(float resolution,
                        unsigned short block_depth,
                        float sf2,
                        float ell,
                        float free_thresh,
                        float occupied_thresh,
                        float var_thresh,
                        float prior_A,
                        float prior_B,
                        bool original_size,
                        float min_W)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow    (2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);

        OcTree::max_depth = block_depth;

        OcTreeNode::sf2 = sf2;
        OcTreeNode::ell = ell;
        OcTreeNode::free_thresh = free_thresh;
        OcTreeNode::occupied_thresh = occupied_thresh;
        OcTreeNode::var_thresh = var_thresh;
        OcTreeNode::prior_A = prior_A;
        OcTreeNode::prior_B = prior_B;
        OcTreeNode::original_size = original_size;
        OcTreeNode::min_W = min_W;
    }

    BGKLVOctoMap::~BGKLVOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void BGKLVOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKLVOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        OcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKLVOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPLineCloud xy;
        GPLineCloud rays;
        vector<int> ray_idx;

        if(ds_resolution > resolution){
            ds_resolution = resolution;
        }

        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, rays, ray_idx);
        assert (ray_idx.size() == xy.size());

#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        //define all blocks from point cloud input
        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        //insert training data into rtree
        for (int k = 0; k < xy.size(); ++k) {
            float p[] = {xy[k].first.x0(), xy[k].first.y0(), xy[k].first.z0()};
            rtree.Insert(p, p, k);
        }
        /////////////////////////////////////////////////

        ////////// Training & Prediction ////////////////
        /////////////////////////////////////////////////

        //define set of blocks that will be predicted
        vector<BlockHashKey> test_blocks;

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        //create key for each block from point cloud, begin loop to process each block
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];

#ifdef OPENMP
#pragma omp critical
#endif
        {
            //run in parallel, add block to block_arr if it doesn't already exist (block_arr is maintained)
            if (block_arr.find(key) == block_arr.end())
                block_arr.emplace(key, new Block(hash_key_to_block(key)));
        };
            Block *block = block_arr[key];
            bool Block_has_info = false;

            //half the region of influence
            point3f half_size(OcTreeNode::ell, OcTreeNode::ell, OcTreeNode::ell);

            //process each node within a block
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                
                float block_size = block->get_size(leaf_it);
                //skip larger blocks than base resolution
                if (block_size > Block::resolution)
                    continue;

                point3f p = block->get_loc(leaf_it);
                point3f lim_min = p - half_size;
                point3f lim_max = p + half_size;

                if(!has_gp_points_in_bbox(lim_min,lim_max))
                    continue;

                //find data for each node individually, xy_idx is raw data
                vector<int> xy_idx;
                get_gp_points_in_bbox(lim_min, lim_max, xy_idx);

                if(xy_idx.size() < 1)
                    continue;

                vector<int> ray_keys(rays.size(), 0); //rays.size number of 0's
                vector<float> block_x, block_y;

#ifdef OPENMP
#pragma omp critical
#endif
            {
                //run in parallel, define data that exists in node influence as hits or rays
                for (int j = 0; j < xy_idx.size(); ++j) {
                    if (ray_idx[xy_idx[j]] == -1) {
                        block_x.push_back(xy[xy_idx[j]].first.x0());
                        block_x.push_back(xy[xy_idx[j]].first.y0());
                        block_x.push_back(xy[xy_idx[j]].first.z0());
                        block_x.push_back(xy[xy_idx[j]].first.x0());
                        block_x.push_back(xy[xy_idx[j]].first.y0());
                        block_x.push_back(xy[xy_idx[j]].first.z0());
                        block_y.push_back(1.0f);
                    }
                    else if (ray_keys[ray_idx[xy_idx[j]]] == 0) {
                        ray_keys[ray_idx[xy_idx[j]]] = 1;
                        block_x.push_back(rays[ray_idx[xy_idx[j]]].first.x0());
                        block_x.push_back(rays[ray_idx[xy_idx[j]]].first.y0());
                        block_x.push_back(rays[ray_idx[xy_idx[j]]].first.z0());
                        block_x.push_back(rays[ray_idx[xy_idx[j]]].first.x1());
                        block_x.push_back(rays[ray_idx[xy_idx[j]]].first.y1());
                        block_x.push_back(rays[ray_idx[xy_idx[j]]].first.z1());
                        block_y.push_back(0.0f);
                    }

                }
            
            };

                //push training data into node (legacy code, used to push into block)
                BGKLV3f *bgklv = new BGKLV3f(OcTreeNode::sf2, OcTreeNode::ell);
                bgklv->train(block_x, block_y);

#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << bgklv_arr.size());
#endif

        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////

                vector<float> xs;
                vector<float> ybar, kbar;
                //p was defined as the current node earlier
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());

                //use training data in bgklv to predict ybar and kbar at xs
                bgklv->predict(xs, ybar, kbar);

                //update active node with predictions
                OcTreeNode &node = leaf_it.get_node();

                    if (kbar[0] > 0.001f){
                        node.update(ybar[0], kbar[0]);
                    }
                
                Block_has_info = true;
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif
            }

#ifdef OPENMP
#pragma omp critical
#endif
        {
            //run in parallel, after block iteration ends check whether to add to list of blocks that were updated
            if(Block_has_info){
                test_blocks.push_back(key);
            }
        };
        }


        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        ///////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for
#endif
        //only use updated blocks
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            auto block = block_arr.find(key);
            if (block == block_arr.end())
                continue;
            if (OcTreeNode::original_size)
                block->second->prune();
        }
#ifdef DEBUG
        Debug_Msg("Pruning done");
#endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////

        //only need to remove raw data from the rtree
        rtree.RemoveAll();
    }

    void BGKLVOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPLineCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(point6f(it->second->get_center()), 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    //method to build training dataset from raw pointcloud data
    void BGKLVOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPLineCloud &xy, GPLineCloud &rays, vector<int> &ray_idx) const {
        //downsample all incoming data
        PCLPointCloud sampled_hits;
        downsample(cloud, sampled_hits, ds_resolution);

        rays.clear();
        ray_idx.clear();
        xy.clear();
        int idx = 0;
        double offset = OcTreeNode::ell*pow(2,0.5);
        double influence = OcTreeNode::ell;
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);
            double l = (p - origin).norm();
            float nx = (p.x() - origin.x()) / l;
            float ny = (p.y() - origin.y()) / l;
            float nz = (p.z() - origin.z()) / l;

            //filter out points too far away, but keep rays up to max_range
            if (max_range > 0) {
                if (l < max_range){
                    l = (float) sqrt((p.x() - origin.x()) * (p.x() - origin.x()) + (p.y() - origin.y()) * (p.y() - origin.y()) + (p.z() - origin.z()) * (p.z() - origin.z()));
                    l = l-offset;
                    xy.emplace_back(point6f(p), 1.0f);
                    ray_idx.push_back(-1);
                }
                else{
                    // continue; //use continue to skip rays up to max_range
                    l = max_range-offset;
                }
            }

            point3f nearest_point = p;
            point3f free_endpt(origin.x() + nx * l, origin.y() + ny * l, origin.z() + nz * l);

            //find points "near" the ray
            PointCloud nearby_points;
            for (auto iter = sampled_hits.begin(); iter != sampled_hits.end(); ++iter) {
                point3f p0(iter->x, iter->y, iter->z);

                //filter out points too far away
                if (max_range > 0) {
                    double range = (p0 - origin).norm();
                    if (range > max_range)
                        continue;
                }
                
                //include free space near the floor (by removing floor points from nearby, currently using x-y plane as floor)
                if(p.z() > (offset+origin.z()) && p0.z() < origin.z()+influence){
                    continue;
                }

                double dist1 = (free_endpt-p0).norm();
                double dist2 = (origin-p0).norm();

                //check if endpt is within influence of ray
                if(dist1 < influence){
                    nearby_points.emplace_back(p0);
                }
                else if(dist1 < l && dist2 < l){
                    nearby_points.emplace_back(p0);
                }
            }

            //search through nearby points to reduce ray as necessary
            point3f line_vec = free_endpt-origin;
            for (auto p1 = nearby_points.begin(); p1 != nearby_points.end(); p1++){
                double dist;
                point3f pnt_vec = *p1 - origin;
                double b = pnt_vec.dot(line_vec);
                if(b > pow(l,2)){
                    continue;
                }
                else{
                    point3f nearest = origin + line_vec*(b/pow(line_vec.norm(),2));
                    dist = (*p1-nearest).norm();
                }

                if(dist < influence){
                    nearest_point = *p1;
                    l = b/line_vec.norm();
                }
            }

            //remove downward rays close to sensor
            if(l < max_range/5.0 && l/(offset-nearest_point.z()) > 0){
               continue;
            }

            free_endpt = point3f(origin.x() + nx * l, origin.y() + ny * l, origin.z() + nz * l);
            point3f free_origin = origin;

            //move free ray origin away from robot
            double mu = 1.0;
            if(l > influence*mu){
                free_origin = point3f(origin.x() + nx * influence*mu, origin.y() + ny * influence*mu, origin.z() + nz * influence*mu);
            }
            else{
                free_origin = free_endpt;
            }

            PointCloud frees;
            beam_sample(free_endpt, free_origin, frees, free_resolution);

            xy.emplace_back(point6f(free_origin.x(), free_origin.y(), free_origin.z()), 0.0f);
            ray_idx.push_back(idx);

            //plaxeholder points along the ray used to check if a ray is near a cell -> yes means use this ray
            for (auto p = frees.begin(); p != frees.end(); ++p) {
                xy.emplace_back(point6f(p->x(), p->y(), p->z()), 0.0f);
                ray_idx.push_back(idx);
            }

            point6f line6f(free_origin, free_endpt);
            rays.emplace_back(line6f, 0.0f);

            ++idx;
        }

    }

    void BGKLVOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
        if (ds_resolution < 0) {
            out = in;
            return;
        }

        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(out);
    }

    void BGKLVOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
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

        float d = l;
        while (d > 0.0) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d -= free_resolution;
        }
    }

    void BGKLVOctoMap::bbox(const GPLineCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x0());
            x.push_back(it->first.x1());
            y.push_back(it->first.y0());
            y.push_back(it->first.y1());
            z.push_back(it->first.z0());
            z.push_back(it->first.z1());
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

    void BGKLVOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int BGKLVOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         vector<int> &out) {
        point3f half_size(OcTreeNode::ell, OcTreeNode::ell, OcTreeNode::ell);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int BGKLVOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(OcTreeNode::ell, OcTreeNode::ell, OcTreeNode::ell);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int BGKLVOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         vector<int> &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKLVOctoMap::search_callback, static_cast<void *>(&out));
    }

    int BGKLVOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKLVOctoMap::count_callback, NULL);
    }

    bool BGKLVOctoMap::count_callback(int k, void *arg) {
        return false;
    }

    bool BGKLVOctoMap::search_callback(int k, void *arg) {
        vector<int> *out = static_cast<vector<int> *>(arg);
        out->push_back(k);
        return true;
    }


    int BGKLVOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int BGKLVOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         vector<int> &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *BGKLVOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    OcTreeNode BGKLVOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
            return OcTreeNode();
        } else {
            return OcTreeNode(block->search(p));
        }
    }

    OcTreeNode BGKLVOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
