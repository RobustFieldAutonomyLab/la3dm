#include <string>
#include <iostream>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <unordered_set>
#include <gpregressor.h>
#include "bgkoctomap.h"
#include "markerarray_pub.h"

tf::TransformListener *listener;
std::string frame_id("/map");
double resolution = 0.1;
double free_resolution = 0.1;
double max_range = 8.0;
bool original_size = false;
la3dm::BGKOctoMap *map;

la3dm::MarkerArrayPub *m_pub;;
int count = 0;

tf::Vector3 last_position;
tf::Quaternion last_orientation;
bool first = true;
double position_change_thresh = 0.1;
double orientation_change_thresh = 0.2;
bool updated = false;

la3dm::MarkerArrayPub *f_pub;
la3dm::MarkerArrayPub *ray_pub;

ros::Publisher grid_pub;
nav_msgs::OccupancyGrid grid;

// la3dm::MarkerArrayPub *ig_pub;
la3dm::MarkerArrayPub *colorbar_pub;

ros::Publisher arrow_pub;
visualization_msgs::MarkerArray arrow_msg;

struct pair_hash {
    inline std::size_t operator()(const std::pair<la3dm::BlockHashKey, la3dm::OcTreeHashKey> &p) const {
        std::hash<la3dm::BlockHashKey> b_hash;
        std::hash<la3dm::OcTreeHashKey > n_hash;
        return b_hash(p.first) ^ n_hash(p.second);
    }
};

void publish_project_2d_map(const la3dm::BGKOctoMap &map) {
    ROS_INFO_STREAM("Projecting 2D map...");
    la3dm::point3f lim_min, lim_max;
    map.get_bbox(lim_min, lim_max);
    float resolution = map.get_resolution();
    unsigned short lim = static_cast<unsigned short>(pow(2, map.get_block_depth() - 1));
    unsigned short block_width = static_cast<unsigned short>((lim_max.x() - lim_min.x() + 0.01) / map.get_block_size());
    unsigned short block_height = static_cast<unsigned short>((lim_max.y() - lim_min.y() + 0.01) /
                                                              map.get_block_size());
    unsigned int width = lim * block_width;
    unsigned int height = lim * block_height;
    grid.info.width = width;
    grid.info.height = height;
    grid.info.origin.position.x = lim_min.x();
    grid.info.origin.position.y = lim_min.y();
    grid.info.resolution = resolution;
    grid.header.frame_id = frame_id;
    grid.header.stamp = ros::Time::now();
    float z0 = 0.05f, z_sensor = 0.35f;

    std::vector<la3dm::point3f> candidates;
    grid.data = std::vector<int8_t>(width * height, -1);
    for (unsigned short bi = 0; bi < block_height; ++bi) {
        float y = lim_min.y() + (bi + 0.5f) * map.get_block_size();
        for (unsigned short bj = 0; bj < block_width; ++bj) {
            float x = lim_min.x() + (bj + 0.5f) * map.get_block_size();
            la3dm::Block *block = map.search(la3dm::block_to_hash_key(x, y, z0));
            if (block == nullptr)
                continue;

            unsigned short ix, iy, iz;
            block->get_index(la3dm::point3f(x, y, z0), ix, iy, iz);
            for (unsigned short i = 0; i < lim; ++i) {
                for (unsigned short j = 0; j < lim; ++j) {
                    la3dm::OcTreeNode &node = (*block)[block->get_node(j, i, iz)];
                    int index = (bi * lim + i) * width + bj * lim + j;
                    if (node.get_state() == la3dm::State::FREE) {
                        grid.data[index] = 0;
                        candidates.emplace_back(lim_min.x() + resolution * (bj * lim + j + 0.5),
                                                lim_min.y() + resolution * (bi * lim + i + 0.5),
                                                z_sensor + 0.5 * resolution);
                    } else if (node.get_state() == la3dm::State::OCCUPIED)
                        grid.data[index] = 100;
                    else
                        grid.data[index] = -1;
                }
            }
        }
    }
}


void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &cloud) {
    tf::StampedTransform transform;
    try {
        listener->lookupTransform(frame_id, cloud->header.frame_id, cloud->header.stamp, transform);
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        return;
    }

    ros::Time start = ros::Time::now();
    la3dm::point3f origin;
    tf::Vector3 translation = transform.getOrigin();
    tf::Quaternion orientation = transform.getRotation();
    if (first || orientation.angleShortestPath(last_orientation) > orientation_change_thresh ||
        translation.distance(last_position) > position_change_thresh) {
        first = false;
        last_position = translation;
        last_orientation = orientation;
        origin.x() = (float) translation.x();
        origin.y() = (float) translation.y();
        origin.z() = (float) translation.z();
        ROS_INFO_STREAM(origin);

        sensor_msgs::PointCloud2 cloud_map;
        pcl_ros::transformPointCloud(frame_id, *cloud, cloud_map, *listener);

        la3dm::PCLPointCloud pcl_cloud;
        pcl::fromROSMsg(cloud_map, pcl_cloud);
        map->insert_pointcloud(pcl_cloud, origin, (float) resolution, (float) free_resolution, (float) max_range);

        ros::Time end = ros::Time::now();
        ROS_INFO_STREAM("One cloud finished in " << (end - start).toSec() << "s");
        updated = true;
    }

    if (count == 0 && updated) {
        la3dm::point3f lim_min, lim_max;
        float min_z, max_z;
        map->get_bbox(lim_min, lim_max);
        min_z = lim_min.z();
        max_z = lim_max.z();
        m_pub->clear();
        for (auto it = map->begin_leaf(); it != map->end_leaf(); ++it) {
            if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
                if (original_size) {
                    la3dm::point3f p = it.get_loc();
                    m_pub->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
                } else {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n) {
                        m_pub->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution());
                    }
                }
            }
        }
        updated = false;

        ///////// Compute Frontiers /////////////////////
        ROS_INFO_STREAM("Computing frontiers");
        f_pub->clear();
        for (auto it = map->begin_leaf(); it != map->end_leaf(); ++it) {
            la3dm::point3f p = it.get_loc();
            if (p.z() > 2.0 || p.z() < 0.3)
                continue;

            if (it.get_node().get_var() > 0.02 &&
                it.get_node().get_prob() < 0.1) {
                f_pub->insert_point3d(p.x(), p.y(), p.z());
            }
        }

        publish_project_2d_map(*map);

        m_pub->publish();
        f_pub->publish();
        grid_pub.publish(grid);
        // ig_pub->publish();
        colorbar_pub->publish();
        arrow_pub.publish(arrow_msg);
    }
    count = (++count) % 10;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "bgkoctomap_server");
    ros::NodeHandle nh("~");

    std::string cloud_topic("/pointcloud");
    std::string map_topic("/occupied_cells_vis_array");
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    double noise = 0.01;
    double l = 100;
    double min_var = 0.001;
    double max_var = 1000;
    double max_known_var = 0.02;
    double ds_resolution = 0.1;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    float var_thresh = 1.0f;
    float prior_A = 1.0f;
    float prior_B = 1.0f;

    nh.param<std::string>("map_topic", map_topic, map_topic);
    nh.param<std::string>("cloud_topic", cloud_topic, cloud_topic);
    nh.param<std::string>("frame_id", frame_id, frame_id);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<double>("noise", noise, noise);
    nh.param<double>("l", l, l);
    nh.param<double>("min_var", min_var, min_var);
    nh.param<double>("max_var", max_var, max_var);
    nh.param<double>("max_known_var", max_known_var, max_known_var);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<bool>("original_size", original_size, original_size);
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<float>("prior_A", prior_A, prior_A);
    nh.param<float>("prior_B", prior_B, prior_B);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
                    "map_topic: " << map_topic << std::endl <<
                    "cloud_topic: " << cloud_topic << std::endl <<
                    "frame_id: " << frame_id << std::endl <<
                    "max_range: " << max_range << std::endl <<
                    "resolution: " << resolution << std::endl <<
                    "block_depth: " << block_depth << std::endl <<
                    "sf2: " << sf2 << std::endl <<
                    "ell: " << ell << std::endl <<
                    "l: " << l << std::endl <<
                    "min_var: " << min_var << std::endl <<
                    "max_var: " << max_var << std::endl <<
                    "max_known_var: " << max_known_var << std::endl <<
                    "free_resolution: " << free_resolution << std::endl <<
                    "ds_resolution: " << ds_resolution << std::endl <<
                    "free_thresh: " << free_thresh << std::endl <<
                    "occupied_thresh: " << occupied_thresh << std::endl <<
                    "original_size: " << original_size << std::endl <<
                    "var_thresh: " << var_thresh << std::endl <<
                    "prior_A: " << prior_A << std::endl <<
                    "prior_B: " << prior_B
    );

    map = new la3dm::BGKOctoMap(resolution, block_depth, sf2, ell, free_thresh, occupied_thresh, var_thresh, prior_A, prior_B);

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(cloud_topic, 0, cloud_callback);
    m_pub = new la3dm::MarkerArrayPub(nh, map_topic, 0.1f);
    listener = new tf::TransformListener();

    f_pub = new la3dm::MarkerArrayPub(nh, "frontier_map", resolution);
    ray_pub = new la3dm::MarkerArrayPub(nh, "ray", resolution);
    // ig_pub = new la3dm::MarkerArrayPub(nh, "ig", resolution);
    colorbar_pub = new la3dm::MarkerArrayPub(nh, "colorbar", resolution);

    grid_pub = nh.advertise<nav_msgs::OccupancyGrid>("/map", 0, false);

    arrow_pub = nh.advertise<visualization_msgs::MarkerArray>("/op_orient", 0, true);

    ROS_INFO_STREAM("Start mapping...");
    ros::spin();
    delete map;
    delete listener;
    delete m_pub;
    delete f_pub;
    delete ray_pub;

    return 0;
}
