#include <string>
#include <iostream>
#include <ros/ros.h>
#include <octomap/octomap.h>
#include <octomap_ros/conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <markerarray_pub.h>
#include <pcl_ros/point_cloud.h>

using std::string;

void load_pcd(std::string filename, octomap::point3d &origin, octomap::Pointcloud &scan) {
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f _origin;
    Eigen::Quaternionf orientaion;
    pcl::io::loadPCDFile(filename, cloud2, _origin, orientaion);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(cloud2, cloud);

    for (auto it = cloud.begin(); it != cloud.end(); ++it) {
        scan.push_back(it->x, it->y, it->z);
    }
    origin.x() = _origin[0];
    origin.y() = _origin[1];
    origin.z() = _origin[2];
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "octomap_static_node");
    ros::NodeHandle nh("~");

    std::string dir;
    std::string prefix;
    int scan_num = 0;
    std::string map_topic("/occupied_cells_vis_array");
    double max_range = -1;
    double resolution = 0.1;
    double min_z = 0;
    double max_z = 0;

    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("prefix", prefix, prefix);
    nh.param<std::string>("topic", map_topic, map_topic);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
                    "dir: " << dir << std::endl <<
                    "prefix: " << prefix << std::endl <<
                    "topic: " << map_topic << std::endl <<
                    "scan_sum: " << scan_num << std::endl <<
                    "max_range: " << max_range << std::endl <<
                    "resolution: " << resolution << std::endl <<
                    "min_z: " << min_z << std::endl <<
                    "max_z: " << max_z << std::endl
    );

    octomap::OcTree oc(resolution);

    ros::Time start = ros::Time::now();
    for (int scan_id = 1; scan_id <= scan_num; ++scan_id) {
        octomap::Pointcloud scan;
        octomap::point3d sensor_origin;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        load_pcd(filename, sensor_origin, scan);

        oc.insertPointCloud(scan, sensor_origin, max_range);
        ROS_INFO_STREAM("Scan " << scan_id << " done");
    }
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");

    ///////// Compute Frontiers /////////////////////
    // ROS_INFO_STREAM("Computing frontiers");
    // la3dm::MarkerArrayPub f_pub(nh, "frontier_map", resolution);
    // for (auto it = oc.begin_leafs(); it != oc.end_leafs(); ++it) {
    //     if (oc.isNodeOccupied(*it))
    //         continue;

    //     if (it.getZ() > 1.0 || it.getZ() < 0.3)
    //         continue;

    //     octomap::OcTreeKey key = it.getKey();
    //     octomap::OcTreeKey nkey;
    //     int n_unknown = 0;
    //     for (nkey[2] = key[2] - 1; nkey[2] <= key[2] + 1; ++nkey[2]) {
    //         for (nkey[1] = key[1] - 1; nkey[1] <= key[1] + 1; ++nkey[1]){
    //             for (nkey[0] = key[0] - 1; nkey[0] <= key[0] + 1; ++nkey[0]){
    //                 if (key != nkey){
    //                     octomap::OcTreeNode* node = oc.search(nkey);
    //                     n_unknown += node == NULL;
    //                 }
    //             }
    //         }
    //     }
    //     if (n_unknown >= 4) {
    //         f_pub.insert_point3d(it.getX(), it.getY(), it.getZ());
    //     }
    // }
    // f_pub.publish();

    ///////// Publish Map /////////////////////
    la3dm::MarkerArrayPub m_pub(nh, map_topic, resolution);
    if (min_z == max_z) {
        double min_x, min_y, max_x, max_y;
        oc.getMetricMin(min_x, min_y, min_z);
        oc.getMetricMax(max_x, max_y, max_z);
    }
    for (auto it = oc.begin_leafs(); it != oc.end_leafs(); ++it) {
        if (oc.isNodeOccupied(*it)) {
            m_pub.insert_point3d(it.getX(), it.getY(), it.getZ(), min_z, max_z, resolution);
        }
    }

    m_pub.publish();
    ros::spin();

    return 0;
}