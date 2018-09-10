#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>

// cuda gpu
#include <ndt_gpu/NormalDistributionsTransform.h>

namespace hdl_graph_slam {

std::string lidar_pose_ndt = "/home/lgw/Documents/project/ndt_mapping/pcd/lidar_pose_ndt.csv";
std::ofstream ndt_pose_outFile(lidar_pose_ndt.c_str(), std::ios::out);

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon
static gpu::GNormalDistributionsTransform anh_gpu_ndt;

// global variables
static pose previous_pose, guess_pose, current_pose, ndt_pose, added_pose, localizer_pose;

static bool initPose = true;
static ros::Publisher ndt_map_pub;
static pcl::PointCloud<pcl::PointXYZI> map, submap;
// pcl::PointCloud<pcl::PointXYZI>::Ptr map;
static double min_add_scan_shift = 1.0;
static int submap_size = 0;
static int submap_num = 0;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;  // current_pose - previous_pose

class ScanMatchingOdometryNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdometryNodelet() {}
  virtual ~ScanMatchingOdometryNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    ndt_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/ndt_map", 10032);
    points_sub = nh.subscribe("/filtered_points", 10032, &ScanMatchingOdometryNodelet::cloud_callback, this); //256
    read_until_pub = nh.advertise<std_msgs::Header>("/scan_matching_odometry/read_until", 10032);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 10032);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    auto& pnh = private_nh;
    odom_frame_id = pnh.param<std::string>("odom_frame_id", "odom");

    // The minimum tranlational distance and rotation angle between keyframes.
    // If this value is zero, frames are always compared with the previous frame
    keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);

    // Registration validation by thresholding
    transform_thresholding = pnh.param<bool>("transform_thresholding", false);
    max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);

    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = voxelgrid;
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::ApproximateVoxelGrid<PointT>> approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" <<std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      boost::shared_ptr<pcl::PassThrough<PointT>> passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }

    registration = select_registration_method(pnh);
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if(!ros::ok()) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud);
    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);

    // In offline estimation, point clouds until the published time will be supplied
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = "/velodyne_points";
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);

  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

Eigen::Matrix4f initGuessPose(){  // (const nav_msgs::Odometry::ConstPtr& novatelInput){

  previous_pose.x = 0.0;
  previous_pose.y = 0.0;
  previous_pose.z = 0.0;
  previous_pose.roll = 0.0;
  previous_pose.pitch = 0.0;
  previous_pose.yaw = 0; 

  ndt_pose.x = 0.0; // 0.0;
  ndt_pose.y = 0.0; // 0.0;
  ndt_pose.z = 0.0;
  ndt_pose.roll = 0.0;
  ndt_pose.pitch = 0.0;
  ndt_pose.yaw = 0.0;

  current_pose.x = 0.0; // 0.0;
  current_pose.y = 0.0; // 0.0;
  current_pose.z = 0.0;
  current_pose.roll = 0.0;
  current_pose.pitch = 0.0;
  current_pose.yaw = 0.0; 

  guess_pose.x = 0.0; // 0.0;
  guess_pose.y = 0.0; // 0.0;
  guess_pose.z = 0.0;
  guess_pose.roll = 0.0;
  guess_pose.pitch = 0.0;
  guess_pose.yaw = 0.0;

  diff_x = 0.0;
  diff_y = 0.0;
  diff_z = 0.0;
  diff_yaw = 0.0;

  Eigen::Matrix4f tmpTransform;
  Eigen::AngleAxisf init_rotation_x(0, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(0, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(0, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation(guess_pose.x, guess_pose.y, guess_pose.z);

  tmpTransform = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix();

  return tmpTransform;
}

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe cloud
   */
  Eigen::Matrix4f matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f initEigen;
    if(initPose == true){
      initEigen = initGuessPose();  // (novatelInput);
      pcl::transformPointCloud(*cloud, *transformed_scan_ptr, initEigen);
      map += *transformed_scan_ptr;
      initPose = false; // return ?
    }

    guess_pose.x = previous_pose.x + diff_x;
    guess_pose.y = previous_pose.y + diff_y;
    guess_pose.z = previous_pose.z + diff_z;
    guess_pose.roll = previous_pose.roll;
    guess_pose.pitch = previous_pose.pitch;
    guess_pose.yaw = previous_pose.yaw + diff_yaw;

    Eigen::AngleAxisf init_rotation_x(guess_pose.roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(guess_pose.pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(guess_pose.yaw, Eigen::Vector3f::UnitZ());

    Eigen::Translation3f init_translation(guess_pose.x, guess_pose.y, guess_pose.z);

    initEigen = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix();

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    // filtered = downsample(cloud);
    // registration->setInputSource(filtered);
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    float voxel_leaf_size = 2;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(cloud);
    voxel_grid_filter.filter(*filtered);

    anh_gpu_ndt.setTransformationEpsilon(trans_eps);
    anh_gpu_ndt.setStepSize(step_size);
    anh_gpu_ndt.setResolution(ndt_res);
    anh_gpu_ndt.setMaximumIterations(max_iter);
    anh_gpu_ndt.setInputSource(filtered);

    pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));

    // downsample(map_ptr);
    // registration->setInputTarget(map_ptr);
    anh_gpu_ndt.setInputTarget(map_ptr);

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    // registration->align(*aligned, initEigen);
    anh_gpu_ndt.align(initEigen);


    /////////****GPU****//////////
    // anh_gpu_ndt.setTransformationEpsilon(trans_eps);
    // anh_gpu_ndt.setStepSize(step_size);
    // anh_gpu_ndt.setResolution(ndt_res);
    // anh_gpu_ndt.setMaximumIterations(max_iter);
    // anh_gpu_ndt.setInputSource(filtered_scan_ptr);
    // anh_gpu_ndt.setInputTarget(map_ptr);

    // anh_gpu_ndt.align(init_guess);
    // fitness_score = anh_gpu_ndt.getFitnessScore();
    // t_localizer = anh_gpu_ndt.getFinalTransformation();
    // has_converged = anh_gpu_ndt.hasConverged();
    // final_num_iteration = anh_gpu_ndt.getFinalNumIteration();
    /////////****GPU****//////////

    // if(!registration->hasConverged()) {
    //   NODELET_INFO_STREAM("scan matching has not converged!!");
    //   NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
    //   return keyframe_pose * prev_trans;
    // }

    // Eigen::Matrix4f trans = registration->getFinalTransformation();
    Eigen::Matrix4f trans = anh_gpu_ndt.getFinalTransformation();
    // Eigen::Matrix4f odom = keyframe_pose * trans;
    pcl::transformPointCloud(*cloud, *transformed_scan_ptr, trans);

    tf::Matrix3x3 mat_l;

    mat_l.setValue(static_cast<double>(trans(0, 0)), static_cast<double>(trans(0, 1)),
                   static_cast<double>(trans(0, 2)), static_cast<double>(trans(1, 0)),
                   static_cast<double>(trans(1, 1)), static_cast<double>(trans(1, 2)),
                   static_cast<double>(trans(2, 0)), static_cast<double>(trans(2, 1)),
                   static_cast<double>(trans(2, 2)));

    // Update ndt_pose.
    current_pose.x = trans(0, 3);
    current_pose.y = trans(1, 3);
    current_pose.z = trans(2, 3);
    mat_l.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw, 1);

    Eigen::Affine3d tmp_T;
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        tmp_T(i, j) = trans(i, j);
      }
    }
    Eigen::Quaterniond ndt_pose_rotation(tmp_T.rotation());
    std::stringstream ss;
    ss << std::setprecision(12) << std::fixed;
    ss << stamp.toSec() << ", ";
    ss << trans(0, 3) << ", ";
    ss << trans(1, 3) << ", ";
    ss << trans(2, 3) << ", ";
    ss << ndt_pose_rotation.coeffs().x() << ", ";
    ss << ndt_pose_rotation.coeffs().y() << ", ";
    ss << ndt_pose_rotation.coeffs().z() << ", ";
    ss << ndt_pose_rotation.coeffs().w() << " ";

    ndt_pose_outFile << ss.str() << std::endl;



    // Calculate the offset (curren_pos - previous_pos)
    diff_x = current_pose.x - previous_pose.x;
    diff_y = current_pose.y - previous_pose.y;
    diff_z = current_pose.z - previous_pose.z;
    diff_yaw = current_pose.yaw - previous_pose.yaw;
    diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    previous_pose.x = current_pose.x;
    previous_pose.y = current_pose.y;
    previous_pose.z = current_pose.z;
    previous_pose.roll = current_pose.roll;
    previous_pose.pitch = current_pose.pitch;
    previous_pose.yaw = current_pose.yaw;

    // ndt_pose_outFile << std::setprecision(20) << stamp.toSec() << " ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.x << " ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.y << " ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.z << " ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.roll << " ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.pitch << " ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.yaw << std::endl;

    // ndt_pose_outFile << std::setprecision(20) << stamp.toSec() << ", ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.x << ", ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.y << ", ";
    // ndt_pose_outFile << std::setprecision(12) << current_pose.z << ", ";
    // ndt_pose_outFile << std::setprecision(12) << ndt_pose_rotation.coeffs().x() << ", ";
    // ndt_pose_outFile << std::setprecision(12) << ndt_pose_rotation.coeffs().y << ", ";
    // ndt_pose_outFile << std::setprecision(12) << ndt_pose_rotation.coeffs().z << ", ";
    // ndt_pose_outFile << std::setprecision(12) << ndt_pose_rotation.coeffs().w << std::endl;


    double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) + pow(current_pose.y - added_pose.y, 2.0));
    if (shift >= min_add_scan_shift)
    {
      submap_size += shift;
      map += *transformed_scan_ptr;
      submap += *transformed_scan_ptr;
      
      // pcl::PointCloud<pcl::PointXYZI>::Ptr map_tmp(new pcl::PointCloud<pcl::PointXYZI>(map));
      // pcl::PointCloud<pcl::PointXYZI>::Ptr submap_tmp(new pcl::PointCloud<pcl::PointXYZI>(submap));
      // //map = 
      // downsample(map_tmp);
      // //submap = 
      // downsample(submap_tmp);

      added_pose.x = current_pose.x;
      added_pose.y = current_pose.y;
      added_pose.z = current_pose.z;
      added_pose.roll = current_pose.roll;
      added_pose.pitch = current_pose.pitch;
      added_pose.yaw = current_pose.yaw;

      keyframe = filtered;
      // registration->setInputTarget(keyframe);
      keyframe_pose = trans;
      keyframe_stamp = stamp;
      // prev_trans.setIdentity();
    }

    // if (submap_size >= max_submap_size)
    if (submap_size >= 150)
    {
      std::string s0 = "/home/lgw/Documents/project/ndt_mapping/pcd/"; // modify 3
      std::string s1 = "submap_";
      std::string s2 = std::to_string(submap_num);
      std::string s3 = ".pcd";
      std::string pcd_filename = s0 + s1 + s2 + s3;

      if (submap.size() != 0)
      {
        // if (pcl::io::savePCDFileASCII(pcd_filename, submap) == -1)
        if (pcl::io::savePCDFileBinary(pcd_filename, submap) == -1)
        {
          std::cout << "Failed saving " << pcd_filename << "." << std::endl;
        }
        std::cout << "Saved " << pcd_filename << " (" << submap.size() << " points)" << std::endl;

        map = submap;
        submap.clear();
        submap_size = 0.0;
      }
      submap_num++;
    }

    sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(map, *map_msg_ptr);
    map_msg_ptr->header.frame_id = "map";
    ndt_map_pub.publish(*map_msg_ptr);

    return trans; // odom;

    // if(!keyframe) {
    //   prev_trans.setIdentity();
    //   keyframe_pose.setIdentity();
    //   keyframe_stamp = stamp;
    //   keyframe = downsample(cloud);
    //   registration->setInputTarget(keyframe);
    //   return Eigen::Matrix4f::Identity();
    // }

    // auto filtered = downsample(cloud);
    // registration->setInputSource(filtered);

    // pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    // registration->align(*aligned, prev_trans);

    // if(!registration->hasConverged()) {
    //   NODELET_INFO_STREAM("scan matching has not converged!!");
    //   NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
    //   return keyframe_pose * prev_trans;
    // }

    // Eigen::Matrix4f trans = registration->getFinalTransformation();
    // Eigen::Matrix4f odom = keyframe_pose * trans;

    // if(transform_thresholding) {
    //   Eigen::Matrix4f delta = prev_trans.inverse() * trans;
    //   double dx = delta.block<3, 1>(0, 3).norm();
    //   double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

    //   if(dx > max_acceptable_trans || da > max_acceptable_angle) {
    //     NODELET_INFO_STREAM("too large transform!!  " << dx << "[m] " << da << "[rad]");
    //     NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
    //     return keyframe_pose * prev_trans;
    //   }
    // }

    // prev_trans = trans;

    // auto keyframe_trans = matrix2transform(stamp, keyframe_pose, odom_frame_id, "keyframe");
    // keyframe_broadcaster.sendTransform(keyframe_trans);

    // double delta_trans = trans.block<3, 1>(0, 3).norm();
    // double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    // double delta_time = (stamp - keyframe_stamp).toSec();
    // if(delta_trans > keyframe_delta_trans || delta_angle > keyframe_delta_angle || delta_time > keyframe_delta_time) {
    //   keyframe = filtered;
    //   registration->setInputTarget(keyframe);

    //   keyframe_pose = odom;
    //   keyframe_stamp = stamp;
    //   prev_trans.setIdentity();
    // }

    // return odom;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& base_frame_id, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_frame_id, base_frame_id);
    odom_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id;

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = base_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    odom_pub.publish(odom);
  }


private:
  // ROS topics
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  ros::Subscriber points_sub;

  ros::Publisher odom_pub;
  tf::TransformBroadcaster odom_broadcaster;
  tf::TransformBroadcaster keyframe_broadcaster;

  std::string odom_frame_id;
  ros::Publisher read_until_pub;

  // keyframe parameters
  double keyframe_delta_trans;  // minimum distance between keyframes
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool transform_thresholding;  //
  double max_acceptable_trans;  //
  double max_acceptable_angle;

  // odometry calculation
  Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe
  Eigen::Matrix4f keyframe_pose;               // keyframe pose
  ros::Time keyframe_stamp;                    // keyframe time
  pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud

  //
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;
};

}

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::ScanMatchingOdometryNodelet, nodelet::Nodelet)
