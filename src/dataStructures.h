
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

#define ANALYZE_ONLY_FIRST_BBOX
//#define PRINT_TABLES_BOUNDING_BOXES
//#define CREATE_PNG_AVI_BOUNDING_BOXES
//#define SHOW_LIDAR_TOPVIEW_WITH_GROUND
//#define SHOW_LIDAR_TOPVIEW_WITHOUTH_GROUND
//#define SHOW_LIDAR_CLUSTERING_WITHOUT_CROPPING
//#define SHOW_LIDAR_CLUSTERING_CROPPING
//#define CREATE_PNG_AVI_TTC_LIDAR
//#define PRINT_TABLE_DIST_RATIO
//#define PRINT_TABLE_MEAN_DIST_RATIO
//#define SHOW_REFUSED_KEYPOINTS
//#define SHOW_TTC_CAMERA
#define SHOW_DISTANCE_LIDAR

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust
	cv::Scalar color;

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

#endif /* dataStructures_h */
