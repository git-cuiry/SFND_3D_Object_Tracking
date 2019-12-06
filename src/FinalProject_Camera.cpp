
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char* argv[])
{
#ifdef ITERATE_ALL_DESCRIPTORS
	for (auto descriptorType : descriptorTypes) {
#endif
#ifdef ITERATE_ALL_DETECTORS
		for (auto detectorType : detectorTypes) {
#endif

			// As you can see here, https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#akaze,
			// AKAZE descriptors can only be used with KAZE or AKAZE keypoints
			if (detectorType.compare("AKAZE") != 0 && descriptorType.compare("AKAZE") == 0 ||
				detectorType.compare("AKAZE") == 0 && descriptorType.compare("AKAZE") != 0)
				continue;

			// As you can see here, https://answers.opencv.org/question/5542/sift-feature-descriptor-doesnt-work-with-orb-keypoinys/?answer=13268#post-id-13268
			// SIFT tunes its OCTAVE automatically while ORB use a fixed number of octaves. I didn't try to to any of the two workaround be cause I don't want to
			// break the style of the exercise so I decided to pass this test.
			if (detectorType.compare("SIFT") == 0 && descriptorType.compare("ORB") == 0)
				continue;

			/* INIT VARIABLES AND DATA STRUCTURES */

	// data location
	string dataPath = "../../../";

	// camera
	string imgBasePath = dataPath + "images/";
	string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
	string imgFileType = ".png";
	int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
	int imgEndIndex = 77;   // last file index to load
	int imgStepWidth = 1;
	int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

	// object detection
	string yoloBasePath = dataPath + "dat/yolo/";
	string yoloClassesFile = yoloBasePath + "coco.names";
	string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
	string yoloModelWeights = yoloBasePath + "yolov3.weights";

	// Lidar
	string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
	string lidarFileType = ".bin";

	// calibration data for camera and lidar
	cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
	cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
	cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation vector

	RT.at<double>(0, 0) = 7.533745e-03; RT.at<double>(0, 1) = -9.999714e-01; RT.at<double>(0, 2) = -6.166020e-04; RT.at<double>(0, 3) = -4.069766e-03;
	RT.at<double>(1, 0) = 1.480249e-02; RT.at<double>(1, 1) = 7.280733e-04; RT.at<double>(1, 2) = -9.998902e-01; RT.at<double>(1, 3) = -7.631618e-02;
	RT.at<double>(2, 0) = 9.998621e-01; RT.at<double>(2, 1) = 7.523790e-03; RT.at<double>(2, 2) = 1.480755e-02; RT.at<double>(2, 3) = -2.717806e-01;
	RT.at<double>(3, 0) = 0.0; RT.at<double>(3, 1) = 0.0; RT.at<double>(3, 2) = 0.0; RT.at<double>(3, 3) = 1.0;

	R_rect_00.at<double>(0, 0) = 9.999239e-01; R_rect_00.at<double>(0, 1) = 9.837760e-03; R_rect_00.at<double>(0, 2) = -7.445048e-03; R_rect_00.at<double>(0, 3) = 0.0;
	R_rect_00.at<double>(1, 0) = -9.869795e-03; R_rect_00.at<double>(1, 1) = 9.999421e-01; R_rect_00.at<double>(1, 2) = -4.278459e-03; R_rect_00.at<double>(1, 3) = 0.0;
	R_rect_00.at<double>(2, 0) = 7.402527e-03; R_rect_00.at<double>(2, 1) = 4.351614e-03; R_rect_00.at<double>(2, 2) = 9.999631e-01; R_rect_00.at<double>(2, 3) = 0.0;
	R_rect_00.at<double>(3, 0) = 0; R_rect_00.at<double>(3, 1) = 0; R_rect_00.at<double>(3, 2) = 0; R_rect_00.at<double>(3, 3) = 1;

	P_rect_00.at<double>(0, 0) = 7.215377e+02; P_rect_00.at<double>(0, 1) = 0.000000e+00; P_rect_00.at<double>(0, 2) = 6.095593e+02; P_rect_00.at<double>(0, 3) = 0.000000e+00;
	P_rect_00.at<double>(1, 0) = 0.000000e+00; P_rect_00.at<double>(1, 1) = 7.215377e+02; P_rect_00.at<double>(1, 2) = 1.728540e+02; P_rect_00.at<double>(1, 3) = 0.000000e+00;
	P_rect_00.at<double>(2, 0) = 0.000000e+00; P_rect_00.at<double>(2, 1) = 0.000000e+00; P_rect_00.at<double>(2, 2) = 1.000000e+00; P_rect_00.at<double>(2, 3) = 0.000000e+00;

	// misc
	double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
	int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
	vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
	bool bVis = false;            // visualize results

	/* MAIN LOOP OVER ALL IMAGES */


	for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth)
	{
		/* LOAD IMAGE INTO BUFFER */

		// assemble filenames for current index
		ostringstream imgNumber;
		imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
		string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

		// load image from file 
		cv::Mat img = cv::imread(imgFullFilename);

		// push image into data frame buffer
		DataFrame frame;
		frame.cameraImg = img;
		dataBuffer.push_back(frame);

		cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


		/* DETECT & CLASSIFY OBJECTS */

		float confThreshold = 0.2;
		float nmsThreshold = 0.4;
		detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
			yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

		cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


		/* CROP LIDAR POINTS */

		// load 3D Lidar points from file
		string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
		std::vector<LidarPoint> lidarPoints;
		loadLidarFromFile(lidarPoints, lidarFullFilename);

#ifdef SHOW_LIDAR_TOPVIEW_WITH_GROUND
		showLidarTopviewAndCreatePng(lidarPoints, cv::Size(10.0, 20.0), (dataBuffer.end() - 1)->cameraImg, dataBuffer.size(), false);
#endif
#ifdef SHOW_LIDAR_TOPVIEW_WITHOUTH_GROUND
		showLidarTopviewAndCreatePng(lidarPoints, cv::Size(10.0, 20.0), (dataBuffer.end() - 1)->cameraImg, dataBuffer.size(), true);
#endif

#ifndef SHOW_LIDAR_CLUSTERING_WITHOUT_CROPPING
		// remove Lidar points based on distance properties
		float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 10.0, maxY = 2.0, minR = 0.1; // focus on ego lane
		cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
#endif

		(dataBuffer.end() - 1)->lidarPoints = lidarPoints;

		cout << "#3 : CROP LIDAR POINTS done" << endl;


		/* CLUSTER LIDAR POINT CLOUD */

		// associate Lidar points with camera-based ROI
		float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
		clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

		// Visualize 3D objects
		bVis = false;
		if (bVis)
		{
			show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(640, 480), true);
		}
		bVis = false;

		cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;


		// REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
		//continue; // skips directly to the next image without processing what comes beneath

		/* DETECT IMAGE KEYPOINTS */

		// convert current image to grayscale
		cv::Mat imgGray;
		cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

#ifndef ITERATE_ALL_DETECTORS
			string detectorType = "SHITOMASI";
#endif

			// extract 2D keypoints from current image
			vector<cv::KeyPoint> keypoints; // create empty feature list for current image

			if (detectorType.compare("SHITOMASI") == 0)
			{
				detKeypointsShiTomasi(keypoints, imgGray, false);
			}
			else if (detectorType.compare("HARRIS") == 0)
			{
				detKeypointsHarris(keypoints, imgGray, false);
			}
			else
			{
				detKeypointsModern(keypoints, imgGray, detectorType, false);
			}

			// optional : limit number of keypoints (helpful for debugging and learning)
			bool bLimitKpts = false;
			if (bLimitKpts)
			{
				int maxKeypoints = 50;

				if (detectorType.compare("SHITOMASI") == 0)
				{ // there is no response info, so keep the first 50 as they are sorted in descending quality order
					keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
				}
				cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
				cout << " NOTE: Keypoints have been limited!" << endl;
			}

			// push keypoints and descriptor for current frame to end of data buffer
			(dataBuffer.end() - 1)->keypoints = keypoints;

			cout << "#5 : DETECT KEYPOINTS done" << endl;


			/* EXTRACT KEYPOINT DESCRIPTORS */
#ifndef ITERATE_ALL_DESCRIPTORS
				string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
#endif



			cv::Mat descriptors;
			descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

			// push descriptors for current frame to end of data buffer
			(dataBuffer.end() - 1)->descriptors = descriptors;

			cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


			if (dataBuffer.size() > 1) // wait until at least two images have been processed
			{

				/* MATCH KEYPOINT DESCRIPTORS */
					vector<cv::DMatch> matches;
					string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
					string _descriptorType = 0 == descriptorType.compare("SIFT") ? "DES_HOG" : "DES_BINARY";
					string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

					matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
						(dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
						matches, _descriptorType, matcherType, selectorType);

					// store matches in current data frame
					(dataBuffer.end() - 1)->kptMatches = matches;

					cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


					/* TRACK 3D OBJECT BOUNDING BOXES */

					//// STUDENT ASSIGNMENT
					//// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
					map<int, int> bbBestMatches;
					matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1)); // associate bounding boxes between current and previous frame using keypoint matches
					//// EOF STUDENT ASSIGNMENT

					// store matches in current data frame
					(dataBuffer.end() - 1)->bbMatches = bbBestMatches;

					// Match colors from previous bounding box
					for (auto& bb : (dataBuffer.end() - 1)->boundingBoxes)
					{
						// This bounding box maybe has a match with a previous bounding box or not. In case it has a match, we get the color from the previous
						for (const auto& match : bbBestMatches)
						{
							if (match.second == bb.boxID)
								bb.color = (dataBuffer.end() - 2)->boundingBoxes[match.first].color;
						}
					}

#ifdef CREATE_PNG_AVI_BOUNDING_BOXES
					auto visImg = (dataBuffer.end() - 1)->cameraImg.clone();

					for (auto& bb : (dataBuffer.end() - 1)->boundingBoxes)
					{
						cv::rectangle(visImg, cv::Point(bb.roi.x, bb.roi.y), cv::Point(bb.roi.x + bb.roi.width, bb.roi.y + bb.roi.height), bb.color, 2);
						cv::imwrite(cv::format("boundingBox%d.png", dataBuffer.size() - 1), visImg);
					}
#endif

#if defined(SHOW_LIDAR_CLUSTERING_WITHOUT_CROPPING) || defined(SHOW_LIDAR_CLUSTERING_CROPPING)
					show3DObjects(*(dataBuffer.end() - 1), cv::Size(10.0, 20.0), (dataBuffer.end() - 1)->cameraImg, imgStartIndex + dataBuffer.size() - 1, P_rect_00, R_rect_00, RT, false);
#endif

					cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


					/* COMPUTE TTC ON OBJECT IN FRONT */
#ifdef CREATE_PNG_AVI_TTC_LIDAR
					auto visImg = (dataBuffer.end() - 1)->cameraImg.clone();
					cv::putText(visImg, cv::format("Frame: %d", imgStartIndex + dataBuffer.size() - 1), cv::Point2d(30, 30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
#endif

					auto ttcLidar = numeric_limits<double>::infinity();

					// loop over all BB match pairs
					for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
					{
						// find bounding boxes associates with current match
						BoundingBox* prevBB = nullptr;
						BoundingBox* currBB = nullptr;
						for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
						{
							if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
							{
								currBB = &(*it2);
							}
						}

						for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
						{
							if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
							{
								prevBB = &(*it2);
							}
						}

						// compute TTC for current match
						if (nullptr == currBB || nullptr == prevBB)
							continue;

						if (currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0) // only compute TTC if we have Lidar points
						{
							//// STUDENT ASSIGNMENT
							//// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
							double ttc;
							computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttc);

							ttcLidar = __min(ttcLidar, ttc);

#ifdef CREATE_PNG_AVI_TTC_LIDAR
							showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
							cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
#endif

							//// EOF STUDENT ASSIGNMENT

							//// STUDENT ASSIGNMENT
							//// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
							//// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
							double ttcCamera;
							clusterKptMatchesWithROI(*prevBB, *currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);

#ifdef SHOW_REFUSED_KEYPOINTS
							cv::Mat result;
							cv::drawMatches(
								(dataBuffer.end() - 2)->cameraImg,
								(dataBuffer.end() - 2)->keypoints,
								(dataBuffer.end() - 1)->cameraImg,
								(dataBuffer.end() - 1)->keypoints,
								currBB->kptMatches,
								result);

							cv::imwrite(cv::format("refused%d.png", dataBuffer.size() - 1), result);
#endif

							if (0 == currBB->kptMatches.size())
								ttcCamera = numeric_limits<double>::quiet_NaN();
							else
								computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
							//// EOF STUDENT ASSIGNMENT

#ifdef SHOW_TTC_CAMERA
							std::ofstream output_stream(detectorType + "-" + descriptorType + ".csv", std::ofstream::out | std::ofstream::app);
							output_stream << ttcCamera << endl;

							cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
							for (const auto& match : currBB->kptMatches)
								cv::circle(visImg, (dataBuffer.end() - 1)->keypoints[match.trainIdx].pt, 2, cv::Scalar(0, 255, 0));

							cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

							cv::putText(visImg, cv::format("Frame: %d", imgStartIndex + dataBuffer.size() - 1), cv::Point2d(30, 30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
							if (ttcCamera == numeric_limits<double>::infinity())
								cv::putText(visImg, cv::format("Time to collision (camera): Infinite", ttcCamera), cv::Point2d(30, 60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
							else
								cv::putText(visImg, cv::format("Time to collision (camera): %.2f seconds", ttcCamera), cv::Point2d(30, 60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));

							cv::imwrite(cv::format("ttcCamera%d.png", dataBuffer.size() - 1), visImg);

							string windowName = "Final Results : TTC";
							cv::namedWindow(windowName, 4);
							cv::imshow(windowName, visImg);
							cout << "Press key to continue to next frame" << endl;
							//cv::waitKey(0);
#endif

#ifdef ANALYZE_ONLY_FIRST_BBOX
							break;
#endif
						} // eof TTC computation
					} // eof loop over all BB matches      

#ifdef SHOW_DISTANCE_LIDAR
			std::ofstream output_stream("distancelidar.csv", std::ofstream::out | std::ofstream::app);
			output_stream << ttcLidar << endl;
			output_stream.close();
#endif
#ifdef CREATE_PNG_AVI_TTC_LIDAR
			std::ofstream output_stream("lidar.csv", std::ofstream::out | std::ofstream::app);
			output_stream << ttcLidar << endl;
			output_stream.close();

			if (ttcLidar == numeric_limits<double>::infinity())
				cv::putText(visImg, cv::format("Time to collision (lidar): Infinite", ttcLidar), cv::Point2d(30, 60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
			else
				cv::putText(visImg, cv::format("Time to collision (lidar): %.2f seconds", ttcLidar), cv::Point2d(30, 60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
			cv::imwrite(cv::format("                             %d.png", dataBuffer.size() - 1), visImg);
#endif
		}

	} // eof loop over all images
#ifdef ITERATE_ALL_DESCRIPTORS
				}
#endif
#ifdef ITERATE_ALL_DETECTORS
			}
#endif

	return 0;
}
