/**
    Real-time Object 2-D Recognition
    Created by Hui Hu for CS 5330 Computer Vision Spring 2023

    Functions to extract and save feature vector
*/

#ifndef FETCH_FEATURES
#define FETCH_FEATURES

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;


// Applies an adaptive threshold to an image and return corresponding binary image
int threshold(Mat &src, Mat &des);

// Segment the image into regions and return the locations of each region
int regionSegment(Mat &src, vector<vector<int>> &regions, Mat &regionMap, int N=1);

// Compute features for each major region
float getFeatureVec(Mat &src, vector<float> &feature, vector<int> region);

// Display the label of one region 
int displayLabel(Mat &src, vector<int> region, char* label, bool isWhite = false);

#endif