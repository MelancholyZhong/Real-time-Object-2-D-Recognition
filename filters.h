/**
    Real-time Object 2-D Recognition
    Created by Yao Zhong for CS 5330 Computer Vision Spring 2023

    Functions to preprocessing the image, including adjust threshold using trackbar.
*/

#ifndef FILTERS
#define FILTERS

//the blur function to blur the image before binary
int blur5x5( cv::Mat &src, cv::Mat &dst );

//Task1:To threshold the image into a binary image
int thresholding( cv::Mat &src, cv::Mat &dst, int threshold);

//Task2: used opencv functions to do the closing operation(2 times of dilate followed by 2 times of erode)
int closing(cv::Mat &src, cv::Mat &dst);

//Allow users to adjust the threshold through this window with a trackbar, the new threshold will be returned 
int adjustThreshold(cv::Mat &frame, int threshold);

//helper funtion to ask for a label and record the feature into csv
int saveNewObject(cv::Mat &frame, cv::Mat &res, std::vector<float> &feature, char* dirName);


#endif