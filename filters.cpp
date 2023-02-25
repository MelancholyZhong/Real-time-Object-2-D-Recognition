/**
    Real-time Object 2-D Recognition
    Created by Yao Zhong for CS 5330 Computer Vision Spring 2023

    Functions to preprocessing the image, including adjust threshold using trackbar.
*/

#include <iostream>
#include <cmath>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "filters.h"
#include "csv_util.h"

//the blur function inherited from project1
int blur5x5( cv::Mat &src, cv::Mat &dst ){
    int g_filter[] = {1,2,4,2,1};
    int multiplied = 1+2+4+2+1;
    cv :: Mat inter = cv::Mat::zeros(src.size(), CV_8UC3);
    //do the horizontal blur first, with the intermideate mat
    for(int i=0;i<src.rows; i++){
        cv::Vec3b *srptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *irptr = inter.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            if(j<=1 || j>=src.cols-2){
                irptr[j][0] = srptr[j][0];
                irptr[j][1] = srptr[j][1];
                irptr[j][2] = srptr[j][2];
            }else{
                irptr[j][0] = (g_filter[0]*srptr[j-2][0] + g_filter[1]*srptr[j-1][0] + g_filter[2]*srptr[j][0]+g_filter[3]*srptr[j+1][0]+g_filter[4]*srptr[j+2][0])/multiplied;
                irptr[j][1] = (g_filter[0]*srptr[j-2][1] + g_filter[1]*srptr[j-1][1] + g_filter[2]*srptr[j][1]+g_filter[3]*srptr[j+1][1]+g_filter[4]*srptr[j+2][1])/multiplied;
                irptr[j][2] = (g_filter[0]*srptr[j-2][2] + g_filter[1]*srptr[j-1][2] + g_filter[2]*srptr[j][2]+g_filter[3]*srptr[j+1][2]+g_filter[4]*srptr[j+2][2])/multiplied;
            }
        }
    }
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    //then do the vertical blur, use intermediate and out to dst
    for(int j=0; j<src.cols; j++){
        for(int i=0;i<src.rows; i++){
            cv::Vec3b *drptr = dst.ptr<cv::Vec3b>(i);
            if(i<=1 || i>=src.rows-2){
                cv::Vec3b *irptr = inter.ptr<cv::Vec3b>(i);
                drptr[j][0] = irptr[j][0];
                drptr[j][1] = irptr[j][1];
                drptr[j][2] = irptr[j][2];
            }else{
                cv::Vec3b *irptrm2 = inter.ptr<cv::Vec3b>(i-2);
                cv::Vec3b *irptrm1 = inter.ptr<cv::Vec3b>(i-1);
                cv::Vec3b *irptr = inter.ptr<cv::Vec3b>(i);
                cv::Vec3b *irptrp1 = inter.ptr<cv::Vec3b>(i+1);
                cv::Vec3b *irptrp2 = inter.ptr<cv::Vec3b>(i+2);
                drptr[j][0] = (g_filter[0]*irptrm2[j][0]+g_filter[1]*irptrm1[j][0]+g_filter[2]*irptr[j][0]+g_filter[3]*irptrp1[j][0]+g_filter[4]*irptrp2[j][0])/multiplied;
                drptr[j][1] = (g_filter[0]*irptrm2[j][1]+g_filter[1]*irptrm1[j][1]+g_filter[2]*irptr[j][1]+g_filter[3]*irptrp1[j][1]+g_filter[4]*irptrp2[j][1])/multiplied;
                drptr[j][2] = (g_filter[0]*irptrm2[j][2]+g_filter[1]*irptrm1[j][2]+g_filter[2]*irptr[j][2]+g_filter[3]*irptrp1[j][2]+g_filter[4]*irptrp2[j][2])/multiplied;            
            }
        }
    }
    return 0;
}

//To threshold the image into a binary image
int thresholding( cv::Mat &src, cv::Mat &dst, int threshold){
    cv::Mat blured;
    blur5x5(src, blured); // blur the image before thresholding
    cv::Mat intermediate =  cv::Mat::zeros(src.size(), CV_8UC1);
    for(int i=0; i<src.rows; i++){
        uchar *irptr = intermediate.ptr<uchar>(i);
        cv::Vec3b *srptr = blured.ptr<cv::Vec3b>(i);
        for(int j=0; j<src.cols; j++){
            if(srptr[j][0]+srptr[j][1]+srptr[j][2] > threshold*3){
                irptr[j] = 0;
            }else{
                irptr[j] = 255;
            }
        }
    }
    dst = intermediate;
    return 0;
}

//Task2: used opencv functions to do the closing operation(2 times of dilate followed by 2 times of erode)
int closing(cv::Mat &src, cv::Mat &dst){
    int size = 1;
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size( 2*size + 1, 2*size+1 ), cv::Point( size, size ) );
    cv::dilate(src, dst,element);
    cv::dilate(dst, dst,element);
    cv::erode(dst, dst, element);
    cv::erode(dst, dst, element);
    return 0;
}

// call-back function for trackbar
static void on_trackbar( int threshold_slider, void* userData){
    cv::Mat frame = *(static_cast<cv::Mat*>(userData));
    cv::Mat res1;
    thresholding(frame, res1, threshold_slider);
    imshow( "Adjust Threshold", res1);
}

//Allow users to adjust the threshold through this window with a trackbar, the new threshold will be returned 
int adjustThreshold(cv::Mat &frame, int threshold){

    cv::namedWindow("Adjust Threshold", 1); // identifies a window
    //threshold settings(with trackbar)
    const int threshold_slider_max = 255;
    int threshold_slider = threshold; //default value of threshold is 150
    char TrackbarName[50];
    std::snprintf( TrackbarName, sizeof(TrackbarName), "Threshold:");
    cv::createTrackbar( TrackbarName, "Adjust Threshold", &threshold_slider, threshold_slider_max, on_trackbar, &frame);
    on_trackbar(threshold_slider, &frame);
    char key = cv::waitKey(0);
    while(key != 'a'){
        key = cv::waitKey(0);
    }
    cv::destroyWindow("Adjust Threshold");
    return threshold_slider;
}


//helper funtion to ask for a label and record the feature into csv
int saveNewObject(cv::Mat &frame, cv::Mat &res, std::vector<float> &feature, char* dirName){

    char label[256];
    std::cout << "intput a label for this object:" << std::endl;
    std::cin >> label;

    append_image_data_csv( dirName , label, feature);

    return 0;
}

