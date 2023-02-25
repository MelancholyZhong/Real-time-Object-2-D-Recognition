/**
    Real-time Object 2-D Recognition
    Created by Yao Zhong  and Hu Hui for CS 5330 Computer Vision Spring 2023

    Functions related to distances and classifiers.
*/


#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "csv_util.h"
#include "fetchFeature.h"
#include "filters.h"
#include "match.h"

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }
    cv::namedWindow("Video", 1); // identifies a window

    // image saving settings
    char dirName[256] = "database.csv";
    std::__fs::filesystem::create_directory("./captured");
    int captured = 0;

    // mode settings
    int mode = 0;               // 0 for only threshold and cleaning mode and 1 for recognition
    int classifier = 0; //0 for nearest 3 neighbor, 1 for nearest neigbor
    int threshold = 150;        // initial threshold
    int capacity = 2;  //recognize at most 2 objects at the same time

    //data preloading
    std::vector<char *> labels; // load the database (will re-load if any change)
    std::vector<std::vector<float>> data;

    //main loop
    for (;;) {
        cv::Mat frame;
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        //preprocessing
        cv::Mat res1; // result of thresholding
        cv::Mat res2; // result of closing
        cv::Mat res3; // used to be bounded and labeled
        cv::Mat regionMap;
        thresholding(frame, res1, threshold);
        closing(res1, res2);
        res3 = res2.clone();
        vector<vector<int>> regions = {};
        regionSegment(res2, regions, regionMap, capacity);


        //Start recognize if mode is 1
        if (mode == 1) {
            if (labels.size() == 0) {
                read_image_data_csv(dirName, labels, data);
            }
            
            for (int i = 0; i < regions.size(); i++) {
                // cout << "(" << regions[i][0] << "," << regions[i][1] << "," << regions[i][2] << "," << regions[i][3] << ")" << endl;

                std::vector<float> feature;
                float moment = getFeatureVec(res3, feature, regions[i]);
                char label[256] = {};
                if(classifier == 0){
                    nearest3(labels, data, feature, label);
                }else{
                    nearestNeighbor(labels, data, feature, label);
                }
                
                char text[256] = {};
                std::snprintf(text, sizeof(text), "%s - %.6f", label, moment);
                displayLabel(res3, regions[i], text, true);
            }
            
        }
        cv::imshow("Video", res3);

        // see if there is a waiting keystroke
        char key = cv::pollKey();
        if (key == 'q') {
            break;
        } else if (key == 's') {
            //save the images for the report
            captured += 1;
            std::string capturedStr = std::to_string(captured);
            std::string fileName1 = "./captured/original_" + capturedStr + ".jpg";
            std::string fileName2 = "./captured/thresholding_" + capturedStr + ".jpg";
            std::string fileName3 = "./captured/closed_" + capturedStr + ".jpg";
            std::string fileName4 = "./captured/oriented_" + capturedStr + ".jpg";
            std::string fileName5 = "./captured/segmented_" + capturedStr + ".jpg";
            cv::imwrite(fileName1, frame);
            cv::imwrite(fileName2, res1);
            cv::imwrite(fileName3, res2);
            cv::imwrite(fileName4, res3);
            cv::imwrite(fileName5, regionMap);
        } else if (key == 'r') {
            //change the mode of recognizing
            mode == 0 ? mode = 1 : mode = 0;
        } else if (key == 'a') {
            //adjust threshold
            int newThreshold = adjustThreshold(frame, threshold);
            threshold = newThreshold;
        } else if (key == 't') {
            //save the new object
            std::vector<float> feature;
            getFeatureVec(res2, feature, regions[0]);
            saveNewObject(frame, res2, feature, dirName);
            labels.clear();
            data.clear();
            read_image_data_csv(dirName, labels, data);
        }else if(key == 'n'){
            //set the classifier to knn
            classifier = 0;
        }else if(key == 'm'){
            //set the classifier to nearest neighbor
            classifier = 1;
        }else if(key == 'l'){
            //save the recognized image only
            captured += 1;
            std::string capturedStr = std::to_string(captured);
            std::string fileName = "./captured/classified_" + capturedStr + ".jpg";
            cv::imwrite(fileName, res3);
        }
    }
    delete capdev;
    return (0);
}