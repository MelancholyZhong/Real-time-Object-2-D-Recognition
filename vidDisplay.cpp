#include <iostream>
#include <string>
#include <filesystem>
#include<vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "filters.h"
#include "match.h"
#include "csv_util.h"



int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        
        //image saving settings
        char dirName[256] = "database.csv";
        std::__fs::filesystem::create_directory("./captured");
        int captured = 0;

        //mode settings
        int mode = 0; //0 for only threshold and cleaning mode and 1 for recognition

        //initial threshold
        int threshold = 150;

        //load the database(will re-load if any change)
        std::vector<char *> labels;
        std::vector<std::vector<float>> data;

        
        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }

                cv::Mat res1;
                cv::Mat res2;
                thresholding(frame, res1, threshold);
                closing(res1, res2);
                cv::imshow("Video", res2);
                if(mode == 1){
                        if(labels.size()==0){
                                read_image_data_csv( dirName, labels, data );
                        }
                        std::vector<float> feature;
                        feature.push_back(0.8);
                        feature.push_back(0.7);
                        char label[256];
                        nearest3(labels, data, feature, label);
                        std::cout<< label <<std::endl;
                        //cv::waitKey(1000); //dont want too much 
                }
                
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                if( key == 'q') {
                    break;
                }else if (key == 's'){
                        captured += 1;
                        std::string capturedStr = std::to_string(captured);
                        std::string fileName1 = "./captured/original_" + capturedStr + ".jpg";
                        std::string fileName2 = "./captured/thresholding_" + capturedStr + ".jpg";
                        std::string fileName3 = "./captured/closed_" + capturedStr + ".jpg";
                        cv::imwrite(fileName1,frame);
                        cv::imwrite(fileName2,res1);
                        cv::imwrite(fileName3,res2);
                }else if(key == 'm'){
                        mode == 0 ? mode = 1 : mode = 0;
                }else if(key == 'a'){
                        int newThreshold = adjustThreshold(frame, threshold);
                        threshold = newThreshold;
                        //std::cout << threshold << std::endl;
                }else if(key == 't'){
                        //For now i used a dummy feature vector.
                        std::vector<float> feature;
                        feature.push_back(0.9);
                        feature.push_back(0.9);
                        saveNewObject(frame, res2, feature, dirName);
                        labels.clear();
                        data.clear();
                        read_image_data_csv( dirName, labels, data );
                }
        }

        delete capdev;
        return(0);
}