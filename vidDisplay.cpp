#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "filters.h"


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

        std::__fs::filesystem::create_directory("./captured");
        int captured = 0;


        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }
                
                cv::Mat res1;
                thresholding(frame, res1, 150);
                cv::Mat res2;
                closing(res1, res2);
                cv::convertScaleAbs(res1, res1);
                cv::imshow("Video", res2);

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
                }
        }

        delete capdev;
        return(0);
}