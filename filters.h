#ifndef FILTERS
#define FILTERS

int blur5x5( cv::Mat &src, cv::Mat &dst );

int thresholding( cv::Mat &src, cv::Mat &dst, int threshold);

int closing(cv::Mat &src, cv::Mat &dst);


#endif