#ifndef FILTERS
#define FILTERS

int blur5x5( cv::Mat &src, cv::Mat &dst );

int thresholding( cv::Mat &src, cv::Mat &dst, int threshold);

int closing(cv::Mat &src, cv::Mat &dst);

int adjustThreshold(cv::Mat &frame, int threshold);

int saveNewObject(cv::Mat &frame, cv::Mat &res, std::vector<float> &feature, char* dirName);


#endif