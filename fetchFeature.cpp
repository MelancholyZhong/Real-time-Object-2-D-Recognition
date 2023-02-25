/**
    Real-time Object 2-D Recognition
    Created by Hui Hu for CS 5330 Computer Vision Spring 2023

    Functions to extract and save feature vector
*/

// basic
#include <map>
#include <math.h> // atan2
#include <queue>  // min heap
#include <string>
#include <vector>
// opencv
#include <opencv2/core/types.hpp> // Rect Moment Point
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// customized function to convert an image to gray scale
int greyscale(Mat &src, Mat &dst) {
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    Mat_<uchar> dstVec = dst;
    Mat_<Vec3b> srcVec = src;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // value = 0.299⋅R+0.587⋅G+0.114⋅B
            float value = 0.114 * srcVec(i, j)[0] + 0.587 * srcVec(i, j)[1] + 0.299 * srcVec(i, j)[2];
            dstVec(i, j) = int(value);
        }
    }
    return 0;
}

// Implement a 5x5 Gaussian filter as separable 1x5 filters ([1 2 4 2 1] vertical and horizontal)
int gaussianBlur(Mat &src, Mat &dst) {
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    Mat_<uchar> dstVec = dst;
    Mat_<float> tmpVec = Mat::zeros(dst.rows, dst.cols, CV_32FC1); // temporary vector, increases precision
    Mat_<uchar> srcVec = src;

    const int filter[5] = {1, 2, 4, 2, 1};

    int rows = src.rows;
    int cols = src.cols;
    int r, c, k; // loop variables
    unsigned short count;

    // Separable filter (time-efficient)
    // Vertical direction
    for (r = 2; r < rows - 2; r++) {
        for (c = 0; c < cols; c++) {
            for (k = r - 2; k <= r + 2; k++) {
                tmpVec(r, c) += filter[k - r + 2] * srcVec(k, c);
            }
        }
    }

    // Horizontal direction
    for (r = 2; r < rows - 2; r++) {
        for (c = 2; c < cols - 2; c++) {
            count = 0;
            for (k = c - 2; k <= c + 2; k++) {
                count += filter[k - c + 2] * tmpVec(r, k);
            }
            dstVec(r, c) = count / 100;
        }
    }

    // Deal with the edges
    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
            if (r < 2 || r >= rows - 2 || c < 2 || c >= cols - 2) {
                dstVec(r, c) = dstVec(min(max(r, 2), rows - 3), min(max(c, 2), cols - 3));
            }
        }
    }

    return 0;
}

// Get the mean of the (blockSize x blockSize) neighbourhood area for pixel (r, c)
uchar getMean(Mat_<uchar> &srcVec, int r, int c, int blockSize) {
    // the edge of neighbourhood area
    int rStart = max(0, r - blockSize / 2);
    int rEnd = min(srcVec.rows, r + blockSize / 2 + 1);
    int cStart = max(0, c - blockSize / 2);
    int cEnd = min(srcVec.cols, c + blockSize / 2 + 1);

    // save all values in neighbourhood area
    vector<uchar> vec = {};
    for (int r = rStart; r < rEnd; r++) {
        for (int c = cStart; c < cEnd; c++) {
            vec.push_back(srcVec(r, c));
        }
    }

    // sort vector to find the mean value
    sort(vec.begin(), vec.end());
    int a = (vec.size() - 1) / 2;
    int b = vec.size() / 2;
    uchar mean = vec[a] / 2 + vec[b] / 2;
    // cout << srcVec(r, c) << "-" << mean << "==" << vec[vec.size()-1] << endl;
    return mean;
}

// Applies an adaptive threshold to an image and return corresponding binary image
int threshold(Mat &src, Mat &dst) {
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat_<uchar> dstVec = dst;
    Mat greyImg, blurredImg;
    int r, c;
    int blockSize = 11;
    int C = 2;

    // Get greyscale image
    greyscale(src, greyImg);

    // Gaussian blur
    gaussianBlur(greyImg, blurredImg);

    // Adaptive threshold
    Mat_<uchar> tempVec = blurredImg;
    for (r = 0; r < blurredImg.rows; r++) {
        for (c = 0; c < blurredImg.cols; c++) {
            // Calculated individual threshold for each pixel
            uchar thresholdValue = getMean(tempVec, r, c, blockSize) - C;
            if (tempVec(r, c) > thresholdValue) {
                dstVec(r, c) = uchar(255);
            } else {
                dstVec(r, c) = 0;
            }
            // cout << int(dstVec(r, c)) << endl;
        }
    }

    return 0;
}

// Union Find (Disjoint Set) Data Structure
class UnionFind {
    int *root, *rank;

  public:
    // Constructor, create an empty union find data structure with N isolated points.
    UnionFind(int N) {
        root = new int[N];
        rank = new int[N];
        for (int i = 0; i < N; i++)
            // at first, each point's root is itself and rank is 1
            root[i] = i, rank[i] = 1;
    }
    // Destructor
    ~UnionFind() {
        delete[] root;
        delete[] rank;
    }

    // Return the root of component corresponding to given point.
    int find(int point) {
        int r = point;
        while (r != root[r])
            r = root[r];
        // shorten the height of tree to speed up future search
        while (point != r) {
            int temp = root[point];
            root[point] = r;
            point = temp;
        }
        return r;
    }

    // Merge two sets
    void merge(int x, int y) {
        int i = find(x);
        int j = find(y);
        if (i == j)
            return;
        // make smaller root point to larger one
        if (rank[i] < rank[j]) {
            root[i] = j, rank[j] += rank[i];
        } else {
            root[j] = i, rank[i] += rank[j];
        }
    }
};

// Global variable
map<int, int> area; // store (root, area) pairs

// To compare two region according to their size of area
class Comparator {
  public:
    int operator()(const int &region1, const int &region2) {
        return area[region1] > area[region2];
    }
};

// Segment the image into regions and return the locations of each region
int regionSegment(Mat &src, vector<vector<int>> &regions, int N) {
    area = {};
    Mat_<uchar> srcVec = src;
    UnionFind UF = UnionFind(src.cols * src.cols);

    // Segment the image into regions
    int cols = src.cols;
    int r, c;

    for (c = 0; c < cols; c++) {
        for (r = 0; r < src.rows; r++) {
            // 4-connected
            if (r - 1 > 0 && srcVec(r, c) > 0 && srcVec(r - 1, c) > 0) {
                UF.merge(r + c * cols, (r - 1) + c * cols); // upper point
            }
            if (c - 1 > 0 && srcVec(r, c) > 0 && srcVec(r, c - 1) > 0) {
                UF.merge(r + c * cols, r + (c - 1) * cols); // left point
            }
        }
    }

    // Record basic information of each region (area, position)
    map<int, vector<int>> position;
    for (r = 0; r < src.rows; r++) {
        for (c = 0; c < src.cols; c++) {
            if (srcVec(r, c) > 0) {
                // continue; // exclude the background point

                int root = UF.find(r + c * cols);
                auto search = area.find(root);
                if (search != area.end()) {
                    // update the area, position
                    area[root] += 1;
                    position[root][0] = std::min(position[root][0], r); // lowerest
                    position[root][1] = std::min(position[root][1], c); // leftmost
                    position[root][2] = std::max(position[root][2], r); // highest
                    position[root][3] = std::max(position[root][3], c); // rightmost
                } else {
                    // add new item
                    area[root] = 1;
                    position[root] = {r, c, r, c};
                }
            }
        }
    }

    // Find the first N largest regions
    priority_queue<int, vector<int>, Comparator> minHeap;
    for (auto region : area) {
        if (region.second > 1000) { // exclude spot
            // Add new item to min-Heap
            minHeap.push(region.first);
            if (minHeap.size() > N) {
                minHeap.pop();
            }
        }
    }

    // Add the positions of first N largest regions to res vector
    // If the regions that meet our requirement are less than N, we only return the qualified regions
    while (!minHeap.empty()) {
        int top = minHeap.top();
        regions.push_back(position[top]); // regions are sorted in increasing order of area
        // cout << position[top][0] << " " << position[top][1] << " " << position[top][2] << " " << position[top][3] << endl;
        minHeap.pop();
    }

    return 0;
}

// Compute features for one region
float getFeatureVec(Mat &src, vector<float> &feature, vector<int> region) {
    // Extract the region
    Mat crop = src(Range(region[0], region[2]), Range(region[1], region[3]));

    vector<vector<Point>> contours;
    findContours(crop, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(region[1], region[0]));
    if (contours.size() > 0) {
        RotatedRect minRect = minAreaRect(contours[0]);
        Point2f rect_points[4];
        minRect.points(rect_points);
        // cout << rect_points[0] << "," << rect_points[1] << "," << rect_points[2] << "," << rect_points[3] << endl;
        for (int j = 0; j < 4; j++) {
            line(src, rect_points[j], rect_points[(j + 1) % 4], Scalar(255), 3, LINE_AA);
        }
    }

    // Compute moments of the region
    Moments momentValue = moments(crop, true);
    double hu[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    HuMoments(momentValue, hu);

    double centroid_x = int(momentValue.m10 / momentValue.m00);
    double centroid_y = int(momentValue.m01 / momentValue.m00);
    double mu11 = momentValue.mu11 / momentValue.m00;
    double mu20 = momentValue.mu20 / momentValue.m00;
    double mu02 = momentValue.mu02 / momentValue.m00;
    double theta = 0.5 * atan2(2 * mu11, (mu20 - mu02));
    line(src, Point(int(centroid_x - cos(theta) * 500) + region[1], int(centroid_y - sin(theta) * 500) + region[0]), Point(int(centroid_x + cos(theta) * 500) + region[1], int(centroid_y + sin(theta) * 500) + region[0]), Scalar(128), 3, LINE_AA);

    // Log scale hu moments to male its value bigger
    for (int i = 0; i < 7; i++) {
        hu[i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
    }

    feature = {};
    for (int i = 0; i < 6; i++) {
        feature.push_back(float(hu[i]));
    }

    return feature[0]; // return the first value to display
}


// Display the label of one region
int displayLabel(Mat &src, vector<int> region, char *label, bool isWhite) {
    // add rectangle frame
    int width = region[3] - region[1] + 1;
    int height = region[2] - region[0] + 1;
    int thickness = 3;
    Rect frame(region[1], region[0], width, height);
    if (isWhite) {
        rectangle(src, frame, Scalar(255), thickness); // white
    } else {
        rectangle(src, frame, Scalar(41, 185, 251), thickness); // yellow
    }

    // add red label and moment value
    string text(label);
    Point org(region[1], max(region[0] - 10, 25));
    double fontScale = 1;
    if (isWhite) {
        putText(src, text, org, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255), thickness, LINE_AA);
    } else {
        putText(src, text, org, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(29, 68, 241), thickness, LINE_AA);
    }

    return 0;
}