/**
    Real-time Object 2-D Recognition
    Created by Hui Hu for CS 5330 Computer Vision Spring 2023

    Functions to extract and save feature vector
*/

// basic
#include <map>
#include <math.h> // atan2
#include <queue>  // min heap
#include <vector>
// opencv
#include <opencv2/core/types.hpp> // Rect Moment
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Threshold the input image and return a binary image
// int threshold(Mat &src, Mat &des, int thresholdValue);

// Clean up your thresholded image using morphological filtering
// int cleanup(Mat &src, Mat &des);

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
vector<vector<int>> segmentation(Mat &src, int N) {
  Mat_<char> srcVec = src;
  UnionFind UF(src.rows * src.cols);

  // Segment the image into regions
  int rows = src.rows;
  int r, c;
  for (r = 0; r < rows; r++) {
    for (c = 0; c < src.cols; c++) {
      // 4-connected
      if (r - 1 > 0 && (srcVec(r, c) * srcVec(r - 1, c) > 1)) {
        UF.merge(r * rows + c, (r - 1) * rows + c); // upper point
      }
      if (c - 1 > 0 && (srcVec(r, c) * srcVec(r, c - 1) > 1)) {
        UF.merge(r * rows + c, r * rows + c - 1); // left point
      }
    }
  }

  // Record basic information of each region (area, position)
  map<int, vector<int>> position;
  int root;
  for (r = 0; r < rows; r++) {
    for (c = 0; c < src.cols; c++) {
      if (srcVec(r, c) == 0) {
        continue; // exclude the background point
      }
      root = UF.find(r * rows + c);
      auto search = area.find(root);
      if (search != area.end()) {
        // update the area, position
        area[root] = area[root] + 1;
        position[root][0] = std::min(position[root][0], r); // lowerest
        position[root][1] = std::min(position[root][1], c); // leftmost
        position[root][2] = std::max(position[root][2], r); // highest
        position[root][3] = std::max(position[root][3], c); // rightmost
      } else {
        // add new item
        area[root] = 1;
        position[root] = {r, c, 0, 0};
      }
    }
  }

  // Find the first N largest regions
  priority_queue<int, vector<int>, Comparator> minHeap;
  for (auto region : area) {
    if (region.second > 10) { // exclude spot
      // Add new item to min-Heap
      minHeap.push(region.first);
      if (minHeap.size() > N) {
        minHeap.pop();
      }
    }
  }

  // Add the positions of first N largest regions to res vector
  // If the regions that meet our requirement are less than N, we only return the qualified regions
  vector<vector<int>> result = {};
  while (!minHeap.empty()) {
    result.push_back(position[minHeap.top()]);
    minHeap.pop();
  }

  return result;
}

// Compute features for one region
float getFeatureVec(Mat &src, vector<double> &feature, vector<int> region, char method) {
  // Extract the region
  Mat crop = src(Range(region[1], region[3]), Range(region[0], region[2]));

  // Compute moments of the region
  Moments momentValue = moments(crop);
  double hu[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  HuMoments(momentValue, hu);

  // Log scale hu moments to male its value bigger
  for (int i = 0; i < 7; i++) {
    hu[i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
  }

  feature = {};
  for (int i = 0; i < 6; i++) {
    feature.push_back(hu[i]);
  }

  return feature[0]; // return the first value to display
}

// Save training data into (image) folder and (feature) CSV files
int saveData(Mat &src, vector<float> &feature, char *name);