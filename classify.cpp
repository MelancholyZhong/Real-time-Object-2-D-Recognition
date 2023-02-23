#include <math.h>
#include <opencv2/core/types.hpp> // Point
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// Display the label of one region
int displayLabel(Mat &src, vector<int> region, char *label) {
    // add yellow rectangle frame
    int width = region[3] - region[1] + 1;
    int height = region[2] - region[0] + 1;
    int thickness = 3;
    Rect frame(region[0], region[1], width, height);
    rectangle(src, frame, Scalar(41, 185, 251), thickness);

    // add red label and moment value
    string text(label);
    Point org(region[0], max(region[1] - 10, 25));
    double fontScale = 1;
    putText(src, text, org, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(29, 68, 241), thickness, LINE_AA);

    return 0;
}