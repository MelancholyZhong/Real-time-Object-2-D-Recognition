#include <math.h>
#include <opencv2/core/types.hpp> // Point
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

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