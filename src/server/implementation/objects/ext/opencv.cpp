#include "opencv.hpp"

namespace cv {

void normalization(InputArray _a, OutputArray _b) {
    _a.getMat().copyTo(_b);
    Mat b = _b.getMat();
    Scalar mean, stdDev;
    meanStdDev(b, mean, stdDev);
    b = (b - mean[0]) / stdDev[0];
}

void meanFilter(InputArray _a, OutputArray _b, size_t n, Size s) {
    _a.getMat().copyTo(_b);
    Mat b = _b.getMat();
    for (size_t i = 0 ; i < n; i++) {
        blur(b, b, s);
    }
}

void interpolate(const Rect& a, const Rect& b, Rect& c, double p) {
    double np = 1 - p;
    c.x = a.x * np + b.x * p + 0.5;
    c.y = a.y * np + b.y * p + 0.5;
    c.width = a.width * np + b.width * p + 0.5;
    c.height = a.height * np + b.height * p + 0.5;
}

void printMatInfo(const string& name, InputArray _a) {
    Mat a = _a.getMat();
    cout << name << ": " << a.rows << "x" << a.cols
            << " channels=" << a.channels()
            << " depth=" << a.depth()
            << " isContinuous=" << (a.isContinuous() ? "true" : "false")
            << " isSubmatrix=" << (a.isSubmatrix() ? "true" : "false") << endl;
}

}

