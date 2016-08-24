/*
 * (C) Copyright 2016 Kurento (http://kurento.org/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef __OPENCV_PLUGIN_SAMPLE_OPENCV_IMPL_HPP__
#define __OPENCV_PLUGIN_SAMPLE_OPENCV_IMPL_HPP__

#include <OpenCVProcess.hpp>
#include "OpencvPluginSample.hpp"
#include <EventHandler.hpp>
// #include "EvmGdownIIR.hpp"
using cv::Mat;
using cv::Size;
using std::cout;
using std::endl;

namespace cv {

const Scalar BLACK    (  0,   0,   0);
const Scalar RED      (255,   0,   0);
const Scalar GREEN    (  0, 255,   0);
const Scalar BLUE     (  0,   0, 255);
const Scalar WHITE    (255, 255, 255);

const Scalar ZERO     (0);

void normalization(InputArray _a, OutputArray _b);

void meanFilter(InputArray _a, OutputArray _b, size_t n = 3, Size s = Size(5, 5));

void interpolate(const Rect& a, const Rect& b, Rect& c, double p);

template<typename T>
void detrend(InputArray _z, OutputArray _r, int lambda = 10) {
    CV_DbgAssert((_z.type() == CV_32F || _z.type() == CV_64F)
            && _z.total() == max(_z.size().width, _z.size().height));

    Mat z = _z.total() == (size_t)_z.size().height ? _z.getMat() : _z.getMat().t();
    if (z.total() < 3) {
        z.copyTo(_r);
    } else {
        int t = z.total();
        Mat i = Mat::eye(t, t, z.type());
        Mat d = Mat(Matx<T,1,3>(1, -2, 1));
        Mat d2Aux = Mat::ones(t-2, 1, z.type()) * d;
        Mat d2 = Mat::zeros(t-2, t, z.type());
        for (int k = 0; k < 3; k++) {
            d2Aux.col(k).copyTo(d2.diag(k));
        }
        Mat r = (i - (i + lambda * lambda * d2.t() * d2).inv()) * z;
        r.copyTo(_r);
    }
}

template<typename T>
int countZeros(InputArray _a) {
    CV_DbgAssert(_a.channels() == 1
            && _a.total() == max(_a.size().width, _a.size().height));

    int count = 0;
    if (_a.total() > 0) {
        Mat a = _a.getMat();
        T last = a.at<T>(0);
        for (int i = 1; i < a.total(); i++) {
            T current = a.at<T>(i);
            if ((last < 0 && current >= 0) || (last > 0 && current <= 0)) {
                count++;
            }
            last = current;
        }
    }
    return count;
}

/**
 * Print Mat info such as rows, cols, channels, depth, isContinuous,
 * and isSubmatrix.
 */
void printMatInfo(const string& name, InputArray _a);

/**
 * Same as printMatInfo plus the actual values of the Mat.
 * @see printMatInfo
 */
template<typename T>
void printMat(const string& name, InputArray _a,
        int rows = -1,
        int cols = -1,
        int channels = -1)
{
    printMatInfo(name, _a);

    Mat a = _a.getMat();
    if (-1 == rows) rows = a.rows;
    if (-1 == cols) cols = a.cols;
    if (-1 == channels) channels = a.channels();

    for (int y = 0; y < rows; y++) {
        cout << "[";
        for (int x = 0; x < cols; x++) {
            T* e = &a.at<T>(y, x);
            cout << "(" << e[0];
            for (int c = 1; c < channels; c++) {
                cout << ", " << e[c];
            }
            cout << ")";
        }
        cout << "]" << endl;
    }
    cout << endl;
}

}


class EvmGdownIIR {
public:
    EvmGdownIIR();
    virtual ~EvmGdownIIR();

    void onFrame(const Mat& src, Mat& out);

    bool first;
    Size blurredSize;
    double fHigh;
    double fLow;
    int alpha;

private:
    Mat srcFloat;
    Mat blurred;
    Mat lowpassHigh;
    Mat lowpassLow;
    Mat outFloat;

};

using std::string;
using std::vector;
using cv::Mat;
using cv::Mat1d;
using cv::Mat1i;
using cv::Rect;
using cv::Size;
using cv::CascadeClassifier;

namespace kurento
{
namespace module
{
namespace opencvpluginsample
{

class OpencvPluginSampleOpenCVImpl : public virtual OpenCVProcess
{

public:

  OpencvPluginSampleOpenCVImpl ();

  virtual ~OpencvPluginSampleOpenCVImpl () {};

  virtual void process (cv::Mat &mat);

  void setFilterType (int filterType);
  void setEdgeThreshold (int edgeValue);

  // pulse stuff
  void load(const string& filename);
  void start(int width, int height);
  void onFrame(Mat& frame);

  int test;

  int maxSignalSize;
  double relativeMinFaceSize;
  struct {
    int disabledFaceId;
  } faceDetection;
  double fps;
  struct {
    double alpha;
    bool magnify;
  } evm;

  struct Face {
    int id;
    int deleteIn;
    bool selected;

    Rect box;
    Mat1d timestamps;
    Mat1d raw;
    Mat1d pulse;
    int noPulseIn;
    bool existsPulse;

    Mat1d bpms;
    double bpm;

    struct {
      EvmGdownIIR evm;
      Mat out;
      Rect box;
    } evm;

    struct Peaks {
      Mat1i indices;
      Mat1d timestamps;
      Mat1d values;

      void push(int index, double timestamp, double value);
      void pop();
      void clear();
    } peaks;

    Face(int id, const Rect& box, int deleteIn);
    int nearestBox(const vector<Rect>& boxes);
    void updateBox(const Rect& box);
    void reset();
  };

  void interpolate(const Rect& a, const Rect& b, Rect& c, double p);
  vector<Face> faces;
  // end pulse stuff
  
private:

  // pulse
  int nearestFace(const Rect& box);
  void onFace(Mat& frame, Face& face, const Rect& box);
  void peaks(Face& face);
  void bpm(Face& face);
  void draw(Mat& frame, const Face& face, const Rect& box);

  double now;
  double lastFaceDetectionTimestamp;
  double lastBpmTimestamp;
  Size minFaceSize;
  CascadeClassifier classifier;
  Mat gray;
  vector<Rect> boxes;
  Mat1d powerSpectrum;
  int nextFaceId;
  int deleteFaceIn;
  int holdPulseFor;
  double currentFps;
  // end pulse

  int filterType;
  int edgeValue;

  bool first;

  // Michae's variables
  int width;
  int height;
};

} /* opencvpluginsample */
} /* module */
} /* kurento */

#endif /*  __OPENCV_PLUGIN_SAMPLE_OPENCV_IMPL_HPP__ */
