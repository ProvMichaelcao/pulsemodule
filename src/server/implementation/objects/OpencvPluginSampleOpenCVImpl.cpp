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

#include "OpencvPluginSampleOpenCVImpl.hpp"
#include <KurentoException.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>

// #include "opencv.hpp"

using std::stringstream;
using namespace cv;
using std::cout;
using std::endl;

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



EvmGdownIIR::EvmGdownIIR() {
    first = true;
    blurredSize = Size(10, 10);
    fLow = 70/60./10;
    fHigh = 80/60./10;
    alpha = 200;
}

EvmGdownIIR::~EvmGdownIIR() {
}

void EvmGdownIIR::onFrame(const Mat& src, Mat& out) {
    // convert to float
    src.convertTo(srcFloat, CV_32F);

    // apply spatial filter: blur and downsample
    resize(srcFloat, blurred, blurredSize, 0, 0, CV_INTER_AREA);

    if (first) {
        first = false;
        blurred.copyTo(lowpassHigh);
        blurred.copyTo(lowpassLow);
        src.copyTo(out);
    } else {
        // apply temporal filter: subtraction of two IIR lowpass filters
        lowpassHigh = lowpassHigh * (1-fHigh) + fHigh * blurred;
        lowpassLow = lowpassLow * (1-fLow) + fLow * blurred;
        blurred = lowpassHigh - lowpassLow;

        // amplify
        blurred *= alpha;

        // resize back to original size
        resize(blurred, outFloat, src.size(), 0, 0, CV_INTER_LINEAR);

        // add back to original frame
        outFloat += srcFloat;

        // convert to 8 bit
        outFloat.convertTo(out, CV_8U);
    }
}


String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

namespace kurento
{
namespace module
{
namespace opencvpluginsample
{

OpencvPluginSampleOpenCVImpl::OpencvPluginSampleOpenCVImpl ()
{
  this->filterType = 0;
  this->edgeValue = 125;
  this->maxSignalSize = 100;
  this->relativeMinFaceSize = 0.4;
  this->deleteFaceIn = 1;
  this->holdPulseFor = 30;
  this->fps = 0;
  this->evm.alpha = 100;

  // added by Michael
  this->first = true;
  this->test = 1;
}

/*
 * This function will be called with each new frame. mat variable
 * contains the current frame. You should insert your image processing code
 * here. Any changes in mat, will be sent through the Media Pipeline.
 */

void OpencvPluginSampleOpenCVImpl::load(const string& filename) {
  classifier.load(filename);
}

void OpencvPluginSampleOpenCVImpl::process(Mat& mat) {
  flip(mat, mat, 1);
  // OpencvPluginSampleOpenCVImpl.evm.magnify = true;

  // cv::cvtColor(mat, mat, CV_BGR2RGB);
  onFrame(mat);
  // cv::cvtColor(mat, mat, CV_RGB2BGR);
  cv::putText(mat, std::to_string(faces.size()),
    Point(0, 440), FONT_HERSHEY_SIMPLEX, 1, GREEN, 1, 8, false);
  // cv::putText(mat, std::to_string(width) + " x " + std::to_string(height),
  //   Point(0, 460), FONT_HERSHEY_SIMPLEX, 1, GREEN, 1, 8, false);
  // cv::putText(mat, std::to_string(now), Point(0, 480), FONT_HERSHEY_SIMPLEX, 
  //   1, Scalar(0, 255, 0), 1, 8, false);


}
void OpencvPluginSampleOpenCVImpl::onFrame(Mat &mat)
{
  // initial sample filter code below
  // cv::Mat matBN (mat.rows, mat.cols, CV_8UC1);
  // cv::cvtColor(mat, matBN, COLOR_BGRA2GRAY);

  // if (filterType == 0) {
  //   Canny (matBN, matBN, edgeValue, 125);
  // }
  // cvtColor (matBN, mat, COLOR_GRAY2BGRA);

  if (first) {
    const Size DIMENSIONS = mat.size();
    width = DIMENSIONS.width;
    height = DIMENSIONS.height;
    minFaceSize = Size(min(width, height) * relativeMinFaceSize, min(width, height) * relativeMinFaceSize);
    this->lastFaceDetectionTimestamp = 0;
    this->lastBpmTimestamp = 0;
    this->nextFaceId = 1;
    this->faces.clear();

    // TODO: might not be loading properly
    classifier.load("/tmp/lbpcascade_frontalface.xml");

    std::ofstream myfile;
    myfile.open("/tmp/michaeltestabc.txt");
    myfile << "         ";
    myfile.close(); 

    test = 444444;
    first = false;
  } 

  // onFrame from pulse
  now = (double)cv::getTickCount();

  if ((now - lastFaceDetectionTimestamp) * 1000. / cv::getTickFrequency() >= 1000) {
    lastFaceDetectionTimestamp = now;

    cv::cvtColor(mat, gray, CV_RGB2GRAY);
    classifier.detectMultiScale(mat, boxes, 1.1, 3, 0, minFaceSize);

    if (faces.size() <= boxes.size()) {
      for (size_t i = 0; i < faces.size(); i++) {
        Face& face = faces.at(i);
        int boxIndex = face.nearestBox(boxes);
        face.deleteIn = deleteFaceIn;
        face.updateBox(boxes.at(boxIndex));
        onFace(mat, face, boxes.at(boxIndex));
        boxes.erase(boxes.begin() + boxIndex);
      }

      for (size_t i = 0; i < boxes.size(); i++) {
        faces.push_back(Face(nextFaceId++, boxes.at(i), deleteFaceIn));
        onFace(mat, faces.back(), boxes.at(i));
      }
    } else {
      for (size_t i = 0; i < faces.size(); i++) {
        faces.at(i).selected = false;
      }
      for (size_t i = 0; i < boxes.size(); i++) {
        int faceIndex = nearestFace(boxes.at(i));
        Face& face = faces.at(faceIndex);
        face.selected = true;
        face.deleteIn = deleteFaceIn;
        face.updateBox(boxes.at(i));
        onFace(mat, face, boxes.at(i));
      }
      for (size_t i = 0; i < faces.size(); i++) {
        Face& face = faces.at(i);
        if (!face.selected) {
          if (face.deleteIn <= 0) {
            faces.erase(faces.begin() + i);
            i--;
          } else {
            face.deleteIn--;
            onFace(mat, face, face.box);
          }
        }
      }
    }
  } else {
    for (size_t i = 0; i < faces.size(); i++) {
      Face& face = faces.at(i);
      onFace(mat, face, face.box);
    }
  }
}

int OpencvPluginSampleOpenCVImpl::nearestFace(const Rect& box) {
  int index = -1;
  int min = -1;
  Point p;

  // search for first unselected face
  for (size_t i = 0; i < faces.size(); i++) {
    if (!faces.at(i).selected) {
      index = i;
      p = box.tl() - faces.at(i).box.tl();
      min = p.x * p.x + p.y * p.y;
      break;
    }
  }

  // no unselected face found
  if (index == -1) {
    return -1;
  }

  // compare with remaining unselected faces
  for (size_t i = index + 1; i < faces.size(); i++) {
    if (!faces.at(i).selected) {
      p = box.tl() - faces.at(i).box.tl();
      int d = p.x * p.x + p.y * p.y;
      if (d < min) {
        min = d;
        index = i;
      }
    }
  }

  return index;
}

void OpencvPluginSampleOpenCVImpl::onFace(Mat& mat, Face& face, const Rect& box) {
  Mat roi = !evm.magnify
    || (evm.magnify && face.existsPulse)
    ? mat(face.evm.box) : face.evm.out;

  if (evm.magnify) {
    if (face.evm.evm.first || face.evm.evm.alpha != evm.alpha) {
      face.reset();
    }
    face.evm.evm.alpha = evm.alpha;
    face.evm.evm.onFrame(mat(face.evm.box), roi);
  } else if (!face.evm.evm.first) {
    face.reset();
  }
  if (face.raw.rows >= maxSignalSize) {
    const int total = face.raw.rows;
    face.raw.rowRange(1, total).copyTo(face.raw.rowRange(0, total - 1));
    face.raw.pop_back();
    face.timestamps.rowRange(1, total).copyTo(face.timestamps.rowRange(0, total - 1));
    face.timestamps.pop_back();
  }

  face.raw.push_back<double>(mean(roi)(1));
  face.timestamps.push_back<double>(getTickCount());

  Scalar rawStdDev;
  meanStdDev(face.raw, Scalar(), rawStdDev);
  const bool stable = rawStdDev(0) <= (evm.magnify ? 1 : 0) * evm.alpha * 0.045 + 1;

  if (stable) {
    currentFps = this->fps;
    if (currentFps == 0) {
      const double diff = (face.timestamps(face.timestamps.rows - 1) - face.timestamps(0)) * 1000. / getTickFrequency();
      currentFps = face.timestamps.rows * 1000 / diff;
    }

    detrend<double>(face.raw, face.pulse, currentFps / 2);
    normalization(face.pulse, face.pulse);
    meanFilter(face.pulse, face.pulse);

    peaks(face);
  } else {
    face.existsPulse = false;
    face.reset();
  }

  if (face.existsPulse) {
    bpm(face);
  }
  if (!face.existsPulse) {
    if (face.pulse.rows == face.raw.rows) {
      face.pulse = 0;
    } else {
      face.pulse = Mat1d::zeros(face.raw.rows, 1);
    }
    face.peaks.clear();
    face.bpms.pop_back(face.bpms.rows);
    face.bpm = 0;
  }
  draw(mat, face, box);
}

void OpencvPluginSampleOpenCVImpl::peaks(Face& face) {
  face.peaks.clear();
  int lastIndex = 0;
  // int lastPeakIndex = 0;
  int lastPeakTimestamp = face.timestamps(0);
  int lastPeakValue = face.pulse(0);
  double peakValueThreshold = 0;

  for (int i = 1; i < face.raw.rows; i++) {
    const double diff = (face.timestamps(i) - face.timestamps(lastIndex)) * 1000. / getTickFrequency();
    if (diff >= 200) {
      int relativePeakIndex[2];
      double peakValue;
      minMaxIdx(face.pulse.rowRange(lastIndex, i+1), 0, &peakValue, 0, &relativePeakIndex[0]);
      const int peakIndex = lastIndex + relativePeakIndex[0];

      if (peakValue > peakValueThreshold && lastIndex < peakIndex && peakIndex < i) {
        const double peakTimestamp = face.timestamps(peakIndex);
        const double peakDiff = (peakTimestamp - lastPeakTimestamp) * 1000. / getTickFrequency();
        if (peakDiff <= 200 && peakValue > lastPeakValue) {
          face.peaks.pop();
        }
        if (peakDiff > 200 || peakValue > lastPeakValue) {
          face.peaks.push(peakIndex, peakTimestamp, peakValue);

          // lastPeakIndex = peakIndex;
          lastPeakTimestamp = peakTimestamp;
          lastPeakValue = peakValue;

          peakValueThreshold = 0.6 * mean(face.peaks.values)(0);
        }
      }

      lastIndex = i;
    }
  }

  Scalar peakValuesStdDev;
  meanStdDev(face.peaks.values, Scalar(), peakValuesStdDev);
  const double diff = (face.timestamps(face.raw.rows - 1) - face.timestamps(0)) / getTickFrequency();

  Scalar peakTimestampsStdDev;
  if (face.peaks.indices.rows >= 3) {
    meanStdDev((face.peaks.timestamps.rowRange(1, face.peaks.timestamps.rows) - 
      face.peaks.timestamps.rowRange(0, face.peaks.timestamps.rows - 1)) / getTickFrequency(), 
      Scalar(), peakTimestampsStdDev);
  }

  bool validPulse = 
      2 <= face.peaks.indices.rows &&
      40/60 * diff <= face.peaks.indices.rows &&
      face.peaks.indices.rows <= 240/60 * diff &&
      peakValuesStdDev(0) <= 0.5 &&
      peakTimestampsStdDev(0) <= 0.5;

  if (!face.existsPulse && validPulse) {
    face.noPulseIn = holdPulseFor;
    face.existsPulse = true;
  } else if (face.existsPulse && !validPulse) {
    if (face.noPulseIn > 0) face.noPulseIn--;
    else face.existsPulse = false;
  }
}

void OpencvPluginSampleOpenCVImpl::Face::Peaks::push(int index, double timestamp, double value) {
    indices.push_back<int>(index);
    timestamps.push_back<double>(timestamp);
    values.push_back<double>(value);
}

void OpencvPluginSampleOpenCVImpl::Face::Peaks::pop() {
    indices.pop_back(min(indices.rows, 1));
    timestamps.pop_back(min(timestamps.rows, 1));
    values.pop_back(min(values.rows, 1));
}

void OpencvPluginSampleOpenCVImpl::Face::Peaks::clear() {
    indices.pop_back(indices.rows);
    timestamps.pop_back(timestamps.rows);
    values.pop_back(values.rows);
}

void OpencvPluginSampleOpenCVImpl::bpm(Face& face) {
  dft(face.pulse, powerSpectrum);
  const int total = face.raw.rows;

  const int low = total * 40./60./currentFps + 1;
  const int high = total * 240./60./currentFps + 1;
  powerSpectrum.rowRange(0, min((size_t)low, (size_t)total)) = ZERO;
  powerSpectrum.pop_back(min((size_t)(total - high), (size_t)total));

  pow(powerSpectrum, 2, powerSpectrum);

  if (!powerSpectrum.empty()) {
    int idx[2];
    minMaxIdx(powerSpectrum, 0, 0, 0, &idx[0]);

    face.bpms.push_back<double>(idx[0] * currentFps * 30. / total);
  }

  if (face.bpm == 0 || (now - lastBpmTimestamp) * 1000. / getTickFrequency() >= 1000.) {
    lastBpmTimestamp = getTickCount();

    face.bpm = mean(face.bpms)(0);
    face.bpms.pop_back(face.bpms.rows);

    if (face.bpm <= 40) {
      face.existsPulse = false;
    }
  }
}

void OpencvPluginSampleOpenCVImpl::draw(Mat& mat, const Face& face, const Rect& box) {
  rectangle(mat, box, BLUE);
  rectangle(mat, face.box, BLUE, 2);
  rectangle(mat, face.evm.box, GREEN);

  Point bl = face.box.tl() + Point(0, face.box.height);
  Point g;
  for (int i = 0; i < face.raw.rows; i++) {
    g = bl + Point(i, -face.raw(i) + 50);
    line(mat, g, g, GREEN);
    g = bl + Point(i, -face.pulse(i) * 10 - 50);
    line(mat, g, g, RED, face.existsPulse ? 2 : 1);
  }

  for (int i = 0; i < face.peaks.indices.rows; i++) {
    const int index = face.peaks.indices(i);
    g = bl + Point(index, -face.pulse(index) * 10 - 50);
    circle(mat, g, 1, BLUE, 2);
  }

  stringstream ss;

  ss << face.id;
  putText(mat, ss.str(), face.box.tl(), FONT_HERSHEY_SIMPLEX, 2, BLUE, 2);
  ss.str("");

  ss.precision(3);
  ss << face.bpm;
  putText(mat, ss.str(), bl, FONT_HERSHEY_SIMPLEX, 2, RED, 2);
}

OpencvPluginSampleOpenCVImpl::Face::Face(int id, const Rect& box, int deleteIn) {
  this->id = id;
  this->box = box;
  this->deleteIn = deleteIn;
  this->updateBox(this->box);
  this->existsPulse = false;
  this->noPulseIn = 0;
}

int OpencvPluginSampleOpenCVImpl::Face::nearestBox(const vector<Rect>& boxes) {
  if (boxes.empty()) {
    return -1;
  }
  int index = 0;
  Point p = box.tl() - boxes.at(0).tl();
  int min = p.x * p.x + p.y * p.y;
  for (size_t i = 1; i < boxes.size(); i++) {
    p = box.tl() - boxes.at(i).tl();
    int d = p.x * p.x + p.y * p.y;
    if (d < min) {
      min = d;
      index = i;
    }
  }
  return index;
}

void OpencvPluginSampleOpenCVImpl::Face::updateBox(const Rect& a) {
  // update box position and size
  Point p = box.tl() - a.tl();
  double d = (p.x * p.x + p.y * p.y) / pow(box.width / 3., 2.);

  // TODO: bring interpolate back at some point

  cv::interpolate(box, a, box, min(1.0, d));

  // update EVM box
  Point c = box.tl() + Point(box.size().width * .5, box.size().height * .5);
  Point r(box.width * .275, box.height * .425);
  evm.box = Rect(c - r, c + r);
}

void OpencvPluginSampleOpenCVImpl::Face::reset() {
  // restarts Eulerian video magnification
  evm.evm.first = true;

  // clear raw signal
  raw.pop_back(raw.rows);
  timestamps.pop_back(timestamps.rows);
}



void OpencvPluginSampleOpenCVImpl::setFilterType (int filterType)
{
  this->filterType = filterType;
}

void OpencvPluginSampleOpenCVImpl::setEdgeThreshold (int edgeValue)
{
  this->edgeValue = edgeValue;
}

} /* opencvpluginsample */
} /* module */
} /* kurento */