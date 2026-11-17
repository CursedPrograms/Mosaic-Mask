// Mosaic Mask - the face: YuNet (OpenCV Zoo, MIT) finds it, MediaPipe's Face Mesh (Apache-2.0) fits 468 points on it,
// both as ONNX through ONNX Runtime's C API (onnxruntime.dll, loaded at run time; Bathos's face code). On video the
// mesh tracks: each frame is fitted on the last frame's face, and YuNet only looks again when it's lost.
// The occluder is FaceFusion's XSeg (xseg_1.onnx): what's in front of the face (a hand, hair, a cup).
#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace Face {

bool init(const std::string& modelsDir, std::string& err);   // the runtime and the two face models

// 468 landmarks (pixels) of the main face in a BGR frame, or empty. still: a photo (no tracking)
class Tracker {
public:
    explicit Tracker(bool still = false) : still_(still) {}
    std::vector<cv::Point2f> landmarks(const cv::Mat& bgr);
    void reset() { last_.clear(); }
private:
    bool still_;
    std::vector<cv::Point2f> last_;
};

// FaceFusion's occluder: 1 where the face shows, 0 where something's in front, over box (canvas pixels)
class Occluder {
public:
    ~Occluder();
    bool load(const std::string& modelsDir);   // false: no model (occlusion stays off)
    bool on = false;
    bool available() const { return session_ != nullptr; }
    std::string why;
    cv::Mat mask(const cv::Mat& bgr, const std::vector<cv::Point2f>& lm, cv::Rect box);   // CV_32F, box-sized
private:
    void* session_ = nullptr;
    // XSeg runs on its own thread, as often as it keeps up: a frame never waits for it, the mask follows the face
    struct Async;
    Async* async_ = nullptr;
};

}
