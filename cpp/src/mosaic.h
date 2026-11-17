// Mosaic Mask - the mosaic: every face in faces/ unwrapped onto one shared layout (the pool), the face cut into
// fragments like a patchwork, each a different face changing on its own timer, warped onto the live face every frame
// (the scramble suit of A Scanner Darkly). C++ port of mosaic/face_pool.py, mesh.py and suit.py.
#pragma once
#include "face.h"
#include <opencv2/core.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace Mosaic {

const int CANON = 384;   // the shared face: CANON x CANON pixels

// a piecewise-affine warp from a source that never moves (the shared face): its side worked out once
class TriangleWarp {
public:
    TriangleWarp() = default;
    TriangleWarp(const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Vec3i>& tris);
    void draw(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point2f>& dstPts, cv::Mat* mask = nullptr) const;
private:
    std::vector<cv::Vec3i> tris_;
    struct Src { cv::Rect box; cv::Point2f rel[3]; };
    std::vector<Src> src_;
};

struct Pool {
    std::vector<std::string> names;            // the faces that worked
    std::vector<cv::Mat> textures;             // each on the shared face (CANON x CANON, BGR)
    std::vector<cv::Point2f> layout;           // 468: the shared face (the pool's average)
    std::vector<std::pair<std::string, std::string>> skipped;   // file, why
    size_t size() const { return names.size(); }
    // every image in folder (cached in folder/_canon); progress(done, total)
    void load(const std::string& folder, const std::function<void(int, int)>& progress = nullptr);
};

class Suit {
public:
    Suit(const Pool& pool, int fragments = 24, unsigned seed = 0);
    void setFragments(int n);
    int fragments() const { return wanted_; }
    double swapMs = 350, fadeMs = 120;
    float colourMatch = 0.4f;
    // the mosaic warped onto the face at lm in frame (in place); occluder: what's in front stays in front
    void apply(cv::Mat& frame, const std::vector<cv::Point2f>& lm, Face::Occluder* occluder);
private:
    const Pool& pool_;
    std::mt19937 rng_;
    std::vector<cv::Vec3i> tris_;
    TriangleWarp warp_;
    std::vector<cv::Point2f> subLayout_;
    cv::Mat faceMask_;                          // CANON x CANON, CV_32F
    struct Frag { cv::Rect box; cv::Mat wt; };  // its box, its weights in it (CV_32F)
    std::vector<Frag> frags_;
    std::vector<int> cur_, nxt_;
    std::vector<double> tSwap_;
    int wanted_ = 24;
    float eyeOpen_[2] = {-1, -1};
    cv::Mat canonTexture(double now);
    void matchColour(cv::Mat& canon, const cv::Mat& live) const;
    cv::Mat own(const std::vector<cv::Point2f>& lm, cv::Size size, cv::Point2f offset);
};

double now();   // seconds (a steady clock)

}
