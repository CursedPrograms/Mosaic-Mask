// Mosaic Mask - Record: the masked picture and your (changed) voice, to one .mp4 (recorder.py in C++). The picture is
// written at a steady 30 fps from the clock (each frame as many times as the time it covered, so it keeps in step
// with the sound); the voice goes to a WAV beside it; at Stop ffmpeg joins them (H.264 + AAC). Without ffmpeg the two
// stay as a silent .mp4 and a .wav.
#pragma once
#include <opencv2/core.hpp>
#include <mutex>
#include <string>

namespace cv { class VideoWriter; }

class Recorder {
public:
    explicit Recorder(std::string folder) : folder_(std::move(folder)) {}
    ~Recorder();
    bool active() const { return active_; }
    double seconds() const;
    bool start(cv::Size frameSize, int sampleRate, std::string& err);   // sampleRate 0: no sound
    void frame(const cv::Mat& bgr);
    void sound(const float* block, int n);                             // from the voice's thread
    std::string stop(std::string& message);                            // the file
    const std::string& folder() const { return folder_; }
private:
    std::string folder_, path_, videoPath_, audioPath_;
    cv::VideoWriter* video_ = nullptr;
    FILE* audio_ = nullptr;
    size_t audioBytes_ = 0;
    int rate_ = 0;
    cv::Size size_;
    long long written_ = 0;
    double t0_ = 0;
    bool active_ = false;
    std::mutex lock_;
};
