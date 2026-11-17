// Mosaic Mask - the virtual camera: frames to OBS Studio's virtual camera ("OBS Virtual Camera" in Zoom, Teams,
// Discord, Meet). OBS's DirectShow camera reads a shared-memory queue (OBSVirtualCamVideo: a header and three NV12
// frames); this writes it, as OBS does. OBS must be installed (its camera is registered then); it needn't be running.
#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace VCam {

bool obsInstalled();

class Obs {
public:
    ~Obs() { close(); }
    bool open(int w, int h, double fps, std::string& err);
    void send(const cv::Mat& bgr);   // resized to the size it was opened with
    void close();
    bool isOpen() const { return mem_ != nullptr; }
    int width() const { return w_; }
    int height() const { return h_; }
private:
    void* map_ = nullptr;
    unsigned char* mem_ = nullptr;
    int w_ = 0, h_ = 0;
    unsigned off_[3] = {0, 0, 0};
};

}
