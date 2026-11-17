// mosaic-cli: one masked frame from a picture or video (Mosaic Mask's --snapshot), for trying settings and tests.
//   mosaic-cli <picture or video> out.png [--faces folder] [--fragments 24] [--colour 40] [--frames 3]
#include "face.h"
#include "mosaic.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

static std::string exeDir() {
#ifdef _WIN32
    wchar_t p[MAX_PATH]; GetModuleFileNameW(nullptr, p, MAX_PATH);
    return fs::path(p).parent_path().u8string();
#else
    return fs::read_symlink("/proc/self/exe").parent_path().u8string();
#endif
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("mosaic-cli <picture or video> out.png [--faces folder] [--fragments 24] [--colour 40] [--frames 3]\n"); return 1; }
    std::string src = argv[1], out = argv[2], faces = (fs::u8path(exeDir()) / "faces").u8string();
    int fragments = 24, frames = 3; float colour = 40;
    for (int i = 3; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--faces") faces = argv[++i];
        else if (a == "--fragments") fragments = atoi(argv[++i]);
        else if (a == "--colour") colour = (float)atof(argv[++i]);
        else if (a == "--frames") frames = atoi(argv[++i]);
    }
    std::string err;
    if (!Face::init(exeDir() + "/models", err)) { printf("%s\n", err.c_str()); return 1; }
    cv::Mat frame = cv::imread(src);
    if (frame.empty()) { cv::VideoCapture cap(src); cap.read(frame); }
    if (frame.empty()) { printf("Can't read %s\n", src.c_str()); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    Mosaic::Pool pool;
    pool.load(faces);
    auto t1 = std::chrono::steady_clock::now();
    printf("faces: %zu ready, %zu skipped (%.1f s)\n", pool.size(), pool.skipped.size(), std::chrono::duration<double>(t1 - t0).count());
    Face::Tracker tracker(true);
    auto lm = tracker.landmarks(frame);
    Face::Occluder occ; occ.load(exeDir() + "/models");
    printf("face found: %s; occlusion: %s\n", lm.empty() ? "no" : "yes", occ.on ? "on" : occ.why.c_str());
    cv::Mat result = frame.clone();
    if (pool.size() && !lm.empty()) {
        Mosaic::Suit suit(pool, fragments, 7);
        suit.colourMatch = colour / 100;
        auto t2 = std::chrono::steady_clock::now();
        for (int k = 0; k < frames; ++k) { result = frame.clone(); suit.apply(result, lm, &occ); }
        printf("mosaic: %.1f ms a frame\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count() / frames);
    }
    cv::imwrite(out, result);
    printf("saved %s\n", out.c_str());
    return 0;
}
