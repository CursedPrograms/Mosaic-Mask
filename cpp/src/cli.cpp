// mosaic-cli: one masked frame from a picture or video (Mosaic Mask's --snapshot), for trying settings and tests.
//   mosaic-cli <picture or video> out.png [--faces folder] [--fragments 24] [--colour 40] [--frames 3]
#include "face.h"
#include "mosaic.h"
#include "vcam.h"
#include "voice.h"
#include <fstream>
#include <thread>

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

// a 16-bit mono WAV in and out (the voice test)
static std::vector<float> readWav(const std::string& p, int& rate) {
    std::ifstream f(fs::u8path(p), std::ios::binary); std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<float> out; rate = 0;
    for (size_t i = 12; i + 8 <= b.size();) {
        const std::string id(&b[i], 4); uint32_t n; memcpy(&n, &b[i + 4], 4);
        if (id == "fmt ") { uint32_t r; memcpy(&r, &b[i + 12], 4); rate = (int)r; }
        if (id == "data") { const int16_t* s = (const int16_t*)&b[i + 8]; for (uint32_t k = 0; k < n / 2; ++k) out.push_back(s[k] / 32768.0f); break; }
        i += 8 + n + (n & 1);
    }
    return out;
}
static void writeWav(const std::string& p, const std::vector<float>& x, int rate) {
    std::ofstream f(fs::u8path(p), std::ios::binary);
    const uint32_t data = (uint32_t)x.size() * 2, riff = 36 + data, fmt = 16, r = rate, br = rate * 2; const uint16_t pcm = 1, ch = 1, ba = 2, bits = 16;
    f.write("RIFF", 4); f.write((const char*)&riff, 4); f.write("WAVEfmt ", 8); f.write((const char*)&fmt, 4); f.write((const char*)&pcm, 2); f.write((const char*)&ch, 2);
    f.write((const char*)&r, 4); f.write((const char*)&br, 4); f.write((const char*)&ba, 2); f.write((const char*)&bits, 2); f.write("data", 4); f.write((const char*)&data, 4);
    for (float v : x) { const int16_t s = (int16_t)std::lround(std::max(-1.0f, std::min(1.0f, v)) * 32767); f.write((const char*)&s, 2); }
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::string(argv[1]) == "--voice") {   // tests: mosaic-cli --voice in.wav out.wav female|male|disguise|off
        int rate; auto in = readWav(argv[2], rate);
        if (rate != Voice::SAMPLE_RATE) { printf("the test wav must be mono 16-bit at %d Hz (it's %d)\n", Voice::SAMPLE_RATE, rate); return 1; }
        int p = 0; for (int k = 0; k < Voice::PRESET_COUNT; ++k) if (std::string(argv[4]) == Voice::presetName(k)) p = k;
        writeWav(argv[3], Voice::render(in, p), rate);
        printf("voice: %s -> %s (%s)\n", argv[2], argv[3], Voice::presetName(p));
        return 0;
    }
    if (argc < 3) { printf("mosaic-cli <picture or video> out.png [--faces folder] [--fragments 24] [--colour 40] [--frames 3]\n"); return 1; }
    std::string src = argv[1], out = argv[2], faces = (fs::u8path(exeDir()) / "faces").u8string();
    int fragments = 24, frames = 3; float colour = 40; double vcamSeconds = 0;
    for (int i = 3; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--faces") faces = argv[++i];
        else if (a == "--fragments") fragments = atoi(argv[++i]);
        else if (a == "--colour") colour = (float)atof(argv[++i]);
        else if (a == "--frames") frames = atoi(argv[++i]);
        else if (a == "--vcam") vcamSeconds = atof(argv[++i]);   // tests: stream the masked picture to the virtual camera
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
    if (vcamSeconds > 0) {
        VCam::Obs cam; std::string verr;
        if (!cam.open(result.cols, result.rows, 30, verr)) { printf("virtual camera: %s\n", verr.c_str()); return 1; }
        printf("virtual camera: sending %dx%d for %.0f s\n", cam.width(), cam.height(), vcamSeconds); fflush(stdout);
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds((int)(vcamSeconds * 1000));
        while (std::chrono::steady_clock::now() < end) { cam.send(result); std::this_thread::sleep_for(std::chrono::milliseconds(33)); }
    }
    printf("saved %s\n", out.c_str());
    return 0;
}
