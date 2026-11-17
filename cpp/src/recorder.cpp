// Mosaic Mask - Record (see recorder.h).
#include "recorder.h"
#include "mosaic.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;
static const int FPS = 30;

Recorder::~Recorder() { std::string m; stop(m); }
double Recorder::seconds() const { return active_ ? Mosaic::now() - t0_ : 0.0; }

static void wavHeader(FILE* f, int rate, uint32_t dataBytes) {
    const uint32_t riff = 36 + dataBytes, fmt = 16, r = rate, br = rate * 2; const uint16_t pcm = 1, ch = 1, ba = 2, bits = 16;
    fseek(f, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f); fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&r, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&dataBytes, 4, 1, f);
}

bool Recorder::start(cv::Size frameSize, int sampleRate, std::string& err) {
    std::lock_guard<std::mutex> g(lock_);
    if (active_) return true;
    std::error_code ec; fs::create_directories(fs::u8path(folder_), ec);
    char stamp[64]; const time_t t = time(nullptr); strftime(stamp, sizeof stamp, "%Y-%m-%d %H-%M-%S", localtime(&t));
    const fs::path stem = fs::u8path(folder_) / stamp;
    path_ = stem.u8string() + ".mp4"; videoPath_ = stem.u8string() + ".video.mp4"; audioPath_ = sampleRate ? stem.u8string() + ".wav" : "";
    video_ = new cv::VideoWriter(videoPath_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), FPS, frameSize);
    if (!video_->isOpened()) { delete video_; video_ = nullptr; err = "Couldn't write " + videoPath_; return false; }
    size_ = frameSize; rate_ = sampleRate; audioBytes_ = 0;
    if (sampleRate) { audio_ = _wfopen(fs::u8path(audioPath_).wstring().c_str(), L"wb"); if (audio_) wavHeader(audio_, sampleRate, 0); }
    written_ = 0; t0_ = Mosaic::now(); active_ = true;
    return true;
}

void Recorder::frame(const cv::Mat& bgr) {
    std::lock_guard<std::mutex> g(lock_);
    if (!active_ || !video_) return;
    cv::Mat f = bgr;
    if (bgr.size() != size_) cv::resize(bgr, f, size_);
    const long long due = (long long)((Mosaic::now() - t0_) * FPS) + 1;
    while (written_ < due) { video_->write(f); ++written_; }
}

void Recorder::sound(const float* block, int n) {
    std::lock_guard<std::mutex> g(lock_);
    if (!active_ || !audio_) return;
    for (int i = 0; i < n; ++i) { const int16_t s = (int16_t)std::lround(std::clamp(block[i], -1.0f, 1.0f) * 32767); fwrite(&s, 2, 1, audio_); }
    audioBytes_ += (size_t)n * 2;
}

static std::string ffmpegPath() {
#ifdef _WIN32
    wchar_t p[MAX_PATH]; GetModuleFileNameW(nullptr, p, MAX_PATH);
    const fs::path beside = fs::path(p).parent_path() / "ffmpeg.exe";
    std::error_code ec;
    if (fs::exists(beside, ec)) return beside.u8string();
    wchar_t found[MAX_PATH];
    if (SearchPathW(nullptr, L"ffmpeg.exe", nullptr, MAX_PATH, found, nullptr)) return fs::path(found).u8string();
#endif
    return "";
}

std::string Recorder::stop(std::string& message) {
    {
        std::lock_guard<std::mutex> g(lock_);
        if (!active_) return "";
        active_ = false;
        if (video_) { video_->release(); delete video_; video_ = nullptr; }
        if (audio_) { wavHeader(audio_, rate_, (uint32_t)audioBytes_); fclose(audio_); audio_ = nullptr; }
    }
    std::error_code ec;
    const std::string ff = ffmpegPath();
    if (ff.empty()) {
        if (!audioPath_.empty()) { message = "Saved (no ffmpeg: the sound is in the .wav beside it)"; return videoPath_; }
        fs::rename(fs::u8path(videoPath_), fs::u8path(path_), ec);
        message = "Saved"; return path_;
    }
    std::string cmd = "\"\"" + ff + "\" -y -loglevel error -i \"" + videoPath_ + "\"";
    if (!audioPath_.empty()) cmd += " -i \"" + audioPath_ + "\" -c:a aac -b:a 160k -shortest";
    cmd += " -c:v libx264 -pix_fmt yuv420p -crf 20 -preset veryfast \"" + path_ + "\"\"";
#ifdef _WIN32
    STARTUPINFOW si{}; si.cb = sizeof si; PROCESS_INFORMATION pi{};
    std::wstring wc = fs::u8path("cmd.exe /c " + cmd).wstring();
    if (CreateProcessW(nullptr, wc.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, 120000);
        CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    }
#else
    (void)!system(cmd.c_str());
#endif
    if (!fs::exists(fs::u8path(path_), ec)) { message = "Saved without joining (ffmpeg failed)"; return videoPath_; }
    fs::remove(fs::u8path(videoPath_), ec);
    if (!audioPath_.empty()) fs::remove(fs::u8path(audioPath_), ec);
    message = "Saved";
    return path_;
}
