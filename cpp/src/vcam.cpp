// Mosaic Mask - the virtual camera (see vcam.h): OBS's shared-memory video queue, written the way OBS writes it
// (plugins/win-dshow/shared-memory-queue.c): a header, then three frame slots, each a 64-bit timestamp and an NV12
// picture; the writer fills the next slot and moves read_idx to it.
#include "vcam.h"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace VCam {

#ifdef _WIN32
#pragma pack(push, 8)
struct QueueHeader {
    volatile LONG writeIdx;
    volatile LONG readIdx;
    volatile LONG state;          // 0 invalid, 1 starting, 2 ready, 3 stopping
    uint32_t offsets[3];
    uint32_t type;                // 0 video
    uint32_t cx, cy;
    uint64_t interval;            // 100 ns units a frame
    uint32_t reserved[8];
};
#pragma pack(pop)
static const unsigned FRAME_HEADER = 32;   // the timestamp, padded
static unsigned align32(unsigned v) { return (v + 31u) & ~31u; }

bool obsInstalled() {
    HKEY k;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Classes\\CLSID\\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}", 0, KEY_READ, &k) != ERROR_SUCCESS) return false;
    RegCloseKey(k);
    return true;
}

bool Obs::open(int w, int h, double fps, std::string& err) {
    close();
    if (!obsInstalled()) { err = "OBS Studio isn't installed (its virtual camera comes with it)"; return false; }
    w &= ~1; h &= ~1;   // NV12: even sizes
    const unsigned frame = (unsigned)w * h * 3 / 2;
    unsigned size = align32(sizeof(QueueHeader));
    for (int i = 0; i < 3; ++i) { off_[i] = size; size = align32(size + FRAME_HEADER + frame); }
    HANDLE m = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, size, L"OBSVirtualCamVideo");
    if (!m) { err = "Couldn't make the virtual camera's shared memory"; return false; }
    if (GetLastError() == ERROR_ALREADY_EXISTS) {   // OBS (or another program) is sending to it already
        CloseHandle(m); err = "The virtual camera is in use by OBS or another program"; return false;
    }
    void* p = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) { CloseHandle(m); err = "Couldn't map the virtual camera's memory"; return false; }
    std::memset(p, 0, size);
    QueueHeader* qh = (QueueHeader*)p;
    qh->state = 1;
    qh->type = 0;
    qh->cx = (uint32_t)w; qh->cy = (uint32_t)h;
    qh->interval = (uint64_t)(10000000.0 / (fps > 0 ? fps : 30.0));
    for (int i = 0; i < 3; ++i) qh->offsets[i] = off_[i];
    map_ = m; mem_ = (unsigned char*)p; w_ = w; h_ = h;
    return true;
}

void Obs::send(const cv::Mat& bgr) {
    if (!mem_ || bgr.empty()) return;
    cv::Mat src = bgr;
    if (bgr.cols != w_ || bgr.rows != h_) cv::resize(bgr, src, cv::Size(w_, h_), 0, 0, cv::INTER_AREA);
    cv::Mat i420;
    cv::cvtColor(src, i420, cv::COLOR_BGR2YUV_I420);   // Y, then U, then V planes
    QueueHeader* qh = (QueueHeader*)mem_;
    const LONG inc = InterlockedIncrement(&qh->writeIdx);
    const int idx = (int)(inc % 3);
    unsigned char* slot = mem_ + off_[idx];
    const uint64_t ts = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::memcpy(slot, &ts, sizeof ts);
    unsigned char* y = slot + FRAME_HEADER;
    const size_t ySize = (size_t)w_ * h_;
    std::memcpy(y, i420.data, ySize);
    // NV12: the U and V planes interleaved
    const unsigned char* u = i420.data + ySize;
    const unsigned char* v = u + ySize / 4;
    unsigned char* uv = y + ySize;
    for (size_t i = 0; i < ySize / 4; ++i) { uv[i * 2] = u[i]; uv[i * 2 + 1] = v[i]; }
    InterlockedExchange(&qh->readIdx, inc);
    InterlockedExchange(&qh->state, 2);
}

void Obs::close() {
    if (mem_) { ((QueueHeader*)mem_)->state = 3; UnmapViewOfFile(mem_); mem_ = nullptr; }
    if (map_) { CloseHandle((HANDLE)map_); map_ = nullptr; }
}
#else
bool obsInstalled() { return false; }
bool Obs::open(int, int, double, std::string& err) { err = "The virtual camera is Windows-only for now"; return false; }
void Obs::send(const cv::Mat&) {}
void Obs::close() {}
#endif

}
