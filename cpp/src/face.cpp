// Mosaic Mask - the face (see face.h). The ONNX Runtime plumbing and the YuNet / Face Mesh maths are Bathos's.
#include "face.h"
#include "onnxruntime_c_api.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <opencv2/calib3d.hpp>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace Face {

static const OrtApi* ort = nullptr;
static OrtEnv* env = nullptr;
static OrtSession* yunet = nullptr;
static OrtSession* mesh = nullptr;
static std::mutex runLock;

static bool ok(OrtStatus* s, std::string& err) {
    if (!s) return true;
    err = std::string("ONNX Runtime: ") + ort->GetErrorMessage(s);
    ort->ReleaseStatus(s);
    return false;
}
static std::string exeDir() {
#ifdef _WIN32
    wchar_t p[MAX_PATH]; GetModuleFileNameW(nullptr, p, MAX_PATH);
    return fs::path(p).parent_path().u8string();
#else
    return fs::read_symlink("/proc/self/exe").parent_path().u8string();
#endif
}
static bool loadRuntime(std::string& err) {
    if (!ort) {
        typedef const OrtApiBase*(ORT_API_CALL * GetBase)();
        GetBase getBase = nullptr;
#ifdef _WIN32
        HMODULE h = LoadLibraryW((fs::u8path(exeDir()) / "onnxruntime.dll").wstring().c_str());
        if (!h) h = LoadLibraryW(L"onnxruntime.dll");
        if (h) getBase = (GetBase)GetProcAddress(h, "OrtGetApiBase");
#else
        void* h = dlopen("libonnxruntime.so", RTLD_NOW);
        if (h) getBase = (GetBase)dlsym(h, "OrtGetApiBase");
#endif
        if (getBase) ort = getBase()->GetApi(ORT_API_VERSION);
        if (!ort) { err = "ONNX Runtime (onnxruntime.dll) isn't next to Mosaic Mask"; return false; }
    }
    if (!env && !ok(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "MosaicMask", &env), err)) return false;
    return true;
}
static OrtSession* openSession(const std::string& model, std::string& err) {
    if (!loadRuntime(err)) return nullptr;
    OrtSessionOptions* so;
    if (!ok(ort->CreateSessionOptions(&so), err)) return nullptr;
    ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    ort->SetIntraOpNumThreads(so, std::max(1u, std::thread::hardware_concurrency() / 2));
    OrtSession* s = nullptr;
#ifdef _WIN32
    const bool made = ok(ort->CreateSession(env, fs::u8path(model).wstring().c_str(), so, &s), err);
#else
    const bool made = ok(ort->CreateSession(env, model.c_str(), so, &s), err);
#endif
    ort->ReleaseSessionOptions(so);
    return made ? s : nullptr;
}
// one-input session; the outputs as float vectors
static bool run(OrtSession* s, std::vector<float>& input, const std::vector<int64_t>& shape, std::vector<std::vector<float>>& outs, std::string& err) {
    std::lock_guard<std::mutex> g(runLock);
    OrtMemoryInfo* mem;
    if (!ok(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem), err)) return false;
    OrtValue* in = nullptr;
    const bool made = ok(ort->CreateTensorWithDataAsOrtValue(mem, input.data(), input.size() * sizeof(float), shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in), err);
    ort->ReleaseMemoryInfo(mem);
    if (!made) return false;
    OrtAllocator* alloc;
    ort->GetAllocatorWithDefaultOptions(&alloc);
    size_t nOut = 0;
    ort->SessionGetOutputCount(s, &nOut);
    char* inName;
    ort->SessionGetInputName(s, 0, alloc, &inName);
    std::vector<char*> outNames(nOut);
    for (size_t i = 0; i < nOut; ++i) ort->SessionGetOutputName(s, i, alloc, &outNames[i]);
    std::vector<OrtValue*> res(nOut, nullptr);
    const char* inNames[] = {inName};
    const bool ran = ok(ort->Run(s, nullptr, inNames, &in, 1, (const char* const*)outNames.data(), nOut, res.data()), err);
    alloc->Free(alloc, inName);
    for (auto n : outNames) alloc->Free(alloc, n);
    ort->ReleaseValue(in);
    if (!ran) return false;
    outs.assign(nOut, {});
    for (size_t i = 0; i < nOut; ++i) {
        OrtTensorTypeAndShapeInfo* info;
        ort->GetTensorTypeAndShape(res[i], &info);
        size_t count = 0;
        ort->GetTensorShapeElementCount(info, &count);
        ort->ReleaseTensorTypeAndShapeInfo(info);
        float* d;
        ort->GetTensorMutableData(res[i], (void**)&d);
        outs[i].assign(d, d + count);
        ort->ReleaseValue(res[i]);
    }
    return true;
}

bool init(const std::string& modelsDir, std::string& err) {
    if (!yunet) { yunet = openSession(modelsDir + "/face_yunet.onnx", err); if (!yunet) return false; }
    if (!mesh) { mesh = openSession(modelsDir + "/face_mesh.onnx", err); if (!mesh) return false; }
    return true;
}

static float sampleBGR(const cv::Mat& m, float x, float y, int c) {
    x = std::clamp(x, 0.0f, m.cols - 1.0f); y = std::clamp(y, 0.0f, m.rows - 1.0f);
    const int x0 = (int)x, y0 = (int)y, x1 = std::min(m.cols - 1, x0 + 1), y1 = std::min(m.rows - 1, y0 + 1);
    const float fx = x - x0, fy = y - y0;
    auto P = [&](int xx, int yy) { return (float)m.ptr<uint8_t>(yy)[xx * 3 + c]; };
    return (P(x0, y0) * (1 - fx) + P(x1, y0) * fx) * (1 - fy) + (P(x0, y1) * (1 - fx) + P(x1, y1) * fx) * fy;
}

struct Det { float x, y, w, h, score; float kp[10]; };
// YuNet: the picture letterboxed into 640 x 640, BGR 0..255; outputs per stride 8/16/32: cls, obj, bbox, kps
static std::vector<Det> detect(const cv::Mat& bgr) {
    const int S = 640;
    const float scale = (float)S / std::max(bgr.cols, bgr.rows);
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(std::max(1, (int)std::lround(bgr.cols * scale)), std::max(1, (int)std::lround(bgr.rows * scale))), 0, 0, cv::INTER_LINEAR);
    std::vector<float> in((size_t)3 * S * S, 0.0f);
    for (int y = 0; y < small.rows; ++y) {
        const uint8_t* p = small.ptr<uint8_t>(y);
        for (int x = 0; x < small.cols; ++x)
            for (int c = 0; c < 3; ++c) in[(size_t)c * S * S + (size_t)y * S + x] = p[x * 3 + c];
    }
    std::vector<std::vector<float>> o; std::string err;
    if (!yunet || !run(yunet, in, {1, 3, S, S}, o, err) || o.size() < 12) return {};
    std::vector<Det> found;
    const int strides[3] = {8, 16, 32};
    for (int k = 0; k < 3; ++k) {
        const int st = strides[k], cols = S / st, n = cols * cols;
        const auto &cls = o[k], &obj = o[3 + k], &box = o[6 + k], &kps = o[9 + k];
        for (int i = 0; i < n; ++i) {
            const float sc = std::sqrt(std::clamp(cls[i], 0.0f, 1.0f) * std::clamp(obj[i], 0.0f, 1.0f));
            if (sc < 0.6f) continue;
            const int r = i / cols, c = i % cols;
            Det f;
            const float cx = (c + box[i * 4]) * st, cy = (r + box[i * 4 + 1]) * st;
            const float bw = std::exp(box[i * 4 + 2]) * st, bh = std::exp(box[i * 4 + 3]) * st;
            f.x = (cx - bw / 2) / scale; f.y = (cy - bh / 2) / scale; f.w = bw / scale; f.h = bh / scale; f.score = sc;
            for (int p = 0; p < 5; ++p) { f.kp[p * 2] = (c + kps[i * 10 + p * 2]) * st / scale; f.kp[p * 2 + 1] = (r + kps[i * 10 + p * 2 + 1]) * st / scale; }
            found.push_back(f);
        }
    }
    std::sort(found.begin(), found.end(), [](const Det& a, const Det& b) { return a.w * a.h * a.score > b.w * b.h * b.score; });   // the main (biggest) face first
    return found;
}
// Face Mesh on a square crop (centre, side, angle): 468 points in picture pixels; its face score (a logit)
static bool fitMesh(const cv::Mat& bgr, float cx, float cy, float side, float angle, std::vector<cv::Point2f>& pts, float& score) {
    const int N = 192;
    const float ca = std::cos(angle), sa = std::sin(angle), k = side / N;
    std::vector<float> in((size_t)N * N * 3);   // NHWC RGB 0..1
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const float u = (x + 0.5f - N / 2.0f) * k, v = (y + 0.5f - N / 2.0f) * k;
            const float px = cx + u * ca - v * sa, py = cy + u * sa + v * ca;
            for (int c = 0; c < 3; ++c) in[((size_t)y * N + x) * 3 + c] = sampleBGR(bgr, px, py, 2 - c) / 255.0f;
        }
    std::vector<std::vector<float>> o; std::string err;
    if (!mesh || !run(mesh, in, {1, N, N, 3}, o, err) || o.size() < 2) return false;
    const auto& lm = o[0].size() >= 1404 ? o[0] : o[1];
    score = (o[0].size() >= 1404 ? o[1] : o[0])[0];
    pts.resize(468);
    for (int i = 0; i < 468; ++i) {
        const float u = (lm[i * 3] - N / 2.0f) * k, v = (lm[i * 3 + 1] - N / 2.0f) * k;
        pts[i] = cv::Point2f(cx + u * ca - v * sa, cy + u * sa + v * ca);
    }
    return true;
}
// the crop MediaPipe tracks with: the mesh's own box, upright by the eyes' outer corners (33, 263), 1.5x
static void cropOf(const std::vector<cv::Point2f>& pts, float& cx, float& cy, float& side, float& angle) {
    angle = std::atan2(pts[263].y - pts[33].y, pts[263].x - pts[33].x);
    const float ca = std::cos(-angle), sa = std::sin(-angle);
    float lo[2] = {1e9f, 1e9f}, hi[2] = {-1e9f, -1e9f};
    for (const auto& p : pts) {
        const float u = p.x * ca - p.y * sa, v = p.x * sa + p.y * ca;
        lo[0] = std::min(lo[0], u); hi[0] = std::max(hi[0], u); lo[1] = std::min(lo[1], v); hi[1] = std::max(hi[1], v);
    }
    const float mu = (lo[0] + hi[0]) / 2, mv = (lo[1] + hi[1]) / 2;
    cx = mu * std::cos(angle) - mv * std::sin(angle); cy = mu * std::sin(angle) + mv * std::cos(angle);
    side = std::max(hi[0] - lo[0], hi[1] - lo[1]) * 1.5f;
}

std::vector<cv::Point2f> Tracker::landmarks(const cv::Mat& bgr) {
    if (bgr.empty()) return {};
    std::vector<cv::Point2f> pts;
    float score = 0, cx, cy, side, angle;
    if (!still_ && last_.size() == 468) {   // tracking: fitted on the last frame's face
        cropOf(last_, cx, cy, side, angle);
        if (fitMesh(bgr, cx, cy, side, angle, pts, score) && score >= 0) { last_ = pts; return pts; }
    }
    const auto faces = detect(bgr);
    if (faces.empty()) { last_.clear(); return {}; }
    const Det& f = faces[0];
    angle = std::atan2(f.kp[3] - f.kp[1], f.kp[2] - f.kp[0]);
    cx = f.x + f.w / 2; cy = f.y + f.h / 2; side = std::max(f.w, f.h) * 1.5f;
    if (!fitMesh(bgr, cx, cy, side, angle, pts, score) || score < 0) { last_.clear(); return {}; }
    // a second fit on the mesh's own box: a tighter crop, as MediaPipe tracks
    cropOf(pts, cx, cy, side, angle);
    std::vector<cv::Point2f> again; float s2 = 0;
    if (fitMesh(bgr, cx, cy, side, angle, again, s2) && s2 >= 0) pts.swap(again);
    last_ = still_ ? std::vector<cv::Point2f>() : pts;
    return pts;
}

// ---------- the occluder (FaceFusion's XSeg) ----------
static const int XS = 256;
// InsightFace's ArcFace five points (112 x 112) moved to 256 x 256 the way InsightFace does for 128 / 256
static cv::Point2f templ(int i) {
    static const float A[5][2] = {{38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
    return cv::Point2f(A[i][0] * XS / 128.0f + 8.0f * XS / 128.0f, A[i][1] * XS / 128.0f);
}
static std::vector<cv::Point2f> fivePoints(const std::vector<cv::Point2f>& lm) {
    return {(lm[33] + lm[133]) * 0.5f, (lm[362] + lm[263]) * 0.5f, lm[1], lm[61], lm[291]};
}
bool Occluder::load(const std::string& modelsDir) {
    std::string err;
    const std::string model = modelsDir + "/xseg_1.onnx";
    std::error_code ec;
    if (!fs::exists(fs::u8path(model), ec)) { why = "no models/xseg_1.onnx"; return false; }
    session_ = openSession(model, err);
    if (!session_) { why = err; return false; }
    on = true;
    return true;
}
struct Occluder::Async {
    std::thread worker;
    std::mutex m;
    std::condition_variable cv;
    cv::Mat crop;                 // the next lined-up face to look at (empty: nothing waiting)
    cv::Mat mask;                 // the last mask, in the lined-up frame
    bool quit = false;
};
Occluder::~Occluder() {
    if (async_) {
        { std::lock_guard<std::mutex> g(async_->m); async_->quit = true; }
        async_->cv.notify_all();
        if (async_->worker.joinable()) async_->worker.join();
        delete async_;
    }
}
cv::Mat Occluder::mask(const cv::Mat& bgr, const std::vector<cv::Point2f>& lm, cv::Rect box) {
    cv::Mat ones(box.height, box.width, CV_32F, cv::Scalar(1.0f));
    if (!session_ || lm.size() != 468) return ones;
    std::vector<cv::Point2f> dst(5); for (int i = 0; i < 5; ++i) dst[i] = templ(i);
    cv::Mat m = cv::estimateAffinePartial2D(fivePoints(lm), dst);
    if (m.empty()) return ones;
    if (!async_) {
        async_ = new Async;
        Async* a = async_; void* sess = session_;
        a->worker = std::thread([a, sess] {
            while (true) {
                cv::Mat crop;
                {
                    std::unique_lock<std::mutex> lk(a->m);
                    a->cv.wait(lk, [a] { return a->quit || !a->crop.empty(); });
                    if (a->quit) return;
                    crop = a->crop; a->crop.release();
                }
                std::vector<float> in((size_t)XS * XS * 3);
                for (int y = 0; y < XS; ++y) { const uint8_t* p = crop.ptr<uint8_t>(y); for (int x = 0; x < XS * 3; ++x) in[(size_t)y * XS * 3 + x] = p[x] / 255.0f; }
                std::vector<std::vector<float>> o; std::string err;
                if (!run((OrtSession*)sess, in, {1, XS, XS, 3}, o, err) || o.empty() || o[0].size() < (size_t)XS * XS) continue;
                cv::Mat out(XS, XS, CV_32F, o[0].data());
                cv::Mat c; cv::min(cv::max(out, 0.0), 1.0, c);
                cv::GaussianBlur(c, c, cv::Size(0, 0), 5);
                cv::Mat firm; cv::max(c, 0.5, firm); firm = (firm - 0.5) * 2.0;   // firm, softened (as FaceFusion)
                std::lock_guard<std::mutex> g(a->m);
                a->mask = firm.clone();
            }
        });
    }
    cv::Mat crop;
    cv::warpAffine(bgr, crop, m, cv::Size(XS, XS), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::Mat last;
    {
        std::lock_guard<std::mutex> g(async_->m);
        async_->crop = crop;      // the newest face replaces any still waiting
        last = async_->mask;
    }
    async_->cv.notify_one();
    if (last.empty()) return ones;
    cv::Mat back; cv::invertAffineTransform(m, back);
    back.at<double>(0, 2) -= box.x; back.at<double>(1, 2) -= box.y;
    cv::Mat res;
    // outside the lined-up crop (the forehead's top, the hair line) counts as face
    cv::warpAffine(last, res, back, box.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(1.0f));
    return res;
}

}
