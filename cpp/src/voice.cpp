// Mosaic Mask - the voice (see voice.h): voice.py's phase vocoder, pitch tracker and disguise in C++, on miniaudio.
#define NOMINMAX
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DECODING
#define MA_NO_GENERATION
#include "miniaudio.h"
#include "voice.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

namespace Voice {

static const int SR = SAMPLE_RATE;
static const int N = 2048;                  // the FFT: 93 ms of voice
static const int HOP = N / 4;               // a new piece every 23 ms
static const int LIFTER = (int)(SR * 0.0016);   // the envelope keeps cepstral detail under 1.6 ms
static const double PI = 3.14159265358979323846;
using cd = std::complex<double>;

const char* presetName(int p) { static const char* n[] = {"off", "female", "male", "disguise"}; return p >= 0 && p < PRESET_COUNT ? n[p] : "?"; }

// ---- an in-place radix-2 FFT (inverse: conjugate trick)
static void fft(std::vector<cd>& a, bool inverse) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) { size_t bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) std::swap(a[i], a[j]); }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2 * PI / len * (inverse ? 1 : -1);
        const cd wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1);
            for (size_t j = 0; j < len / 2; ++j) { const cd u = a[i + j], v = a[i + j + len / 2] * w; a[i + j] = u + v; a[i + j + len / 2] = u - v; w *= wl; }
        }
    }
    if (inverse) for (auto& x : a) x /= (double)n;
}
static std::vector<cd> rfft(const double* x, int n, int size) {   // the first size/2 + 1 bins of a (zero-padded) real signal
    std::vector<cd> a(size, 0.0);
    for (int i = 0; i < n && i < size; ++i) a[i] = x[i];
    fft(a, false);
    a.resize(size / 2 + 1);
    return a;
}
static std::vector<double> irfft(const std::vector<cd>& spec, int size) {   // a real signal from its half spectrum
    std::vector<cd> a(size);
    for (int k = 0; k <= size / 2; ++k) a[k] = spec[k];
    for (int k = 1; k < size / 2; ++k) a[size - k] = std::conj(spec[k]);
    fft(a, true);
    std::vector<double> out(size);
    for (int i = 0; i < size; ++i) out[i] = a[i].real();
    return out;
}

// ---- your pitch, learned as you talk: autocorrelation on each 93 ms, the median of the voiced ones lately
struct PitchTracker {
    std::deque<double> recent;
    size_t keep = (size_t)(6.0 * SR / HOP);
    double median = 0;
    double update(const std::vector<double>& frame) {
        const int n = (int)frame.size();
        double mean = 0; for (double v : frame) mean += v; mean /= n;
        std::vector<double> x(n); double rms = 0;
        for (int i = 0; i < n; ++i) { const double w = 0.5 - 0.5 * std::cos(2 * PI * i / (n - 1)); x[i] = (frame[i] - mean); rms += x[i] * x[i]; x[i] *= w; }
        if (std::sqrt(rms / n) < 0.01) return median;   // quiet: nothing to learn
        auto spec = rfft(x.data(), n, 2 * n);
        for (auto& c : spec) c = std::norm(c);
        auto ac = irfft(spec, 2 * n);
        const int lo = SR / 400, hi = SR / 60;           // voices: 60 - 400 Hz
        int lag = lo; for (int i = lo; i < hi; ++i) if (ac[i] > ac[lag]) lag = i;
        if (ac[lag] < 0.45 * ac[0]) return median;       // not clearly voiced
        recent.push_back((double)SR / lag);
        while (recent.size() > keep) recent.pop_front();
        if (recent.size() >= 8) { std::vector<double> s(recent.begin(), recent.end()); std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end()); median = s[s.size() / 2]; }
        return median;
    }
};
static void ratios(int preset, double yours, double& pitch, double& formant) {
    if (yours <= 0) { pitch = preset == FEMALE ? 1.6 : 0.65; formant = preset == FEMALE ? 1.15 : 0.87; return; }   // until it's heard you
    pitch = std::clamp((preset == FEMALE ? 210.0 : 115.0) / yours, 0.45, 2.6);
    formant = std::clamp(std::pow(pitch, 0.25), 0.82, 1.22);
}

// ---- disguise: pitch and formants on slow random walks round 1.0
struct Disguise {
    double p = 1, f = 1, tp = 1, tf = 1, next = 0;
    std::mt19937 rng{std::random_device{}()};
    void step(double dt, double& pitch, double& formant) {
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::uniform_real_distribution<double> a(-0.33, 0.33), b(-0.22, 0.22), c(1.5, 4.0);
        if (now >= next) { tp = std::pow(2.0, a(rng)); tf = std::pow(2.0, b(rng)); next = now + c(rng); }
        const double k = std::min(1.0, dt / 1.2);
        p += (tp - p) * k; f += (tf - f) * k;
        pitch = p; formant = f;
    }
};

// ---- the phase vocoder
struct Shifter {
    std::vector<double> window, inbuf, lastPhase, sumPhase, outbuf;
    double expected, norm;
    Shifter() : window(N), inbuf(N, 0.0), lastPhase(N / 2 + 1, 0.0), sumPhase(N / 2 + 1, 0.0), outbuf(N, 0.0) {
        double s = 0;
        for (int i = 0; i < N; ++i) { window[i] = 0.5 - 0.5 * std::cos(2 * PI * i / N); s += window[i] * window[i]; }   // np.hanning(N+1)[:-1]
        expected = 2 * PI * HOP / N;
        norm = 1.0 / (s / HOP);
    }
    void process(const float* block, float* out, double pitch, double formant) {
        std::move(inbuf.begin() + HOP, inbuf.end(), inbuf.begin());
        for (int i = 0; i < HOP; ++i) inbuf[N - HOP + i] = block[i];
        std::vector<double> x(N); for (int i = 0; i < N; ++i) x[i] = inbuf[i] * window[i];
        auto spec = rfft(x.data(), N, N);
        const int bins = N / 2 + 1;
        std::vector<double> mag(bins), freq(bins), logm(bins);
        for (int k = 0; k < bins; ++k) {
            mag[k] = std::abs(spec[k]);
            const double ph = std::arg(spec[k]);
            double delta = ph - lastPhase[k] - k * expected;
            lastPhase[k] = ph;
            delta = std::fmod(delta + PI, 2 * PI); if (delta < 0) delta += 2 * PI; delta -= PI;
            freq[k] = k + delta / expected;
        }
        // the envelope (formants) and the fine structure (harmonics), apart
        std::vector<cd> lm(bins); for (int k = 0; k < bins; ++k) lm[k] = std::log(mag[k] + 1e-9);
        auto cep = irfft(lm, N);
        for (int i = LIFTER; i < N - LIFTER; ++i) cep[i] = 0;
        auto envSpec = rfft(cep.data(), N, N);
        std::vector<double> env(bins), fine(bins);
        for (int k = 0; k < bins; ++k) { env[k] = std::exp(envSpec[k].real()); fine[k] = mag[k] / std::max(env[k], 1e-9); }
        // the harmonics move by the pitch ratio, the envelope by the formant ratio
        std::vector<double> newFine(bins, 0.0), newFreq(bins, 0.0), newEnv(bins, 0.0);
        for (int k = 0; k < bins; ++k) {
            const int t = (int)std::lround(k * pitch);
            if (t < bins) { newFine[t] += fine[k]; newFreq[t] = freq[k] * pitch; }
            const double src = k / formant;
            if (src <= bins - 1) { const int i0 = (int)src; const double fr = src - i0; newEnv[k] = env[i0] * (1 - fr) + env[std::min(bins - 1, i0 + 1)] * fr; }
        }
        // phases carry on: each bin advances by its new frequency
        std::vector<cd> outSpec(bins);
        for (int k = 0; k < bins; ++k) {
            sumPhase[k] += newFreq[k] * expected;
            sumPhase[k] = std::fmod(sumPhase[k] + PI, 2 * PI); if (sumPhase[k] < 0) sumPhase[k] += 2 * PI; sumPhase[k] -= PI;
            outSpec[k] = std::polar(newFine[k] * newEnv[k], sumPhase[k]);
        }
        auto frame = irfft(outSpec, N);
        for (int i = 0; i < N; ++i) outbuf[i] += frame[i] * window[i];
        for (int i = 0; i < HOP; ++i) out[i] = (float)(outbuf[i] * norm);
        std::move(outbuf.begin() + HOP, outbuf.end(), outbuf.begin());
        std::fill(outbuf.end() - HOP, outbuf.end(), 0.0);
    }
};

// ---- the devices and the stream
struct Ring {   // float samples between the audio threads and the worker
    std::mutex m; std::deque<float> q; size_t max;
    explicit Ring(size_t mx) : max(mx) {}
    void push(const float* p, size_t n) { std::lock_guard<std::mutex> g(m); q.insert(q.end(), p, p + n); while (q.size() > max) q.pop_front(); }
    size_t pop(float* p, size_t n) { std::lock_guard<std::mutex> g(m); const size_t k = std::min(n, q.size()); std::copy(q.begin(), q.begin() + k, p); q.erase(q.begin(), q.begin() + k); return k; }
    size_t size() { std::lock_guard<std::mutex> g(m); return q.size(); }
};

struct Changer::Impl {
    ma_context ctx{}; bool ctxOk = false;
    ma_device mic{}, cable{}, monitor{};
    bool micOn = false, cableOn = false, monOn = false;
    Ring in{SR}, toCable{SR / 3}, toMon{SR / 3};
    std::thread worker; std::atomic<bool> run{false};
    std::condition_variable cv; std::mutex cvm;
    PitchTracker tracker; Disguise disguise; Shifter shifter;
    std::vector<double> raw = std::vector<double>(N, 0.0);
    std::atomic<float> pitchHz{0};
};

static void onMic(ma_device* dev, void*, const void* in, ma_uint32 frames) {
    auto* d = (Changer::Impl*)dev->pUserData;   // (set below)
    d->in.push((const float*)in, frames);
    d->cv.notify_one();
}
static void onCable(ma_device* dev, void* out, const void*, ma_uint32 frames) {
    auto* d = (Changer::Impl*)dev->pUserData;
    const size_t got = d->toCable.pop((float*)out, frames);
    std::fill((float*)out + got, (float*)out + frames, 0.0f);
}
static void onMonitor(ma_device* dev, void* out, const void*, ma_uint32 frames) {
    auto* d = (Changer::Impl*)dev->pUserData;
    const size_t got = d->toMon.pop((float*)out, frames);
    std::fill((float*)out + got, (float*)out + frames, 0.0f);
}

static bool findDevice(ma_context* ctx, bool input, const std::string& part, ma_device_id& id) {
    ma_device_info *play, *cap; ma_uint32 np, nc;
    if (ma_context_get_devices(ctx, &play, &np, &cap, &nc) != MA_SUCCESS) return false;
    ma_device_info* list = input ? cap : play; const ma_uint32 n = input ? nc : np;
    std::string want = part; for (auto& c : want) c = (char)tolower((unsigned char)c);
    for (ma_uint32 i = 0; i < n; ++i) {
        std::string name = list[i].name; for (auto& c : name) c = (char)tolower((unsigned char)c);
        if (name.find(want) != std::string::npos) { id = list[i].id; return true; }
    }
    return false;
}

std::vector<std::string> devices(bool input) {
    std::vector<std::string> out;
    ma_context c;
    if (ma_context_init(nullptr, 0, nullptr, &c) != MA_SUCCESS) return out;
    ma_device_info *play, *cap; ma_uint32 np, nc;
    if (ma_context_get_devices(&c, &play, &np, &cap, &nc) == MA_SUCCESS)
        for (ma_uint32 i = 0; i < (input ? nc : np); ++i) out.push_back((input ? cap : play)[i].name);
    ma_context_uninit(&c);
    return out;
}

std::vector<float> render(const std::vector<float>& in, int preset) {
    PitchTracker tracker; Disguise disguise; Shifter shifter;
    std::vector<double> raw(N, 0.0);
    std::vector<float> out; out.reserve(in.size());
    std::vector<float> block(HOP), o(HOP);
    for (size_t s = 0; s + HOP <= in.size(); s += HOP) {
        std::copy(in.begin() + s, in.begin() + s + HOP, block.begin());
        std::move(raw.begin() + HOP, raw.end(), raw.begin());
        for (int i = 0; i < HOP; ++i) raw[N - HOP + i] = block[i];
        double pitch = 1, formant = 1;
        if (preset == FEMALE || preset == MALE) ratios(preset, tracker.update(raw), pitch, formant);
        else if (preset == DISGUISE) disguise.step((double)HOP / SR, pitch, formant);
        if (preset == OFF) o = block; else shifter.process(block.data(), o.data(), pitch, formant);
        out.insert(out.end(), o.begin(), o.end());
    }
    return out;
}

Changer::Changer() : d_(new Impl) {}
Changer::~Changer() { stop(); }
bool Changer::running() const { return d_->run; }
bool Changer::toCall() const { return d_->cableOn; }
float Changer::yourPitch() const { return d_->pitchHz; }

bool Changer::start(const std::string& mic, const std::string& cable, const std::string& monitor, std::string& err) {
    stop();
    Impl& d = *d_;
    if (ma_context_init(nullptr, 0, nullptr, &d.ctx) != MA_SUCCESS) { err = "No sound system"; return false; }
    d.ctxOk = true;
    auto open = [&](ma_device& dev, ma_device_type type, const ma_device_id* id, ma_device_data_proc cb) {
        ma_device_config c = ma_device_config_init(type);
        if (type == ma_device_type_capture) { c.capture.pDeviceID = id; c.capture.format = ma_format_f32; c.capture.channels = 1; }
        else { c.playback.pDeviceID = id; c.playback.format = ma_format_f32; c.playback.channels = 1; }
        c.sampleRate = SR; c.periodSizeInFrames = HOP; c.dataCallback = cb; c.pUserData = &d;
        return ma_device_init(&d.ctx, &c, &dev) == MA_SUCCESS;
    };
    ma_device_id micId{}, cableId{}, monId{};
    const bool micNamed = !mic.empty() && findDevice(&d.ctx, true, mic, micId);
    if (!open(d.mic, ma_device_type_capture, micNamed ? &micId : nullptr, onMic)) { err = "No microphone found"; return false; }
    d.micOn = true;
    if (findDevice(&d.ctx, false, cable.empty() ? "CABLE Input" : cable, cableId) && open(d.cable, ma_device_type_playback, &cableId, onCable)) d.cableOn = true;
    if (!monitor.empty() && findDevice(&d.ctx, false, monitor, monId) && open(d.monitor, ma_device_type_playback, &monId, onMonitor)) d.monOn = true;
    why = d.cableOn ? "" : "No VB-Audio Virtual Cable: the voice goes into recordings only";
    d.run = true;
    d.worker = std::thread([this] {
        Impl& d = *d_;
        std::vector<float> block(HOP), out(HOP);
        auto last = std::chrono::steady_clock::now();
        while (d.run) {
            {
                std::unique_lock<std::mutex> lk(d.cvm);
                d.cv.wait_for(lk, std::chrono::milliseconds(50), [&] { return !d.run || d.in.size() >= (size_t)HOP; });
            }
            while (d.run && d.in.size() >= (size_t)HOP) {
                d.in.pop(block.data(), HOP);
                const auto now = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(now - last).count(); last = now;
                std::move(d.raw.begin() + HOP, d.raw.end(), d.raw.begin());
                for (int i = 0; i < HOP; ++i) d.raw[N - HOP + i] = block[i];
                const int p = preset;
                double pitch = 1, formant = 1;
                if (p == FEMALE || p == MALE) { d.pitchHz = (float)d.tracker.update(d.raw); ratios(p, d.pitchHz, pitch, formant); }
                else if (p == DISGUISE) d.disguise.step(dt, pitch, formant);
                if (p == OFF) out = block;
                else d.shifter.process(block.data(), out.data(), pitch, formant);
                for (auto& v : out) v = std::clamp(v, -1.0f, 1.0f);
                if (d.cableOn) d.toCable.push(out.data(), HOP);
                if (d.monOn) d.toMon.push(out.data(), HOP);
                if (tap) tap(out.data(), HOP);
            }
        }
    });
    ma_device_start(&d.mic);
    if (d.cableOn) ma_device_start(&d.cable);
    if (d.monOn) ma_device_start(&d.monitor);
    return true;
}

void Changer::stop() {
    Impl& d = *d_;
    d.run = false;
    d.cv.notify_all();
    if (d.worker.joinable()) d.worker.join();
    if (d.micOn) { ma_device_uninit(&d.mic); d.micOn = false; }
    if (d.cableOn) { ma_device_uninit(&d.cable); d.cableOn = false; }
    if (d.monOn) { ma_device_uninit(&d.monitor); d.monOn = false; }
    if (d.ctxOk) { ma_context_uninit(&d.ctx); d.ctxOk = false; }
}

}
