// Mosaic Mask - the mosaic (see mosaic.h): face_pool.py, mesh.py and suit.py in C++.
#include "mosaic.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace Mosaic {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- the mesh: MediaPipe's indices (mesh.py)
// the points the warp follows: the face's outline, lips, eyes, brows, the nose's ridge and wings, the cheeks
static const int SUBSET[] = {0, 1, 2, 4, 5, 6, 7, 10, 13, 14, 17, 19, 21, 33, 37, 39, 40, 46, 50, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 98, 101, 103, 105, 107, 109, 115, 118, 123, 127, 129, 131, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 187, 191, 195, 197, 205, 207, 234, 246, 249, 251, 263, 267, 269, 270, 276, 280, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 330, 332, 334, 336, 338, 344, 347, 352, 356, 358, 360, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 411, 415, 425, 427, 454, 466};
static const int OVAL[] = {10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454};
static const int NSUB = sizeof SUBSET / sizeof SUBSET[0], NOVAL = sizeof OVAL / sizeof OVAL[0];
// anchors for the fragments: eyes, brows, the bridge, nose, mouth, chin, cheeks, jaw, temples, forehead
static const int ANCHORS[] = {159, 386, 105, 334, 6, 4, 13, 152, 50, 280, 172, 397, 54, 284, 10};
static const cv::Point2f EYE_L_AT(130, 165), EYE_R_AT(254, 165), MOUTH_AT(192, 280);
// your own blinks and mouth: the inside of the lips, the eyes, and how open each eye is
static const int INNER_LIPS[] = {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95};
static const int EYES[2][16] = {{33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7},
                                {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382}};
static const int EYE_OPEN[2][4] = {{159, 145, 33, 133}, {386, 374, 362, 263}};   // top, bottom, corner, corner
static const float CLOSING = 0.85f, SHUT = 0.68f;

static std::vector<cv::Point2f> pick(const std::vector<cv::Point2f>& lm, const int* idx, int n) {
    std::vector<cv::Point2f> v(n); for (int i = 0; i < n; ++i) v[i] = lm[idx[i]]; return v;
}
static std::vector<cv::Point2f> toCanon(const std::vector<cv::Point2f>& lm) {
    const std::vector<cv::Point2f> a = {(lm[33] + lm[133]) * 0.5f, (lm[362] + lm[263]) * 0.5f, (lm[13] + lm[14]) * 0.5f};
    const std::vector<cv::Point2f> b = {EYE_L_AT, EYE_R_AT, MOUTH_AT};
    cv::Mat m = cv::estimateAffinePartial2D(a, b);
    std::vector<cv::Point2f> out;
    if (m.empty()) return lm;
    cv::transform(lm, out, m);
    return out;
}
// Delaunay triangles over the points, inside the shared square
static std::vector<cv::Vec3i> triangles(const std::vector<cv::Point2f>& pts) {
    cv::Subdiv2D sub(cv::Rect(0, 0, CANON, CANON));
    std::map<std::pair<long, long>, int> index;
    auto key = [](float x, float y) { return std::make_pair(std::lround(x * 100), std::lround(y * 100)); };
    for (int i = 0; i < (int)pts.size(); ++i) {
        const float x = std::clamp(pts[i].x, 0.5f, CANON - 1.5f), y = std::clamp(pts[i].y, 0.5f, CANON - 1.5f);
        index[key(x, y)] = i;
        sub.insert(cv::Point2f(x, y));
    }
    std::vector<cv::Vec6f> list; sub.getTriangleList(list);
    std::vector<cv::Vec3i> tris;
    for (const auto& t : list) {
        int c[3]; bool all = true;
        for (int k = 0; k < 3 && all; ++k) { auto it = index.find(key(t[k * 2], t[k * 2 + 1])); if (it == index.end()) all = false; else c[k] = it->second; }
        if (all) tris.push_back(cv::Vec3i(c[0], c[1], c[2]));
    }
    return tris;
}

TriangleWarp::TriangleWarp(const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Vec3i>& tris) : tris_(tris) {
    for (const auto& t : tris) {
        std::vector<cv::Point2f> s = {srcPts[t[0]], srcPts[t[1]], srcPts[t[2]]};
        Src e; e.box = cv::boundingRect(s);
        for (int k = 0; k < 3; ++k) e.rel[k] = s[k] - cv::Point2f((float)e.box.x, (float)e.box.y);
        src_.push_back(e);
    }
}
void TriangleWarp::draw(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point2f>& dstPts, cv::Mat* mask) const {
    const cv::Rect frame(0, 0, dst.cols, dst.rows), srcFrame(0, 0, src.cols, src.rows);
    for (size_t i = 0; i < tris_.size(); ++i) {
        const auto& t = tris_[i]; const Src& s = src_[i];
        const cv::Rect sb = s.box & srcFrame;
        if (sb.width <= 0 || sb.height <= 0) continue;
        std::vector<cv::Point2f> d = {dstPts[t[0]], dstPts[t[1]], dstPts[t[2]]};
        const cv::Rect db = cv::boundingRect(d) & frame;
        if (db.width <= 0 || db.height <= 0) continue;
        cv::Point2f dRel[3], sRel[3];
        for (int k = 0; k < 3; ++k) { dRel[k] = d[k] - cv::Point2f((float)db.x, (float)db.y); sRel[k] = s.rel[k] - cv::Point2f((float)(sb.x - s.box.x), (float)(sb.y - s.box.y)); }
        cv::Mat m = cv::getAffineTransform(sRel, dRel);
        cv::Mat patch;
        cv::warpAffine(src(sb), patch, m, db.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        cv::Mat triMask = cv::Mat::zeros(db.size(), CV_8U);
        cv::Point ip[3]; for (int k = 0; k < 3; ++k) ip[k] = cv::Point((int)std::lround(dRel[k].x), (int)std::lround(dRel[k].y));
        cv::fillConvexPoly(triMask, ip, 3, cv::Scalar(255), cv::LINE_8);
        cv::Mat roi = dst(db);
        patch.copyTo(roi, triMask);
        if (mask) (*mask)(db).setTo(cv::Scalar(1.0f), triMask);
    }
}

// ---- the pool (face_pool.py)
static cv::Mat readImage(const fs::path& p) {   // OpenCV's imread can't open Unicode paths on Windows: through memory
    std::ifstream f(p, std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) return cv::Mat();
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}
static void writeImage(const fs::path& p, const cv::Mat& m) {
    std::vector<uchar> buf; cv::imencode(".png", m, buf);
    std::ofstream(p, std::ios::binary).write((const char*)buf.data(), buf.size());
}
static std::string hashOf(const std::string& s) {   // FNV-1a, 64 bit, as hex
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    char b[17]; snprintf(b, sizeof b, "%016llx", (unsigned long long)h); return b;
}
void Pool::load(const std::string& folder, const std::function<void(int, int)>& progress) {
    names.clear(); textures.clear(); layout.clear(); skipped.clear();
    std::error_code ec;
    const fs::path dir = fs::u8path(folder), cache = dir / "_canon";
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().u8string(); for (auto& c : ext) c = (char)tolower((unsigned char)c);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".bmp" || ext == ".jfif") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    fs::create_directories(cache, ec);
    Face::Tracker tracker(true);
    struct Found { std::string name; cv::Mat img; std::vector<cv::Point2f> lm; std::string stamp; };
    std::vector<Found> found;
    for (size_t n = 0; n < files.size(); ++n) {
        const std::string name = files[n].filename().u8string();
        cv::Mat img = readImage(files[n]);
        if (img.empty()) { skipped.push_back({name, "can't read it"}); continue; }
        if (std::max(img.cols, img.rows) > 1400) { const double s = 1400.0 / std::max(img.cols, img.rows); cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA); }
        auto lm = tracker.landmarks(img);
        if (lm.size() != 468) { skipped.push_back({name, "no face found"}); continue; }
        auto t = fs::last_write_time(files[n], ec).time_since_epoch().count();
        found.push_back({name, img, lm, std::to_string((long long)t)});
        if (progress) progress((int)n + 1, (int)files.size());
    }
    if (found.empty()) return;
    // the shared face: the average of every face, aligned by the eyes and mouth
    layout.assign(468, cv::Point2f(0, 0));
    for (const auto& f : found) { auto c = toCanon(f.lm); for (int i = 0; i < 468; ++i) layout[i] += c[i]; }
    for (auto& p : layout) p *= 1.0f / found.size();
    const auto subLayout = pick(layout, SUBSET, NSUB);
    const auto tris = triangles(subLayout);
    std::ostringstream st; for (const auto& p : subLayout) st << std::lround(p.x * 10) << ',' << std::lround(p.y * 10) << ';';
    const std::string stamp = hashOf(st.str());
    for (const auto& f : found) {
        const fs::path cached = cache / (hashOf(f.name + f.stamp + stamp) + ".png");
        cv::Mat tex = fs::exists(cached, ec) ? readImage(cached) : cv::Mat();
        if (tex.empty() || tex.cols != CANON) {
            tex = cv::Mat::zeros(CANON, CANON, CV_8UC3);
            TriangleWarp(pick(f.lm, SUBSET, NSUB), tris).draw(f.img, tex, subLayout);
            writeImage(cached, tex);
        }
        textures.push_back(tex);
        names.push_back(f.name);
    }
    std::ofstream js(cache / "pool.txt");
    js << "faces " << names.size() << "\n"; for (const auto& s : skipped) js << "skipped " << s.first << ": " << s.second << "\n";
}

// ---- the suit (suit.py)
Suit::Suit(const Pool& pool, int fragments, unsigned seed) : pool_(pool), rng_(seed ? seed : (unsigned)std::random_device{}()) {
    subLayout_ = pick(pool.layout, SUBSET, NSUB);
    tris_ = triangles(subLayout_);
    warp_ = TriangleWarp(subLayout_, tris_);
    std::vector<cv::Point> hull, oval;
    for (int i = 0; i < NOVAL; ++i) oval.push_back(cv::Point((int)pool.layout[OVAL[i]].x, (int)pool.layout[OVAL[i]].y));
    cv::convexHull(oval, hull);
    faceMask_ = cv::Mat::zeros(CANON, CANON, CV_32F);
    cv::fillConvexPoly(faceMask_, hull, cv::Scalar(1.0f));
    setFragments(fragments);
}
void Suit::setFragments(int n) {
    const int nA = sizeof ANCHORS / sizeof ANCHORS[0];
    n = std::max(nA, n); wanted_ = n;
    std::vector<cv::Point2f> pts;
    for (int a : ANCHORS) pts.push_back(pool_.layout[a]);
    std::vector<cv::Point> inside; cv::findNonZero(faceMask_ > 0.5f, inside);
    while ((int)pts.size() < n && !inside.empty()) { const auto& p = inside[rng_() % inside.size()]; pts.push_back(cv::Point2f((float)p.x, (float)p.y)); }
    // the patchwork: each pixel to its nearest point (a Voronoi map), soft edges between pieces, shares summing to 1
    cv::Mat label(CANON, CANON, CV_32S);
    for (int y = 0; y < CANON; ++y)
        for (int x = 0; x < CANON; ++x) {
            int best = 0; float bd = 1e30f;
            for (int i = 0; i < (int)pts.size(); ++i) { const float dx = x - pts[i].x, dy = y - pts[i].y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
            label.at<int>(y, x) = best;
        }
    std::vector<cv::Mat> w(pts.size());
    cv::Mat sum = cv::Mat::zeros(CANON, CANON, CV_32F);
    for (size_t i = 0; i < pts.size(); ++i) {
        cv::Mat on = (label == (int)i); on.convertTo(w[i], CV_32F, 1.0 / 255);
        cv::GaussianBlur(w[i], w[i], cv::Size(0, 0), 3.0);
        sum += w[i];
    }
    cv::max(sum, 1e-6, sum);
    frags_.clear();
    for (auto& x : w) {
        x /= sum;
        std::vector<cv::Point> nz; cv::findNonZero(x > 1e-3f, nz);
        if (nz.empty()) continue;
        const cv::Rect b = cv::boundingRect(nz);
        frags_.push_back({b, x(b).clone()});
    }
    const double t = now();
    const int count = (int)pool_.size();
    cur_.assign(frags_.size(), 0); nxt_.assign(frags_.size(), 0); tSwap_.assign(frags_.size(), t);
    std::uniform_real_distribution<double> u(0, swapMs / 1000);
    for (size_t i = 0; i < frags_.size(); ++i) { cur_[i] = nxt_[i] = count ? (int)(rng_() % count) : 0; tSwap_[i] = t + u(rng_); }
}
cv::Mat Suit::canonTexture(double t) {
    cv::Mat out = cv::Mat::zeros(CANON, CANON, CV_32FC3);
    const double fade = std::max(fadeMs / 1000, 1e-3);
    const int count = (int)pool_.size();
    std::uniform_real_distribution<double> jitter(0.5, 1.5);
    for (size_t i = 0; i < frags_.size(); ++i) {
        if (t >= tSwap_[i]) {   // time for a new face
            cur_[i] = nxt_[i];
            if (count > 1) { const int c = (int)(rng_() % (count - 1)); nxt_[i] = c + (c >= cur_[i]); }
            tSwap_[i] = t + swapMs / 1000 * jitter(rng_);
        }
        const float a = 1.0f - (float)std::min(1.0, std::max(0.0, (tSwap_[i] - t) / fade));   // the last fade before a swap
        const Frag& f = frags_[i];
        cv::Mat frag; pool_.textures[cur_[i]](f.box).convertTo(frag, CV_32FC3);
        if (a > 0) { cv::Mat n; pool_.textures[nxt_[i]](f.box).convertTo(n, CV_32FC3); frag = frag * (1 - a) + n * a; }
        cv::Mat wt3; cv::Mat chans[3] = {f.wt, f.wt, f.wt}; cv::merge(chans, 3, wt3);
        cv::Mat roi = out(f.box);
        roi += frag.mul(wt3);
    }
    return out;
}
void Suit::matchColour(cv::Mat& canon, const cv::Mat& live) const {
    if (colourMatch <= 0 || live.empty()) return;
    cv::Mat small, fm, lv;
    cv::resize(canon, small, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
    cv::resize(faceMask_, fm, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
    cv::resize(live, lv, cv::Size(48, 48), 0, 0, cv::INTER_AREA);
    lv.convertTo(lv, CV_32FC3);
    cv::Scalar sm, ss, dm, ds;
    cv::meanStdDev(small, sm, ss, fm > 0.5f);
    cv::meanStdDev(lv, dm, ds);
    const float k = colourMatch;
    std::vector<cv::Mat> ch; cv::split(canon, ch);
    for (int c = 0; c < 3; ++c) {
        const double s1 = ss[c] + 1e-3, s2 = ds[c] + 1e-3;
        const double gain = 1 + (s2 / s1 - 1) * k, shift = (dm[c] - sm[c] * s2 / s1) * k;
        ch[c] = ch[c] * gain + shift;
    }
    cv::merge(ch, canon);
}
cv::Mat Suit::own(const std::vector<cv::Point2f>& lm, cv::Size size, cv::Point2f offset) {
    cv::Mat m = cv::Mat::zeros(size, CV_32F);
    auto poly = [&](const int* idx, int n, cv::Point2f centreScale, float value) {
        std::vector<cv::Point2f> p = pick(lm, idx, n);
        cv::Point2f c(0, 0); for (const auto& q : p) c += q; c *= 1.0f / n;
        std::vector<cv::Point> ip;
        for (const auto& q : p) { const cv::Point2f r = c + cv::Point2f((q.x - c.x) * centreScale.x, (q.y - c.y) * centreScale.y) - offset; ip.push_back(cv::Point((int)std::lround(r.x), (int)std::lround(r.y))); }
        cv::Mat layer = cv::Mat::zeros(size, CV_32F);
        std::vector<std::vector<cv::Point>> pp = {ip};
        cv::fillPoly(layer, pp, cv::Scalar(value));
        cv::max(m, layer, m);
    };
    poly(INNER_LIPS, 20, cv::Point2f(1, 1), 1.0f);   // the inside of the mouth is always yours
    for (int k = 0; k < 2; ++k) {   // as an eye closes it fades to your own lid (measured against your own open eye)
        const auto& o = EYE_OPEN[k];
        const float gap = (float)cv::norm(lm[o[0]] - lm[o[1]]) / std::max((float)cv::norm(lm[o[2]] - lm[o[3]]), 1e-3f);
        eyeOpen_[k] = eyeOpen_[k] < 0 ? gap : std::max(eyeOpen_[k] * 0.998f, gap);
        const float ref = eyeOpen_[k];
        const float shut = std::clamp((CLOSING * ref - gap) / ((CLOSING - SHUT) * ref), 0.0f, 1.0f);
        if (shut > 0) poly(EYES[k], 16, cv::Point2f(1.25f, 1.9f), shut);
    }
    const int k = std::max(3, (int)(std::min(size.width, size.height) * 0.03) | 1);
    cv::GaussianBlur(m, m, cv::Size(k, k), 0);
    return m;
}
void Suit::apply(cv::Mat& frame, const std::vector<cv::Point2f>& lm, Face::Occluder* occluder) {
    if (lm.size() != 468 || pool_.size() == 0) return;
    cv::Mat canon = canonTexture(now());
    std::vector<cv::Point> ovalPts, hull;
    for (int i = 0; i < NOVAL; ++i) ovalPts.push_back(cv::Point((int)lm[OVAL[i]].x, (int)lm[OVAL[i]].y));
    cv::convexHull(ovalPts, hull);
    const cv::Rect fb = cv::boundingRect(hull);
    const int pad = (int)(std::max(fb.width, fb.height) * 0.08) + 2;
    const cv::Rect box = cv::Rect(fb.x - pad, fb.y - pad, fb.width + 2 * pad, fb.height + 2 * pad) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (box.width < 8 || box.height < 8) return;
    matchColour(canon, frame(fb & cv::Rect(0, 0, frame.cols, frame.rows)));
    cv::Mat canon8; canon.convertTo(canon8, CV_8UC3);
    cv::Mat crop = frame(box);
    cv::Mat warped = crop.clone();
    cv::Mat mask = cv::Mat::zeros(box.size(), CV_32F);
    std::vector<cv::Point2f> dst = pick(lm, SUBSET, NSUB);
    const cv::Point2f off((float)box.x, (float)box.y);
    for (auto& p : dst) p -= off;
    warp_.draw(canon8, warped, dst, &mask);
    // a soft edge all round, so it melts into the head instead of a cut-out
    cv::Mat face = cv::Mat::zeros(box.size(), CV_32F);
    std::vector<cv::Point> h2; for (const auto& p : hull) h2.push_back(p - cv::Point(box.x, box.y));
    cv::fillConvexPoly(face, h2, cv::Scalar(1.0f));
    const int sz = std::max(3, (int)(std::min(fb.width, fb.height) * 0.06) | 1);
    cv::erode(face, face, cv::Mat::ones(sz, sz, CV_8U));
    cv::GaussianBlur(face, face, cv::Size(0, 0), sz / 2.0);
    cv::Mat alpha = mask.mul(face).mul(1.0 - own(lm, box.size(), off));
    if (occluder && occluder->on) alpha = alpha.mul(occluder->mask(frame, lm, box));
    cv::Mat a3, w32, c32; cv::Mat ch[3] = {alpha, alpha, alpha}; cv::merge(ch, 3, a3);
    warped.convertTo(w32, CV_32FC3); crop.convertTo(c32, CV_32FC3);
    cv::Mat outF = w32.mul(a3) + c32.mul(cv::Scalar(1, 1, 1) - a3);
    outF.convertTo(crop, CV_8UC3);
}

}
