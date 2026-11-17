// Mosaic Mask - the window (SDL2, OpenGL 3, Dear ImGui; Atelier's look: greys, the Atelier blue, controls on the left).
// The camera, the mosaic, the virtual camera and the recording run on their own thread; the window shows the latest
// finished frame. Keys: M mosaic, O occlusion, V virtual camera, A voice, R reload faces, C record / stop, Q / Esc quit.
//
//   MosaicMask [--camera N] [--source picture-or-video] [--faces folder] [--voice female] [--no-virtual]
//              [--lang de] [--capture out.bmp]      (tests: a screenshot once the faces are in and a frame is done)
#include "face.h"
#include "lang.h"
#include "mosaic.h"
#include "recorder.h"
#include "splash.h"
#include "vcam.h"
#include "voice.h"

#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#endif

#ifndef MOSAIC_VERSION
#define MOSAIC_VERSION "2.0.0"
#endif

namespace fs = std::filesystem;

// ---- the look: Atelier's greys and blue
static const ImU32 C_BG = IM_COL32(0x1a, 0x1a, 0x1a, 255), C_PANEL = IM_COL32(0x21, 0x21, 0x21, 255), C_WELL = IM_COL32(0x2a, 0x2a, 0x2a, 255),
                   C_LINE = IM_COL32(0x33, 0x33, 0x33, 255), C_TEXT = IM_COL32(0xd6, 0xd6, 0xd6, 255), C_DIM = IM_COL32(0x9a, 0x9a, 0x9a, 255),
                   C_BLUE = IM_COL32(0x00, 0x7b, 0xb7, 255), C_DEEP = IM_COL32(0x00, 0x5c, 0x8a, 255), C_RED = IM_COL32(0xd4, 0x3c, 0x3c, 255);
static ImVec4 col4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }
static void applyStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0; s.FrameRounding = 2; s.GrabRounding = 2; s.PopupRounding = 2; s.ScrollbarRounding = 2;
    s.WindowBorderSize = 0; s.FrameBorderSize = 0; s.PopupBorderSize = 1;
    s.WindowPadding = ImVec2(12, 10); s.FramePadding = ImVec2(7, 4); s.ItemSpacing = ImVec2(6, 7);
    ImVec4* c = s.Colors;
    c[ImGuiCol_Text] = col4(C_TEXT); c[ImGuiCol_TextDisabled] = col4(C_DIM);
    c[ImGuiCol_WindowBg] = col4(C_PANEL); c[ImGuiCol_ChildBg] = col4(C_PANEL); c[ImGuiCol_PopupBg] = col4(C_PANEL);
    c[ImGuiCol_Border] = col4(C_LINE); c[ImGuiCol_Separator] = col4(C_LINE);
    c[ImGuiCol_FrameBg] = col4(C_WELL); c[ImGuiCol_FrameBgHovered] = col4(IM_COL32(0x2e, 0x2e, 0x2e, 255)); c[ImGuiCol_FrameBgActive] = col4(IM_COL32(0x33, 0x33, 0x33, 255));
    c[ImGuiCol_MenuBarBg] = col4(C_BG);
    c[ImGuiCol_CheckMark] = col4(C_BLUE); c[ImGuiCol_SliderGrab] = col4(C_BLUE); c[ImGuiCol_SliderGrabActive] = col4(C_DEEP);
    c[ImGuiCol_Button] = col4(IM_COL32(0x33, 0x33, 0x33, 255)); c[ImGuiCol_ButtonHovered] = col4(C_DEEP); c[ImGuiCol_ButtonActive] = col4(C_BLUE);
    c[ImGuiCol_Header] = col4(IM_COL32(0x33, 0x33, 0x33, 255)); c[ImGuiCol_HeaderHovered] = col4(C_DEEP); c[ImGuiCol_HeaderActive] = col4(C_BLUE);
}
static bool fileExists(const std::string& f) { std::error_code ec; return fs::exists(fs::u8path(f), ec); }
static std::string exeDir() { char* b = SDL_GetBasePath(); std::string s = b ? b : ""; if (b) SDL_free(b); return s; }
static std::string prefsFile() { char* p = SDL_GetPrefPath("Cursed Entertainment", "Mosaic Mask"); std::string s = p ? std::string(p) + "mosaicmask.ini" : ""; if (p) SDL_free(p); return s; }
// the faces and recordings: Documents\Mosaic Mask, or "Mosaic Mask" in the user's folder when Windows guards Documents
static std::string homeDir() {
#ifdef _WIN32
    PWSTR p = nullptr; fs::path docs;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &p))) { docs = fs::path(p) / "Mosaic Mask"; CoTaskMemFree(p); }
    std::error_code ec;
    if (!docs.empty()) { fs::create_directories(docs, ec); const fs::path t = docs / ".write-test"; { std::ofstream f(t); if (f) { f.close(); fs::remove(t, ec); return docs.u8string(); } } }
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Profile, 0, nullptr, &p))) { fs::path home = fs::path(p) / "Mosaic Mask"; CoTaskMemFree(p); fs::create_directories(home, ec); return home.u8string(); }
#endif
    return exeDir();
}
static void openFolder(const std::string& f) {
#ifdef _WIN32
    std::error_code ec; fs::create_directories(fs::u8path(f), ec);
    ShellExecuteW(nullptr, L"open", fs::u8path(f).wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#endif
}
static void openUrl(const char* u) {
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", u, nullptr, nullptr, SW_SHOWNORMAL);
#endif
}

// ---- the state the window and the worker share
struct Settings { int fragments = 24, swapMs = 350, fadeMs = 120, colour = 40; };
static std::mutex mtx;
static Settings settings;
static std::atomic<bool> maskOn{true}, wantVirtual{true}, wantReload{true}, running{true}, occlusionOn{true};
static cv::Mat latest; static bool latestFace = false; static long long latestNo = 0;
static float fps = 0;
static std::string status, facesNote, virtualNote;
static int poolSize = 0, skippedCount = 0;
static float loadProgress = -1;
static Face::Occluder occluder;
static Voice::Changer voice;
static std::unique_ptr<Recorder> recorder;
static std::string facesDir, sourceArg; static int cameraArg = 0;
static std::string micName, monitorName, recNote;
static bool voiceOk = false;

static void say(const std::string& s) { std::lock_guard<std::mutex> g(mtx); status = s; }

// ---- the worker: the camera, the mosaic, the virtual camera, the recording
static void work() {
    cv::VideoCapture cap; cv::Mat still;
    bool isStill = false;
    if (!sourceArg.empty()) {
        std::string ext = fs::u8path(sourceArg).extension().u8string(); for (auto& c : ext) c = (char)tolower((unsigned char)c);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".bmp" || ext == ".jfif") { still = cv::imread(sourceArg); isStill = !still.empty(); }
        else cap.open(sourceArg);
    } else {
        cap.open(cameraArg, cv::CAP_DSHOW);
        if (cap.isOpened()) { cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720); }
    }
    if (!isStill && !cap.isOpened()) { say(T("No camera: is another program using it?")); }
    Face::Tracker tracker(isStill);
    Mosaic::Pool pool;
    std::unique_ptr<Mosaic::Suit> suit;
    VCam::Obs cam;
    double tLast = Mosaic::now();
    while (running) {
        if (wantReload) {
            wantReload = false;
            { std::lock_guard<std::mutex> g(mtx); facesNote = T("Reading the faces..."); loadProgress = 0; }
            pool.load(facesDir, [](int done, int total) { std::lock_guard<std::mutex> g(mtx); loadProgress = total ? (float)done / total : 1.0f; });
            suit = pool.size() ? std::make_unique<Mosaic::Suit>(pool, settings.fragments) : nullptr;
            std::lock_guard<std::mutex> g(mtx);
            poolSize = (int)pool.size(); skippedCount = (int)pool.skipped.size(); loadProgress = -1;
        }
        cv::Mat frame;
        if (isStill) { frame = still.clone(); std::this_thread::sleep_for(std::chrono::milliseconds(33)); }
        else if (cap.isOpened()) {
            if (!cap.read(frame) && !sourceArg.empty()) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); cap.read(frame); }
        }
        if (frame.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
        // the virtual camera follows its switch
        if (wantVirtual && !cam.isOpen()) {
            std::string err;
            if (!cam.open(frame.cols, frame.rows, 30, err)) { wantVirtual = false; std::lock_guard<std::mutex> g(mtx); virtualNote = err; }
            else { std::lock_guard<std::mutex> g(mtx); virtualNote.clear(); }
        } else if (!wantVirtual && cam.isOpen()) cam.close();
        Settings s; { std::lock_guard<std::mutex> g(mtx); s = settings; }
        if (suit) {
            if (s.fragments != suit->fragments()) suit->setFragments(s.fragments);
            suit->swapMs = std::max(40, s.swapMs); suit->fadeMs = s.fadeMs; suit->colourMatch = s.colour / 100.0f;
        }
        occluder.on = occlusionOn && occluder.available();
        const auto lm = tracker.landmarks(frame);
        cv::Mat out = frame;
        if (suit && maskOn && lm.size() == 468) { out = frame.clone(); suit->apply(out, lm, &occluder); }
        if (cam.isOpen()) cam.send(out);
        if (recorder) recorder->frame(out);
        const double now = Mosaic::now();
        std::lock_guard<std::mutex> g(mtx);
        fps = fps * 0.9f + 0.1f / (float)std::max(now - tLast, 1e-3); tLast = now;
        latest = out; latestFace = lm.size() == 468; ++latestNo;
    }
    cam.close();
}

// ---- preferences: language, devices, the sliders
static std::map<std::string, std::string> prefs;
static void loadPrefs() {
    std::ifstream f(fs::u8path(prefsFile())); std::string line;
    while (std::getline(f, line)) { const size_t eq = line.find('='); if (eq != std::string::npos) prefs[line.substr(0, eq)] = line.substr(eq + 1); }
    auto num = [](const std::string& k, int d) { auto it = prefs.find(k); return it == prefs.end() ? d : atoi(it->second.c_str()); };
    settings.fragments = num("fragments", 24); settings.swapMs = num("swap", 350); settings.fadeMs = num("fade", 120); settings.colour = num("colour", 40);
    micName = prefs["mic"]; monitorName = prefs["monitor"];
}
static void savePrefs() {
    std::ofstream f(fs::u8path(prefsFile()));
    f << "lang=" << Lang::current() << "\nfragments=" << settings.fragments << "\nswap=" << settings.swapMs << "\nfade=" << settings.fadeMs
      << "\ncolour=" << settings.colour << "\nmic=" << micName << "\nmonitor=" << monitorName << "\n";
}

static void loadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    const char* ui = "C:/Windows/Fonts/segoeui.ttf";
    if (!fileExists(ui)) { io.Fonts->AddFontDefault(); return; }
    static const ImWchar ranges[] = {0x0020, 0x024F, 0x0400, 0x04FF, 0x2010, 0x2027, 0x2190, 0x21FF, 0x25A0, 0x25FF, 0x2600, 0x26FF, 0};
    auto add = [&](const char* file, float size) {
        ImFont* f = io.Fonts->AddFontFromFileTTF(file, size, nullptr, ranges);
        ImFontConfig m; m.MergeMode = true;
        if (Lang::needsCJK()) {
            const char* cjk = Lang::current() == "ko" ? "C:/Windows/Fonts/malgun.ttf" : Lang::current() == "zh" ? "C:/Windows/Fonts/msyh.ttc" : "C:/Windows/Fonts/YuGothM.ttc";
            const ImWchar* r = Lang::current() == "ko" ? io.Fonts->GetGlyphRangesKorean() : Lang::current() == "zh" ? io.Fonts->GetGlyphRangesChineseSimplifiedCommon() : io.Fonts->GetGlyphRangesJapanese();
            if (fileExists(cjk)) io.Fonts->AddFontFromFileTTF(cjk, size, &m, r);
        }
        return f;
    };
    add(ui, 16.0f);
    ImFont* title = add(fileExists("C:/Windows/Fonts/segoeuil.ttf") ? "C:/Windows/Fonts/segoeuil.ttf" : ui, 46.0f);
    ImFont* smallF = add(ui, 13.0f);
    Splash::setFonts(title, smallF);
}

// ---- the window's parts
static GLuint tex = 0; static int texW = 0, texH = 0; static long long texNo = -1;
static bool showAbout = false;
static void section(const char* name) { ImGui::Spacing(); ImGui::TextColored(col4(C_BLUE), "%s", name); }
static bool slider(const char* label, int* v, int lo, int hi, const char* fmt, const char* tip) {
    ImGui::TextUnformatted(label);
    ImGui::SetNextItemWidth(-1);
    const bool c = ImGui::SliderInt((std::string("##") + label).c_str(), v, lo, hi, fmt);
    if (tip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    return c;
}
static void nextVoice() { voice.preset = (voice.preset + 1) % Voice::PRESET_COUNT; }
static void toggleRecord() {
    if (!recorder) return;
    if (recorder->active()) { std::string m; const std::string p = recorder->stop(m); recNote = p.empty() ? m : m + ": " + fs::u8path(p).filename().u8string(); return; }
    cv::Size size; { std::lock_guard<std::mutex> g(mtx); if (latest.empty()) return; size = latest.size(); }
    std::string err;
    if (!recorder->start(size, voiceOk ? Voice::SAMPLE_RATE : 0, err)) recNote = err;
}

static void controls() {
    Settings s; { std::lock_guard<std::mutex> g(mtx); s = settings; }
    section(T("FACES"));
    {
        std::lock_guard<std::mutex> g(mtx);
        if (loadProgress >= 0) ImGui::ProgressBar(loadProgress, ImVec2(-1, 0), T("Reading the faces..."));
        else if (poolSize) ImGui::Text(T("%d faces"), poolSize);
        else ImGui::TextWrapped("%s", T("No faces yet: put pictures of AI-generated people in the faces folder, then Reload"));
        if (skippedCount && loadProgress < 0) ImGui::TextDisabled(T("%d skipped (no face found)"), skippedCount);
    }
    if (ImGui::Button(T("Reload  (R)"), ImVec2(130, 0))) wantReload = true;
    ImGui::SameLine(); if (ImGui::Button(T("Open the folder"), ImVec2(-1, 0))) openFolder(facesDir);
    section(T("PICTURE"));
    bool m = maskOn; if (ImGui::Checkbox(T("Mosaic  (M)"), &m)) maskOn = m;
    bool o = occlusionOn;
    ImGui::BeginDisabled(!occluder.available());
    if (ImGui::Checkbox(occluder.available() ? T("Occlusion  (O)") : T("Occlusion: no model"), &o)) occlusionOn = o;
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", T("Hands, hair, a cup in front of your face stay in front of the mosaic"));
    bool v = wantVirtual; if (ImGui::Checkbox(T("Virtual camera  (V)"), &v)) wantVirtual = v;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", T("In your call, pick the camera \"OBS Virtual Camera\" (OBS Studio must be installed)"));
    { std::lock_guard<std::mutex> g(mtx); if (!virtualNote.empty()) ImGui::TextWrapped("%s", T(virtualNote.c_str())); }
    bool ch = false;
    ch |= slider(T("Fragments"), &s.fragments, 15, 80, "%d", T("How many pieces the face is cut into: more is finer and busier"));
    ch |= slider(T("Swap"), &s.swapMs, 40, 2000, "%d ms", T("How often each piece changes to someone else"));
    ch |= slider(T("Fade"), &s.fadeMs, 0, 600, "%d ms", T("How long a piece blends into the next face: the shimmer"));
    ch |= slider(T("Colour"), &s.colour, 0, 100, "%d%%", T("How much the mosaic takes on your skin tone: 0 is everyone's own skin, a patchwork"));
    if (ch) { std::lock_guard<std::mutex> g(mtx); settings = s; }
    section(T("VOICE  (A)"));
    if (!voiceOk) ImGui::TextDisabled("%s", T("No microphone found"));
    else {
        for (int p = 0; p < Voice::PRESET_COUNT; ++p) {
            if (p) ImGui::SameLine(0, 2);
            const bool on = voice.preset == p;
            if (on) ImGui::PushStyleColor(ImGuiCol_Button, col4(C_BLUE));
            if (ImGui::Button(T(Voice::presetName(p)), ImVec2((ImGui::GetContentRegionAvail().x - (Voice::PRESET_COUNT - 1 - p) * 2) / (Voice::PRESET_COUNT - p), 0))) voice.preset = p;
            if (on) ImGui::PopStyleColor();
        }
        if (!voice.toCall()) ImGui::TextWrapped("%s", T("No VB-Audio Virtual Cable: the voice goes into recordings only"));
        else ImGui::TextDisabled("%s", T("In your call, pick the microphone \"CABLE Output\""));
        if (voice.preset == Voice::FEMALE || voice.preset == Voice::MALE) {
            const float p = voice.yourPitch();
            if (p > 0) ImGui::TextDisabled(T("Your pitch: %.0f Hz"), p); else ImGui::TextDisabled("%s", T("Learning your pitch as you talk..."));
        }
    }
    auto devicePick = [&](const char* label, std::string& name, bool input, bool allowNone) {
        static std::map<std::string, std::vector<std::string>> cache;
        auto& list = cache[label]; if (list.empty()) list = Voice::devices(input);
        ImGui::TextUnformatted(label); ImGui::SetNextItemWidth(-1);
        const std::string shown = name.empty() ? (allowNone ? T("(none)") : T("(the default)")) : name;
        if (ImGui::BeginCombo((std::string("##") + label).c_str(), shown.c_str())) {
            if (ImGui::Selectable(allowNone ? T("(none)") : T("(the default)"), name.empty())) { name.clear(); return true; }
            for (const auto& d : list) if (ImGui::Selectable(d.c_str(), d == name)) { name = d; ImGui::EndCombo(); return true; }
            ImGui::EndCombo();
        }
        return false;
    };
    bool dev = devicePick(T("Microphone"), micName, true, false);
    dev |= devicePick(T("Also hear yourself on"), monitorName, false, true);
    if (dev) { const int p = voice.preset; std::string err; voiceOk = voice.start(micName, "CABLE Input", monitorName, err); voice.preset = p; savePrefs(); }
    section(T("RECORD  (C)"));
    const bool rec = recorder && recorder->active();
    ImGui::PushStyleColor(ImGuiCol_Button, col4(rec ? C_RED : C_BLUE));
    char label[64];
    if (rec) { const int sec = (int)recorder->seconds(); snprintf(label, sizeof label, "%s   %d:%02d", T("Stop"), sec / 60, sec % 60); }
    else snprintf(label, sizeof label, "%s", T("Record"));
    if (ImGui::Button(label, ImVec2(-1, 38))) toggleRecord();
    ImGui::PopStyleColor();
    ImGui::TextWrapped("%s", recNote.empty() ? T("The picture and your voice, to one .mp4") : recNote.c_str());
    if (ImGui::Button(T("Open the recordings"), ImVec2(-1, 0)) && recorder) openFolder(recorder->folder());
    ImGui::Spacing(); ImGui::Separator();
    std::lock_guard<std::mutex> g(mtx);
    ImGui::TextDisabled("%.0f fps  -  %s", fps, latestFace ? T("face found") : T("no face in view"));
    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
}

static void picture(ImVec2 pos, ImVec2 size) {
    {
        std::lock_guard<std::mutex> g(mtx);
        if (!latest.empty() && latestNo != texNo) {
            cv::Mat rgba; cv::cvtColor(latest, rgba, cv::COLOR_BGR2RGBA);
            if (!tex) glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
            texW = rgba.cols; texH = rgba.rows; texNo = latestNo;
        }
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), C_BG);
    if (!tex) { dl->AddText(ImVec2(pos.x + 20, pos.y + 20), C_DIM, T("Starting the camera...")); return; }
    const float k = std::min(size.x / texW, size.y / texH), w = texW * k, h = texH * k;
    const ImVec2 a(pos.x + (size.x - w) / 2, pos.y + (size.y - h) / 2);
    dl->AddImage((ImTextureID)(intptr_t)tex, a, ImVec2(a.x + w, a.y + h));
    if (recorder && recorder->active()) dl->AddCircleFilled(ImVec2(a.x + 22, a.y + 22), 8, C_RED);
}

int main(int argc, char** argv) {
    std::string capture, lang, voiceArg;
    bool noVirtual = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--capture" && i + 1 < argc) capture = argv[++i];
        else if (a == "--camera" && i + 1 < argc) cameraArg = atoi(argv[++i]);
        else if (a == "--source" && i + 1 < argc) sourceArg = argv[++i];
        else if (a == "--faces" && i + 1 < argc) facesDir = argv[++i];
        else if (a == "--voice" && i + 1 < argc) voiceArg = argv[++i];
        else if (a == "--lang" && i + 1 < argc) lang = argv[++i];
        else if (a == "--no-virtual") noVirtual = true;
    }
    loadPrefs();
    if (lang.empty()) lang = prefs["lang"];
    Lang::load(exeDir() + "lang", lang);
    const std::string home = homeDir();
    if (facesDir.empty()) facesDir = (fs::u8path(home) / "faces").u8string();
    { std::error_code ec; fs::create_directories(fs::u8path(facesDir), ec); }
    recorder = std::make_unique<Recorder>((fs::u8path(home) / "recordings").u8string());
    wantVirtual = !noVirtual && capture.empty();

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) return 1;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3); SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE); SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_Window* window = SDL_CreateWindow("Mosaic Mask", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_HIDDEN);
    if (!window) return 1;
    SDL_SetWindowMinimumSize(window, 980, 620);
    SDL_GLContext gl = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl); SDL_GL_SetSwapInterval(1);
    SDL_Surface* icon = SDL_LoadBMP((exeDir() + "assets/icon.bmp").c_str());
    if (icon) SDL_SetWindowIcon(window, icon);
    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    loadFonts(); applyStyle();
    SDL_Window* splashWin = nullptr;
    if (capture.empty()) {
        splashWin = SDL_CreateWindow("Mosaic Mask", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)Splash::W, (int)Splash::H, SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS);
        if (splashWin && SDL_GL_MakeCurrent(splashWin, gl) != 0) { SDL_DestroyWindow(splashWin); splashWin = nullptr; SDL_GL_MakeCurrent(window, gl); }
        if (splashWin && icon) SDL_SetWindowIcon(splashWin, icon);
    }
    if (icon) SDL_FreeSurface(icon);
    ImGui_ImplSDL2_InitForOpenGL(splashWin ? splashWin : window, gl);
    ImGui_ImplOpenGL3_Init("#version 330");
    // the face models, the occluder, the voice (while the splash is up)
    std::string err;
    std::atomic<bool> loaded{false};
    std::thread loader([&] {
        if (!Face::init(exeDir() + "models", err)) say(err);
        occluder.load(exeDir() + "models");
        if (capture.empty()) {
            std::string verr;
            voiceOk = voice.start(micName, "CABLE Input", monitorName, verr);
            for (int p = 0; p < Voice::PRESET_COUNT; ++p) if (voiceArg == Voice::presetName(p)) voice.preset = p;
            voice.tap = [](const float* b, int n) { if (recorder) recorder->sound(b, n); };
        }
        loaded = true;
    });
    if (splashWin) {
        const Uint32 start = SDL_GetTicks();
        while (true) {
            SDL_Event e; while (SDL_PollEvent(&e)) {}
            ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplSDL2_NewFrame(); ImGui::NewFrame();
            Splash::draw(ImGui::GetForegroundDrawList(), 0, 0, T("Loading the face models..."), std::min(1.0f, (SDL_GetTicks() - start) / 1500.0f));
            ImGui::Render(); glViewport(0, 0, (int)Splash::W, (int)Splash::H); glClearColor(0.1f, 0.1f, 0.1f, 1); glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); SDL_GL_SwapWindow(splashWin);
            if (SDL_GetTicks() - start > 1500 && loaded) break;
            SDL_Delay(16);
        }
    }
    loader.join();
    if (splashWin) {
        ImGui_ImplSDL2_Shutdown(); SDL_GL_MakeCurrent(window, gl); SDL_DestroyWindow(splashWin);
        ImGui_ImplSDL2_InitForOpenGL(window, gl);
    }
    if (capture.empty()) { SDL_ShowWindow(window); SDL_MaximizeWindow(window); SDL_RaiseWindow(window); }
    std::thread worker(work);

    int frames = 0; long long shotAfter = -1;
    bool quit = false;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_KEYDOWN && !ImGui::GetIO().WantTextInput) {
                switch (e.key.keysym.sym) {
                case SDLK_m: maskOn = !maskOn; break;
                case SDLK_o: if (occluder.available()) occlusionOn = !occlusionOn; break;
                case SDLK_v: wantVirtual = !wantVirtual; break;
                case SDLK_a: if (voiceOk) nextVoice(); break;
                case SDLK_r: wantReload = true; break;
                case SDLK_c: toggleRecord(); break;
                case SDLK_q: case SDLK_ESCAPE: quit = true; break;
                default: break;
                }
            }
        }
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplSDL2_NewFrame(); ImGui::NewFrame();
        const ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("main", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu(T("File"))) {
                if (ImGui::MenuItem(T("Open the faces folder"))) openFolder(facesDir);
                if (ImGui::MenuItem(T("Open the recordings")) && recorder) openFolder(recorder->folder());
                ImGui::Separator();
                if (ImGui::MenuItem(T("Quit"), "Q")) quit = true;
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu(T("Language"))) {
                for (const auto& p : Lang::packs())
                    if (ImGui::MenuItem(p.name.c_str(), nullptr, Lang::current() == p.code)) { Lang::load(exeDir() + "lang", p.code); savePrefs(); say(T("The new language shows fully after a restart (its letters)")); }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu(T("Help"))) {
                if (ImGui::MenuItem(T("About Mosaic Mask"))) showAbout = true;
                if (ImGui::MenuItem("GitHub")) openUrl("https://github.com/CursedPrograms/Mosaic-Mask");
                if (ImGui::MenuItem("Ko-fi")) openUrl("https://ko-fi.com/cursedentertainment");
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        const float leftW = 320;
        ImGui::BeginChild("controls", ImVec2(leftW, 0), false);
        controls();
        ImGui::EndChild();
        ImGui::SameLine();
        const ImVec2 p = ImGui::GetCursorScreenPos(), avail = ImGui::GetContentRegionAvail();
        picture(p, avail);
        ImGui::End();
        if (showAbout) { ImGui::OpenPopup("##about"); showAbout = false; }
        ImGui::SetNextWindowSize(ImVec2(Splash::W, Splash::H));
        if (ImGui::BeginPopupModal("##about", nullptr, ImGuiWindowFlags_NoDecoration)) {
            const ImVec2 o = ImGui::GetWindowPos();
            Splash::draw(ImGui::GetWindowDrawList(), o.x, o.y, T("Click to close"), -1);
            if (ImGui::IsMouseClicked(0) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        ImGui::Render();
        int w, h; SDL_GL_GetDrawableSize(window, &w, &h);
        glViewport(0, 0, w, h); glClearColor(0.1f, 0.1f, 0.1f, 1); glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ++frames;
        if (!capture.empty()) {   // tests: once the faces are in and a few masked frames are done
            bool ready; { std::lock_guard<std::mutex> g(mtx); ready = !wantReload && loadProgress < 0 && latestNo > 5; }
            if (ready && shotAfter < 0) shotAfter = frames + 10;
            if ((shotAfter > 0 && frames >= shotAfter) || frames > 3000) {
                std::vector<unsigned char> px((size_t)w * h * 4), flip(px.size());
                glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                for (int y = 0; y < h; ++y) memcpy(&flip[(size_t)y * w * 4], &px[(size_t)(h - 1 - y) * w * 4], (size_t)w * 4);
                SDL_Surface* s = SDL_CreateRGBSurfaceWithFormatFrom(flip.data(), w, h, 32, w * 4, SDL_PIXELFORMAT_ABGR8888);
                SDL_SaveBMP(s, capture.c_str()); SDL_FreeSurface(s);
                quit = true;
            }
        }
        SDL_GL_SwapWindow(window);
    }
    running = false;
    worker.join();
    if (recorder && recorder->active()) { std::string m; recorder->stop(m); }
    voice.stop();
    savePrefs();
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplSDL2_Shutdown(); ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl); SDL_DestroyWindow(window); SDL_Quit();
    return 0;
}
