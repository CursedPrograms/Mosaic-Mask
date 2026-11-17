// Mosaic Mask - the splash: a face made of patchwork tiles (each a different grey, a few in blue, as the mosaic's pieces
// are each a different stranger), the name, the version, the credits and what's loading. Atelier greys and blue, solid.
#include "splash.h"

#include <imgui.h>

#include <cfloat>
#include <cmath>

#ifndef MOSAIC_VERSION
#define MOSAIC_VERSION "2.0.0"
#endif

namespace Splash {

static ImFont *titleFont = nullptr, *smallFont = nullptr;
void setFonts(ImFont* title, ImFont* small) { titleFont = title; smallFont = small; }

static const ImU32 BG = IM_COL32(0x1a, 0x1a, 0x1a, 255), LINE = IM_COL32(0x33, 0x33, 0x33, 255),
                   TEXT = IM_COL32(0xd6, 0xd6, 0xd6, 255), DIM = IM_COL32(0x9a, 0x9a, 0x9a, 255),
                   BLUE = IM_COL32(0x00, 0x7b, 0xb7, 255);

static void text(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 col, const char* s) { dl->AddText(f ? f : ImGui::GetFont(), size, p, col, s); }
static float textWidth(ImFont* f, float size, const char* s) { return (f ? f : ImGui::GetFont())->CalcTextSizeA(size, FLT_MAX, 0, s).x; }
static float hash(int i, int k) { unsigned h = (unsigned)i * 374761393u + (unsigned)k * 668265263u; h = (h ^ (h >> 13)) * 1274126177u; return (h ^ (h >> 16)) / 4294967295.0f; }

void draw(ImDrawList* dl, float x, float y, const char* status, float progress) {
    const ImVec2 o(x, y), end(x + W, y + H);
    dl->PushClipRect(o, end, true);
    dl->AddRectFilled(o, end, BG);
    // the face: an oval of tiles, each its own shade; eyes and a mouth left dark; a few tiles in blue
    const float cx = x + 470, cy = y + 196, rx = 112, ry = 150, t = 17;
    int n = 0;
    for (float ty = cy - ry; ty < cy + ry; ty += t)
        for (float tx = cx - rx; tx < cx + rx; tx += t, ++n) {
            const float px = tx + t / 2, py = ty + t / 2, u = (px - cx) / rx, v = (py - cy) / ry;
            if (u * u + v * v > 1) continue;
            const bool eye = (std::fabs(u + 0.38f) < 0.17f || std::fabs(u - 0.38f) < 0.17f) && std::fabs(v + 0.12f) < 0.06f;
            const bool mouth = std::fabs(u) < 0.32f && std::fabs(v - 0.48f) < 0.05f;
            ImU32 c;
            if (eye || mouth) c = IM_COL32(0x22, 0x22, 0x22, 255);
            else if (hash(n, 3) < 0.09f) c = BLUE;
            else { const int g = 0x4a + (int)(hash(n, 1) * 0x5a); c = IM_COL32(g, g, g, 255); }
            dl->AddRectFilled(ImVec2(tx + 1, ty + 1), ImVec2(tx + t - 1, ty + t - 1), c, 2);
        }
    // the name: "Mosaic" in the text colour, "Mask" in blue
    const float ts = 48, tx = x + 40, ty = y + 92;
    text(dl, titleFont, ts, ImVec2(tx, ty), TEXT, "Mosaic ");
    text(dl, titleFont, ts, ImVec2(tx + textWidth(titleFont, ts, "Mosaic "), ty), BLUE, "Mask");
    text(dl, nullptr, 16, ImVec2(tx + 2, ty + 60), DIM, "Version " MOSAIC_VERSION);
    text(dl, nullptr, 16, ImVec2(tx + 2, ty + 82), DIM, "A disguise for video calls");
    const float cs = 13, cry = y + 306;
    text(dl, smallFont, cs, ImVec2(tx, cry), DIM, "Face Mesh (Google, Apache-2.0), YuNet (OpenCV, MIT), XSeg (FaceFusion),");
    text(dl, smallFont, cs, ImVec2(tx, cry + 17), DIM, "OpenCV, ONNX Runtime, miniaudio, Dear ImGui, SDL2.");
    text(dl, smallFont, cs, ImVec2(tx, cry + 34), DIM, "Farica\xe2\x98\x86Kimora \xc2\xa9 Cursed Entertainment");
    if (status) text(dl, nullptr, 16, ImVec2(tx, y + 368), TEXT, status);
    if (progress >= 0) {
        const float p = progress > 1 ? 1 : progress;
        dl->AddRectFilled(ImVec2(x, end.y - 3), end, LINE);
        dl->AddRectFilled(ImVec2(x, end.y - 3), ImVec2(x + W * p, end.y), BLUE);
    }
    dl->AddRect(o, end, LINE);
    dl->PopClipRect();
}

}
