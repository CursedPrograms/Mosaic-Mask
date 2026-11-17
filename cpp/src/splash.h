// Mosaic Mask - the splash: shown in its own small window while Mosaic Mask starts, and in Help > About.
#pragma once

struct ImDrawList;
struct ImFont;

namespace Splash {
constexpr float W = 640, H = 400;
void setFonts(ImFont* title, ImFont* small);
void draw(ImDrawList* dl, float x, float y, const char* status, float progress);
}
