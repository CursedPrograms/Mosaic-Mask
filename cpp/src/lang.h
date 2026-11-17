// Mosaic Mask - language packs: every word on screen goes through T("English"). A pack is lang/<code>.txt,
// one "English = translation" a line; anything it doesn't have stays English.
#pragma once
#include <string>
#include <vector>

namespace Lang {
struct Pack { std::string code, name; };
const std::vector<Pack>& packs();           // English first, then the packs found in lang/
bool load(const std::string& dir, const std::string& code);   // "" or "en": English
const std::string& current();
bool needsCJK();                            // Japanese, Chinese, Korean: a CJK font
}
const char* T(const char* english);
