// Mosaic Mask - language packs: see lang.h.
#include "lang.h"

#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace Lang {
static std::unordered_map<std::string, std::string> words;
static std::string cur = "en";
const std::vector<Pack>& packs() {
    static const std::vector<Pack> all = {
        {"en", "English"}, {"ja", "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"}, {"zh", "\xe4\xb8\xad\xe6\x96\x87"}, {"es", "Espa\xc3\xb1ol"},
        {"ko", "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4"}, {"fr", "Fran\xc3\xa7" "ais"}, {"de", "Deutsch"}, {"ru", "\xd0\xa0\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xb8\xd0\xb9"},
        {"sr", "\xd0\xa1\xd1\x80\xd0\xbf\xd1\x81\xd0\xba\xd0\xb8"}, {"af", "Afrikaans"}, {"yua", "Maaya t'aan"}, {"pdc", "Deitsch"}};
    return all;
}
bool load(const std::string& dir, const std::string& code) {
    words.clear(); cur = code.empty() ? "en" : code;
    if (cur == "en") return true;
    std::ifstream f(fs::u8path(dir) / fs::u8path(cur + ".txt"), std::ios::binary);
    if (!f) { cur = "en"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find(" = ");
        if (eq == std::string::npos) continue;
        std::string v = line.substr(eq + 3);
        for (size_t i = 0; (i = v.find("\\n", i)) != std::string::npos;) v.replace(i, 2, "\n");
        words[line.substr(0, eq)] = v;
    }
    return true;
}
const std::string& current() { return cur; }
bool needsCJK() { return cur == "ja" || cur == "zh" || cur == "ko"; }
}

const char* T(const char* english) {
    if (Lang::words.empty()) return english;
    const auto it = Lang::words.find(english);
    return it == Lang::words.end() ? english : it->second.c_str();
}
