// Mosaic Mask - the voice: your microphone, changed live, out to VB-Audio Virtual Cable (the call picks "CABLE Output"
// as its microphone) and, if you like, your headphones. voice.py in C++, on miniaudio (public domain / MIT-0):
//   female    your voice taken to a woman's pitch (about 210 Hz), the formants up with it
//   male      to a man's pitch (about 115 Hz), the formants down with it
//   disguise  pitch and formants drifting slowly, at random, round your own: nobody in particular
//   off       your own voice (still through the cable, so the call's microphone doesn't change)
// A phase vocoder on one continuous stream (93 ms FFT, 23 ms hop): each bin's true frequency from its phase advance,
// the spectrum split into its envelope (cepstral, the formants) and fine structure (the harmonics), the harmonics
// moved by the pitch ratio, the envelope by the formant ratio, the phases carried on. Your own pitch is learned as you
// talk (autocorrelation, the median of the last seconds), so female and male aim at a pitch, not a fixed step.
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Voice {

enum Preset { OFF, FEMALE, MALE, DISGUISE, PRESET_COUNT };
const char* presetName(int p);
const int SAMPLE_RATE = 24000;

std::vector<std::string> devices(bool input);   // names, as the system gives them
// offline: a recording (mono, SAMPLE_RATE) through a voice, as the live stream would change it (tests)
std::vector<float> render(const std::vector<float>& in, int preset);

class Changer {
public:
    Changer();
    ~Changer();
    // mic: part of its name ("" the default); cable: the cable's playback side; monitor: also play here ("" none)
    bool start(const std::string& mic, const std::string& cable, const std::string& monitor, std::string& err);
    void stop();
    std::atomic<int> preset{OFF};
    bool running() const;
    bool toCall() const;                         // the virtual cable was found
    float yourPitch() const;                     // Hz (0: not learned yet)
    std::string why;                             // what's missing
    std::function<void(const float*, int)> tap;  // every block of the voice that goes out (recordings)
    struct Impl;                                 // (the audio threads' side)
private:
    std::unique_ptr<Impl> d_;
};

}
