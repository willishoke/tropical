// Audition-only host wrapper around the unmodified Mutable Instruments Clouds
// reverb header. The reference source tree is supplied with --clouds-root by
// run_reference.py and is not part of the Tropical product build.

#include "clouds/dsp/frame.h"
#include "clouds/dsp/fx/reverb.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv)
{
  if (argc != 8)
  {
    std::cerr
      << "usage: clouds_reference INPUT.f32 OUTPUT.f32 "
         "TIME DIFFUSION LP INPUT_GAIN AMOUNT\n";
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
  if (!input) return 3;
  const std::streamsize bytes = input.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0)
    return 4;
  input.seekg(0);
  std::vector<float> mono(
    static_cast<std::size_t>(bytes) / sizeof(float), 0.0f);
  input.read(
    reinterpret_cast<char *>(mono.data()),
    static_cast<std::streamsize>(mono.size() * sizeof(float)));
  if (!input) return 5;

  std::vector<clouds::FloatFrame> frames(mono.size());
  for (std::size_t i = 0; i < mono.size(); ++i)
    frames[i] = {mono[i], mono[i]};

  // Static storage reproduces firmware zero-initialization for the reverb's
  // decay states as well as its 12-bit, 16k-sample delay memory.
  static uint16_t memory[16384];
  static clouds::Reverb reverb;
  reverb.Init(memory);
  reverb.set_time(std::strtof(argv[3], nullptr));
  reverb.set_diffusion(std::strtof(argv[4], nullptr));
  reverb.set_lp(std::strtof(argv[5], nullptr));
  reverb.set_input_gain(std::strtof(argv[6], nullptr));
  reverb.set_amount(std::strtof(argv[7], nullptr));
  reverb.Process(frames.data(), frames.size());

  std::ofstream output(argv[2], std::ios::binary);
  if (!output) return 6;
  output.write(
    reinterpret_cast<const char *>(frames.data()),
    static_cast<std::streamsize>(frames.size() * sizeof(clouds::FloatFrame)));
  return output ? 0 : 7;
}
