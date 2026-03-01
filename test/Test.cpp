/* 
 * * * * * * * *
 * E G R E S S *
 * * * * * * * *
 */


#include "../lib/RtAudio.h"
#include "../src/Graph.hpp"
#include <curses.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <utility>


// Callback function automatically invoked when buffer is empty
// Adapted from RtAudio documentation:
// https://www.music.mcgill.ca/~gary/rtaudio/playback.html
int fillBuffer 
( void * outputBuffer, 
  void * inputBuffer, 
  unsigned int nBufferFrames,
  double streamTime, 
  RtAudioStreamStatus status, 
  void * graph 
)
{
  double * buffer = (double *) outputBuffer;
  Graph * r = (Graph *) graph;
  if (status)
    std::cout << "Stream underflow detected!" << std::endl;

  // fill graph mixer buffer and update values
  r->process();

  // write interleaved audio data
  for (auto i = 0; i < nBufferFrames; ++i) 
  {
    for (auto j = 0; j < 2; j++) 
    {
      *buffer++ = r->outputBuffer.at(i);
    }
  }
  return 0;
}

void closeStream(RtAudio & dac)
{
  try 
  {
    dac.stopStream();
  }

  catch (RtAudioError& e) 
  {
    e.printMessage();
  }

  if (dac.isStreamOpen()) 
  {
    dac.closeStream();
  }
}

void SQR_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(440));

  graph.addOutput(std::make_pair("vco1", VCO::SQR));
}

void SIN_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(440));

  graph.addOutput(std::make_pair("vco1", VCO::SIN));
}

void TRI_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(440));

  graph.addOutput(std::make_pair("vco1", VCO::TRI));
}

void SAW_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(440));

  graph.addOutput(std::make_pair("vco1", VCO::SAW));
}

void AM_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(1000));
  graph.addModule("vco2", std::make_unique<VCO>(200));
  graph.addModule("vca1", std::make_unique<VCA>());

  graph.connect("vco1", VCO::SIN, "vca1", VCA::IN1);
  graph.connect("vco2", VCO::SIN, "vca1", VCA::IN2);
  
  graph.addOutput(std::make_pair("vca1", VCA::OUT));
}

void MUX_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(1000));
  graph.addModule("vco2", std::make_unique<VCO>(2000));
  graph.addModule("lfo1", std::make_unique<VCO>(100));
  graph.addModule("mux1", std::make_unique<MUX>());

  graph.connect("vco1", VCO::SIN, "mux1", MUX::IN1);
  graph.connect("vco2", VCO::SIN, "mux1", MUX::IN2);
  graph.connect("lfo1", VCO::SIN, "mux1", MUX::CTRL);
  
  graph.addOutput(std::make_pair("mux1", MUX::OUT));
}

void FM_test(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(1000));
  graph.addModule("vco2", std::make_unique<VCO>(200));
  graph.addModule("c1", std::make_unique<CONST>(3));

  graph.connect("vco2", VCO::SIN, "vco1", VCO::FM);
  graph.connect("c1", CONST::OUT, "vco1", VCO::FM_INDEX);
  
  graph.addOutput(std::make_pair("vco1", VCO::SIN));
}

void ENV_test(Graph & graph)
{
  graph.addModule("env1", std::make_unique<ENV>(1, 5));
  graph.addModule("lfo1", std::make_unique<VCO>(200));

  graph.connect("lfo1", VCO::SQR, "env1", ENV::TRIG);

  graph.addOutput(std::make_pair("env1", ENV::OUT));
}

/*
 * Patch with chaotic behavior
 */
void FM_chaos(Graph & graph)
{
  graph.addModule("vco1", std::make_unique<VCO>(200.1));
  graph.addModule("vco2", std::make_unique<VCO>(300.0));
  graph.addModule("vco3", std::make_unique<VCO>(100.01));
  graph.addModule("vca1", std::make_unique<VCA>());
  graph.addModule("lfo1", std::make_unique<VCO>(.01));
  graph.addModule("lfo2", std::make_unique<VCO>(10));
  graph.addModule("c", std::make_unique<CONST>(1.5));

  graph.connect("vco2", VCO::SIN, "vco1", VCO::FM);
  graph.connect("vco3", VCO::SIN, "vco2", VCO::FM);
  graph.connect("vco1", VCO::SIN, "vco3", VCO::FM);
  graph.connect("lfo1", VCO::SAW, "vco3", VCO::FM);
  graph.connect("vco1", VCO::SIN, "vca1", VCA::IN1);
  graph.connect("lfo1", VCO::SIN, "vca1", VCA::IN2);

  graph.connect("c", CONST::OUT, "vco3", VCO::FM_INDEX);
  graph.connect("c", CONST::OUT, "vco1", VCO::FM_INDEX);
  graph.connect("lfo1", VCO::SIN, "vco2", VCO::FM_INDEX);

  graph.addOutput(std::make_pair("vca1", VCA::OUT));
  graph.addOutput(std::make_pair("vco2", VCO::SIN));
}

int main(int argc, char * argv[])
{
  unsigned int bufferFrames = 2048;  

  Graph graph(bufferFrames);

  //SQR_test(graph);
  //SAW_test(graph);
  //SIN_test(graph);
  //TRI_test(graph);
  //FM_test(graph);
  //AM_test(graph);
  //MUX_test(graph);
  //FM_chaos(graph);
  // Fill single buffer, output to stdout
  ENV_test(graph);
  graph.process();

  return 0;
}
