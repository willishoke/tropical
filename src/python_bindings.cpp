#include "Graph.hpp"

#include "../lib/rtaudio/RtAudio.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PythonGraph
{
  public:
    explicit PythonGraph(unsigned int buffer_length) : graph_(buffer_length) {}

    bool destroy_module(const std::string & module_name)
    {
      return graph_.remove_module(module_name);
    }

    bool connect(
      const std::string & from_module,
      unsigned int from_output_id,
      const std::string & to_module,
      unsigned int to_input_id)
    {
      return graph_.connect(from_module, from_output_id, to_module, to_input_id);
    }

    bool disconnect(
      const std::string & from_module,
      unsigned int from_output_id,
      const std::string & to_module,
      unsigned int to_input_id)
    {
      return graph_.remove_connection(from_module, from_output_id, to_module, to_input_id);
    }

    bool add_output(const std::string & module_name, unsigned int output_id)
    {
      return graph_.addOutput(std::make_pair(module_name, output_id));
    }

    void process()
    {
      graph_.process();
    }

    std::vector<double> output_buffer() const
    {
      return graph_.outputBuffer;
    }

    Graph & graph()
    {
      return graph_;
    }

    std::string next_name(const std::string & prefix)
    {
      const auto id = ++name_counters_[prefix];
      return prefix + std::to_string(id);
    }

    bool add_vco(const std::string & module_name, double frequency_hz)
    {
      return graph_.addModule(module_name, std::make_unique<VCO>(frequency_hz));
    }

    bool add_mux(const std::string & module_name)
    {
      return graph_.addModule(module_name, std::make_unique<MUX>());
    }

    bool add_vca(const std::string & module_name)
    {
      return graph_.addModule(module_name, std::make_unique<VCA>());
    }

    bool add_env(const std::string & module_name, double rise_ms, double fall_ms)
    {
      return graph_.addModule(module_name, std::make_unique<ENV>(rise_ms, fall_ms));
    }

    bool add_delay(const std::string & module_name, double buffer_size_samples)
    {
      return graph_.addModule(module_name, std::make_unique<DELAY>(buffer_size_samples));
    }

    bool add_const(const std::string & module_name, double value)
    {
      return graph_.addModule(module_name, std::make_unique<CONST>(value));
    }

    bool add_lowpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<LOWPASS>(freq_hz, res));
    }

    bool add_highpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<HIGHPASS>(freq_hz, res));
    }

    bool add_bandpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<BANDPASS>(freq_hz, res));
    }

    bool add_notch(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<NOTCH>(freq_hz, res));
    }

    bool add_allpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<ALLPASS>(freq_hz, res));
    }

  private:
    Graph graph_;
    std::unordered_map<std::string, uint64_t> name_counters_;
};

static PythonGraph & default_graph()
{
  static PythonGraph graph(1024);
  return graph;
}

struct OutputPort
{
  PythonGraph * graph;
  std::string module_name;
  unsigned int output_id;
};

struct InputPort
{
  PythonGraph * graph;
  std::string module_name;
  unsigned int input_id;
};

static void validate_same_graph(const OutputPort & out, const InputPort & in)
{
  if (out.graph != in.graph)
  {
    throw std::invalid_argument("Ports belong to different graphs.");
  }
}

static bool connect_ports(const OutputPort & out, const InputPort & in)
{
  validate_same_graph(out, in);
  return out.graph->connect(out.module_name, out.output_id, in.module_name, in.input_id);
}

static bool disconnect_ports(const OutputPort & out, const InputPort & in)
{
  validate_same_graph(out, in);
  return out.graph->disconnect(out.module_name, out.output_id, in.module_name, in.input_id);
}

static bool add_output_port(const OutputPort & out)
{
  return out.graph->add_output(out.module_name, out.output_id);
}

static std::vector<OutputPort> incoming_ports(const InputPort & in)
{
  std::vector<OutputPort> results;
  const auto sources = in.graph->graph().incoming_connections(in.module_name, in.input_id);
  results.reserve(sources.size());
  for (const auto & source : sources)
  {
    results.push_back(OutputPort{in.graph, source.first, source.second});
  }
  return results;
}

class PyVCO
{
  public:
    explicit PyVCO(double frequency_hz)
      : graph_(&default_graph()), name_(graph_->next_name("vco"))
    {
      if (!graph_->add_vco(name_, frequency_hz))
      {
        throw std::invalid_argument("Failed to create VCO '" + name_ + "'.");
      }
    }

    PyVCO(PythonGraph & graph, std::string name, double frequency_hz)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_vco(name_, frequency_hz))
      {
        throw std::invalid_argument("Failed to create VCO '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    OutputPort saw() const { return OutputPort{graph_, name_, VCO::SAW}; }
    OutputPort tri() const { return OutputPort{graph_, name_, VCO::TRI}; }
    OutputPort sin() const { return OutputPort{graph_, name_, VCO::SIN}; }
    OutputPort sqr() const { return OutputPort{graph_, name_, VCO::SQR}; }

    InputPort fm() const { return InputPort{graph_, name_, VCO::FM}; }
    InputPort fm_index() const { return InputPort{graph_, name_, VCO::FM_INDEX}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyMUX
{
  public:
    PyMUX() : graph_(&default_graph()), name_(graph_->next_name("mux"))
    {
      if (!graph_->add_mux(name_))
      {
        throw std::invalid_argument("Failed to create MUX '" + name_ + "'.");
      }
    }

    PyMUX(PythonGraph & graph, std::string name) : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_mux(name_))
      {
        throw std::invalid_argument("Failed to create MUX '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort in1() const { return InputPort{graph_, name_, MUX::IN1}; }
    InputPort in2() const { return InputPort{graph_, name_, MUX::IN2}; }
    InputPort ctrl() const { return InputPort{graph_, name_, MUX::CTRL}; }
    OutputPort out() const { return OutputPort{graph_, name_, MUX::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyVCA
{
  public:
    PyVCA() : graph_(&default_graph()), name_(graph_->next_name("vca"))
    {
      if (!graph_->add_vca(name_))
      {
        throw std::invalid_argument("Failed to create VCA '" + name_ + "'.");
      }
    }

    PyVCA(PythonGraph & graph, std::string name) : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_vca(name_))
      {
        throw std::invalid_argument("Failed to create VCA '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort in1() const { return InputPort{graph_, name_, VCA::IN1}; }
    InputPort in2() const { return InputPort{graph_, name_, VCA::IN2}; }
    OutputPort out() const { return OutputPort{graph_, name_, VCA::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyENV
{
  public:
    PyENV(double rise_ms, double fall_ms) : graph_(&default_graph()), name_(graph_->next_name("env"))
    {
      if (!graph_->add_env(name_, rise_ms, fall_ms))
      {
        throw std::invalid_argument("Failed to create ENV '" + name_ + "'.");
      }
    }

    PyENV(PythonGraph & graph, std::string name, double rise_ms, double fall_ms)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_env(name_, rise_ms, fall_ms))
      {
        throw std::invalid_argument("Failed to create ENV '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort trig() const { return InputPort{graph_, name_, ENV::TRIG}; }
    InputPort rise() const { return InputPort{graph_, name_, ENV::RISE}; }
    InputPort fall() const { return InputPort{graph_, name_, ENV::FALL}; }
    OutputPort out() const { return OutputPort{graph_, name_, ENV::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyDELAY
{
  public:
    explicit PyDELAY(double buffer_size_samples)
      : graph_(&default_graph()), name_(graph_->next_name("delay"))
    {
      if (!graph_->add_delay(name_, buffer_size_samples))
      {
        throw std::invalid_argument("Failed to create DELAY '" + name_ + "'.");
      }
    }

    PyDELAY(PythonGraph & graph, std::string name, double buffer_size_samples)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_delay(name_, buffer_size_samples))
      {
        throw std::invalid_argument("Failed to create DELAY '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort in() const { return InputPort{graph_, name_, DELAY::IN}; }
    OutputPort out() const { return OutputPort{graph_, name_, DELAY::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyCONST
{
  public:
    explicit PyCONST(double value) : graph_(&default_graph()), name_(graph_->next_name("const"))
    {
      if (!graph_->add_const(name_, value))
      {
        throw std::invalid_argument("Failed to create CONST '" + name_ + "'.");
      }
    }

    PyCONST(PythonGraph & graph, std::string name, double value)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_const(name_, value))
      {
        throw std::invalid_argument("Failed to create CONST '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    OutputPort out() const { return OutputPort{graph_, name_, CONST::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyLOWPASS
{
  public:
    PyLOWPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("lowpass"))
    {
      if (!graph_->add_lowpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create LOWPASS '" + name_ + "'.");
      }
    }

    PyLOWPASS(PythonGraph & graph, std::string name, double freq_hz, double res = 0.707)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_lowpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create LOWPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, LOWPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, LOWPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, LOWPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, LOWPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyHIGHPASS
{
  public:
    PyHIGHPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("highpass"))
    {
      if (!graph_->add_highpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create HIGHPASS '" + name_ + "'.");
      }
    }

    PyHIGHPASS(PythonGraph & graph, std::string name, double freq_hz, double res = 0.707)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_highpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create HIGHPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, HIGHPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, HIGHPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, HIGHPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, HIGHPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyBANDPASS
{
  public:
    PyBANDPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("bandpass"))
    {
      if (!graph_->add_bandpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create BANDPASS '" + name_ + "'.");
      }
    }

    PyBANDPASS(PythonGraph & graph, std::string name, double freq_hz, double res = 0.707)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_bandpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create BANDPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, BANDPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, BANDPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, BANDPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, BANDPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyNOTCH
{
  public:
    PyNOTCH(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("notch"))
    {
      if (!graph_->add_notch(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create NOTCH '" + name_ + "'.");
      }
    }

    PyNOTCH(PythonGraph & graph, std::string name, double freq_hz, double res = 0.707)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_notch(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create NOTCH '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, NOTCH::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, NOTCH::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, NOTCH::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, NOTCH::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyALLPASS
{
  public:
    PyALLPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("allpass"))
    {
      if (!graph_->add_allpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create ALLPASS '" + name_ + "'.");
      }
    }

    PyALLPASS(PythonGraph & graph, std::string name, double freq_hz, double res = 0.707)
      : graph_(&graph), name_(std::move(name))
    {
      if (!graph_->add_allpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create ALLPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, ALLPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, ALLPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, ALLPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, ALLPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PythonDAC
{
  public:
    explicit PythonDAC(unsigned int sample_rate = 44100, unsigned int channels = 2)
      : PythonDAC(default_graph(), sample_rate, channels)
    {
    }

    explicit PythonDAC(PythonGraph & graph, unsigned int sample_rate = 44100, unsigned int channels = 2)
      : graph_(graph), sample_rate_(sample_rate), channels_(channels), running_(false)
    {
    }

    ~PythonDAC()
    {
      stop();
    }

    void start()
    {
      if (running_)
      {
        return;
      }

      if (audio_.getDeviceCount() < 1)
      {
        throw std::runtime_error("No audio output devices found.");
      }

      RtAudio::StreamParameters out_params;
      out_params.deviceId = audio_.getDefaultOutputDevice();
      out_params.nChannels = channels_;
      out_params.firstChannel = 0;

      unsigned int buffer_frames = graph_.graph().getBufferLength();

      audio_.openStream(
        &out_params,
        nullptr,
        RTAUDIO_FLOAT64,
        sample_rate_,
        &buffer_frames,
        &PythonDAC::fill_buffer,
        this);

      audio_.startStream();
      running_ = true;
    }

    void stop()
    {
      if (!audio_.isStreamOpen())
      {
        running_ = false;
        return;
      }

      if (audio_.isStreamRunning())
      {
        audio_.stopStream();
      }
      audio_.closeStream();
      running_ = false;
    }

    bool is_running() const
    {
      return running_;
    }

  private:
    static int fill_buffer(
      void * output_buffer,
      void *,
      unsigned int n_buffer_frames,
      double,
      RtAudioStreamStatus,
      void * user_data)
    {
      auto * self = static_cast<PythonDAC *>(user_data);
      auto * out = static_cast<double *>(output_buffer);

      self->graph_.graph().process();
      const auto & source = self->graph_.graph().outputBuffer;

      for (unsigned int i = 0; i < n_buffer_frames; ++i)
      {
        const double sample = i < source.size() ? source[i] : 0.0;
        for (unsigned int c = 0; c < self->channels_; ++c)
        {
          *out++ = sample;
        }
      }

      return 0;
    }

    PythonGraph & graph_;
    RtAudio audio_;
    unsigned int sample_rate_;
    unsigned int channels_;
    bool running_;
};

PYBIND11_MODULE(egress, m)
{
  m.doc() = "Python frontend for egress graph/modules and DAC control.";

  py::class_<PythonGraph>(m, "Graph")
    .def(py::init<unsigned int>(), py::arg("buffer_length"))
    .def("destroy_module", &PythonGraph::destroy_module, py::arg("module_name"))
    .def(
      "connect",
      &PythonGraph::connect,
      py::arg("from_module"),
      py::arg("from_output_id"),
      py::arg("to_module"),
      py::arg("to_input_id"))
    .def(
      "disconnect",
      &PythonGraph::disconnect,
      py::arg("from_module"),
      py::arg("from_output_id"),
      py::arg("to_module"),
      py::arg("to_input_id"))
    .def("add_output", &PythonGraph::add_output, py::arg("module_name"), py::arg("output_id"))
    .def("process", &PythonGraph::process)
    .def("output_buffer", &PythonGraph::output_buffer);

  m.def("graph", []() -> PythonGraph & { return default_graph(); }, py::return_value_policy::reference);

  py::class_<OutputPort>(m, "OutputPort")
    .def_property_readonly("module_name", [](const OutputPort & p) { return p.module_name; })
    .def_property_readonly("output_id", [](const OutputPort & p) { return p.output_id; });

  py::class_<InputPort>(m, "InputPort")
    .def_property_readonly("module_name", [](const InputPort & p) { return p.module_name; })
    .def_property_readonly("input_id", [](const InputPort & p) { return p.input_id; });

  m.def("connect", &connect_ports, py::arg("out"), py::arg("in"));
  m.def("disconnect", &disconnect_ports, py::arg("out"), py::arg("in"));
  m.def("add_output", &add_output_port, py::arg("out"));
  m.def("incoming", &incoming_ports, py::arg("in"));

  py::class_<PyVCO>(m, "VCO")
    .def(py::init<double>(), py::arg("frequency_hz"))
    .def(py::init<PythonGraph &, std::string, double>(), py::arg("graph"), py::arg("name"), py::arg("frequency_hz"))
    .def_property_readonly("name", &PyVCO::name)
    .def_property_readonly("saw", &PyVCO::saw)
    .def_property_readonly("tri", &PyVCO::tri)
    .def_property_readonly("sin", &PyVCO::sin)
    .def_property_readonly("sqr", &PyVCO::sqr)
    .def_property_readonly("fm", &PyVCO::fm)
    .def_property_readonly("fm_index", &PyVCO::fm_index);

  py::class_<PyMUX>(m, "MUX")
    .def(py::init<>())
    .def(py::init<PythonGraph &, std::string>(), py::arg("graph"), py::arg("name"))
    .def_property_readonly("name", &PyMUX::name)
    .def_property_readonly("input1", &PyMUX::in1)
    .def_property_readonly("input2", &PyMUX::in2)
    .def_property_readonly("control", &PyMUX::ctrl)
    .def_property_readonly("output", &PyMUX::out);

  py::class_<PyVCA>(m, "VCA")
    .def(py::init<>())
    .def(py::init<PythonGraph &, std::string>(), py::arg("graph"), py::arg("name"))
    .def_property_readonly("name", &PyVCA::name)
    .def_property_readonly("input1", &PyVCA::in1)
    .def_property_readonly("input2", &PyVCA::in2)
    .def_property_readonly("output", &PyVCA::out);

  py::class_<PyENV>(m, "ENV")
    .def(py::init<double, double>(), py::arg("rise_ms"), py::arg("fall_ms"))
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("rise_ms"), py::arg("fall_ms"))
    .def_property_readonly("name", &PyENV::name)
    .def_property_readonly("trig", &PyENV::trig)
    .def_property_readonly("rise", &PyENV::rise)
    .def_property_readonly("fall", &PyENV::fall)
    .def_property_readonly("output", &PyENV::out);

  py::class_<PyDELAY>(m, "DELAY")
    .def(py::init<double>(), py::arg("buffer_size_samples"))
    .def(py::init<PythonGraph &, std::string, double>(), py::arg("graph"), py::arg("name"), py::arg("buffer_size_samples"))
    .def_property_readonly("name", &PyDELAY::name)
    .def_property_readonly("input", &PyDELAY::in)
    .def_property_readonly("output", &PyDELAY::out);

  py::class_<PyCONST>(m, "CONST")
    .def(py::init<double>(), py::arg("value"))
    .def(py::init<PythonGraph &, std::string, double>(), py::arg("graph"), py::arg("name"), py::arg("value"))
    .def_property_readonly("name", &PyCONST::name)
    .def_property_readonly("output", &PyCONST::out);

  py::class_<PyLOWPASS>(m, "LOWPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyLOWPASS::name)
    .def_property_readonly("input", &PyLOWPASS::in)
    .def_property_readonly("freq", &PyLOWPASS::freq)
    .def_property_readonly("res", &PyLOWPASS::res)
    .def_property_readonly("output", &PyLOWPASS::out);

  py::class_<PyHIGHPASS>(m, "HIGHPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyHIGHPASS::name)
    .def_property_readonly("input", &PyHIGHPASS::in)
    .def_property_readonly("freq", &PyHIGHPASS::freq)
    .def_property_readonly("res", &PyHIGHPASS::res)
    .def_property_readonly("output", &PyHIGHPASS::out);

  py::class_<PyBANDPASS>(m, "BANDPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyBANDPASS::name)
    .def_property_readonly("input", &PyBANDPASS::in)
    .def_property_readonly("freq", &PyBANDPASS::freq)
    .def_property_readonly("res", &PyBANDPASS::res)
    .def_property_readonly("output", &PyBANDPASS::out);

  py::class_<PyNOTCH>(m, "NOTCH")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyNOTCH::name)
    .def_property_readonly("input", &PyNOTCH::in)
    .def_property_readonly("freq", &PyNOTCH::freq)
    .def_property_readonly("res", &PyNOTCH::res)
    .def_property_readonly("output", &PyNOTCH::out);

  py::class_<PyALLPASS>(m, "ALLPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyALLPASS::name)
    .def_property_readonly("input", &PyALLPASS::in)
    .def_property_readonly("freq", &PyALLPASS::freq)
    .def_property_readonly("res", &PyALLPASS::res)
    .def_property_readonly("output", &PyALLPASS::out);

  py::class_<PythonDAC>(m, "DAC")
    .def(py::init<unsigned int, unsigned int>(), py::arg("sample_rate") = 44100, py::arg("channels") = 2)
    .def(py::init<PythonGraph &, unsigned int, unsigned int>(), py::arg("graph"), py::arg("sample_rate") = 44100, py::arg("channels") = 2)
    .def("start", &PythonDAC::start)
    .def("stop", &PythonDAC::stop)
    .def("is_running", &PythonDAC::is_running);

  py::enum_<VCO::Ins>(m, "VCOIn")
    .value("FM", VCO::Ins::FM)
    .value("FM_INDEX", VCO::Ins::FM_INDEX)
    .value("IN_COUNT", VCO::Ins::IN_COUNT);

  py::enum_<VCO::Outs>(m, "VCOOut")
    .value("SAW", VCO::Outs::SAW)
    .value("TRI", VCO::Outs::TRI)
    .value("SIN", VCO::Outs::SIN)
    .value("SQR", VCO::Outs::SQR)
    .value("OUT_COUNT", VCO::Outs::OUT_COUNT);
}
