#include "Graph.hpp"

#include "../lib/rtaudio/RtAudio.h"

#include <memory>
#include <stdexcept>
#include <string>
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

    bool add_vco(const std::string & module_name, int frequency_hz)
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

  private:
    Graph graph_;
};

class PyVCO
{
  public:
    PyVCO(PythonGraph & graph, std::string name, int frequency_hz) : name_(std::move(name))
    {
      if (!graph.add_vco(name_, frequency_hz))
      {
        throw std::invalid_argument("Failed to create VCO '" + name_ + "' (duplicate name or invalid graph).");
      }
    }

    const std::string & name() const
    {
      return name_;
    }

  private:
    std::string name_;
};

class PyMUX
{
  public:
    PyMUX(PythonGraph & graph, std::string name) : name_(std::move(name))
    {
      if (!graph.add_mux(name_))
      {
        throw std::invalid_argument("Failed to create MUX '" + name_ + "' (duplicate name or invalid graph).");
      }
    }

    const std::string & name() const
    {
      return name_;
    }

  private:
    std::string name_;
};

class PyVCA
{
  public:
    PyVCA(PythonGraph & graph, std::string name) : name_(std::move(name))
    {
      if (!graph.add_vca(name_))
      {
        throw std::invalid_argument("Failed to create VCA '" + name_ + "' (duplicate name or invalid graph).");
      }
    }

    const std::string & name() const
    {
      return name_;
    }

  private:
    std::string name_;
};

class PyENV
{
  public:
    PyENV(PythonGraph & graph, std::string name, double rise_ms, double fall_ms) : name_(std::move(name))
    {
      if (!graph.add_env(name_, rise_ms, fall_ms))
      {
        throw std::invalid_argument("Failed to create ENV '" + name_ + "' (duplicate name or invalid graph).");
      }
    }

    const std::string & name() const
    {
      return name_;
    }

  private:
    std::string name_;
};

class PyDELAY
{
  public:
    PyDELAY(PythonGraph & graph, std::string name, double buffer_size_samples) : name_(std::move(name))
    {
      if (!graph.add_delay(name_, buffer_size_samples))
      {
        throw std::invalid_argument("Failed to create DELAY '" + name_ + "' (duplicate name or invalid graph).");
      }
    }

    const std::string & name() const
    {
      return name_;
    }

  private:
    std::string name_;
};

class PyCONST
{
  public:
    PyCONST(PythonGraph & graph, std::string name, double value) : name_(std::move(name))
    {
      if (!graph.add_const(name_, value))
      {
        throw std::invalid_argument("Failed to create CONST '" + name_ + "' (duplicate name or invalid graph).");
      }
    }

    const std::string & name() const
    {
      return name_;
    }

  private:
    std::string name_;
};

class PythonDAC
{
  public:
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

  py::class_<PyVCO>(m, "VCO")
    .def(py::init<PythonGraph &, std::string, int>(), py::arg("graph"), py::arg("name"), py::arg("frequency_hz"))
    .def_property_readonly("name", &PyVCO::name);

  py::class_<PyMUX>(m, "MUX")
    .def(py::init<PythonGraph &, std::string>(), py::arg("graph"), py::arg("name"))
    .def_property_readonly("name", &PyMUX::name);

  py::class_<PyVCA>(m, "VCA")
    .def(py::init<PythonGraph &, std::string>(), py::arg("graph"), py::arg("name"))
    .def_property_readonly("name", &PyVCA::name);

  py::class_<PyENV>(m, "ENV")
    .def(py::init<PythonGraph &, std::string, double, double>(), py::arg("graph"), py::arg("name"), py::arg("rise_ms"), py::arg("fall_ms"))
    .def_property_readonly("name", &PyENV::name);

  py::class_<PyDELAY>(m, "DELAY")
    .def(py::init<PythonGraph &, std::string, double>(), py::arg("graph"), py::arg("name"), py::arg("buffer_size_samples"))
    .def_property_readonly("name", &PyDELAY::name);

  py::class_<PyCONST>(m, "CONST")
    .def(py::init<PythonGraph &, std::string, double>(), py::arg("graph"), py::arg("name"), py::arg("value"))
    .def_property_readonly("name", &PyCONST::name);

  py::class_<PythonDAC>(m, "DAC")
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

  py::enum_<MUX::Ins>(m, "MUXIn")
    .value("IN1", MUX::Ins::IN1)
    .value("IN2", MUX::Ins::IN2)
    .value("CTRL", MUX::Ins::CTRL)
    .value("IN_COUNT", MUX::Ins::IN_COUNT);

  py::enum_<MUX::Outs>(m, "MUXOut")
    .value("OUT", MUX::Outs::OUT)
    .value("OUT_COUNT", MUX::Outs::OUT_COUNT);

  py::enum_<VCA::Ins>(m, "VCAIn")
    .value("IN1", VCA::Ins::IN1)
    .value("IN2", VCA::Ins::IN2)
    .value("IN_COUNT", VCA::Ins::IN_COUNT);

  py::enum_<VCA::Outs>(m, "VCAOut")
    .value("OUT", VCA::Outs::OUT)
    .value("OUT_COUNT", VCA::Outs::OUT_COUNT);

  py::enum_<ENV::Ins>(m, "ENVIn")
    .value("TRIG", ENV::Ins::TRIG)
    .value("RISE", ENV::Ins::RISE)
    .value("FALL", ENV::Ins::FALL)
    .value("IN_COUNT", ENV::Ins::IN_COUNT);

  py::enum_<ENV::Outs>(m, "ENVOut")
    .value("OUT", ENV::Outs::OUT)
    .value("OUT_COUNT", ENV::Outs::OUT_COUNT);

  py::enum_<DELAY::Ins>(m, "DELAYIn")
    .value("IN", DELAY::Ins::IN)
    .value("IN_COUNT", DELAY::Ins::IN_COUNT);

  py::enum_<DELAY::Outs>(m, "DELAYOut")
    .value("OUT", DELAY::Outs::OUT)
    .value("OUT_COUNT", DELAY::Outs::OUT_COUNT);

  py::enum_<CONST::Outs>(m, "CONSTOut")
    .value("OUT", CONST::Outs::OUT)
    .value("OUT_COUNT", CONST::Outs::OUT_COUNT);
}
