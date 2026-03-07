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
  PythonGraph * graph = nullptr;
  std::string module_name;
  unsigned int output_id = 0;
};

struct InputPort
{
  PythonGraph * graph = nullptr;
  std::string module_name;
  unsigned int input_id = 0;
};

struct SignalExpr
{
  PythonGraph * graph = nullptr;
  Graph::ExprSpecPtr spec;
};

static PythonGraph * merge_graphs(PythonGraph * lhs, PythonGraph * rhs)
{
  if (lhs != nullptr && rhs != nullptr && lhs != rhs)
  {
    throw std::invalid_argument("Expression operands belong to different graphs.");
  }
  return lhs != nullptr ? lhs : rhs;
}

static SignalExpr make_signal_expr(PythonGraph * graph, Graph::ExprSpecPtr spec)
{
  SignalExpr expr;
  expr.graph = graph;
  expr.spec = std::move(spec);
  return expr;
}

static SignalExpr make_literal_expr(double value)
{
  return make_signal_expr(nullptr, Graph::literal_expr(value));
}

static SignalExpr make_output_expr(const OutputPort & out)
{
  return make_signal_expr(out.graph, Graph::ref_expr(out.module_name, out.output_id));
}

static SignalExpr current_input_expr(const InputPort & input)
{
  auto spec = input.graph->graph().get_input_expr(input.module_name, input.input_id);
  if (!spec)
  {
    spec = Graph::literal_expr(0.0);
  }
  return make_signal_expr(input.graph, std::move(spec));
}

static SignalExpr coerce_expr(const py::handle & value)
{
  if (py::isinstance<SignalExpr>(value))
  {
    return value.cast<SignalExpr>();
  }

  if (py::isinstance<OutputPort>(value))
  {
    return make_output_expr(value.cast<OutputPort>());
  }

  if (py::isinstance<InputPort>(value))
  {
    return current_input_expr(value.cast<InputPort>());
  }

  if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value))
  {
    return make_literal_expr(value.cast<double>());
  }

  throw std::invalid_argument("Expected an output port, input expression, arithmetic expression, or numeric literal.");
}

static SignalExpr make_unary_expr(Graph::ExprKind kind, const SignalExpr & operand)
{
  return make_signal_expr(operand.graph, Graph::unary_expr(kind, operand.spec));
}

static SignalExpr make_binary_expr(Graph::ExprKind kind, const SignalExpr & lhs, const SignalExpr & rhs)
{
  return make_signal_expr(merge_graphs(lhs.graph, rhs.graph), Graph::binary_expr(kind, lhs.spec, rhs.spec));
}

static SignalExpr add_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(Graph::ExprKind::Add, lhs, coerce_expr(rhs));
}

static SignalExpr sub_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(Graph::ExprKind::Sub, lhs, coerce_expr(rhs));
}

static SignalExpr mul_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(Graph::ExprKind::Mul, lhs, coerce_expr(rhs));
}

static SignalExpr div_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(Graph::ExprKind::Div, lhs, coerce_expr(rhs));
}

static void assign_input_expr(const InputPort & input, Graph::ExprSpecPtr expr)
{
  if (!input.graph->graph().set_input_expr(input.module_name, input.input_id, std::move(expr)))
  {
    throw std::invalid_argument("Failed to assign expression to input.");
  }
}

static void assign_input(const InputPort & input, const py::handle & value)
{
  if (value.is_none())
  {
    assign_input_expr(input, nullptr);
    return;
  }

  const SignalExpr expr = coerce_expr(value);
  if (expr.graph != nullptr && expr.graph != input.graph)
  {
    throw std::invalid_argument("Assigned expression belongs to a different graph.");
  }

  assign_input_expr(input, expr.spec);
}

static bool connect_ports(const OutputPort & out, const InputPort & in)
{
  if (out.graph != in.graph)
  {
    throw std::invalid_argument("Ports belong to different graphs.");
  }
  return out.graph->connect(out.module_name, out.output_id, in.module_name, in.input_id);
}

static bool disconnect_ports(const OutputPort & out, const InputPort & in)
{
  if (out.graph != in.graph)
  {
    throw std::invalid_argument("Ports belong to different graphs.");
  }
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

static py::object input_iadd(const InputPort & input, const py::handle & rhs)
{
  return py::cast(add_expr(current_input_expr(input), rhs));
}

static py::object input_isub(const InputPort & input, const py::handle & rhs)
{
  return py::cast(sub_expr(current_input_expr(input), rhs));
}

static py::object input_imul(const InputPort & input, const py::handle & rhs)
{
  return py::cast(mul_expr(current_input_expr(input), rhs));
}

static py::object input_idiv(const InputPort & input, const py::handle & rhs)
{
  return py::cast(div_expr(current_input_expr(input), rhs));
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

  private:
    explicit PythonDAC(PythonGraph & graph, unsigned int sample_rate, unsigned int channels)
      : graph_(graph), sample_rate_(sample_rate), channels_(channels), running_(false)
    {
    }

  public:
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

  py::class_<SignalExpr>(m, "SignalExpr")
    .def("__add__", [](const SignalExpr & lhs, const py::object & rhs) { return add_expr(lhs, rhs); })
    .def("__radd__", [](const SignalExpr & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__sub__", [](const SignalExpr & lhs, const py::object & rhs) { return sub_expr(lhs, rhs); })
    .def("__rsub__", [](const SignalExpr & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__mul__", [](const SignalExpr & lhs, const py::object & rhs) { return mul_expr(lhs, rhs); })
    .def("__rmul__", [](const SignalExpr & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__truediv__", [](const SignalExpr & lhs, const py::object & rhs) { return div_expr(lhs, rhs); })
    .def("__rtruediv__", [](const SignalExpr & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__neg__", [](const SignalExpr & expr) { return make_unary_expr(Graph::ExprKind::Neg, expr); });

  py::class_<OutputPort>(m, "OutputPort")
    .def_property_readonly("module_name", [](const OutputPort & p) { return p.module_name; })
    .def_property_readonly("output_id", [](const OutputPort & p) { return p.output_id; })
    .def("__add__", [](const OutputPort & lhs, const py::object & rhs) { return add_expr(make_output_expr(lhs), rhs); })
    .def("__radd__", [](const OutputPort & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__sub__", [](const OutputPort & lhs, const py::object & rhs) { return sub_expr(make_output_expr(lhs), rhs); })
    .def("__rsub__", [](const OutputPort & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__mul__", [](const OutputPort & lhs, const py::object & rhs) { return mul_expr(make_output_expr(lhs), rhs); })
    .def("__rmul__", [](const OutputPort & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__truediv__", [](const OutputPort & lhs, const py::object & rhs) { return div_expr(make_output_expr(lhs), rhs); })
    .def("__rtruediv__", [](const OutputPort & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__neg__", [](const OutputPort & out) { return make_unary_expr(Graph::ExprKind::Neg, make_output_expr(out)); });

  py::class_<InputPort>(m, "InputPort")
    .def_property_readonly("module_name", [](const InputPort & p) { return p.module_name; })
    .def_property_readonly("input_id", [](const InputPort & p) { return p.input_id; })
    .def_property_readonly("expr", [](const InputPort & p) { return current_input_expr(p); })
    .def("assign", [](const InputPort & in, const py::object & value) { assign_input(in, value); }, py::arg("value"))
    .def("__add__", [](const InputPort & lhs, const py::object & rhs) { return add_expr(current_input_expr(lhs), rhs); })
    .def("__radd__", [](const InputPort & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__sub__", [](const InputPort & lhs, const py::object & rhs) { return sub_expr(current_input_expr(lhs), rhs); })
    .def("__rsub__", [](const InputPort & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__mul__", [](const InputPort & lhs, const py::object & rhs) { return mul_expr(current_input_expr(lhs), rhs); })
    .def("__rmul__", [](const InputPort & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__truediv__", [](const InputPort & lhs, const py::object & rhs) { return div_expr(current_input_expr(lhs), rhs); })
    .def("__rtruediv__", [](const InputPort & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__iadd__", [](const InputPort & lhs, const py::object & rhs) { return input_iadd(lhs, rhs); })
    .def("__isub__", [](const InputPort & lhs, const py::object & rhs) { return input_isub(lhs, rhs); })
    .def("__imul__", [](const InputPort & lhs, const py::object & rhs) { return input_imul(lhs, rhs); })
    .def("__itruediv__", [](const InputPort & lhs, const py::object & rhs) { return input_idiv(lhs, rhs); })
    .def("__neg__", [](const InputPort & p) { return make_unary_expr(Graph::ExprKind::Neg, current_input_expr(p)); });

  m.def("connect", &connect_ports, py::arg("out"), py::arg("in"));
  m.def("disconnect", &disconnect_ports, py::arg("out"), py::arg("in"));
  m.def("add_output", &add_output_port, py::arg("out"));
  m.def("incoming", &incoming_ports, py::arg("in"));

  py::class_<PyVCO>(m, "VCO")
    .def(py::init<double>(), py::arg("frequency_hz"))
    .def_property_readonly("name", &PyVCO::name)
    .def_property_readonly("saw", &PyVCO::saw)
    .def_property_readonly("tri", &PyVCO::tri)
    .def_property_readonly("sin", &PyVCO::sin)
    .def_property_readonly("sqr", &PyVCO::sqr)
    .def_property("fm", &PyVCO::fm, [](PyVCO & self, const py::object & value) { assign_input(self.fm(), value); })
    .def_property("fm_index", &PyVCO::fm_index, [](PyVCO & self, const py::object & value) { assign_input(self.fm_index(), value); });

  py::class_<PyMUX>(m, "MUX")
    .def(py::init<>())
    .def_property_readonly("name", &PyMUX::name)
    .def_property("input1", &PyMUX::in1, [](PyMUX & self, const py::object & value) { assign_input(self.in1(), value); })
    .def_property("input2", &PyMUX::in2, [](PyMUX & self, const py::object & value) { assign_input(self.in2(), value); })
    .def_property("control", &PyMUX::ctrl, [](PyMUX & self, const py::object & value) { assign_input(self.ctrl(), value); })
    .def_property_readonly("output", &PyMUX::out);

  py::class_<PyVCA>(m, "VCA")
    .def(py::init<>())
    .def_property_readonly("name", &PyVCA::name)
    .def_property("input1", &PyVCA::in1, [](PyVCA & self, const py::object & value) { assign_input(self.in1(), value); })
    .def_property("input2", &PyVCA::in2, [](PyVCA & self, const py::object & value) { assign_input(self.in2(), value); })
    .def_property_readonly("output", &PyVCA::out);

  py::class_<PyENV>(m, "ENV")
    .def(py::init<double, double>(), py::arg("rise_ms"), py::arg("fall_ms"))
    .def_property_readonly("name", &PyENV::name)
    .def_property("trig", &PyENV::trig, [](PyENV & self, const py::object & value) { assign_input(self.trig(), value); })
    .def_property("rise", &PyENV::rise, [](PyENV & self, const py::object & value) { assign_input(self.rise(), value); })
    .def_property("fall", &PyENV::fall, [](PyENV & self, const py::object & value) { assign_input(self.fall(), value); })
    .def_property_readonly("output", &PyENV::out);

  py::class_<PyDELAY>(m, "DELAY")
    .def(py::init<double>(), py::arg("buffer_size_samples"))
    .def_property_readonly("name", &PyDELAY::name)
    .def_property("input", &PyDELAY::in, [](PyDELAY & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property_readonly("output", &PyDELAY::out);

  py::class_<PyCONST>(m, "CONST")
    .def(py::init<double>(), py::arg("value"))
    .def_property_readonly("name", &PyCONST::name)
    .def_property_readonly("output", &PyCONST::out);

  py::class_<PyLOWPASS>(m, "LOWPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyLOWPASS::name)
    .def_property("input", &PyLOWPASS::in, [](PyLOWPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyLOWPASS::freq, [](PyLOWPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyLOWPASS::res, [](PyLOWPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyLOWPASS::out);

  py::class_<PyHIGHPASS>(m, "HIGHPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyHIGHPASS::name)
    .def_property("input", &PyHIGHPASS::in, [](PyHIGHPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyHIGHPASS::freq, [](PyHIGHPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyHIGHPASS::res, [](PyHIGHPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyHIGHPASS::out);

  py::class_<PyBANDPASS>(m, "BANDPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyBANDPASS::name)
    .def_property("input", &PyBANDPASS::in, [](PyBANDPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyBANDPASS::freq, [](PyBANDPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyBANDPASS::res, [](PyBANDPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyBANDPASS::out);

  py::class_<PyNOTCH>(m, "NOTCH")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyNOTCH::name)
    .def_property("input", &PyNOTCH::in, [](PyNOTCH & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyNOTCH::freq, [](PyNOTCH & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyNOTCH::res, [](PyNOTCH & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyNOTCH::out);

  py::class_<PyALLPASS>(m, "ALLPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyALLPASS::name)
    .def_property("input", &PyALLPASS::in, [](PyALLPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyALLPASS::freq, [](PyALLPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyALLPASS::res, [](PyALLPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyALLPASS::out);

  py::class_<PythonDAC>(m, "DAC")
    .def(py::init<unsigned int, unsigned int>(), py::arg("sample_rate") = 44100, py::arg("channels") = 2)
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
