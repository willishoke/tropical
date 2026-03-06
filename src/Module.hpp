#include <vector>
#include <math.h>

using Signal = double;

class Module
{
  public:
    virtual ~Module() {}

    Module(unsigned int inSize, unsigned int outSize)
    {
      this->inputs.resize(inSize);
      this->outputs.resize(outSize);

    }

    virtual void process() = 0;

    void postprocess()
    {
      for (auto & in : inputs)
      {
        in = 0.0;
      }
     
      // apply thresholding to restrict output to range [-10.0, 10.0] 
      for (auto & out : outputs)
      {
        out = fmin(out, 10.0);
        out = fmax(out, -10.0);
      }
    }
   
    std::string module_name;

  protected:
    std::vector<Signal> inputs;
    std::vector<Signal> outputs;
    unsigned int sampleRate;

  private:
    friend class Graph;
};


// saw core lets us "easily" get 4 output waveforms
// core takes on values in range [0.0, 1.0]
// output in range [-5.0, 5.0]
class VCO : public Module
{ 
  public:
    // invoke base class constructor
    VCO(double freq) : Module(IN_COUNT, OUT_COUNT) 
    {
      frequency = freq;
      core = 0.0;
    }
  
    enum Ins
    {
      FM,
      FM_INDEX,
      IN_COUNT
    };
  
    enum Outs
    {
      SAW,
      TRI,
      SIN,
      SQR,
      OUT_COUNT
    };

    void process() 
    {
      // calculate FM value
      double fm = pow(2, inputs[FM_INDEX] * inputs[FM] / 5.0);

      // apply exponential FM
      double freq = frequency * fm;

      const double dt = fmin(fabs(freq) / 44100.0, 0.5);

      // increment phase and wrap to [0, 1)
      core += freq / 44100.0;
      while (core >= 1.0) core -= 1.0;
      while (core < 0.0) core += 1.0;

      // polyBLEP band-limited saw
      double saw = 2.0 * core - 1.0;
      saw -= poly_blep(core, dt);

      // polyBLEP band-limited square
      double sqr = core < 0.5 ? 1.0 : -1.0;
      sqr += poly_blep(core, dt);
      double half_phase = core + 0.5;
      if (half_phase >= 1.0) half_phase -= 1.0;
      sqr -= poly_blep(half_phase, dt);

      // triangle from integrated band-limited square
      tri_state += sqr * dt * 4.0;
      tri_state = fmin(1.0, fmax(-1.0, tri_state));

      // sine is naturally band-limited; use fast polynomial approximation
      double sine = fast_sin_from_phase(core);

      outputs[SAW] = 5.0 * saw;
      outputs[TRI] = 5.0 * tri_state;
      outputs[SIN] = 5.0 * sine;
      outputs[SQR] = 5.0 * sqr;
      
      // invoke postprocessing routine
      Module::postprocess();

      inputs[FM_INDEX] = 5.0;
    }

  private:
    static double poly_blep(double t, double dt)
    {
      if (dt <= 0.0 || dt >= 1.0)
      {
        return 0.0;
      }

      if (t < dt)
      {
        t /= dt;
        return t + t - t * t - 1.0;
      }

      if (t > 1.0 - dt)
      {
        t = (t - 1.0) / dt;
        return t * t + t + t + 1.0;
      }

      return 0.0;
    }

    static double fast_sin_from_phase(double phase)
    {
      // Map phase [0, 1) -> x in [-pi, pi)
      double x = (phase - 0.5) * 2.0 * M_PI;

      // Fast sine approximation:
      // y = Bx + Cx|x|, then corrective term P(y|y| - y) + y
      constexpr double B = 4.0 / M_PI;
      constexpr double C = -4.0 / (M_PI * M_PI);
      constexpr double P = 0.225;

      const double y = B * x + C * x * fabs(x);
      return P * (y * fabs(y) - y) + y;
    }

    double frequency;
    double core;
    double tri_state = 0.0;
};

class MUX : public Module
{
  public:
    MUX() : Module(IN_COUNT, OUT_COUNT) {}
  
    enum Ins
    {
      IN1,
      IN2,
      CTRL,
      IN_COUNT
    };
  
    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process() 
    {
      // route input depending on polarity of control signal
      outputs[OUT] = inputs[CTRL] > 0.0 ? inputs[IN1] : inputs[IN2];

      // invoke postprocessing routine
      Module::postprocess();
    }
};

// 4-quadrant multiplier
// two inputs, one output

class VCA : public Module
{
  public:
    VCA() : Module(IN_COUNT, OUT_COUNT) {}
 
    enum Ins
    {
      IN1,
      IN2,
      IN_COUNT
    };
  
    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process() 
    {
      // update output value, downscaling to avoid clipping
      outputs[OUT] = inputs[IN1] * inputs[IN2] / 5.0;

      // clean up
      Module::postprocess();
    }
};

class ENV: public Module
{
  public:
    ENV(double rise, double fall) : 
      Module(IN_COUNT, OUT_COUNT)
    {
      this->rise = rise;
      this->fall = fall;
      
      // initialize core value to 0
      this->core = 0.0;

      // module remains idle until positive value appears at TRIG
      this->stage = ENV::Stage::IDLE;
    }

    enum Ins
    {
      TRIG,
      RISE,
      FALL,
      IN_COUNT
    };
  
    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process() 
    {
      if (this->stage == ENV::Stage::IDLE)
      {
        if (this->inputs[TRIG] >= 0.0)
        {
          // update stage tag
          this->stage = ENV::Stage::RISING;

          // need to calculate frequency from wavelength
          double step = 1.0 / (this->rise * 44.1);

          // increment core value
          this->core += step;
        }
      }

      else if (this->stage == ENV::Stage::RISING)
      {
        // need to calculate frequency from wavelength
        double step = 1.0 / (this->rise * 44.1);

        // increment core value
        this->core += step;

        if (this->core >= 1.0)
        {
          this->core = 1.0;
          this->stage = ENV::Stage::FALLING;
        } 
      }

      else if (this->stage == ENV::Stage::FALLING)
      {
        // need to calculate frequency from wavelength
        double step = 1.0 / (this->fall * 44.1);

        // increment core value
        this->core -= step;

        if (this->core <= 0.0)
        {
          this->core = 0.0;
          this->stage = ENV::Stage::IDLE;
        }
      }

      // update output value
      outputs[OUT] = 5.0 * this->core;

      // clean up
      Module::postprocess();
    }

  private:
    enum class Stage
    {
      IDLE,     // Module is waiting for trigger
      RISING,   // Module is in rise phase
      FALLING,  // Module is in fall phase
    } stage;

    double rise; // Value in milliseconds
    double fall;  // Value in milliseconds
    double core;   // Value in range [0.0, 5.0]
};

class BiquadCore
{
  public:
    enum class Type
    {
      LOWPASS,
      HIGHPASS,
      BANDPASS,
      NOTCH,
      ALLPASS
    };

    void set(Type type, double freq_hz, double res, double sample_rate = 44100.0)
    {
      const double nyquist = sample_rate * 0.5 - 1.0;
      const double freq = fmin(fmax(freq_hz, 10.0), nyquist);
      const double q_clamped = fmin(fmax(res, 0.05), 50.0);

      const double omega = 2.0 * M_PI * freq / sample_rate;
      const double sin_omega = sin(omega);
      const double cos_omega = cos(omega);
      const double alpha = sin_omega / (2.0 * q_clamped);

      double b0 = 0.0;
      double b1 = 0.0;
      double b2 = 0.0;
      double a0 = 1.0 + alpha;
      double a1 = -2.0 * cos_omega;
      double a2 = 1.0 - alpha;

      switch (type)
      {
        case Type::LOWPASS:
          b0 = (1.0 - cos_omega) * 0.5;
          b1 = 1.0 - cos_omega;
          b2 = (1.0 - cos_omega) * 0.5;
          break;
        case Type::HIGHPASS:
          b0 = (1.0 + cos_omega) * 0.5;
          b1 = -(1.0 + cos_omega);
          b2 = (1.0 + cos_omega) * 0.5;
          break;
        case Type::BANDPASS:
          b0 = alpha;
          b1 = 0.0;
          b2 = -alpha;
          break;
        case Type::NOTCH:
          b0 = 1.0;
          b1 = -2.0 * cos_omega;
          b2 = 1.0;
          break;
        case Type::ALLPASS:
          b0 = 1.0 - alpha;
          b1 = -2.0 * cos_omega;
          b2 = 1.0 + alpha;
          break;
      }

      this->b0 = b0 / a0;
      this->b1 = b1 / a0;
      this->b2 = b2 / a0;
      this->a1 = a1 / a0;
      this->a2 = a2 / a0;
    }

    double process(double x)
    {
      const double y = b0 * x + z1;
      z1 = b1 * x - a1 * y + z2;
      z2 = b2 * x - a2 * y;
      return y;
    }

  private:
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double z1 = 0.0;
    double z2 = 0.0;
};

class LOWPASS : public Module
{
  public:
    LOWPASS(double freq_hz, double res = 0.707)
      : Module(IN_COUNT, OUT_COUNT), freq_hz(freq_hz), res(res) {}

    enum Ins
    {
      IN,
      FREQ,
      RES,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      const double center = freq_hz * pow(2.0, inputs[FREQ] / 5.0);
      const double q_value = res + inputs[RES];
      core.set(BiquadCore::Type::LOWPASS, center, q_value);
      outputs[OUT] = core.process(inputs[IN]);
      Module::postprocess();
    }

  private:
    double freq_hz;
    double res;
    BiquadCore core;
};

class HIGHPASS : public Module
{
  public:
    HIGHPASS(double freq_hz, double res = 0.707)
      : Module(IN_COUNT, OUT_COUNT), freq_hz(freq_hz), res(res) {}

    enum Ins
    {
      IN,
      FREQ,
      RES,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      const double center = freq_hz * pow(2.0, inputs[FREQ] / 5.0);
      const double q_value = res + inputs[RES];
      core.set(BiquadCore::Type::HIGHPASS, center, q_value);
      outputs[OUT] = core.process(inputs[IN]);
      Module::postprocess();
    }

  private:
    double freq_hz;
    double res;
    BiquadCore core;
};

class BANDPASS : public Module
{
  public:
    BANDPASS(double freq_hz, double res = 0.707)
      : Module(IN_COUNT, OUT_COUNT), freq_hz(freq_hz), res(res) {}

    enum Ins
    {
      IN,
      FREQ,
      RES,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      const double center = freq_hz * pow(2.0, inputs[FREQ] / 5.0);
      const double q_value = res + inputs[RES];
      core.set(BiquadCore::Type::BANDPASS, center, q_value);
      outputs[OUT] = core.process(inputs[IN]);
      Module::postprocess();
    }

  private:
    double freq_hz;
    double res;
    BiquadCore core;
};

class NOTCH : public Module
{
  public:
    NOTCH(double freq_hz, double res = 0.707)
      : Module(IN_COUNT, OUT_COUNT), freq_hz(freq_hz), res(res) {}

    enum Ins
    {
      IN,
      FREQ,
      RES,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      const double center = freq_hz * pow(2.0, inputs[FREQ] / 5.0);
      const double q_value = res + inputs[RES];
      core.set(BiquadCore::Type::NOTCH, center, q_value);
      outputs[OUT] = core.process(inputs[IN]);
      Module::postprocess();
    }

  private:
    double freq_hz;
    double res;
    BiquadCore core;
};

class ALLPASS : public Module
{
  public:
    ALLPASS(double freq_hz, double res = 0.707)
      : Module(IN_COUNT, OUT_COUNT), freq_hz(freq_hz), res(res) {}

    enum Ins
    {
      IN,
      FREQ,
      RES,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      const double center = freq_hz * pow(2.0, inputs[FREQ] / 5.0);
      const double q_value = res + inputs[RES];
      core.set(BiquadCore::Type::ALLPASS, center, q_value);
      outputs[OUT] = core.process(inputs[IN]);
      Module::postprocess();
    }

  private:
    double freq_hz;
    double res;
    BiquadCore core;
};

class DELAY : public Module
{
  public:
    DELAY(double time) : Module(IN_COUNT, OUT_COUNT)
    {
      bufferSize = time;
      bufferPosition = 0;
      buffer.resize(time);       
    }

    enum Ins
    {
      IN,
      IN_COUNT
    };
  
    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process() 
    {
      // write most recent value to buffer
      buffer[bufferPosition++] = inputs[IN];
      bufferPosition %= bufferSize;

      // update output value
      outputs[OUT] = buffer[bufferPosition];

      // clean up
      Module::postprocess();
    }

  private:
    unsigned int bufferSize; 
    unsigned int bufferPosition;
    std::vector<Signal> buffer;
};

class CONST : public Module
{
  public:
    CONST(Signal s) : Module(IN_COUNT, OUT_COUNT)
    {
      value = s; 
    }

    enum Ins
    {
      IN_COUNT
    };
  
    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process() 
    {
      // update output value
      outputs[OUT] = value;

      // clean up
      Module::postprocess();
    }

  private:
    Signal value; 
};

class ADD : public Module
{
  public:
    ADD() : Module(IN_COUNT, OUT_COUNT) {}

    enum Ins
    {
      IN1,
      IN2,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      outputs[OUT] = inputs[IN1] + inputs[IN2];
      Module::postprocess();
    }
};

class SUB : public Module
{
  public:
    SUB() : Module(IN_COUNT, OUT_COUNT) {}

    enum Ins
    {
      IN1,
      IN2,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      outputs[OUT] = inputs[IN1] - inputs[IN2];
      Module::postprocess();
    }
};

class MUL : public Module
{
  public:
    MUL() : Module(IN_COUNT, OUT_COUNT) {}

    enum Ins
    {
      IN1,
      IN2,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      outputs[OUT] = inputs[IN1] * inputs[IN2];
      Module::postprocess();
    }
};

class DIV : public Module
{
  public:
    DIV() : Module(IN_COUNT, OUT_COUNT) {}

    enum Ins
    {
      IN1,
      IN2,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      const double denominator = inputs[IN2];
      outputs[OUT] = fabs(denominator) < 1e-12 ? 0.0 : inputs[IN1] / denominator;
      Module::postprocess();
    }
};

class NEG : public Module
{
  public:
    NEG() : Module(IN_COUNT, OUT_COUNT) {}

    enum Ins
    {
      IN,
      IN_COUNT
    };

    enum Outs
    {
      OUT,
      OUT_COUNT
    };

    void process()
    {
      outputs[OUT] = -inputs[IN];
      Module::postprocess();
    }
};
