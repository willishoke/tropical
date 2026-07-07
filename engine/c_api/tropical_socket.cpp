#include "c_api/tropical_socket.hpp"
#include <vector>

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace tropical_socket
{

using json = nlohmann::json;

// Linux: suppress SIGPIPE per-send via MSG_NOSIGNAL. macOS lacks it and uses
// the SO_NOSIGPIPE socket option instead (set on each accepted fd below).
#ifdef MSG_NOSIGNAL
static constexpr int kSendFlags = MSG_NOSIGNAL;
#else
static constexpr int kSendFlags = 0;
#endif

SocketServer::SocketServer(tropical_runtime::FlatRuntime * rt, std::string addr)
  : runtime_(rt), addr_(std::move(addr))
{
}

SocketServer::~SocketServer()
{
  stop();
}

bool SocketServer::start()
{
  if (addr_.empty()) { error_ = "socket: empty address"; return false; }
  if (addr_.size() >= sizeof(((struct sockaddr_un *)nullptr)->sun_path))
  {
    error_ = "socket: address too long for sockaddr_un";
    return false;
  }

  if (::pipe(wake_pipe_) != 0)
  {
    error_ = std::string("socket: pipe() failed: ") + std::strerror(errno);
    return false;
  }

  listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd_ < 0)
  {
    error_ = std::string("socket: socket() failed: ") + std::strerror(errno);
    return false;
  }

  ::unlink(addr_.c_str());  // clear a stale node from a prior run

  struct sockaddr_un un;
  std::memset(&un, 0, sizeof(un));
  un.sun_family = AF_UNIX;
  std::strncpy(un.sun_path, addr_.c_str(), sizeof(un.sun_path) - 1);

  if (::bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&un), sizeof(un)) != 0)
  {
    error_ = std::string("socket: bind('") + addr_ + "') failed: " + std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  if (::listen(listen_fd_, 16) != 0)
  {
    error_ = std::string("socket: listen() failed: ") + std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  accept_thread_ = std::thread([this] { accept_loop(); });
  return true;
}

void SocketServer::stop()
{
  bool was = shutdown_.exchange(true);
  if (was) return;

  // Wake the accept poll and the Lean next_control wait.
  if (wake_pipe_[1] >= 0) { const char b = 1; (void)::write(wake_pipe_[1], &b, 1); }
  ctrl_cv_.notify_all();

  if (accept_thread_.joinable()) accept_thread_.join();

  // Close all client fds so their recv() returns and the loops exit.
  {
    std::lock_guard<std::mutex> lk(clients_mtx_);
    for (auto & [id, fd] : client_fds_) ::close(fd);
    client_fds_.clear();
  }
  for (auto & t : client_threads_) if (t.joinable()) t.join();
  client_threads_.clear();

  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  if (wake_pipe_[0] >= 0) { ::close(wake_pipe_[0]); wake_pipe_[0] = -1; }
  if (wake_pipe_[1] >= 0) { ::close(wake_pipe_[1]); wake_pipe_[1] = -1; }
  if (!addr_.empty()) ::unlink(addr_.c_str());
}

void SocketServer::accept_loop()
{
  while (!shutdown_.load(std::memory_order_acquire))
  {
    struct pollfd fds[2];
    fds[0].fd = listen_fd_;   fds[0].events = POLLIN; fds[0].revents = 0;
    fds[1].fd = wake_pipe_[0]; fds[1].events = POLLIN; fds[1].revents = 0;
    const int r = ::poll(fds, 2, -1);
    if (r < 0) { if (errno == EINTR) continue; break; }
    if (fds[1].revents & POLLIN) break;  // shutdown signalled
    if (!(fds[0].revents & POLLIN)) continue;

    const int cfd = ::accept(listen_fd_, nullptr, nullptr);
    if (cfd < 0) { if (errno == EINTR || errno == EAGAIN) continue; break; }

#ifdef SO_NOSIGPIPE
    { int on = 1; ::setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on)); }
#endif

    const uint64_t id = next_client_id_.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(clients_mtx_);
      client_fds_[id] = cfd;
      // accept_loop is the sole writer of client_threads_ (joined in stop()
      // after this thread exits), so no extra guard is needed.
      client_threads_.emplace_back([this, cfd, id] { client_loop(cfd, id); });
    }
  }
}

void SocketServer::client_loop(int fd, uint64_t id)
{
  std::string buf;
  char tmp[4096];
  while (!shutdown_.load(std::memory_order_acquire))
  {
    const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) break;  // peer closed or error
    buf.append(tmp, static_cast<std::size_t>(n));

    std::size_t pos;
    while ((pos = buf.find('\n')) != std::string::npos)
    {
      std::string line = buf.substr(0, pos);
      buf.erase(0, pos + 1);
      // strip a trailing CR (\r\n clients)
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (!line.empty()) handle_line(fd, id, line);
    }
    if (buf.size() > kMaxLine) buf.clear();  // runaway: drop, don't grow unbounded
  }
  close_client(id);
}

void SocketServer::handle_line(int fd, uint64_t id, const std::string & line)
{
  // Read just enough to route by method. A control payload is forwarded as the
  // ORIGINAL bytes (number lexical forms preserved); only data-plane methods
  // are interpreted here.
  std::string method;
  bool parsed = false;
  try
  {
    json j = json::parse(line);
    parsed = true;
    if (j.contains("method") && j["method"].is_string())
      method = j["method"].get<std::string>();
  }
  catch (...)
  {
    parsed = false;  // malformed: hand to the control plane so Lean returns the
                     // standard parse-error envelope, matching --rpc behavior
  }

  if (parsed && (method == "set_param" || method == "get_telemetry"
              || method == "render_window" || method == "playback_position"))
  {
    const std::string resp = handle_data(line);
    write_line(fd, resp.data(), resp.size());
    return;
  }
  enqueue_control(id, line, method);
}

// Host-contract param dispatch (design/host-param-dispatch.md): the write
// discipline is a fact about the compiled patch, read from the plan's
// param_disciplines table — the client never chooses a verb. Every branch is
// a synchronous slot write off the audio thread, like the raw path; the whole
// resolve-read-write runs under one lock so a concurrent hot-swap can never
// land values at stale slot indices. The math mirrors the Lean reference
// (Engine.lean handleSetParamGlide / handleSetParamFreq /
// handleSetParamVelocity) exactly — including the quantized-increment floor
// arithmetic — and the conformance differential
// (tests/web/param_dispatch_conformance.test.ts) gates the agreement.
static std::string dispatch_set_param(tropical_runtime::FlatRuntime * rt,
                                      const json & id, const std::string & name,
                                      double value)
{
  auto ok = [&] {
    return json{{"jsonrpc", "2.0"}, {"id", id},
                {"result", {{"name", name}, {"value", value}}}}.dump();
  };
  auto err = [&](const std::string & msg) {
    return json{{"jsonrpc", "2.0"}, {"id", id},
                {"error", {{"code", -32603}, {"message", msg}}}}.dump();
  };

  return rt->with_active_state_sync(
    [&](tropical_runtime::KernelState & st) -> std::string
  {
    auto slot_of = [&st](const std::string & slot_name) -> uint32_t {
      for (uint32_t i = 0; i < st.slot_names.size(); ++i)
        if (st.slot_names[i] == slot_name) return i;
      return UINT32_MAX;
    };
    // Missing companions read as 0.0 and write as no-ops, matching the Lean
    // reference's per-slot defaults.
    auto read  = [&st](uint32_t i) { return i < st.slots.size() ? st.slots[i] : 0.0; };
    auto write = [&st](uint32_t i, double v) { if (i < st.slots.size()) st.slots[i] = v; };

    const tropical_runtime::ParamDiscipline * pd = st.find_discipline(name);
    const std::string disc = pd ? pd->discipline : std::string{"raw"};

    if (disc == "glide")
    {
      // Re-anchor the closed-form smoothstep ramp so it departs from the
      // CURRENT value (no jump): evaluate f(now) from the companions, then
      // v0 := current, v1 := target, t0 := now. The base slot does not exist
      // for glided params — only the companions are written.
      const uint32_t v0i = slot_of("param:" + name + "#v0");
      const uint32_t v1i = slot_of("param:" + name + "#v1");
      const uint32_t t0i = slot_of("param:" + name + "#t0");
      if (v0i == UINT32_MAX)
        return err("set_param: no glide slots for '" + name + "'");
      const double now = static_cast<double>(st.sample_index);
      // dur matches the kernel's ramp window; the table carries it (0.02 s
      // for playground knobs), with the kernel's 20 ms as the fallback.
      const double dur_sec = pd->glide_dur_sec > 0.0 ? pd->glide_dur_sec : 0.02;
      const double dur = st.sample_rate * dur_sec;
      const double v0 = read(v0i);
      const double v1 = read(v1i);
      const double t0 = read(t0i);
      const double r = (now - t0) / dur;
      const double s = r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
      write(v0i, v0 + (v1 - v0) * (s * s * (3.0 - 2.0 * s)));
      write(v1i, value);
      write(t0i, now);
      return ok();
    }

    if (disc == "anchor")
    {
      // Phase-anchored frequency: bump #phase by the phase the change would
      // have jumped (Δφ = (inc0 − inc1)·now / 2^32 cycles, inc the phasor's
      // own quantized increment), wrap to [0, 1), then write the base slot.
      const uint32_t fi = slot_of("param:" + name);
      if (fi == UINT32_MAX)
        return err("set_param: unknown param '" + name + "'");
      const uint32_t pi = slot_of("param:" + name + "#phase");
      if (pi == UINT32_MAX)
      {
        // No #phase companion in the loaded plan: degrade to raw.
        write(fi, value);
        return ok();
      }
      const double now  = static_cast<double>(st.sample_index);
      const double sr   = st.sample_rate;
      const double inc0 = std::floor(read(fi) * 4294967296.0 / sr);
      const double inc1 = std::floor(value * 4294967296.0 / sr);
      const double dcyc = ((inc0 - inc1) * now) / 4294967296.0;
      const double ph   = read(pi) + dcyc;
      write(pi, ph - std::floor(ph));   // frac → [0, 1)
      write(fi, value);
      return ok();
    }

    if (disc == "velocity")
    {
      // Master-clock re-base: tau_base += (v_current − target)·now/SR keeps
      // M(n) value-continuous across the velocity change.
      const uint32_t vi = slot_of("param:" + name);
      if (vi == UINT32_MAX)
        return err("set_param: unknown param '" + name + "'");
      // The origin slot is the declared companion; the Lean reference
      // derives the same name by substitution ("velocity" → "tau_base").
      std::string tau;
      if (!pd->companions.empty()) tau = pd->companions.front();
      else
      {
        tau = name;
        const std::size_t pos = tau.find("velocity");
        if (pos != std::string::npos) tau.replace(pos, 8, "tau_base");
      }
      const uint32_t ti = slot_of("param:" + tau);
      if (ti == UINT32_MAX)
        return err("set_param: no origin slot '" + tau + "'");
      const double now = static_cast<double>(st.sample_index);
      write(ti, read(ti) + (read(vi) - value) * now / st.sample_rate);
      write(vi, value);
      return ok();
    }

    // raw (or a name absent from the table): plain base-slot write —
    // today's behavior, and the whole contract for old plans.
    const uint32_t bi = slot_of("param:" + name);
    if (bi == UINT32_MAX)
      return err("set_param: unknown param '" + name + "'");
    write(bi, value);
    return ok();
  });
}

std::string SocketServer::handle_data(const std::string & line)
{
  json id = nullptr;
  std::string method;
  try
  {
    json j = json::parse(line);
    if (j.contains("id")) id = j["id"];
    method = j.value("method", std::string{});

    if (method == "set_param")
    {
      const json & p = j.at("params");
      const std::string name = p.at("name").get<std::string>();
      const double value = p.at("value").get<double>();
      if (!std::isfinite(value))
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32603}, {"message", "set_param: value must be finite"}}}}.dump();
      return dispatch_set_param(runtime_, id, name, value);
    }

    if (method == "get_telemetry")
    {
      return json{{"jsonrpc", "2.0"}, {"id", id},
                  {"result", {
                    {"recompile_version", static_cast<uint64_t>(runtime_->recompile_version())},
                    {"buffer_length", static_cast<uint32_t>(runtime_->getBufferLength())},
                    {"slot_count", static_cast<uint32_t>(runtime_->slot_count())}}}}.dump();
    }

    // Master clock: the audio thread's current sample index (advances one
    // buffer per process(), paced by the device when playing). A slave
    // consumer renders a window ending here for drift-free sync.
    if (method == "playback_position")
    {
      return json{{"jsonrpc", "2.0"}, {"id", id},
                  {"result", {{"position", static_cast<uint64_t>(runtime_->current_sample_index())}}}}.dump();
    }

    // Random-access waveform read: evaluate the active (stateless) kernel over
    // [start, start+count) and return each named slot's per-sample trajectory.
    if (method == "render_window")
    {
      const json & p = j.at("params");
      const uint64_t start = p.at("start").get<uint64_t>();
      uint32_t count = p.at("count").get<uint32_t>();
      if (count > 16384u)
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: count too large (max 16384)"}}}}.dump();
      const auto names = p.at("slots").get<std::vector<std::string>>();
      std::vector<uint32_t> ids;
      ids.reserve(names.size());
      for (const auto & nm : names)
      {
        const uint32_t sid = runtime_->slot_index(nm);
        if (sid == UINT32_MAX)
          return json{{"jsonrpc", "2.0"}, {"id", id},
                      {"error", {{"code", -32602}, {"message", "render_window: unknown slot '" + nm + "'"}}}}.dump();
        ids.push_back(sid);
      }
      std::vector<double> out(static_cast<size_t>(count) * ids.size(), 0.0);
      if (!runtime_->render_window(start, count, ids.data(),
                                   static_cast<uint32_t>(ids.size()), out.data()))
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32603}, {"message", "render_window: active kernel is not fused"}}}}.dump();
      json values = json::array();
      for (size_t k = 0; k < ids.size(); ++k)
      {
        json ch = json::array();
        for (uint32_t i = 0; i < count; ++i) ch.push_back(out[k * count + i]);
        values.push_back(std::move(ch));
      }
      return json{{"jsonrpc", "2.0"}, {"id", id},
                  {"result", {{"start", start}, {"count", count}, {"values", std::move(values)}}}}.dump();
    }
  }
  catch (const std::exception & e)
  {
    return json{{"jsonrpc", "2.0"}, {"id", id},
                {"error", {{"code", -32603}, {"message", std::string("data-plane error: ") + e.what()}}}}.dump();
  }

  return json{{"jsonrpc", "2.0"}, {"id", id},
              {"error", {{"code", -32601}, {"message", "unknown data-plane method"}}}}.dump();
}

void SocketServer::send_superseded(const ControlMsg & msg)
{
  json id = nullptr;
  try
  {
    json j = json::parse(msg.bytes);
    if (j.contains("id")) id = j["id"];
  }
  catch (...) { /* malformed: reply with a null id */ }
  const std::string resp = json{{"jsonrpc", "2.0"}, {"id", id},
                                {"result", {{"superseded", true}}}}.dump();
  send_response(msg.client_id, resp.data(), resp.size());
}

void SocketServer::enqueue_control(uint64_t client_id, std::string line, std::string method)
{
  std::vector<ControlMsg> superseded;
  {
    std::lock_guard<std::mutex> lk(ctrl_mtx_);
    // Compile coalescing: a `load_patch_graph` is a FULL-graph snapshot, so a new
    // one supersedes any still-queued (not-yet-dispatched) one — while the user
    // drags, we never waste a compile on an intermediate graph, we jump straight
    // to the latest. Incremental control requests (wire/add_instance/…) are NOT
    // full snapshots, so they are left in order. The one in-flight compile the
    // Lean thread already pulled runs to completion (LLVM has no mid-compile
    // interrupt); this drops only the BACKLOG. All of it lives here in the C++
    // queue, so the Lean control loop stays single-threaded and serial.
    if (method == "load_patch_graph")
    {
      std::deque<ControlMsg> kept;
      for (auto & m : ctrl_q_)
      {
        if (m.method == "load_patch_graph") superseded.push_back(std::move(m));
        else kept.push_back(std::move(m));
      }
      ctrl_q_ = std::move(kept);
    }
    ctrl_q_.push_back(ControlMsg{client_id, std::move(line), std::move(method)});
  }
  ctrl_cv_.notify_one();
  // Resolve the dropped requests' promises (outside the queue lock — send_response
  // takes clients_mtx_) so an async client that awaited them doesn't hang.
  for (const auto & m : superseded) send_superseded(m);
}

bool SocketServer::next_control(uint64_t * out_client_id, char ** out_bytes, std::size_t * out_len)
{
  std::unique_lock<std::mutex> lk(ctrl_mtx_);
  ctrl_cv_.wait(lk, [this] { return shutdown_.load(std::memory_order_acquire) || !ctrl_q_.empty(); });
  if (ctrl_q_.empty()) return false;  // shutdown with nothing left to drain

  ControlMsg msg = std::move(ctrl_q_.front());
  ctrl_q_.pop_front();
  lk.unlock();

  const std::size_t n = msg.bytes.size();
  char * buf = static_cast<char *>(std::malloc(n + 1));
  if (!buf) return false;
  std::memcpy(buf, msg.bytes.data(), n);
  buf[n] = '\0';

  if (out_client_id) *out_client_id = msg.client_id;
  if (out_bytes) *out_bytes = buf;
  if (out_len) *out_len = n;
  return true;
}

void SocketServer::write_line(int fd, const char * bytes, std::size_t len)
{
  std::lock_guard<std::mutex> lk(clients_mtx_);
  std::string out(bytes, len);
  out.push_back('\n');
  std::size_t off = 0;
  while (off < out.size())
  {
    const ssize_t w = ::send(fd, out.data() + off, out.size() - off, kSendFlags);
    if (w <= 0) break;  // closed/error: drop the rest
    off += static_cast<std::size_t>(w);
  }
}

void SocketServer::send_response(uint64_t client_id, const char * bytes, std::size_t len)
{
  int fd = -1;
  {
    std::lock_guard<std::mutex> lk(clients_mtx_);
    auto it = client_fds_.find(client_id);
    if (it == client_fds_.end()) return;  // disconnected
    fd = it->second;
    std::string out(bytes, len);
    out.push_back('\n');
    std::size_t off = 0;
    while (off < out.size())
    {
      const ssize_t w = ::send(fd, out.data() + off, out.size() - off, kSendFlags);
      if (w <= 0) break;
      off += static_cast<std::size_t>(w);
    }
  }
}

void SocketServer::close_client(uint64_t client_id)
{
  std::lock_guard<std::mutex> lk(clients_mtx_);
  auto it = client_fds_.find(client_id);
  if (it == client_fds_.end()) return;
  ::close(it->second);
  client_fds_.erase(it);
}

}  // namespace tropical_socket
