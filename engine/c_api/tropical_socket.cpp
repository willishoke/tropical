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
// land values at stale slot indices. The math mirrors the Lean set_param
// discipline implementations exactly — including the quantized-increment
// floor arithmetic — and the conformance differential
// (tests/web/param_dispatch_conformance.test.ts) gates the agreement.
static std::string dispatch_set_param(tropical_runtime::FlatRuntime * rt,
                                      const json & id, const std::string & name,
                                      double value)
{
  const auto result = rt->dispatch_param_sync(name, value);
  if (!result.ok)
    return json{{"jsonrpc", "2.0"}, {"id", id},
                {"error", {{"code", -32603},
                           {"message", result.error}}}}.dump();
  return json{{"jsonrpc", "2.0"}, {"id", id},
              {"result", {
                {"name", name},
                {"value", value},
                {"observed_sample_index", result.observed_sample_index},
                {"effective_sample_index", result.effective_sample_index},
              }}}.dump();
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
      const uint32_t requested_count = p.at("count").get<uint32_t>();
      if (requested_count == 0 || requested_count > 16384u)
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: count must be in [1,16384]"}}}}.dump();
      const std::string anchor = p.value("anchor", std::string{});
      if (!anchor.empty() && anchor != "playback")
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: anchor must be 'playback'"}}}}.dump();
      const bool playback_anchored = anchor == "playback";
      if (!playback_anchored && !p.contains("start"))
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: start is required without a playback anchor"}}}}.dump();
      const uint64_t playback_position = playback_anchored
        ? runtime_->current_sample_index()
        : 0;
      const uint64_t start = playback_anchored
        ? (playback_position >= requested_count
             ? playback_position - requested_count
             : 0)
        : p.at("start").get<uint64_t>();
      const uint32_t point_budget =
        p.value("point_budget", requested_count);
      if (point_budget == 0 || point_budget > 16384u)
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: point_budget must be in [1,16384]"}}}}.dump();
      const uint32_t stride =
        (requested_count + point_budget - 1) / point_budget;
      const uint32_t count =
        (requested_count + stride - 1) / stride;
      const auto names = p.at("slots").get<std::vector<std::string>>();
      std::vector<double> out(
        static_cast<size_t>(count) * names.size(), 0.0);
      const auto render = runtime_->render_window_observation_by_name(
        start, count, stride, names, out.data());
      if (!render.unknown_slot.empty())
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: unknown slot '" + render.unknown_slot + "'"}}}}.dump();
      if (render.status == tropical_runtime::RenderWindowStatus::Preempted)
      {
        json result = {
          {"start", start},
          {"count", 0},
          {"span", requested_count},
          {"stride", stride},
          {"preempted", true},
          {"program_version", render.program_version},
          {"control_version", render.control_version},
          {"effective_sample_index", render.effective_sample_index},
          {"values", json::array()},
        };
        if (playback_anchored)
        {
          result["anchor"] = "playback";
          result["playback_position"] = playback_position;
        }
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"result", std::move(result)}}.dump();
      }
      if (render.status == tropical_runtime::RenderWindowStatus::Unsupported)
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32603}, {"message", "render_window: active kernel is not fused"}}}}.dump();
      if (render.status != tropical_runtime::RenderWindowStatus::Rendered)
        return json{{"jsonrpc", "2.0"}, {"id", id},
                    {"error", {{"code", -32602}, {"message", "render_window: invalid render request"}}}}.dump();
      json values = json::array();
      for (size_t k = 0; k < names.size(); ++k)
      {
        json ch = json::array();
        for (uint32_t i = 0; i < count; ++i) ch.push_back(out[k * count + i]);
        values.push_back(std::move(ch));
      }
      json result = {
        {"start", start},
        {"count", count},
        {"span", requested_count},
        {"stride", stride},
        {"preempted", false},
        {"program_version", render.program_version},
        {"control_version", render.control_version},
        {"effective_sample_index", render.effective_sample_index},
        {"values", std::move(values)},
      };
      if (playback_anchored)
      {
        result["anchor"] = "playback";
        result["playback_position"] = playback_position;
      }
      return json{{"jsonrpc", "2.0"}, {"id", id},
                  {"result", std::move(result)}}.dump();
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
