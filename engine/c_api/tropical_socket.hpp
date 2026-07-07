#pragma once

/*
 * tropical_socket.hpp — a single Unix-domain socket endpoint over the
 * FlatRuntime, with an internal control/data plane split.
 *
 * One socket, many clients. Each client gets a reader thread. Requests are
 * newline-delimited JSON-RPC 2.0 (the same line format the --rpc stdio path
 * speaks). Routing is by method name:
 *
 *   DATA plane (handled here, never touches Lean): `set_param`, `get_telemetry`.
 *     set_param resolves param:{name} -> slot and writes it via
 *     FlatRuntime::set_slot_by_name_sync (hot-swap-safe). Telemetry reads the
 *     runtime's recompile counter / slot count directly. Pure C++, no Lean hop.
 *
 *   CONTROL plane (everything else — structural mutations, queries): the
 *     original request line is enqueued; the single Lean driver thread PULLS it
 *     via next_control(), dispatches through Engine.handleTool, and pushes the
 *     response back via send_response(). C++ never interprets a control payload
 *     beyond reading `method` to route; the original bytes are forwarded intact.
 *
 * Call direction across the FFI stays strictly Lean->C++: Lean pulls (blocks in
 * next_control on a condvar) and pushes responses; C++ never calls into Lean.
 */

#include "runtime/FlatRuntime.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tropical_socket
{

struct ControlMsg
{
  uint64_t    client_id;  // routes the response back to the originating fd
  std::string bytes;      // original request line (no trailing newline)
  std::string method;     // parsed JSON-RPC method (for queue coalescing)
};

class SocketServer
{
public:
  SocketServer(tropical_runtime::FlatRuntime * rt, std::string addr);
  ~SocketServer();  // stop() + joins

  SocketServer(const SocketServer &) = delete;
  SocketServer & operator=(const SocketServer &) = delete;

  // Bind + listen on the Unix-domain socket and spawn the accept thread.
  // Returns false on failure (see error()).
  bool start();
  void stop();

  const std::string & error() const { return error_; }

  // ── Lean control-plane FFI surface (called only from the Lean driver) ──────
  // Block until a control message arrives or the socket shuts down. Returns
  // false on shutdown-with-empty-queue. On true, *out_client_id is set and the
  // request line is copied into a malloc'd NUL-terminated buffer (*out_bytes /
  // *out_len) the caller frees with tropical_free_buffer.
  bool next_control(uint64_t * out_client_id, char ** out_bytes, std::size_t * out_len);

  // Send a response line (a newline is appended) to one client. No-op if the
  // client has disconnected.
  void send_response(uint64_t client_id, const char * bytes, std::size_t len);

private:
  void accept_loop();
  void client_loop(int fd, uint64_t client_id);
  void handle_line(int fd, uint64_t client_id, const std::string & line);
  std::string handle_data(const std::string & line);  // returns a JSON-RPC response line
  void enqueue_control(uint64_t client_id, std::string line, std::string method);
  // Resolve a coalesced-away control request's promise with a `superseded`
  // result, so an async client that awaited it doesn't hang.
  void send_superseded(const ControlMsg & msg);
  void close_client(uint64_t client_id);
  // Write a full line (caller supplies bytes without trailing newline); a
  // newline is appended. Serializes all writes to client fds under clients_mtx_.
  void write_line(int fd, const char * bytes, std::size_t len);

  tropical_runtime::FlatRuntime * runtime_;
  std::string addr_;
  int listen_fd_ = -1;
  int wake_pipe_[2] = {-1, -1};  // self-pipe to wake accept()/recv on shutdown
  std::string error_;

  std::atomic<bool> shutdown_{false};
  std::thread accept_thread_;

  std::mutex clients_mtx_;  // guards client_fds_ and serializes all fd writes
  std::unordered_map<uint64_t, int> client_fds_;
  std::vector<std::thread> client_threads_;
  std::atomic<uint64_t> next_client_id_{1};

  std::mutex ctrl_mtx_;
  std::condition_variable ctrl_cv_;
  std::deque<ControlMsg> ctrl_q_;  // deque (not queue): coalescing scans + drops mid-queue

  static constexpr std::size_t kMaxLine = 4u * 1024u * 1024u;  // drop a runaway line
};

}  // namespace tropical_socket
