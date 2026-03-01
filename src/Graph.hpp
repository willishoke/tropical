#include "Module.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using inputID = std::pair<std::string, unsigned int>;
using outputID = std::pair<std::string, unsigned int>;
using mPtr = std::unique_ptr<Module>;

/*
 * Graph stores modules and routes signals between them.
 * The audio thread owns active DSP state and applies pending edits once per buffer.
 */
class Graph
{
  public:
    explicit Graph(unsigned int bufferLength)
      : bufferLength_(bufferLength), outputBuffer(bufferLength, 0.0)
    {
    }

    void process()
    {
      apply_pending_commands();

      for (unsigned int i = 0; i < bufferLength_; ++i)
      {
        // Fan-in mixing: each destination input accumulates all routed source outputs.
        for (const auto & route : routes_)
        {
          Module * src = modules_[route.src_module_id].module.get();
          Module * dst = modules_[route.dst_module_id].module.get();
          dst->inputs[route.dst_input_id] += src->outputs[route.src_output_id];
        }

        for (uint32_t module_id : execution_order_)
        {
          modules_[module_id].module->process();
        }

        double mixed = 0.0;
        for (const auto & tap : mix_)
        {
          mixed += modules_[tap.module_id].module->outputs[tap.output_id] / 20.0;
        }
        outputBuffer[i] = mixed;
      }
    }

    bool addModule(std::string name, mPtr new_module)
    {
      if (!new_module)
      {
        return false;
      }

      const unsigned int in_count = static_cast<unsigned int>(new_module->inputs.size());
      const unsigned int out_count = static_cast<unsigned int>(new_module->outputs.size());

      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (control_modules_.find(name) != control_modules_.end())
      {
        return false;
      }

      control_modules_.emplace(name, ModuleShape{in_count, out_count});

      Command command;
      command.type = CommandType::AddModule;
      command.module_name = std::move(name);
      command.module = std::move(new_module);
      command_queue_.push_back(std::move(command));
      return true;
    }

    bool addOutput(outputID output)
    {
      const std::string & module_name = output.first;
      const unsigned int output_id = output.second;

      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto module_it = control_modules_.find(module_name);
      if (module_it == control_modules_.end())
      {
        return false;
      }

      if (output_id >= module_it->second.out_count)
      {
        return false;
      }

      control_mix_.push_back(output);

      Command command;
      command.type = CommandType::AddOutput;
      command.module_name = module_name;
      command.src_output_id = output_id;
      command_queue_.push_back(std::move(command));
      return true;
    }

    bool connect(
      std::string src_module,
      unsigned int src_output_id,
      std::string dst_module,
      unsigned int dst_input_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto src_it = control_modules_.find(src_module);
      auto dst_it = control_modules_.find(dst_module);
      if (src_it == control_modules_.end() || dst_it == control_modules_.end())
      {
        return false;
      }

      if (src_output_id >= src_it->second.out_count)
      {
        return false;
      }

      if (dst_input_id >= dst_it->second.in_count)
      {
        return false;
      }

      control_connections_.push_back(ConnectionDesc{
        std::move(src_module), src_output_id, std::move(dst_module), dst_input_id});

      Command command;
      command.type = CommandType::Connect;
      command.module_name = control_connections_.back().src_module;
      command.src_output_id = src_output_id;
      command.other_module_name = control_connections_.back().dst_module;
      command.dst_input_id = dst_input_id;
      command_queue_.push_back(std::move(command));
      return true;
    }

    bool remove_connection(
      std::string src_module,
      unsigned int src_output_id,
      std::string dst_module,
      unsigned int dst_input_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);

      bool removed = false;
      for (auto it = control_connections_.begin(); it != control_connections_.end(); )
      {
        const bool matches = it->src_module == src_module &&
                             it->src_output_id == src_output_id &&
                             it->dst_module == dst_module &&
                             it->dst_input_id == dst_input_id;
        if (matches)
        {
          it = control_connections_.erase(it);
          removed = true;
        }
        else
        {
          ++it;
        }
      }

      if (!removed)
      {
        return false;
      }

      Command command;
      command.type = CommandType::RemoveConnection;
      command.module_name = std::move(src_module);
      command.src_output_id = src_output_id;
      command.other_module_name = std::move(dst_module);
      command.dst_input_id = dst_input_id;
      command_queue_.push_back(std::move(command));
      return true;
    }

    bool remove_module(const std::string & module_name)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (control_modules_.find(module_name) == control_modules_.end())
      {
        return false;
      }

      control_modules_.erase(module_name);

      control_connections_.erase(
        std::remove_if(
          control_connections_.begin(),
          control_connections_.end(),
          [&module_name](const ConnectionDesc & c)
          {
            return c.src_module == module_name || c.dst_module == module_name;
          }),
        control_connections_.end());

      control_mix_.erase(
        std::remove_if(
          control_mix_.begin(),
          control_mix_.end(),
          [&module_name](const outputID & out)
          {
            return out.first == module_name;
          }),
        control_mix_.end());

      Command command;
      command.type = CommandType::RemoveModule;
      command.module_name = module_name;
      command_queue_.push_back(std::move(command));
      return true;
    }

    unsigned int getBufferLength() const
    {
      return bufferLength_;
    }

  private:
    struct ModuleShape
    {
      unsigned int in_count;
      unsigned int out_count;
    };

    struct ConnectionDesc
    {
      std::string src_module;
      unsigned int src_output_id;
      std::string dst_module;
      unsigned int dst_input_id;
    };

    struct ModuleSlot
    {
      std::string name;
      mPtr module;
      bool active = false;
    };

    struct Route
    {
      uint32_t src_module_id;
      unsigned int src_output_id;
      uint32_t dst_module_id;
      unsigned int dst_input_id;
    };

    struct MixTap
    {
      uint32_t module_id;
      unsigned int output_id;
    };

    enum class CommandType
    {
      AddModule,
      RemoveModule,
      Connect,
      RemoveConnection,
      AddOutput
    };

    struct Command
    {
      CommandType type{};
      std::string module_name;
      std::string other_module_name;
      unsigned int src_output_id = 0;
      unsigned int dst_input_id = 0;
      mPtr module;
    };

    void apply_pending_commands()
    {
      std::vector<Command> local_commands;
      {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (command_queue_.empty())
        {
          return;
        }
        local_commands.swap(command_queue_);
      }

      bool needs_order_rebuild = false;
      for (auto & command : local_commands)
      {
        switch (command.type)
        {
          case CommandType::AddModule:
            apply_add_module(command.module_name, std::move(command.module));
            needs_order_rebuild = true;
            break;
          case CommandType::RemoveModule:
            apply_remove_module(command.module_name);
            needs_order_rebuild = true;
            break;
          case CommandType::Connect:
            apply_connect(
              command.module_name,
              command.src_output_id,
              command.other_module_name,
              command.dst_input_id);
            needs_order_rebuild = true;
            break;
          case CommandType::RemoveConnection:
            apply_remove_connection(
              command.module_name,
              command.src_output_id,
              command.other_module_name,
              command.dst_input_id);
            needs_order_rebuild = true;
            break;
          case CommandType::AddOutput:
            apply_add_output(command.module_name, command.src_output_id);
            break;
        }
      }

      if (needs_order_rebuild)
      {
        rebuild_execution_order();
      }
    }

    void apply_add_module(std::string module_name, mPtr module)
    {
      if (!module)
      {
        return;
      }

      if (name_to_id_.find(module_name) != name_to_id_.end())
      {
        return;
      }

      uint32_t module_id = 0;
      if (!free_ids_.empty())
      {
        module_id = free_ids_.back();
        free_ids_.pop_back();
        modules_[module_id].name = std::move(module_name);
        modules_[module_id].module = std::move(module);
        modules_[module_id].active = true;
      }
      else
      {
        module_id = static_cast<uint32_t>(modules_.size());
        modules_.push_back(ModuleSlot{std::move(module_name), std::move(module), true});
      }

      name_to_id_[modules_[module_id].name] = module_id;
    }

    void apply_remove_module(const std::string & module_name)
    {
      auto name_it = name_to_id_.find(module_name);
      if (name_it == name_to_id_.end())
      {
        return;
      }

      const uint32_t module_id = name_it->second;
      name_to_id_.erase(name_it);

      routes_.erase(
        std::remove_if(
          routes_.begin(),
          routes_.end(),
          [module_id](const Route & r)
          {
            return r.src_module_id == module_id || r.dst_module_id == module_id;
          }),
        routes_.end());

      mix_.erase(
        std::remove_if(
          mix_.begin(),
          mix_.end(),
          [module_id](const MixTap & tap)
          {
            return tap.module_id == module_id;
          }),
        mix_.end());

      modules_[module_id].module.reset();
      modules_[module_id].active = false;
      modules_[module_id].name.clear();
      free_ids_.push_back(module_id);
    }

    void apply_connect(
      const std::string & src_module,
      unsigned int src_output_id,
      const std::string & dst_module,
      unsigned int dst_input_id)
    {
      auto src_it = name_to_id_.find(src_module);
      auto dst_it = name_to_id_.find(dst_module);
      if (src_it == name_to_id_.end() || dst_it == name_to_id_.end())
      {
        return;
      }

      routes_.push_back(Route{src_it->second, src_output_id, dst_it->second, dst_input_id});
    }

    void apply_remove_connection(
      const std::string & src_module,
      unsigned int src_output_id,
      const std::string & dst_module,
      unsigned int dst_input_id)
    {
      auto src_it = name_to_id_.find(src_module);
      auto dst_it = name_to_id_.find(dst_module);
      if (src_it == name_to_id_.end() || dst_it == name_to_id_.end())
      {
        return;
      }

      const uint32_t src_id = src_it->second;
      const uint32_t dst_id = dst_it->second;

      routes_.erase(
        std::remove_if(
          routes_.begin(),
          routes_.end(),
          [src_id, src_output_id, dst_id, dst_input_id](const Route & r)
          {
            return r.src_module_id == src_id &&
                   r.src_output_id == src_output_id &&
                   r.dst_module_id == dst_id &&
                   r.dst_input_id == dst_input_id;
          }),
        routes_.end());
    }

    void apply_add_output(const std::string & module_name, unsigned int output_id)
    {
      auto name_it = name_to_id_.find(module_name);
      if (name_it == name_to_id_.end())
      {
        return;
      }

      mix_.push_back(MixTap{name_it->second, output_id});
    }

    void rebuild_execution_order()
    {
      const std::size_t module_count = modules_.size();
      std::vector<unsigned int> indegree(module_count, 0);
      std::vector<std::vector<uint32_t>> adjacency(module_count);
      std::vector<uint32_t> active_ids;
      active_ids.reserve(module_count);

      for (uint32_t id = 0; id < modules_.size(); ++id)
      {
        if (modules_[id].active)
        {
          active_ids.push_back(id);
        }
      }

      for (const auto & route : routes_)
      {
        if (!modules_[route.src_module_id].active || !modules_[route.dst_module_id].active)
        {
          continue;
        }
        adjacency[route.src_module_id].push_back(route.dst_module_id);
        ++indegree[route.dst_module_id];
      }

      std::deque<uint32_t> ready;
      for (uint32_t id : active_ids)
      {
        if (indegree[id] == 0)
        {
          ready.push_back(id);
        }
      }

      std::vector<uint32_t> ordered;
      ordered.reserve(active_ids.size());

      while (!ready.empty())
      {
        uint32_t id = ready.front();
        ready.pop_front();
        ordered.push_back(id);

        for (uint32_t dst_id : adjacency[id])
        {
          if (--indegree[dst_id] == 0)
          {
            ready.push_back(dst_id);
          }
        }
      }

      if (ordered.size() == active_ids.size())
      {
        execution_order_ = std::move(ordered);
        return;
      }

      // Cycle fallback: preserve previous order for active modules, then append the rest.
      std::vector<uint8_t> seen(module_count, 0);
      std::vector<uint32_t> fallback;
      fallback.reserve(active_ids.size());

      for (uint32_t id : execution_order_)
      {
        if (id < modules_.size() && modules_[id].active && !seen[id])
        {
          fallback.push_back(id);
          seen[id] = 1;
        }
      }
      for (uint32_t id : active_ids)
      {
        if (!seen[id])
        {
          fallback.push_back(id);
          seen[id] = 1;
        }
      }
      execution_order_ = std::move(fallback);
    }

    unsigned int bufferLength_ = 0;

    // Audio-thread-owned active state (cache-friendly contiguous traversal).
    std::vector<ModuleSlot> modules_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
    std::vector<uint32_t> execution_order_;
    std::vector<Route> routes_;
    std::vector<MixTap> mix_;
    std::vector<uint32_t> free_ids_;

    // Control-thread mirror state for immediate validation.
    std::unordered_map<std::string, ModuleShape> control_modules_;
    std::vector<ConnectionDesc> control_connections_;
    std::vector<outputID> control_mix_;

    // Cross-thread command queue (applied once per buffer by audio thread).
    std::mutex pending_mutex_;
    std::vector<Command> command_queue_;

  public:
    // Public for existing callback and Python API compatibility.
    std::vector<double> outputBuffer;
};
