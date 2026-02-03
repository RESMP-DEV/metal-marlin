#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
// Forward declaration for non-ObjC++ compilation units
typedef void* MTLCommandBuffer;
#endif

namespace metal_marlin {

// Async event that can be signaled by GPU completion and waited on by CPU.
class AsyncEvent {
public:
    AsyncEvent();

    // Shared state allows safe capture by completion handlers.
    AsyncEvent(const AsyncEvent&) = default;
    AsyncEvent& operator=(const AsyncEvent&) = default;

    // Reset event to unsignaled.
    void reset() noexcept;

    // Manually signal event (CPU-side).
    void signal() noexcept;

    // Check if event has been signaled.
    [[nodiscard]] bool is_signaled() const noexcept;

#ifdef __OBJC__
    // Signal when the provided command buffer completes.
    void signal_on_complete(id<MTLCommandBuffer> command_buffer);
#endif

    // Block until signaled.
    void wait() const;

    // Block until signaled or timeout expires.
    [[nodiscard]] bool wait_for(std::chrono::nanoseconds timeout) const;

private:
    struct State {
        mutable std::mutex mutex;
        mutable std::condition_variable cv;
        std::atomic<bool> signaled{false};
    };

    std::shared_ptr<State> state_;
};

} // namespace metal_marlin
