/**
 * @file events.cpp
 * @brief Asynchronous event waiting for overlapping CPU/GPU work
 *
 * This module provides a lightweight AsyncEvent that can be armed on a
 * Metal command buffer and waited on later from the CPU. It enables
 * overlapping CPU-side work with in-flight GPU execution without blocking
 * until the results are needed.
 */

#include "events.hpp"

namespace metal_marlin {

AsyncEvent::AsyncEvent()
    : state_(std::make_shared<State>()) {}

void AsyncEvent::reset() noexcept {
    state_->signaled.store(false, std::memory_order_release);
}

void AsyncEvent::signal() noexcept {
    state_->signaled.store(true, std::memory_order_release);
    state_->cv.notify_all();
}

bool AsyncEvent::is_signaled() const noexcept {
    return state_->signaled.load(std::memory_order_acquire);
}

#ifdef __OBJC__
void AsyncEvent::signal_on_complete(id<MTLCommandBuffer> command_buffer) {
    reset();

    if (!command_buffer) {
        signal();
        return;
    }

    auto state = state_;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        state->signaled.store(true, std::memory_order_release);
        state->cv.notify_all();
    }];
}
#endif

void AsyncEvent::wait() const {
    if (is_signaled()) {
        return;
    }

    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [this]() {
        return state_->signaled.load(std::memory_order_acquire);
    });
}

bool AsyncEvent::wait_for(std::chrono::nanoseconds timeout) const {
    if (is_signaled()) {
        return true;
    }

    if (timeout.count() <= 0) {
        return is_signaled();
    }

    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->cv.wait_for(lock, timeout, [this]() {
        return state_->signaled.load(std::memory_order_acquire);
    });
}

} // namespace metal_marlin
