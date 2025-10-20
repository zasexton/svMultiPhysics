/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_SCOPED_SUBSCRIPTION_H
#define SVMP_SCOPED_SUBSCRIPTION_H

#include "MeshObserver.h"
#include <memory>
#include <utility>

namespace svmp {

/**
 * @brief RAII wrapper for managing observer subscriptions
 *
 * Automatically unsubscribes the observer when the ScopedSubscription
 * goes out of scope, ensuring proper cleanup without manual management.
 *
 * Example usage:
 * @code
 * {
 *   MyObserver observer;
 *   ScopedSubscription sub(mesh.event_bus(), &observer);
 *   // Observer is now subscribed
 *   // ... do work ...
 * } // Observer is automatically unsubscribed here
 * @endcode
 */
class ScopedSubscription {
public:
  /**
   * @brief Default constructor - creates an inactive subscription
   */
  ScopedSubscription() = default;

  /**
   * @brief Subscribe a non-owning observer to an event bus
   *
   * @param bus The event bus to subscribe to
   * @param observer The observer to subscribe (lifetime must exceed this subscription)
   */
  ScopedSubscription(MeshEventBus* bus, MeshObserver* observer)
      : bus_(bus), observer_(observer), owned_(false) {
    if (bus_ && observer_) {
      bus_->subscribe(observer_);
    }
  }

  /**
   * @brief Subscribe an owned observer to an event bus
   *
   * @param bus The event bus to subscribe to
   * @param observer Shared ownership of the observer
   */
  ScopedSubscription(MeshEventBus* bus, std::shared_ptr<MeshObserver> observer)
      : bus_(bus), observer_(observer.get()), owned_observer_(observer), owned_(true) {
    if (bus_ && owned_observer_) {
      bus_->subscribe(owned_observer_);
    }
  }

  /**
   * @brief Destructor - automatically unsubscribes the observer
   */
  ~ScopedSubscription() {
    unsubscribe();
  }

  // Disable copy constructor and assignment
  ScopedSubscription(const ScopedSubscription&) = delete;
  ScopedSubscription& operator=(const ScopedSubscription&) = delete;

  /**
   * @brief Move constructor
   */
  ScopedSubscription(ScopedSubscription&& other) noexcept
      : bus_(other.bus_),
        observer_(other.observer_),
        owned_observer_(std::move(other.owned_observer_)),
        owned_(other.owned_) {
    other.bus_ = nullptr;
    other.observer_ = nullptr;
    other.owned_ = false;
  }

  /**
   * @brief Move assignment operator
   */
  ScopedSubscription& operator=(ScopedSubscription&& other) noexcept {
    if (this != &other) {
      unsubscribe();
      bus_ = other.bus_;
      observer_ = other.observer_;
      owned_observer_ = std::move(other.owned_observer_);
      owned_ = other.owned_;
      other.bus_ = nullptr;
      other.observer_ = nullptr;
      other.owned_ = false;
    }
    return *this;
  }

  /**
   * @brief Manually unsubscribe the observer
   *
   * After calling this, the subscription becomes inactive.
   */
  void unsubscribe() {
    if (bus_ && observer_) {
      bus_->unsubscribe(observer_);
    }
    reset();
  }

  /**
   * @brief Reset the subscription to inactive state
   */
  void reset() {
    bus_ = nullptr;
    observer_ = nullptr;
    owned_observer_.reset();
    owned_ = false;
  }

  /**
   * @brief Release ownership without unsubscribing
   *
   * After calling this, the observer remains subscribed but this
   * ScopedSubscription will no longer manage it.
   *
   * @return The observer pointer (or nullptr if owned)
   */
  MeshObserver* release() {
    MeshObserver* released = owned_ ? nullptr : observer_;
    bus_ = nullptr;
    observer_ = nullptr;
    owned_observer_.reset();
    owned_ = false;
    return released;
  }

  /**
   * @brief Check if the subscription is active
   */
  bool is_active() const {
    return bus_ != nullptr && observer_ != nullptr;
  }

  /**
   * @brief Get the subscribed observer
   */
  MeshObserver* observer() const { return observer_; }

  /**
   * @brief Get the event bus
   */
  MeshEventBus* bus() const { return bus_; }

  /**
   * @brief Check if the observer is owned by this subscription
   */
  bool is_owned() const { return owned_; }

private:
  MeshEventBus* bus_ = nullptr;
  MeshObserver* observer_ = nullptr;
  std::shared_ptr<MeshObserver> owned_observer_;
  bool owned_ = false;
};

/**
 * @brief Helper class to manage multiple subscriptions
 *
 * Useful for managing subscriptions from multiple observers to the same bus
 * or from one observer to multiple buses.
 */
class ScopedSubscriptionGroup {
public:
  ScopedSubscriptionGroup() = default;
  ~ScopedSubscriptionGroup() = default;

  // Disable copy
  ScopedSubscriptionGroup(const ScopedSubscriptionGroup&) = delete;
  ScopedSubscriptionGroup& operator=(const ScopedSubscriptionGroup&) = delete;

  // Enable move
  ScopedSubscriptionGroup(ScopedSubscriptionGroup&&) = default;
  ScopedSubscriptionGroup& operator=(ScopedSubscriptionGroup&&) = default;

  /**
   * @brief Add a subscription with a non-owning observer
   */
  void add(MeshEventBus* bus, MeshObserver* observer) {
    subscriptions_.emplace_back(bus, observer);
  }

  /**
   * @brief Add a subscription with an owned observer
   */
  void add(MeshEventBus* bus, std::shared_ptr<MeshObserver> observer) {
    subscriptions_.emplace_back(bus, observer);
  }

  /**
   * @brief Add an existing scoped subscription
   */
  void add(ScopedSubscription&& subscription) {
    subscriptions_.push_back(std::move(subscription));
  }

  /**
   * @brief Unsubscribe all observers
   */
  void unsubscribe_all() {
    for (auto& sub : subscriptions_) {
      sub.unsubscribe();
    }
  }

  /**
   * @brief Clear all subscriptions
   */
  void clear() {
    subscriptions_.clear();
  }

  /**
   * @brief Get the number of active subscriptions
   */
  size_t size() const {
    size_t count = 0;
    for (const auto& sub : subscriptions_) {
      if (sub.is_active()) {
        ++count;
      }
    }
    return count;
  }

  /**
   * @brief Check if there are any active subscriptions
   */
  bool empty() const {
    return size() == 0;
  }

private:
  std::vector<ScopedSubscription> subscriptions_;
};

/**
 * @brief Create a scoped subscription (convenience function)
 */
template<typename... Args>
ScopedSubscription make_scoped_subscription(Args&&... args) {
  return ScopedSubscription(std::forward<Args>(args)...);
}

} // namespace svmp

#endif // SVMP_SCOPED_SUBSCRIPTION_H