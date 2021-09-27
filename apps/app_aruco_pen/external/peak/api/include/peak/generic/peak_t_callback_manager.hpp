/*!
 * \file    peak_t_callback_manager.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <unordered_map>
#include <memory>
#include <mutex>
#include <utility>


namespace peak
{
namespace core
{

template <class CCallbackHandleType, class CallbackType>
class TCallbackManager
{
public:
    using RegisterCallbackFunction = std::function<CCallbackHandleType(void* callbackContext)>;
    using UnregisterCallbackFunction = std::function<void(CCallbackHandleType callbackHandle)>;

    TCallbackManager() = delete;
    TCallbackManager(
        const RegisterCallbackFunction& registerFunction, const UnregisterCallbackFunction& unregisterFunction)
        : m_registerFunction(registerFunction)
        , m_unregisterFunction(unregisterFunction)
        , m_callbacksByHandle()
        , m_callbacksByHandleMutex()
    {}
    ~TCallbackManager() = default;
    TCallbackManager(const TCallbackManager& other) = delete;
    TCallbackManager& operator=(const TCallbackManager& other) = delete;
    TCallbackManager(TCallbackManager&& other) = delete;
    TCallbackManager& operator=(TCallbackManager&& other) = delete;

    CCallbackHandleType RegisterCallback(const CallbackType& callback)
    {
        auto _callback = std::make_unique<CallbackType>(callback);
        auto callbackHandle = m_registerFunction(_callback.get());

        {
            std::lock_guard<std::mutex> lock(m_callbacksByHandleMutex);

            m_callbacksByHandle.emplace(callbackHandle, std::move(_callback));
        }

        return callbackHandle;
    }
    void UnregisterCallback(CCallbackHandleType callbackHandle)
    {
        m_unregisterFunction(callbackHandle);

        {
            std::lock_guard<std::mutex> lock(m_callbacksByHandleMutex);

            m_callbacksByHandle.erase(callbackHandle);
        }
    }
    void UnregisterAllCallbacks()
    {
        std::lock_guard<std::mutex> lock(m_callbacksByHandleMutex);

        for (const auto& mapIt : m_callbacksByHandle)
        {
            auto callbackHandle = mapIt.first;
            m_unregisterFunction(callbackHandle);
        }

        m_callbacksByHandle.clear();
    }

private:
    RegisterCallbackFunction m_registerFunction;
    UnregisterCallbackFunction m_unregisterFunction;

    std::unordered_map<CCallbackHandleType, std::unique_ptr<CallbackType>> m_callbacksByHandle;

    mutable std::mutex m_callbacksByHandleMutex;
};

template <class CallbackHandleType, class CallbackType>
class TTriggerCallbackManager
{
public:
    TTriggerCallbackManager() = default;
    ~TTriggerCallbackManager() = default;
    TTriggerCallbackManager(const TTriggerCallbackManager& other) = delete;
    TTriggerCallbackManager& operator=(const TTriggerCallbackManager& other) = delete;
    TTriggerCallbackManager(TTriggerCallbackManager&& other) = delete;
    TTriggerCallbackManager& operator=(TTriggerCallbackManager&& other) = delete;

    CallbackHandleType RegisterCallback(const CallbackType& callback)
    {
        auto _callback = std::make_unique<CallbackType>(callback);
        auto callbackHandle = reinterpret_cast<CallbackHandleType>(_callback.get());

        {
            std::lock_guard<std::mutex> lock(m_callbacksByHandleMutex);

            m_callbacksByHandle.emplace(callbackHandle, std::move(_callback));
        }

        return callbackHandle;
    }
    void UnregisterCallback(CallbackHandleType callbackHandle)
    {
        std::lock_guard<std::mutex> lock(m_callbacksByHandleMutex);

        m_callbacksByHandle.erase(callbackHandle);
    }

    template <class... Args>
    void TriggerCallbacks(Args&&... args) const
    {
        std::lock_guard<std::mutex> lock(m_callbacksByHandleMutex);

        for (const auto& mapIt : m_callbacksByHandle)
        {
            const auto& callback = mapIt.second;
            callback->operator()(std::forward<Args>(args)...);
        }
    }

private:
    std::unordered_map<CallbackHandleType, std::unique_ptr<CallbackType>> m_callbacksByHandle;

    mutable std::mutex m_callbacksByHandleMutex;
};

} /* namespace core */
} /* namespace peak */
