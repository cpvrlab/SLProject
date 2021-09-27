/*!
 * \file    peak_system.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_common_structs.hpp>
#include <peak/common/peak_event_supporting_module.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/generic/peak_t_callback_manager.hpp>
#include <peak/interface/peak_interface_descriptor.hpp>
#include <peak/system/peak_common_system_enums.hpp>

#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

class ProducerLibrary;

struct InterfaceFoundCallbackContainer;

/*!
 * \brief Represents a GenTL System module.
 *
 * This class allows to query information about the GenTL System module and to enumerate its
 * \link InterfaceDescriptor InterfaceDescriptors\endlink, which allow to open the corresponding Interface.
 *
 */
class System
    : public EventSupportingModule
    , public std::enable_shared_from_this<System>
{
public:
    /*! The type of interface found callbacks. */
    using InterfaceFoundCallback = std::function<void(const std::shared_ptr<InterfaceDescriptor>& foundInterface)>;
    /*! The type of interface found callback handles. */
    using InterfaceFoundCallbackHandle = InterfaceFoundCallback*;
    /*! The type of interface lost callbacks. */
    using InterfaceLostCallback = std::function<void(const std::string& lostInterfaceId)>;
    /*! The type of interface lost callback handles. */
    using InterfaceLostCallbackHandle = InterfaceLostCallback*;

    System() = delete;
    ~System() override;
    System(const System& other) = delete;
    System& operator=(const System& other) = delete;
    System(System&& other) = delete;
    System& operator=(System&& other) = delete;

    /*! @copydoc ProducerLibrary::Key() */
    std::string Key() const;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the ID.
     *
     * \return ID
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ID() const;
    /*! @copydoc SystemDescriptor::DisplayName() */
    std::string DisplayName() const;
    /*! @copydoc SystemDescriptor::VendorName() */
    std::string VendorName() const;
    /*! @copydoc SystemDescriptor::ModelName() */
    std::string ModelName() const;
    /*! @copydoc SystemDescriptor::Version() */
    std::string Version() const;
    /*! @copydoc SystemDescriptor::TLType() */
    std::string TLType() const;
    /*! @copydoc SystemDescriptor::CTIFileName() */
    std::string CTIFileName() const;
    /*! @copydoc SystemDescriptor::CTIFullPath() */
    std::string CTIFullPath() const;
    /*! @copydoc SystemDescriptor::GenTLVersionMajor() */
    uint32_t GenTLVersionMajor() const;
    /*! @copydoc SystemDescriptor::GenTLVersionMinor() */
    uint32_t GenTLVersionMinor() const;
    /*! @copydoc SystemDescriptor::CharacterEncoding() */
    peak::core::CharacterEncoding CharacterEncoding() const;

    /*!
     * \brief Searches for interfaces.
     *
     * This function triggers an update of the internal interface list. The callbacks registered on the system
     * will be triggered if an interface is found or lost.
     *
     * \param[in] timeout_ms The time to wait for new interfaces in milliseconds. In any case the
     *                       GenTL Producer must make sure that this operation is completed in a
     *                       reasonable amount of time depending on the underlying technology.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UpdateInterfaces(Timeout timeout_ms);
    /*!
     * \brief Returns the interface list.
     *
     * \return Interface list
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<InterfaceDescriptor>> Interfaces() const;
    /*! @copydoc SystemDescriptor::ParentLibrary() */
    std::shared_ptr<ProducerLibrary> ParentLibrary() const;

    /*!
     * \brief Registers a callback for signaling a found interface.
     *
     * This function registers a callback which gets called every time a new interface is found. Pass the callback
     * handle returned by this function to UnregisterInterfaceFoundCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if a new interface is found.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    InterfaceFoundCallbackHandle RegisterInterfaceFoundCallback(const InterfaceFoundCallback& callback);
    /*!
     * \brief Unregisters a interface found callback.
     *
     * This function unregisters a interface found callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterInterfaceFoundCallback(InterfaceFoundCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a lost interface.
     *
     * This function registers a callback which gets called every time a interface is lost. Pass the callback
     * handle returned by this function to UnregisterInterfaceLostCallback() to unregister the callback.
     *
     * \param[in] callback The callback to call if a interface is lost.
     *
     * \return Callback handle
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    InterfaceLostCallbackHandle RegisterInterfaceLostCallback(const InterfaceLostCallback& callback);
    /*!
     * \brief Unregisters a interface lost callback.
     *
     * This function unregisters a interface lost callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterInterfaceLostCallback(InterfaceLostCallbackHandle callbackHandle);

private:
    struct InterfaceFoundCallbackContainer
    {
        std::weak_ptr<System> _System;
        System::InterfaceFoundCallback Callback;
    };
    static void PEAK_CALL_CONV InterfaceFoundCallbackCWrapper(
        PEAK_INTERFACE_DESCRIPTOR_HANDLE foundInterface, void* context);
    static void PEAK_CALL_CONV InterfaceLostCallbackCWrapper(
        const char* lostInterfaceId, size_t lostInterfaceIdSize, void* context);
    std::shared_ptr<InterfaceDescriptor> GetOrAddFoundInterface(PEAK_INTERFACE_DESCRIPTOR_HANDLE foundInterface);

    PEAK_MODULE_HANDLE ModuleHandle() const override;
    PEAK_EVENT_SUPPORTING_MODULE_HANDLE EventSupportingModuleHandle() const override;

    void InitializeUpdateMechanismIfNecessary();

    friend ClassCreator<System>;
    System(PEAK_SYSTEM_HANDLE systemHandle, const std::weak_ptr<ProducerLibrary>& parentLibrary);
    PEAK_SYSTEM_HANDLE m_backendHandle;

    std::weak_ptr<ProducerLibrary> m_parentLibrary;

    std::unique_ptr<TCallbackManager<PEAK_INTERFACE_FOUND_CALLBACK_HANDLE, InterfaceFoundCallbackContainer>>
        m_interfaceFoundCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_INTERFACE_LOST_CALLBACK_HANDLE, InterfaceLostCallback>>
        m_interfaceLostCallbackManager;

    std::vector<std::shared_ptr<InterfaceDescriptor>> m_interfaces;
    std::unordered_map<std::string, std::shared_ptr<InterfaceDescriptor>> m_interfacesByKey;
    std::unordered_map<std::string, std::string> m_interfacesKeyById;
    mutable std::mutex m_interfacesMutex;
    std::once_flag m_updateMechanismInitializedFlag;

    std::string m_key;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline System::System(PEAK_SYSTEM_HANDLE systemHandle, const std::weak_ptr<ProducerLibrary>& parentLibrary)
    : m_backendHandle(systemHandle)
    , m_parentLibrary(parentLibrary)
    , m_interfaceFoundCallbackManager()
    , m_interfaceLostCallbackManager()
    , m_key(QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetKey(systemHandle, key, keySize);
    }))
{
    m_interfaceFoundCallbackManager =
        std::make_unique<TCallbackManager<PEAK_INTERFACE_FOUND_CALLBACK_HANDLE, InterfaceFoundCallbackContainer>>(
            [&](void* callbackContext) {
                return QueryNumericFromCInterfaceFunction<PEAK_INTERFACE_FOUND_CALLBACK_HANDLE>(
                    [&](PEAK_INTERFACE_FOUND_CALLBACK_HANDLE* interfaceFoundCallbackHandle) {
                        return PEAK_C_ABI_PREFIX PEAK_System_RegisterInterfaceFoundCallback(m_backendHandle,
                            InterfaceFoundCallbackCWrapper, callbackContext, interfaceFoundCallbackHandle);
                    });
            },
            [&](PEAK_INTERFACE_FOUND_CALLBACK_HANDLE callbackHandle) {
                CallAndCheckCInterfaceFunction([&] {
                    return PEAK_C_ABI_PREFIX PEAK_System_UnregisterInterfaceFoundCallback(
                        m_backendHandle, callbackHandle);
                });
            });

    m_interfaceLostCallbackManager =
        std::make_unique<TCallbackManager<PEAK_INTERFACE_LOST_CALLBACK_HANDLE, InterfaceLostCallback>>(
            [&](void* callbackContext) {
                return QueryNumericFromCInterfaceFunction<PEAK_INTERFACE_LOST_CALLBACK_HANDLE>(
                    [&](PEAK_INTERFACE_LOST_CALLBACK_HANDLE* interfaceLostCallbackHandle) {
                        return PEAK_C_ABI_PREFIX PEAK_System_RegisterInterfaceLostCallback(m_backendHandle,
                            InterfaceLostCallbackCWrapper, callbackContext, interfaceLostCallbackHandle);
                    });
            },
            [&](PEAK_INTERFACE_LOST_CALLBACK_HANDLE callbackHandle) {
                CallAndCheckCInterfaceFunction([&] {
                    return PEAK_C_ABI_PREFIX PEAK_System_UnregisterInterfaceLostCallback(
                        m_backendHandle, callbackHandle);
                });
            });
}

inline System::~System()
{
    try
    {
        m_interfaceFoundCallbackManager->UnregisterAllCallbacks();
        m_interfaceLostCallbackManager->UnregisterAllCallbacks();
    }
    catch (const Exception&)
    {}

    (void)PEAK_C_ABI_PREFIX PEAK_System_Destruct(m_backendHandle);
}

inline std::string System::Key() const
{
    return m_key;
}

inline RawInformation System::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetInfo(m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline std::string System::ID() const
{
    return QueryStringFromCInterfaceFunction([&](char* id, size_t* idSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetID(m_backendHandle, id, idSize);
    });
}

inline std::string System::DisplayName() const
{
    return QueryStringFromCInterfaceFunction([&](char* displayName, size_t* displayNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetDisplayName(m_backendHandle, displayName, displayNameSize);
    });
}

inline std::string System::VendorName() const
{
    return QueryStringFromCInterfaceFunction([&](char* vendorName, size_t* vendorNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetVendorName(m_backendHandle, vendorName, vendorNameSize);
    });
}

inline std::string System::ModelName() const
{
    return QueryStringFromCInterfaceFunction([&](char* modelName, size_t* modelNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetModelName(m_backendHandle, modelName, modelNameSize);
    });
}

inline std::string System::Version() const
{
    return QueryStringFromCInterfaceFunction([&](char* version, size_t* versionSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetVersion(m_backendHandle, version, versionSize);
    });
}

inline std::string System::TLType() const
{
    return QueryStringFromCInterfaceFunction([&](char* tlType, size_t* tlTypeSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetTLType(m_backendHandle, tlType, tlTypeSize);
    });
}

inline std::string System::CTIFileName() const
{
    return QueryStringFromCInterfaceFunction([&](char* ctiFileName, size_t* ctiFileNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetCTIFileName(m_backendHandle, ctiFileName, ctiFileNameSize);
    });
}

inline std::string System::CTIFullPath() const
{
    return QueryStringFromCInterfaceFunction([&](char* ctiFullPath, size_t* ctiFullPathSize) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetCTIFullPath(m_backendHandle, ctiFullPath, ctiFullPathSize);
    });
}

inline uint32_t System::GenTLVersionMajor() const
{
    return QueryNumericFromCInterfaceFunction<uint32_t>([&](uint32_t* gentlVersionMajor) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetGenTLVersionMajor(m_backendHandle, gentlVersionMajor);
    });
}

inline uint32_t System::GenTLVersionMinor() const
{
    return QueryNumericFromCInterfaceFunction<uint32_t>([&](uint32_t* gentlVersionMinor) {
        return PEAK_C_ABI_PREFIX PEAK_System_GetGenTLVersionMinor(m_backendHandle, gentlVersionMinor);
    });
}

inline peak::core::CharacterEncoding System::CharacterEncoding() const
{
    return static_cast<peak::core::CharacterEncoding>(
        QueryNumericFromCInterfaceFunction<PEAK_CHARACTER_ENCODING>(
            [&](PEAK_CHARACTER_ENCODING* characterEncoding) {
                return PEAK_C_ABI_PREFIX PEAK_System_GetCharacterEncoding(m_backendHandle, characterEncoding);
            }));
}

inline void System::UpdateInterfaces(Timeout timeout_ms)
{
    InitializeUpdateMechanismIfNecessary();

    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_System_UpdateInterfaces(m_backendHandle, timeout_ms); });
}

inline std::vector<std::shared_ptr<InterfaceDescriptor>> System::Interfaces() const
{
    std::lock_guard<std::mutex> lock(m_interfacesMutex);
    return m_interfaces;
}

inline std::shared_ptr<ProducerLibrary> System::ParentLibrary() const
{
    return LockOrThrow(m_parentLibrary);
}

inline System::InterfaceFoundCallbackHandle System::RegisterInterfaceFoundCallback(
    const System::InterfaceFoundCallback& callback)
{
    return reinterpret_cast<InterfaceFoundCallbackHandle>(
        m_interfaceFoundCallbackManager->RegisterCallback(InterfaceFoundCallbackContainer{ shared_from_this(),
            [callback](const std::shared_ptr<InterfaceDescriptor>& foundInterface) { callback(foundInterface); } }));
}

inline void System::UnregisterInterfaceFoundCallback(System::InterfaceFoundCallbackHandle callbackHandle)
{
    m_interfaceFoundCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_INTERFACE_FOUND_CALLBACK_HANDLE>(callbackHandle));
}

inline System::InterfaceLostCallbackHandle System::RegisterInterfaceLostCallback(
    const System::InterfaceLostCallback& callback)
{
    return reinterpret_cast<InterfaceLostCallbackHandle>(m_interfaceLostCallbackManager->RegisterCallback(callback));
}

inline void System::UnregisterInterfaceLostCallback(System::InterfaceLostCallbackHandle callbackHandle)
{
    m_interfaceLostCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_INTERFACE_LOST_CALLBACK_HANDLE>(callbackHandle));
}

inline void PEAK_CALL_CONV System::InterfaceFoundCallbackCWrapper(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE foundInterface, void* context)
{
    auto callbackContainer = static_cast<InterfaceFoundCallbackContainer*>(context);

    auto interfaceDescriptor = callbackContainer->_System.lock()->GetOrAddFoundInterface(foundInterface);

    callbackContainer->Callback(interfaceDescriptor);
}

inline void PEAK_CALL_CONV System::InterfaceLostCallbackCWrapper(
    const char* lostInterfaceId, size_t lostInterfaceIdSize, void* context)
{
    auto callback = static_cast<System::InterfaceLostCallback*>(context);

    callback->operator()(std::string(lostInterfaceId, lostInterfaceIdSize - 1));
}

inline std::shared_ptr<InterfaceDescriptor> System::GetOrAddFoundInterface(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE foundInterface)
{
    std::lock_guard<std::mutex> lock(m_interfacesMutex);

    const auto interfaceKey = QueryStringFromCInterfaceFunction([&](char* key, size_t* keySize) {
        return PEAK_C_ABI_PREFIX PEAK_InterfaceDescriptor_GetKey(foundInterface, key, keySize);
    });
    auto keyInterfacePairIterator = m_interfacesByKey.find(interfaceKey);
    if (keyInterfacePairIterator == m_interfacesByKey.end())
    {
        auto interfaceDescriptor = std::static_pointer_cast<InterfaceDescriptor>(
            std::make_shared<ClassCreator<InterfaceDescriptor>>(
                foundInterface, std::weak_ptr<System>{ shared_from_this() }));
        m_interfaces.emplace_back(interfaceDescriptor);
        m_interfacesByKey.emplace(interfaceKey, interfaceDescriptor);
        m_interfacesKeyById.emplace(interfaceDescriptor->ID(), interfaceKey);

        return interfaceDescriptor;
    }

    return keyInterfacePairIterator->second;
}

inline PEAK_MODULE_HANDLE System::ModuleHandle() const
{
    auto moduleHandle = QueryNumericFromCInterfaceFunction<PEAK_MODULE_HANDLE>(
        [&](PEAK_MODULE_HANDLE* _moduleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_System_ToModule(m_backendHandle, _moduleHandle);
        });

    return moduleHandle;
}

inline PEAK_EVENT_SUPPORTING_MODULE_HANDLE System::EventSupportingModuleHandle() const
{
    auto eventSupportingModuleHandle = QueryNumericFromCInterfaceFunction<PEAK_EVENT_SUPPORTING_MODULE_HANDLE>(
        [&](PEAK_EVENT_SUPPORTING_MODULE_HANDLE* _eventSupportingModuleHandle) {
            return PEAK_C_ABI_PREFIX PEAK_System_ToEventSupportingModule(
                m_backendHandle, _eventSupportingModuleHandle);
        });

    return eventSupportingModuleHandle;
}

inline void System::InitializeUpdateMechanismIfNecessary()
{
    std::call_once(m_updateMechanismInitializedFlag, [&] {
        (void)RegisterInterfaceFoundCallback([](const std::shared_ptr<InterfaceDescriptor>&) {
            // Registering an empty callback is enough since InterfaceFoundCallbackCWrapper() handles everything else.
        });
        (void)RegisterInterfaceLostCallback([&](const std::string& lostInterfaceId) {
            std::lock_guard<std::mutex> lock(m_interfacesMutex);
            const auto lostInterfaceKey = m_interfacesKeyById.at(lostInterfaceId);
            m_interfaces.erase(std::remove_if(std::begin(m_interfaces), std::end(m_interfaces),
                                   [lostInterfaceKey](const std::shared_ptr<InterfaceDescriptor>& interface_) {
                                       return interface_->Key() == lostInterfaceKey;
                                   }),
                std::end(m_interfaces));
            m_interfacesByKey.erase(lostInterfaceKey);
            m_interfacesKeyById.erase(lostInterfaceId);
        });
    });
}

} /* namespace core */
} /* namespace peak */
