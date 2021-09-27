/*!
 * \file    peak_firmware_update_progress_observer.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/device/peak_firmware_update_information.hpp>
#include <peak/generic/peak_t_callback_manager.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{

/*!
 * \brief The possible firmware update step types.
 */
enum class FirmwareUpdateStep
{
    CheckPreconditions,
    AcquireUpdateData,
    WriteFeature,
    ExecuteFeature,
    AssertFeature,
    UploadFile,
    ResetDevice
};

/*!
 * \brief Allows to observe a firmware update process.
 *
 * Observe a firmware update process by registering several callbacks.
 * These callbacks get called when the observer is passed to the FirmwareUpdater::UpdateDevice() function. The
 * callbacks inform about changes of the status of the firmware update process.
 *
 */
class FirmwareUpdateProgressObserver
{
public:
    /*! The type of update started callbacks. */
    using UpdateStartedCallback = std::function<void(
        const std::shared_ptr<FirmwareUpdateInformation>& updateInformation, uint32_t estimatedDuration_ms)>;
    /*! The type of update started callback handles. */
    using UpdateStartedCallbackHandle = UpdateStartedCallback*;
    /*! The type of update step started callbacks. */
    using UpdateStepStartedCallback = std::function<void(
        FirmwareUpdateStep updateStep, uint32_t estimatedDuration_ms, const std::string& description)>;
    /*! The type of update step started callback handles. */
    using UpdateStepStartedCallbackHandle = UpdateStepStartedCallback*;
    /*! The type of update step progress changed callbacks. */
    using UpdateStepProgressChangedCallback =
        std::function<void(FirmwareUpdateStep updateStep, double progressPercentage)>;
    /*! The type of update step progress changed callback handles. */
    using UpdateStepProgressChangedCallbackHandle = UpdateStepProgressChangedCallback*;
    /*! The type of update step finished callbacks. */
    using UpdateStepFinishedCallback = std::function<void(FirmwareUpdateStep updateStep)>;
    /*! The type of update step finished callback handles. */
    using UpdateStepFinishedCallbackHandle = UpdateStepFinishedCallback*;
    /*! The type of update finished callbacks. */
    using UpdateFinishedCallback = std::function<void(void)>;
    /*! The type of update finished callback handles. */
    using UpdateFinishedCallbackHandle = UpdateFinishedCallback*;
    /*! The type of update failed callbacks. */
    using UpdateFailedCallback = std::function<void(const std::string& errorDescription)>;
    /*! The type of update failed callback handles. */
    using UpdateFailedCallbackHandle = UpdateFailedCallback*;

    FirmwareUpdateProgressObserver();
    ~FirmwareUpdateProgressObserver();
    FirmwareUpdateProgressObserver(const FirmwareUpdateProgressObserver& other) = delete;
    FirmwareUpdateProgressObserver& operator=(const FirmwareUpdateProgressObserver& other) = delete;
    FirmwareUpdateProgressObserver(FirmwareUpdateProgressObserver&& other) = delete;
    FirmwareUpdateProgressObserver& operator=(FirmwareUpdateProgressObserver&& other) = delete;

    /*!
     * \brief Registers a callback for signaling a started update.
     *
     * \param[in] callback The callback to call if a update is started.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    UpdateStartedCallbackHandle RegisterUpdateStartedCallback(UpdateStartedCallback callback);
    /*!
     * \brief Unregisters an update started callback.
     *
     * This function unregisters an update started callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterUpdateStartedCallback(UpdateStartedCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a started update step.
     *
     * \param[in] callback The callback to call if a update step is started.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    UpdateStepStartedCallbackHandle RegisterUpdateStepStartedCallback(UpdateStepStartedCallback callback);
    /*!
     * \brief Unregisters an update step started callback.
     *
     * This function unregisters an update step started callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterUpdateStepStartedCallback(UpdateStepStartedCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling progress in an update step.
     *
     * \param[in] callback The callback to call if progress is made in a update step.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    UpdateStepProgressChangedCallbackHandle RegisterUpdateStepProgressChangedCallback(
        UpdateStepProgressChangedCallback callback);
    /*!
     * \brief Unregisters an update step progress changed callback.
     *
     * This function unregisters an update step progress changed callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterUpdateStepProgressChangedCallback(UpdateStepProgressChangedCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a finished update step.
     *
     * \param[in] callback The callback to call if a update step is finished.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    UpdateStepFinishedCallbackHandle RegisterUpdateStepFinishedCallback(UpdateStepFinishedCallback callback);
    /*!
     * \brief Unregisters an update step finished callback.
     *
     * This function unregisters an update step finished callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterUpdateStepFinishedCallback(UpdateStepFinishedCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a finished update.
     *
     * \param[in] callback The callback to call if a update is finished.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    UpdateFinishedCallbackHandle RegisterUpdateFinishedCallback(UpdateFinishedCallback callback);
    /*!
     * \brief Unregisters an update finished callback.
     *
     * This function unregisters an update finished callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterUpdateFinishedCallback(UpdateFinishedCallbackHandle callbackHandle);
    /*!
     * \brief Registers a callback for signaling a failed update.
     *
     * \param[in] callback The callback to call if a update has failed.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    UpdateFailedCallbackHandle RegisterUpdateFailedCallback(UpdateFailedCallback callback);
    /*!
     * \brief Unregisters an update failed callback.
     *
     * This function unregisters an update failed callback by taking its handle.
     *
     * \param[in] callbackHandle The handle of the callback to unregister.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void UnregisterUpdateFailedCallback(UpdateFailedCallbackHandle callbackHandle);

private:
    static void PEAK_CALL_CONV FirmwareUpdateStartedCallbackCWrapper(
        PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE updateInformation, uint32_t estimatedDuration_ms, void* context);
    static void PEAK_CALL_CONV FirmwareUpdateStepStartedCallbackCWrapper(PEAK_FIRMWARE_UPDATE_STEP updateStep,
        uint32_t estimatedDuration_ms, const char* description, size_t descriptionSize, void* context);
    static void PEAK_CALL_CONV FirmwareUpdateStepProgressChangedCallbackCWrapper(
        PEAK_FIRMWARE_UPDATE_STEP updateStep, double progressPercentage, void* context);
    static void PEAK_CALL_CONV FirmwareUpdateStepFinishedCallbackCWrapper(
        PEAK_FIRMWARE_UPDATE_STEP updateStep, void* context);
    static void PEAK_CALL_CONV FirmwareUpdateFinishedCallbackCWrapper(void* context);
    static void PEAK_CALL_CONV FirmwareUpdateFailedCallbackCWrapper(
        const char* errorDescription, size_t errorDescriptionSize, void* context);

    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE m_backendHandle;

    std::unique_ptr<TCallbackManager<PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE, UpdateStartedCallback>>
        m_updateStartedCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE, UpdateStepStartedCallback>>
        m_updateStepStartedCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE,
        UpdateStepProgressChangedCallback>>
        m_updateStepProgressChangedCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE, UpdateStepFinishedCallback>>
        m_updateStepFinishedCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE, UpdateFinishedCallback>>
        m_updateFinishedCallbackManager;
    std::unique_ptr<TCallbackManager<PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE, UpdateFailedCallback>>
        m_updateFailedCallbackManager;

    friend class FirmwareUpdater;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::string ToString(FirmwareUpdateStep entry)
{
    std::string entryString;

    if (entry == FirmwareUpdateStep::CheckPreconditions)
    {
        entryString = "CheckPreconditions";
    }
    else if (entry == FirmwareUpdateStep::AcquireUpdateData)
    {
        entryString = "AcquireUpdateData";
    }
    else if (entry == FirmwareUpdateStep::WriteFeature)
    {
        entryString = "WriteFeature";
    }
    else if (entry == FirmwareUpdateStep::ExecuteFeature)
    {
        entryString = "ExecuteFeature";
    }
    else if (entry == FirmwareUpdateStep::AssertFeature)
    {
        entryString = "AssertFeature";
    }
    else if (entry == FirmwareUpdateStep::UploadFile)
    {
        entryString = "UploadFile";
    }
    else if (entry == FirmwareUpdateStep::ResetDevice)
    {
        entryString = "ResetDevice";
    }

    return entryString;
}

inline FirmwareUpdateProgressObserver::FirmwareUpdateProgressObserver()
    : m_backendHandle(QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE>(
        [](PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE* firmwareUpdateProgressObserverHandle) {
            return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_Construct(
                firmwareUpdateProgressObserverHandle);
        }))
    , m_updateStartedCallbackManager()
    , m_updateStepStartedCallbackManager()
    , m_updateStepProgressChangedCallbackManager()
    , m_updateStepFinishedCallbackManager()
    , m_updateFinishedCallbackManager()
    , m_updateFailedCallbackManager()
{
    m_updateStartedCallbackManager = std::make_unique<
        TCallbackManager<PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE, UpdateStartedCallback>>(
        [&](void* callbackContext) {
            return QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE* firmwareUpdateStartedCallbackHandle) {
                    return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback(
                        m_backendHandle, FirmwareUpdateStartedCallbackCWrapper, callbackContext,
                        firmwareUpdateStartedCallbackHandle);
                });
        },
        [&](PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE callbackHandle) {
            CallAndCheckCInterfaceFunction([&] {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback(
                    m_backendHandle, callbackHandle);
            });
        });
    m_updateStepStartedCallbackManager = std::make_unique<
        TCallbackManager<PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE, UpdateStepStartedCallback>>(
        [&](void* callbackContext) {
            return QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE* firmwareUpdateStepStartedCallbackHandle) {
                    return PEAK_C_ABI_PREFIX
                        PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback(m_backendHandle,
                            FirmwareUpdateStepStartedCallbackCWrapper, callbackContext,
                            firmwareUpdateStepStartedCallbackHandle);
                });
        },
        [&](PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE callbackHandle) {
            CallAndCheckCInterfaceFunction([&] {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback(
                    m_backendHandle, callbackHandle);
            });
        });
    m_updateStepProgressChangedCallbackManager = std::make_unique<TCallbackManager<
        PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE, UpdateStepProgressChangedCallback>>(
        [&](void* callbackContext) {
            return QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE*
                        firmwareUpdateStepProgressChangedCallbackHandle) {
                    return PEAK_C_ABI_PREFIX
                        PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback(
                            m_backendHandle, FirmwareUpdateStepProgressChangedCallbackCWrapper, callbackContext,
                            firmwareUpdateStepProgressChangedCallbackHandle);
                });
        },
        [&](PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE callbackHandle) {
            CallAndCheckCInterfaceFunction([&] {
                return PEAK_C_ABI_PREFIX
                    PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback(
                        m_backendHandle, callbackHandle);
            });
        });
    m_updateStepFinishedCallbackManager = std::make_unique<
        TCallbackManager<PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE, UpdateStepFinishedCallback>>(
        [&](void* callbackContext) {
            return QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE* firmwareUpdateStepFinishedCallbackHandle) {
                    return PEAK_C_ABI_PREFIX
                        PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback(m_backendHandle,
                            FirmwareUpdateStepFinishedCallbackCWrapper, callbackContext,
                            firmwareUpdateStepFinishedCallbackHandle);
                });
        },
        [&](PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE callbackHandle) {
            CallAndCheckCInterfaceFunction([&] {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback(
                    m_backendHandle, callbackHandle);
            });
        });
    m_updateFinishedCallbackManager = std::make_unique<
        TCallbackManager<PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE, UpdateFinishedCallback>>(
        [&](void* callbackContext) {
            return QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE* firmwareUpdateFinishedCallbackHandle) {
                    return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback(
                        m_backendHandle, FirmwareUpdateFinishedCallbackCWrapper, callbackContext,
                        firmwareUpdateFinishedCallbackHandle);
                });
        },
        [&](PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE callbackHandle) {
            CallAndCheckCInterfaceFunction([&] {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback(
                    m_backendHandle, callbackHandle);
            });
        });
    m_updateFailedCallbackManager =
        std::make_unique<TCallbackManager<PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE, UpdateFailedCallback>>(
            [&](void* callbackContext) {
                return QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE>(
                    [&](PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE* firmwareUpdateFailedCallbackHandle) {
                        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback(
                            m_backendHandle, FirmwareUpdateFailedCallbackCWrapper, callbackContext,
                            firmwareUpdateFailedCallbackHandle);
                    });
            },
            [&](PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE callbackHandle) {
                CallAndCheckCInterfaceFunction([&] {
                    return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback(
                        m_backendHandle, callbackHandle);
                });
            });
}

inline FirmwareUpdateProgressObserver::~FirmwareUpdateProgressObserver()
{
    try
    {
        m_updateStartedCallbackManager->UnregisterAllCallbacks();
        m_updateStepStartedCallbackManager->UnregisterAllCallbacks();
        m_updateStepProgressChangedCallbackManager->UnregisterAllCallbacks();
        m_updateStepFinishedCallbackManager->UnregisterAllCallbacks();
        m_updateFinishedCallbackManager->UnregisterAllCallbacks();
        m_updateFailedCallbackManager->UnregisterAllCallbacks();
    }
    catch (const Exception&)
    {}

    (void)PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateProgressObserver_Destruct(m_backendHandle);
}

inline FirmwareUpdateProgressObserver::UpdateStartedCallbackHandle FirmwareUpdateProgressObserver::
    RegisterUpdateStartedCallback(FirmwareUpdateProgressObserver::UpdateStartedCallback callback)
{
    return reinterpret_cast<UpdateStartedCallbackHandle>(m_updateStartedCallbackManager->RegisterCallback(callback));
}

inline void FirmwareUpdateProgressObserver::UnregisterUpdateStartedCallback(
    FirmwareUpdateProgressObserver::UpdateStartedCallbackHandle callbackHandle)
{
    m_updateStartedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE>(callbackHandle));
}

inline FirmwareUpdateProgressObserver::UpdateStepStartedCallbackHandle FirmwareUpdateProgressObserver::
    RegisterUpdateStepStartedCallback(FirmwareUpdateProgressObserver::UpdateStepStartedCallback callback)
{
    return reinterpret_cast<UpdateStepStartedCallbackHandle>(
        m_updateStepStartedCallbackManager->RegisterCallback(callback));
}

inline void FirmwareUpdateProgressObserver::UnregisterUpdateStepStartedCallback(
    FirmwareUpdateProgressObserver::UpdateStepStartedCallbackHandle callbackHandle)
{
    m_updateStepStartedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE>(callbackHandle));
}

inline FirmwareUpdateProgressObserver::UpdateStepProgressChangedCallbackHandle FirmwareUpdateProgressObserver::
    RegisterUpdateStepProgressChangedCallback(
        FirmwareUpdateProgressObserver::UpdateStepProgressChangedCallback callback)
{
    return reinterpret_cast<UpdateStepProgressChangedCallbackHandle>(
        m_updateStepProgressChangedCallbackManager->RegisterCallback(callback));
}

inline void FirmwareUpdateProgressObserver::UnregisterUpdateStepProgressChangedCallback(
    FirmwareUpdateProgressObserver::UpdateStepProgressChangedCallbackHandle callbackHandle)
{
    m_updateStepProgressChangedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE>(callbackHandle));
}

inline FirmwareUpdateProgressObserver::UpdateStepFinishedCallbackHandle FirmwareUpdateProgressObserver::
    RegisterUpdateStepFinishedCallback(FirmwareUpdateProgressObserver::UpdateStepFinishedCallback callback)
{
    return reinterpret_cast<UpdateStepFinishedCallbackHandle>(
        m_updateStepFinishedCallbackManager->RegisterCallback(callback));
}

inline void FirmwareUpdateProgressObserver::UnregisterUpdateStepFinishedCallback(
    FirmwareUpdateProgressObserver::UpdateStepFinishedCallbackHandle callbackHandle)
{
    m_updateStepFinishedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE>(callbackHandle));
}

inline FirmwareUpdateProgressObserver::UpdateFinishedCallbackHandle FirmwareUpdateProgressObserver::
    RegisterUpdateFinishedCallback(FirmwareUpdateProgressObserver::UpdateFinishedCallback callback)
{
    return reinterpret_cast<UpdateFinishedCallbackHandle>(m_updateFinishedCallbackManager->RegisterCallback(callback));
}

inline void FirmwareUpdateProgressObserver::UnregisterUpdateFinishedCallback(
    FirmwareUpdateProgressObserver::UpdateFinishedCallbackHandle callbackHandle)
{
    m_updateFinishedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE>(callbackHandle));
}

inline FirmwareUpdateProgressObserver::UpdateFailedCallbackHandle FirmwareUpdateProgressObserver::
    RegisterUpdateFailedCallback(FirmwareUpdateProgressObserver::UpdateFailedCallback callback)
{
    return reinterpret_cast<UpdateFailedCallbackHandle>(m_updateFailedCallbackManager->RegisterCallback(callback));
}

inline void FirmwareUpdateProgressObserver::UnregisterUpdateFailedCallback(
    FirmwareUpdateProgressObserver::UpdateFailedCallbackHandle callbackHandle)
{
    m_updateFailedCallbackManager->UnregisterCallback(
        reinterpret_cast<PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE>(callbackHandle));
}

inline void PEAK_CALL_CONV FirmwareUpdateProgressObserver::FirmwareUpdateStartedCallbackCWrapper(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE updateInformation, uint32_t estimatedDuration_ms, void* context)
{
    auto callback = static_cast<FirmwareUpdateProgressObserver::UpdateStartedCallback*>(context);

    callback->operator()(
        std::make_shared<ClassCreator<FirmwareUpdateInformation>>(updateInformation), estimatedDuration_ms);
}

inline void PEAK_CALL_CONV FirmwareUpdateProgressObserver::FirmwareUpdateStepStartedCallbackCWrapper(
    PEAK_FIRMWARE_UPDATE_STEP updateStep, uint32_t estimatedDuration_ms, const char* description,
    size_t descriptionSize, void* context)
{
    auto callback = static_cast<FirmwareUpdateProgressObserver::UpdateStepStartedCallback*>(context);

    callback->operator()(static_cast<FirmwareUpdateStep>(updateStep), estimatedDuration_ms,
        std::string(description, descriptionSize - 1));
}

inline void PEAK_CALL_CONV FirmwareUpdateProgressObserver::FirmwareUpdateStepProgressChangedCallbackCWrapper(
    PEAK_FIRMWARE_UPDATE_STEP updateStep, double progressPercentage, void* context)
{
    auto callback = static_cast<FirmwareUpdateProgressObserver::UpdateStepProgressChangedCallback*>(context);

    callback->operator()(static_cast<FirmwareUpdateStep>(updateStep), progressPercentage);
}

inline void PEAK_CALL_CONV FirmwareUpdateProgressObserver::FirmwareUpdateStepFinishedCallbackCWrapper(
    PEAK_FIRMWARE_UPDATE_STEP updateStep, void* context)
{
    auto callback = static_cast<FirmwareUpdateProgressObserver::UpdateStepFinishedCallback*>(context);

    callback->operator()(static_cast<FirmwareUpdateStep>(updateStep));
}

inline void PEAK_CALL_CONV FirmwareUpdateProgressObserver::FirmwareUpdateFinishedCallbackCWrapper(void* context)
{
    auto callback = static_cast<FirmwareUpdateProgressObserver::UpdateFinishedCallback*>(context);

    callback->operator()();
}

inline void PEAK_CALL_CONV FirmwareUpdateProgressObserver::FirmwareUpdateFailedCallbackCWrapper(
    const char* errorDescription, size_t errorDescriptionSize, void* context)
{
    auto callback = static_cast<FirmwareUpdateProgressObserver::UpdateFailedCallback*>(context);

    callback->operator()(std::string(errorDescription, errorDescriptionSize - 1));
}

} /* namespace core */
} /* namespace peak */
