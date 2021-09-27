/*!
 * \file    peak_firmware_updater.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/device/peak_device_descriptor.hpp>
#include <peak/device/peak_firmware_update_information.hpp>
#include <peak/device/peak_firmware_update_progress_observer.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <memory>
#include <string>
#include <vector>


namespace peak
{
namespace core
{
using namespace std::chrono_literals;

/*!
 * \brief Allows to update the firmware of a device.
 *
 * To update the device firmware, call CollectFirmwareUpdateInformation() with a *.guf file and the device to update,
 * select one of the firmware updates in the list, then pass it to UpdateDevice() to exectue the update.
 *
 * To observe the update progress, pass a FirmwareUpdateProgressObserver to the UpdateDevice(). This is optional, but
 * very helpful, since firmware updates can take several minutes.
 *
 * \note The DeviceDescriptor passed to CollectFirmwareUpdateInformation() and UpdateDevice() will be opened in
 *       Exclusive mode. Therefore, if the Device is opened in any other application, both methods will fail with an
 *       InternalErrorException.
 *
 */
class FirmwareUpdater final
{
public:
    FirmwareUpdater();
    ~FirmwareUpdater();
    FirmwareUpdater(const FirmwareUpdater& other) = delete;
    FirmwareUpdater& operator=(const FirmwareUpdater& other) = delete;
    FirmwareUpdater(FirmwareUpdater&& other) = delete;
    FirmwareUpdater& operator=(FirmwareUpdater&& other) = delete;

    /*!
     * \brief Collects all firmware update information of a given *.guf file.
     *
     * \param[in] gufPath The path of the *.guf file containing the firmware update.
     *
     * \return A list of firmware update information.
     *
     * \since 1.2
     *
     * \throws InvalidArgumentException One of the submitted arguments is outside the valid range or is not supported.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<FirmwareUpdateInformation>> CollectFirmwareUpdateInformation(
        const std::string& gufPath) const;
    /*!
     * \brief Collects the firmware update information of a given *.guf file fitting to a given device.
     *
     * \param[in] gufPath The path of the *.guf file containing the firmware update.
     * \param[in] device The device to update.
     *
     * \return A list of firmware update information fitting to the given device
     *
     * \since 1.0
     *
     * \throws InvalidArgumentException One of the submitted arguments is outside the valid range or is not supported.
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<std::shared_ptr<FirmwareUpdateInformation>> CollectFirmwareUpdateInformation(
        const std::string& gufPath, const std::shared_ptr<DeviceDescriptor>& device) const;
    /*!
     * \brief Updates a given device by using a given firmware update information.
     *
     * This is a blocking call, i.e. it only returns once the update is done. Depending on the device, the update may
     * take several minutes. To watch the progress during the update, pass a FirmwareUpdateProgressObserver.
     *
     * \param[in] device The device to update.
     * \param[in] updateInformation The firmware update information to update the device.
     * \param[in] progressObserver The progress observer to observe the update process. It is optional.
     * \param[in] deviceResetDiscoveryTimeout Time to wait for a device to reboot during the update.
     *
     * \note The DeviceDescriptor passed to this function will be invalid after the update. Update the
     *       DeviceManager/Interface and retrieve a new DeviceDescriptor.
     *
     * \since 1.0
     * \since 1.2 Added deviceResetDiscoveryTimeout parameter.
     *
     * \throws InvalidArgumentException One of the submitted arguments is outside the valid range or is not supported.
     * \throws InternalErrorException An internal error has occurred.
     * \throws TimeoutException The deviceResetDiscoveryTimeout was exceeded.
     */
    void UpdateDevice(const std::shared_ptr<DeviceDescriptor>& device,
        const std::shared_ptr<FirmwareUpdateInformation>& updateInformation,
        const FirmwareUpdateProgressObserver* progressObserver = nullptr,
        Timeout deviceResetDiscoveryTimeout = Timeout(60000ms));

private:
    PEAK_FIRMWARE_UPDATER_HANDLE m_backendHandle;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline FirmwareUpdater::FirmwareUpdater()
    : m_backendHandle(QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATER_HANDLE>(
        [](PEAK_FIRMWARE_UPDATER_HANDLE* firmwareUpdaterHandle) {
            return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_Construct(firmwareUpdaterHandle);
        }))
{}

inline FirmwareUpdater::~FirmwareUpdater()
{
    (void)PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_Destruct(m_backendHandle);
}

inline std::vector<std::shared_ptr<FirmwareUpdateInformation>> FirmwareUpdater::CollectFirmwareUpdateInformation(
    const std::string& gufPath) const
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation(
            m_backendHandle, gufPath.c_str(), gufPath.size() + 1);
    });

    auto numFirmwareUpdateInformation = QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* _numFirmwareUpdateInformation) {
            return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation(
                m_backendHandle, _numFirmwareUpdateInformation);
        });

    std::vector<std::shared_ptr<FirmwareUpdateInformation>> firmwareUpdateInformation;
    for (size_t x = 0; x < numFirmwareUpdateInformation; ++x)
    {
        auto firmwareUpdateInformationHandle =
            QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE* _firmwareUpdateInformationHandle) {
                    return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_GetFirmwareUpdateInformation(
                        m_backendHandle, x, _firmwareUpdateInformationHandle);
                });

        firmwareUpdateInformation.emplace_back(
            std::make_shared<ClassCreator<FirmwareUpdateInformation>>(firmwareUpdateInformationHandle));
    }

    return firmwareUpdateInformation;
}

inline std::vector<std::shared_ptr<FirmwareUpdateInformation>> FirmwareUpdater::CollectFirmwareUpdateInformation(
    const std::string& gufPath, const std::shared_ptr<DeviceDescriptor>& device) const
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation(
            m_backendHandle, gufPath.c_str(), gufPath.size() + 1, device->m_backendHandle);
    });

    auto numFirmwareUpdateInformation = QueryNumericFromCInterfaceFunction<size_t>(
        [&](size_t* _numFirmwareUpdateInformation) {
            return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation(
                m_backendHandle, _numFirmwareUpdateInformation);
        });

    std::vector<std::shared_ptr<FirmwareUpdateInformation>> firmwareUpdateInformation;
    for (size_t x = 0; x < numFirmwareUpdateInformation; ++x)
    {
        auto firmwareUpdateInformationHandle =
            QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE>(
                [&](PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE* _firmwareUpdateInformationHandle) {
                    return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_GetFirmwareUpdateInformation(
                        m_backendHandle, x, _firmwareUpdateInformationHandle);
                });

        firmwareUpdateInformation.emplace_back(
            std::make_shared<ClassCreator<FirmwareUpdateInformation>>(firmwareUpdateInformationHandle));
    }

    return firmwareUpdateInformation;
}

inline void FirmwareUpdater::UpdateDevice(const std::shared_ptr<DeviceDescriptor>& device,
    const std::shared_ptr<FirmwareUpdateInformation>& updateInformation,
    const FirmwareUpdateProgressObserver* progressObserver /* = nullptr */,
    Timeout deviceResetDiscoveryTimeout /* = Timeout{ 60000ms }*/)
{
    CallAndCheckCInterfaceFunction([&] {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout(m_backendHandle,
            device->m_backendHandle, updateInformation->m_backendHandle,
            progressObserver ? progressObserver->m_backendHandle : nullptr, deviceResetDiscoveryTimeout);
    });
}

} /* namespace core */
} /* namespace peak */
