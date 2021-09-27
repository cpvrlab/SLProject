/*!
 * \file    peak_firmware_update_information.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <string>


namespace peak
{
namespace core
{

/*!
 * \brief Specifies if the device persists user sets and/or sequencer sets during the update.
 */
enum class FirmwareUpdatePersistence
{
    None, //!< All sets are reset to default values during the update.
    Full //!< The device guarantees, that all sets are persisted during the update.
};

/*!
 * \brief The style of the version value.
 *
 * This is needed to actually interpret and sort the versions. Using this information an update software is able
 * to inform the user if an update would actually be a downgrade or if it was already applied to the device.
 */
enum class FirmwareUpdateVersionStyle
{
    /*!
     * The version consists of any number of parts separated by dots. If a part consists of decimal characters only,
     * it is compared numerically, otherwise it is compared using strcmp(). This leads to the following ordering:
     * 1.1.a < 1.10.a < 1.10.b <1.10.b.a
     */
    Dotted,
    /*!
     * The style as specified in 'Semantic Versioning 2.0.0' ( http://semver.org/ ).
     */
    Semantic
};

/*!
 * \brief Represents a single firmware update information.
 *
 * This class allows to query information from the *.guf file about a single firmware update information.
 *
 * See GenICam FWUpdate Standard.
 *
 */
class FirmwareUpdateInformation
{
public:
    FirmwareUpdateInformation() = delete;
    ~FirmwareUpdateInformation() = default;
    FirmwareUpdateInformation(const FirmwareUpdateInformation& other) = delete;
    FirmwareUpdateInformation& operator=(const FirmwareUpdateInformation& other) = delete;
    FirmwareUpdateInformation(FirmwareUpdateInformation&& other) = delete;
    FirmwareUpdateInformation& operator=(FirmwareUpdateInformation&& other) = delete;

    /*!
     * \brief Checks whether the information held are valid.
     *
     * \return True, if the information held are valid.
     * \return False otherwise.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    bool IsValid() const;
    /*!
     * \brief Returns the file name of the package this firmware update information belongs to.
     *
     * \return File name
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string FileName() const;
    /*!
     * \brief Returns the description of the firmware update.
     *
     * \return Description
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Description() const;
    /*!
     * \brief Returns the version of the firmware update.
     *
     * \return Version
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string Version() const;
    /*!
     * \brief Regular expression to extract the device version from the DeviceFirmwareVersion node.
     *
     * The first matched group is used as result. This is needed for devices which encode more information than just
     * the firmware version inside the DeviceFirmwareVersion node. The default value is: ^(.*)$
     *
     * \return Version extraction pattern
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string VersionExtractionPattern() const;
    /*!
     * \brief The style of the Version() value.
     *
     * This is needed to actually interpret and sort the versions. Using this information an update software is
     * able to inform the user if an update would actually be a downgrade or if it was already applied to the device.
     *
     * \return Version style
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    FirmwareUpdateVersionStyle VersionStyle() const;
    /*!
     * \brief Release notes of the firmware update.
     *
     * \return Release notes
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ReleaseNotes() const;
    /*!
     * \brief A link to a webpage with more release notes.
     *
     * This webpage can contain addition details not contained in ReleaseNotes().
     *
     * \return URL where the release notes of the firmware update can be found
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::string ReleaseNotesURL() const;
    /*!
     * \brief Specifies if the device persists user sets during the update.
     *
     * \return User set persistence
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    FirmwareUpdatePersistence UserSetPersistence() const;
    /*!
     * \brief Specifies if the device persists sequencer sets during the update.
     *
     * \return Sequencer set persistence
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    FirmwareUpdatePersistence SequencerSetPersistence() const;

private:
    friend ClassCreator<FirmwareUpdateInformation>;
    explicit FirmwareUpdateInformation(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle);
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE m_backendHandle;

    friend class FirmwareUpdater;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline FirmwareUpdateInformation::FirmwareUpdateInformation(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle)
    : m_backendHandle(firmwareUpdateInformationHandle)
{}

inline bool FirmwareUpdateInformation::IsValid() const
{
    return QueryNumericFromCInterfaceFunction<PEAK_BOOL8>([&](PEAK_BOOL8* isValid) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetIsValid(m_backendHandle, isValid);
    }) > 0;
}

inline std::string FirmwareUpdateInformation::FileName() const
{
    return QueryStringFromCInterfaceFunction([&](char* fileName, size_t* fileNameSize) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetFileName(
            m_backendHandle, fileName, fileNameSize);
    });
}

inline std::string FirmwareUpdateInformation::Description() const
{
    return QueryStringFromCInterfaceFunction([&](char* description, size_t* descriptionSize) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetDescription(
            m_backendHandle, description, descriptionSize);
    });
}

inline std::string FirmwareUpdateInformation::Version() const
{
    return QueryStringFromCInterfaceFunction([&](char* version, size_t* versionSize) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetVersion(m_backendHandle, version, versionSize);
    });
}

inline std::string FirmwareUpdateInformation::VersionExtractionPattern() const
{
    return QueryStringFromCInterfaceFunction([&](char* versionExtractionPattern, size_t* versionExtractionPatternSize) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern(
            m_backendHandle, versionExtractionPattern, versionExtractionPatternSize);
    });
}

inline FirmwareUpdateVersionStyle FirmwareUpdateInformation::VersionStyle() const
{
    return static_cast<FirmwareUpdateVersionStyle>(
        QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_VERSION_STYLE>(
            [&](PEAK_FIRMWARE_UPDATE_VERSION_STYLE* versionStyle) {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetVersionStyle(
                    m_backendHandle, versionStyle);
            }));
}

inline std::string FirmwareUpdateInformation::ReleaseNotes() const
{
    return QueryStringFromCInterfaceFunction([&](char* releaseNotes, size_t* releaseNotesSize) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetReleaseNotes(
            m_backendHandle, releaseNotes, releaseNotesSize);
    });
}

inline std::string FirmwareUpdateInformation::ReleaseNotesURL() const
{
    return QueryStringFromCInterfaceFunction([&](char* releaseNotesUrl, size_t* releaseNotesUrlSize) {
        return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetReleaseNotesURL(
            m_backendHandle, releaseNotesUrl, releaseNotesUrlSize);
    });
}

inline FirmwareUpdatePersistence FirmwareUpdateInformation::UserSetPersistence() const
{
    return static_cast<FirmwareUpdatePersistence>(
        QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_PERSISTENCE>(
            [&](PEAK_FIRMWARE_UPDATE_PERSISTENCE* userSetPersistence) {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetUserSetPersistence(
                    m_backendHandle, userSetPersistence);
            }));
}

inline FirmwareUpdatePersistence FirmwareUpdateInformation::SequencerSetPersistence() const
{
    return static_cast<FirmwareUpdatePersistence>(
        QueryNumericFromCInterfaceFunction<PEAK_FIRMWARE_UPDATE_PERSISTENCE>(
            [&](PEAK_FIRMWARE_UPDATE_PERSISTENCE* sequencerSetPersistence) {
                return PEAK_C_ABI_PREFIX PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence(
                    m_backendHandle, sequencerSetPersistence);
            }));
}

inline std::string ToString(FirmwareUpdatePersistence entry)
{
    std::string entryString;

    if (entry == FirmwareUpdatePersistence::None)
    {
        entryString = "None";
    }
    else if (entry == FirmwareUpdatePersistence::Full)
    {
        entryString = "Full";
    }

    return entryString;
}

inline std::string ToString(FirmwareUpdateVersionStyle entry)
{
    std::string entryString;

    if (entry == FirmwareUpdateVersionStyle::Dotted)
    {
        entryString = "Dotted";
    }
    else if (entry == FirmwareUpdateVersionStyle::Semantic)
    {
        entryString = "Semantic";
    }

    return entryString;
}

} /* namespace core */
} /* namespace peak */
