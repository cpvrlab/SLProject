/*!
 * \file    peak_backend.h
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_dll_defines.h>

#ifdef __cplusplus
#    include <cstddef>
#    include <cstdint>

extern "C" {
#else
#    include <stddef.h>
#    include <stdint.h>
#endif

typedef int8_t PEAK_BOOL8;

/*! The enum holding the possible acquisition start modes. */
enum PEAK_ACQUISITION_START_MODE_t
{
    PEAK_ACQUISITION_START_MODE_DEFAULT = 0,

    PEAK_ACQUISITION_START_MODE_CUSTOM = 1000
};
typedef int32_t PEAK_ACQUISITION_START_MODE;

/*! The enum holding the possible acquisition stop modes. */
enum PEAK_ACQUISITION_STOP_MODE_t
{
    PEAK_ACQUISITION_STOP_MODE_DEFAULT = 0,
    PEAK_ACQUISITION_STOP_MODE_KILL,

    PEAK_ACQUISITION_STOP_MODE_CUSTOM = 1000
};
typedef int32_t PEAK_ACQUISITION_STOP_MODE;

/*! The enum holding the possible buffer part types. */
enum PEAK_BUFFER_PART_TYPE_t
{
    PEAK_BUFFER_PART_TYPE_UNKNOWN = 0,
    PEAK_BUFFER_PART_TYPE_IMAGE_2D,
    PEAK_BUFFER_PART_TYPE_PLANE_BI_PLANAR_2D,
    PEAK_BUFFER_PART_TYPE_PLANE_TRI_PLANAR_2D,
    PEAK_BUFFER_PART_TYPE_PLANE_QUAD_PLANAR_2D,
    PEAK_BUFFER_PART_TYPE_IMAGE_3D,
    PEAK_BUFFER_PART_TYPE_PLANE_BI_PLANAR_3D,
    PEAK_BUFFER_PART_TYPE_PLANE_TRI_PLANAR_3D,
    PEAK_BUFFER_PART_TYPE_PLANE_QUAD_PLANAR_3D,
    PEAK_BUFFER_PART_TYPE_CONFIDENCE_MAP,

    PEAK_BUFFER_PART_TYPE_CUSTOM = 1000
};
typedef int32_t PEAK_BUFFER_PART_TYPE;

/*! The enum holding the possible buffer payload types. */
enum PEAK_BUFFER_PAYLOAD_TYPE_t
{
    PEAK_BUFFER_PAYLOAD_TYPE_UNKNOWN = 0,
    PEAK_BUFFER_PAYLOAD_TYPE_IMAGE,
    PEAK_BUFFER_PAYLOAD_TYPE_RAW_DATA,
    PEAK_BUFFER_PAYLOAD_TYPE_FILE,
    PEAK_BUFFER_PAYLOAD_TYPE_CHUNK,
    PEAK_BUFFER_PAYLOAD_TYPE_JPEG,
    PEAK_BUFFER_PAYLOAD_TYPE_JPEG_2000,
    PEAK_BUFFER_PAYLOAD_TYPE_H264,
    PEAK_BUFFER_PAYLOAD_TYPE_CHUNK_ONLY,
    PEAK_BUFFER_PAYLOAD_TYPE_DEVICE_SPECIFIC,
    PEAK_BUFFER_PAYLOAD_TYPE_MULTI_PART,

    PEAK_BUFFER_PAYLOAD_TYPE_CUSTOM = 1000
};
typedef int32_t PEAK_BUFFER_PAYLOAD_TYPE;

/*! The enum holding the possible character encodings. */
enum PEAK_CHARACTER_ENCODING_t
{
    PEAK_CHARACTER_ENCODING_ASCII = 0,
    PEAK_CHARACTER_ENCODING_UTF8
};
typedef int32_t PEAK_CHARACTER_ENCODING;

/*! The enum holding the possible data stream flush modes. */
enum PEAK_DATA_STREAM_FLUSH_MODE_t
{
    PEAK_DATA_STREAM_FLUSH_MODE_INPUT_POOL_TO_OUTPUT_QUEUE = 0,
    PEAK_DATA_STREAM_FLUSH_MODE_DISCARD_OUTPUT_QUEUE,
    PEAK_DATA_STREAM_FLUSH_MODE_ALL_TO_INPUT_POOL,
    PEAK_DATA_STREAM_FLUSH_MODE_UNQUEUED_TO_INPUT_POOL,
    PEAK_DATA_STREAM_FLUSH_MODE_DISCARD_ALL,

    PEAK_DATA_STREAM_FLUSH_MODE_CUSTOM = 1000
};
typedef int32_t PEAK_DATA_STREAM_FLUSH_MODE;

/*! The enum holding the possible device access status. */
enum PEAK_DEVICE_ACCESS_STATUS_t
{
    PEAK_DEVICE_ACCESS_STATUS_READ_WRITE = 1,
    PEAK_DEVICE_ACCESS_STATUS_READ_ONLY,
    PEAK_DEVICE_ACCESS_STATUS_NO_ACCESS,
    PEAK_DEVICE_ACCESS_STATUS_BUSY,
    PEAK_DEVICE_ACCESS_STATUS_OPEN_READ_WRITE,
    PEAK_DEVICE_ACCESS_STATUS_OPEN_READ_ONLY,

    PEAK_DEVICE_ACCESS_STATUS_CUSTOM = 1000
};
typedef int32_t PEAK_DEVICE_ACCESS_STATUS;

/*! The enum holding the possible device access types. */
enum PEAK_DEVICE_ACCESS_TYPE_t
{
    PEAK_DEVICE_ACCESS_TYPE_READ_ONLY = 2,
    PEAK_DEVICE_ACCESS_TYPE_CONTROL,
    PEAK_DEVICE_ACCESS_TYPE_EXCLUSIVE,

    PEAK_DEVICE_ACCESS_TYPE_CUSTOM = 1000
};
typedef int32_t PEAK_DEVICE_ACCESS_TYPE;

/*! The enum holding the possible device information roles. */
enum PEAK_DEVICE_INFORMATION_ROLE_t
{
    PEAK_DEVICE_INFORMATION_ROLE_ID = 0,
    PEAK_DEVICE_INFORMATION_ROLE_VENDOR_NAME,
    PEAK_DEVICE_INFORMATION_ROLE_MODEL_NAME,
    PEAK_DEVICE_INFORMATION_ROLE_TL_TYPE,
    PEAK_DEVICE_INFORMATION_ROLE_DISPLAY_NAME,
    PEAK_DEVICE_INFORMATION_ROLE_ACCESS_STATUS,
    PEAK_DEVICE_INFORMATION_ROLE_USER_DEFINED_NAME,
    PEAK_DEVICE_INFORMATION_ROLE_SERIAL_NUMBER,
    PEAK_DEVICE_INFORMATION_ROLE_VERSION,
    PEAK_DEVICE_INFORMATION_ROLE_TIMESTAMP_TICK_FREQUENCY,

    PEAK_DEVICE_INFORMATION_ROLE_CUSTOM = 1000
};
typedef int32_t PEAK_DEVICE_INFORMATION_ROLE;

/*! The enum holding the possible endianness types. */
enum PEAK_ENDIANNESS_t
{
    PEAK_ENDIANNESS_UNKNOWN = 0,
    PEAK_ENDIANNESS_LITTLE,
    PEAK_ENDIANNESS_BIG
};
typedef int32_t PEAK_ENDIANNESS;

/*! The enum holding the possible event types. */
enum PEAK_EVENT_TYPE_t
{
    PEAK_EVENT_TYPE_ERROR = 0,
    PEAK_EVENT_TYPE_FEATURE_INVALIDATE = 2,
    PEAK_EVENT_TYPE_FEATURE_CHANGE,
    PEAK_EVENT_TYPE_REMOTE_DEVICE,
    PEAK_EVENT_TYPE_MODULE,

    PEAK_EVENT_TYPE_FEATURE_CUSTOM = 1000
};
typedef int32_t PEAK_EVENT_TYPE;

/*! The enum holding the possible firmware update persistence types. */
enum PEAK_FIRMWARE_UPDATE_PERSISTENCE_t
{
    PEAK_FIRMWARE_UPDATE_PERSISTENCE_NONE = 0,
    PEAK_FIRMWARE_UPDATE_PERSISTENCE_FULL
};
typedef int32_t PEAK_FIRMWARE_UPDATE_PERSISTENCE;

/*! The enum holding the possible firmware update steps. */
enum PEAK_FIRMWARE_UPDATE_STEP_t
{
    PEAK_FIRMWARE_UPDATE_STEP_CHECK_PRECONDITIONS = 0,
    PEAK_FIRMWARE_UPDATE_STEP_ACQUIRE_UPDATE_DATA,
    PEAK_FIRMWARE_UPDATE_STEP_WRITE_FEATURE,
    PEAK_FIRMWARE_UPDATE_STEP_EXECUTE_FEATURE,
    PEAK_FIRMWARE_UPDATE_STEP_ASSERT_FEATURE,
    PEAK_FIRMWARE_UPDATE_STEP_UPLOAD_FILE,
    PEAK_FIRMWARE_UPDATE_STEP_RESET_DEVICE
};
typedef int32_t PEAK_FIRMWARE_UPDATE_STEP;

/*! The enum holding the possible firmware update version styles. */
enum PEAK_FIRMWARE_UPDATE_VERSION_STYLE_t
{
    PEAK_FIRMWARE_UPDATE_VERSION_STYLE_DOTTED = 0,
    PEAK_FIRMWARE_UPDATE_VERSION_STYLE_SEMANTIC
};
typedef int32_t PEAK_FIRMWARE_UPDATE_VERSION_STYLE;

/*! The enum holding the possible node access status types. */
enum PEAK_NODE_ACCESS_STATUS_t
{
    PEAK_NODE_ACCESS_STATUS_NOT_IMPLEMENTED = 0,
    PEAK_NODE_ACCESS_STATUS_NOT_AVAILABLE,
    PEAK_NODE_ACCESS_STATUS_WRITE_ONLY,
    PEAK_NODE_ACCESS_STATUS_READ_ONLY,
    PEAK_NODE_ACCESS_STATUS_READ_WRITE
};
typedef int32_t PEAK_NODE_ACCESS_STATUS;

/*! The enum holding the possible node caching modes. */
enum PEAK_NODE_CACHING_MODE_t
{
    PEAK_NODE_CACHING_MODE_NO_CACHE = 0,
    PEAK_NODE_CACHING_MODE_WRITE_THROUGH,
    PEAK_NODE_CACHING_MODE_WRITE_AROUND
};
typedef int32_t PEAK_NODE_CACHING_MODE;

/*! The enum holding the possible node cache use policies. */
enum PEAK_NODE_CACHE_USE_POLICY_t
{
    PEAK_NODE_CACHE_USE_POLICY_USE_CACHE = 0,
    PEAK_NODE_CACHE_USE_POLICY_IGNORE_CACHE
};
typedef int32_t PEAK_NODE_CACHE_USE_POLICY;

/*! The enum holding the possible node display notations. */
enum PEAK_NODE_DISPLAY_NOTATION_t
{
    PEAK_NODE_DISPLAY_NOTATION_AUTOMATIC = 0,
    PEAK_NODE_DISPLAY_NOTATION_FIXED,
    PEAK_NODE_DISPLAY_NOTATION_SCIENTIFIC
};
typedef int32_t PEAK_NODE_DISPLAY_NOTATION;

/*! The enum holding the possible node increment types. */
enum PEAK_NODE_INCREMENT_TYPE_t
{
    PEAK_NODE_INCREMENT_TYPE_NO_INCREMENT = 0,
    PEAK_NODE_INCREMENT_TYPE_FIXED_INCREMENT,
    PEAK_NODE_INCREMENT_TYPE_LIST_INCREMENT
};
typedef int32_t PEAK_NODE_INCREMENT_TYPE;

/*! The enum holding the possible node namespaces. */
enum PEAK_NODE_NAMESPACE_t
{
    PEAK_NODE_NAMESPACE_CUSTOM = 0,
    PEAK_NODE_NAMESPACE_STANDARD
};
typedef int32_t PEAK_NODE_NAMESPACE;

/*! The enum holding the possible node representations. */
enum PEAK_NODE_REPRESENTATION_t
{
    PEAK_NODE_REPRESENTATION_LINEAR = 0,
    PEAK_NODE_REPRESENTATION_LOGARITHMIC,
    PEAK_NODE_REPRESENTATION_BOOLEAN,
    PEAK_NODE_REPRESENTATION_PURE_NUMBER,
    PEAK_NODE_REPRESENTATION_HEX_NUMBER,
    PEAK_NODE_REPRESENTATION_IP4_ADDRESS,
    PEAK_NODE_REPRESENTATION_MAC_ADDRESS
};
typedef int32_t PEAK_NODE_REPRESENTATION;

/*! The enum holding the possible node types. */
enum PEAK_NODE_TYPE_t
{
    PEAK_NODE_TYPE_INTEGER = 0,
    PEAK_NODE_TYPE_BOOLEAN,
    PEAK_NODE_TYPE_COMMAND,
    PEAK_NODE_TYPE_FLOAT,
    PEAK_NODE_TYPE_STRING,
    PEAK_NODE_TYPE_REGISTER,
    PEAK_NODE_TYPE_CATEGORY,
    PEAK_NODE_TYPE_ENUMERATION,
    PEAK_NODE_TYPE_ENUMERATION_ENTRY
};
typedef int32_t PEAK_NODE_TYPE;

/*! The enum holding the possible node visibility types. */
enum PEAK_NODE_VISIBILITY_t
{
    PEAK_NODE_VISIBILITY_BEGINNER = 0,
    PEAK_NODE_VISIBILITY_EXPERT,
    PEAK_NODE_VISIBILITY_GURU,
    PEAK_NODE_VISIBILITY_INVISIBLE
};
typedef int32_t PEAK_NODE_VISIBILITY;

/*! The enum holding the possible pixel format namespaces. */
enum PEAK_PIXEL_FORMAT_NAMESPACE_t
{
    PEAK_PIXEL_FORMAT_NAMESPACE_GEV = 1,
    PEAK_PIXEL_FORMAT_NAMESPACE_IIDC,
    PEAK_PIXEL_FORMAT_NAMESPACE_PFNC_16_BIT,
    PEAK_PIXEL_FORMAT_NAMESPACE_PFNC_32_BIT,

    PEAK_PIXEL_FORMAT_NAMESPACE_CUSTOM = 1000
};
typedef int32_t PEAK_PIXEL_FORMAT_NAMESPACE;

/*! The enum holding the possible port URL scheme types. */
enum PEAK_PORT_URL_SCHEME_t
{
    PEAK_PORT_URL_SCHEME_LOCAL = 0,
    PEAK_PORT_URL_SCHEME_HTTP,
    PEAK_PORT_URL_SCHEME_FILE,

    PEAK_PORT_URL_SCHEME_CUSTOM = 1000
};
typedef int32_t PEAK_PORT_URL_SCHEME;

/*! The enum holding the possible function return codes. */
enum PEAK_RETURN_CODE_t
{
    PEAK_RETURN_CODE_SUCCESS = 0,
    PEAK_RETURN_CODE_ERROR,
    PEAK_RETURN_CODE_NOT_INITIALIZED,
    PEAK_RETURN_CODE_ABORTED,
    PEAK_RETURN_CODE_BAD_ACCESS,
    PEAK_RETURN_CODE_BAD_ALLOC,
    PEAK_RETURN_CODE_BUFFER_TOO_SMALL,
    PEAK_RETURN_CODE_INVALID_ADDRESS,
    PEAK_RETURN_CODE_INVALID_ARGUMENT,
    PEAK_RETURN_CODE_INVALID_CAST,
    PEAK_RETURN_CODE_INVALID_HANDLE,
    PEAK_RETURN_CODE_NOT_FOUND,
    PEAK_RETURN_CODE_OUT_OF_RANGE,
    PEAK_RETURN_CODE_TIMEOUT,
    PEAK_RETURN_CODE_NOT_AVAILABLE,
    PEAK_RETURN_CODE_NOT_IMPLEMENTED
};
typedef int32_t PEAK_RETURN_CODE;

/*! \brief The constant defining an infinite number of buffers. */
const uint64_t PEAK_INFINITE_NUMBER = 0xFFFFFFFFFFFFFFFFULL;

/*! The constant defining an infinite timeout. */
const uint64_t PEAK_INFINITE_TIMEOUT = 0xFFFFFFFFFFFFFFFFULL;

struct PEAK_PRODUCER_LIBRARY;
/*! The type of producer library handles. */
typedef struct PEAK_PRODUCER_LIBRARY* PEAK_PRODUCER_LIBRARY_HANDLE;

struct PEAK_SYSTEM_DESCRIPTOR;
/*! The type of system descriptor handles. */
typedef struct PEAK_SYSTEM_DESCRIPTOR* PEAK_SYSTEM_DESCRIPTOR_HANDLE;

struct PEAK_SYSTEM;
/*! The type of system handles. */
typedef struct PEAK_SYSTEM* PEAK_SYSTEM_HANDLE;

struct PEAK_INTERFACE_DESCRIPTOR;
/*! The type of interface descriptor handles. */
typedef struct PEAK_INTERFACE_DESCRIPTOR* PEAK_INTERFACE_DESCRIPTOR_HANDLE;

struct PEAK_INTERFACE;
/*! The type of interface handles. */
typedef struct PEAK_INTERFACE* PEAK_INTERFACE_HANDLE;

struct PEAK_DEVICE_DESCRIPTOR;
/*! The type of device descriptor handles. */
typedef struct PEAK_DEVICE_DESCRIPTOR* PEAK_DEVICE_DESCRIPTOR_HANDLE;

struct PEAK_DEVICE;
/*! The type of device handles. */
typedef struct PEAK_DEVICE* PEAK_DEVICE_HANDLE;

struct PEAK_REMOTE_DEVICE;
/*! The type of remote device handles. */
typedef struct PEAK_REMOTE_DEVICE* PEAK_REMOTE_DEVICE_HANDLE;

struct PEAK_DATA_STREAM_DESCRIPTOR;
/*! The type of data stream descriptor handles. */
typedef struct PEAK_DATA_STREAM_DESCRIPTOR* PEAK_DATA_STREAM_DESCRIPTOR_HANDLE;

struct PEAK_DATA_STREAM;
/*! The type of data stream handles. */
typedef struct PEAK_DATA_STREAM* PEAK_DATA_STREAM_HANDLE;

struct PEAK_BUFFER;
/*! The type of buffer handles. */
typedef struct PEAK_BUFFER* PEAK_BUFFER_HANDLE;

struct PEAK_BUFFER_CHUNK;
/*! The type of buffer chunk handles. */
typedef struct PEAK_BUFFER_CHUNK* PEAK_BUFFER_CHUNK_HANDLE;

struct PEAK_BUFFER_PART;
/*! The type of buffer part handles. */
typedef struct PEAK_BUFFER_PART* PEAK_BUFFER_PART_HANDLE;

struct PEAK_MODULE_DESCRIPTOR;
/*! The type of module descriptor handles. */
typedef struct PEAK_MODULE_DESCRIPTOR* PEAK_MODULE_DESCRIPTOR_HANDLE;

struct PEAK_MODULE;
/*! The type of module handles. */
typedef struct PEAK_MODULE* PEAK_MODULE_HANDLE;

struct PEAK_NODE_MAP;
/*! The type of node map handles. */
typedef struct PEAK_NODE_MAP* PEAK_NODE_MAP_HANDLE;

struct PEAK_NODE;
/*! The type of node handles. */
typedef struct PEAK_NODE* PEAK_NODE_HANDLE;

struct PEAK_INTEGER_NODE;
/*! The type of integer node handles. */
typedef struct PEAK_INTEGER_NODE* PEAK_INTEGER_NODE_HANDLE;

struct PEAK_BOOLEAN_NODE;
/*! The type of boolean node handles. */
typedef struct PEAK_BOOLEAN_NODE* PEAK_BOOLEAN_NODE_HANDLE;

struct PEAK_COMMAND_NODE;
/*! The type of command node handles. */
typedef struct PEAK_COMMAND_NODE* PEAK_COMMAND_NODE_HANDLE;

struct PEAK_FLOAT_NODE;
/*! The type of float node handles. */
typedef struct PEAK_FLOAT_NODE* PEAK_FLOAT_NODE_HANDLE;

struct PEAK_STRING_NODE;
/*! The type of string node handles. */
typedef struct PEAK_STRING_NODE* PEAK_STRING_NODE_HANDLE;

struct PEAK_REGISTER_NODE;
/*! The type of register node handles. */
typedef struct PEAK_REGISTER_NODE* PEAK_REGISTER_NODE_HANDLE;

struct PEAK_CATEGORY_NODE;
/*! The type of category node handles. */
typedef struct PEAK_CATEGORY_NODE* PEAK_CATEGORY_NODE_HANDLE;

struct PEAK_ENUMERATION_NODE;
/*! The type of enumeration node handles. */
typedef struct PEAK_ENUMERATION_NODE* PEAK_ENUMERATION_NODE_HANDLE;

struct PEAK_ENUMERATION_ENTRY_NODE;
/*! The type of enumeration entry node handles. */
typedef struct PEAK_ENUMERATION_ENTRY_NODE* PEAK_ENUMERATION_ENTRY_NODE_HANDLE;

struct PEAK_PORT;
/*! The type of port handles. */
typedef struct PEAK_PORT* PEAK_PORT_HANDLE;

struct PEAK_PORT_URL;
/*! The type of port URL handles. */
typedef struct PEAK_PORT_URL* PEAK_PORT_URL_HANDLE;

struct PEAK_EVENT_SUPPORTING_MODULE;
/*! The type of event supporting module handles. */
typedef struct PEAK_EVENT_SUPPORTING_MODULE* PEAK_EVENT_SUPPORTING_MODULE_HANDLE;

struct PEAK_EVENT_CONTROLLER;
/*! The type of event controller handles. */
typedef struct PEAK_EVENT_CONTROLLER* PEAK_EVENT_CONTROLLER_HANDLE;

struct PEAK_EVENT;
/*! The type of event handles. */
typedef struct PEAK_EVENT* PEAK_EVENT_HANDLE;

struct PEAK_FIRMWARE_UPDATER;
/*! The type of firmware updater handles. */
typedef struct PEAK_FIRMWARE_UPDATER* PEAK_FIRMWARE_UPDATER_HANDLE;

struct PEAK_FIRMWARE_UPDATE_INFORMATION;
/*! The type of firmware update information handles. */
typedef struct PEAK_FIRMWARE_UPDATE_INFORMATION* PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE;

struct PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER;
/*! The type of firmware update progress observer handles. */
typedef struct PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER* PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE;


/*! The type of interface found callbacks */
typedef void(PEAK_CALL_CONV* PEAK_INTERFACE_FOUND_CALLBACK)(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE foundInterface, void* context);
/*! The type of interface lost callbacks */
typedef void(PEAK_CALL_CONV* PEAK_INTERFACE_LOST_CALLBACK)(
    const char* lostInterfaceId, size_t lostInterfaceIdSize, void* context);
/*! The type of interface found callback handles */
typedef PEAK_INTERFACE_FOUND_CALLBACK* PEAK_INTERFACE_FOUND_CALLBACK_HANDLE;
/*! The type of interface lost callback handles */
typedef PEAK_INTERFACE_LOST_CALLBACK* PEAK_INTERFACE_LOST_CALLBACK_HANDLE;

/*! The type of device found callbacks */
typedef void(PEAK_CALL_CONV* PEAK_DEVICE_FOUND_CALLBACK)(
    PEAK_DEVICE_DESCRIPTOR_HANDLE foundDevice, void* context);
/*! The type of device lost callbacks */
typedef void(PEAK_CALL_CONV* PEAK_DEVICE_LOST_CALLBACK)(
    const char* lostDeviceId, size_t lostDeviceIdSize, void* context);
/*! The type of device found callback handles */
typedef PEAK_DEVICE_FOUND_CALLBACK* PEAK_DEVICE_FOUND_CALLBACK_HANDLE;
/*! The type of device lost callback handles */
typedef PEAK_DEVICE_LOST_CALLBACK* PEAK_DEVICE_LOST_CALLBACK_HANDLE;

/*! The type of buffer revocation callbacks */
typedef void(PEAK_CALL_CONV* PEAK_BUFFER_REVOCATION_CALLBACK)(void* buffer, void* userPtr, void* context);

/*! The type of node changed callbacks */
typedef void(PEAK_CALL_CONV* PEAK_NODE_CHANGED_CALLBACK)(PEAK_NODE_HANDLE changedNode, void* context);
/*! The type of node changed callback handles */
typedef PEAK_NODE_CHANGED_CALLBACK* PEAK_NODE_CHANGED_CALLBACK_HANDLE;

/*! The type of device descriptor information changed callbacks */
typedef void(PEAK_CALL_CONV* PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK)(
    const PEAK_DEVICE_INFORMATION_ROLE* changedRoles, size_t changedRolesSize, void* context);
/*! The type of device descriptor information changed callback handles */
typedef PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK*
    PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE;

/*! The type of firmware update started callbacks */
typedef void(PEAK_CALL_CONV* PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK)(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, uint32_t estimatedDuration_ms,
    void* context);
/*! The type of firmware update started callback handles */
typedef PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK* PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE;
/*! The type of firmware update step started callbacks */
typedef void(PEAK_CALL_CONV* PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK)(PEAK_FIRMWARE_UPDATE_STEP updateStep,
    uint32_t estimatedDuration_ms, const char* description, size_t descriptionSize, void* context);
/*! The type of firmware update step started callback handles */
typedef PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK* PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE;
/*! The type of firmware update step progress changed callbacks */
typedef void(PEAK_CALL_CONV* PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK)(
    PEAK_FIRMWARE_UPDATE_STEP updateStep, double progressPercentage, void* context);
/*! The type of firmware update step progress changed callback handles */
typedef PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK*
    PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE;
/*! The type of firmware update step finished callbacks */
typedef void(PEAK_CALL_CONV* PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK)(
    PEAK_FIRMWARE_UPDATE_STEP updateStep, void* context);
/*! The type of firmware update step finished callback handles */
typedef PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK* PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE;
/*! The type of firmware update finished callbacks */
typedef void(PEAK_CALL_CONV* PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK)(void* context);
/*! The type of firmware update finished callback handles */
typedef PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK* PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE;
/*! The type of firmware update failed callbacks */
typedef void(PEAK_CALL_CONV* PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK)(
    const char* errorDescription, size_t errorDescriptionSize, void* context);
/*! The type of firmware update failed callback handles */
typedef PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK* PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE;


#define PEAK_C_API PEAK_PUBLIC PEAK_RETURN_CODE PEAK_CALL_CONV

#ifndef PEAK_DYNAMIC_LOADING
/*!
 * \brief Initializes the IDS peak API library.
 *
 * This function **must** be called prior to any other function call to allow global initialization of library
 * internals. This function is necessary since automated initialization functionality like within DllMain on MS Windows
 * platforms is very limited.
 *
 * Calling this function multiple times is ok, but note that there is no reference counting, so the first call to
 * PEAK_Library_Close() _will_ close the library.
 *
 * \note Calling any other function before this will return a PEAK_RETURN_CODE_NOT_INITIALIZED error.
 */
PEAK_C_API PEAK_Library_Initialize();
/*!
 * \brief Closes the IDS peak API library and cleans up any resources that are still in use.
 *
 * This function should be called after no function of the library is needed anymore, before unloading the library. It
 * cleans up any resources still in use.
 *
 * \note Calling any other function (except PEAK_Library_Initialize()) after this will return a
 *       PEAK_RETURN_CODE_NOT_INITIALIZED error.
 *
 * \warning It is of particular importance on _MS Windows_ platforms when using "Run-Time Dynamic Linking", especially
 *          when unloading the DLL with FreeLibrary(). Otherwise it is very likely to run into the "Loader-Lock Deadlock
 *          Problem" during DLL unloading.
 */
PEAK_C_API PEAK_Library_Close();
/*!
 * \brief Checks if the IDS peak API library is initialized.
 *
 * \param[out] isInitialized Flag telling whether the IDS peak API library is initialized.
 *
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isInitialized is a null pointer
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Library_IsInitialized(PEAK_BOOL8* isInitialized);
/*!
 * \brief Asks for the major component of the library version.
 *
 * \param[out] libraryVersionMajor The major component of the library version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT libraryVersionMajor is a null pointer
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Library_GetVersionMajor(uint32_t* libraryVersionMajor);
/*!
 * \brief Asks for the minor component of the library version.
 *
 * \param[out] libraryVersionMinor The minor component of the library version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT libraryVersionMinor is a null pointer
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Library_GetVersionMinor(uint32_t* libraryVersionMinor);
/*!
 * \brief Asks for the subminor component of the library version.
 *
 * \param[out] libraryVersionSubminor The subminor component of the library version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT libraryVersionSubminor is a null pointer
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Library_GetVersionSubminor(uint32_t* libraryVersionSubminor);
/*!
 * \brief Queries the last error.
 *
 * This function is normally used by applying a two-step procedure. First of all, you call the function with all
 * arguments except of lastErrorDescription.
 * \code
 *   // Error handling is omitted
 *   PEAK_RETURN_CODE lastErrorCode = PEAK_RETURN_CODE_SUCCESS;
 *   size_t size = 0;
 *   PEAK_Library_GetLastError(&lastErrorCode, NULL, &size);
 * \endcode
 * The function then gives you the last error code and the size of the error description. You could stop now if you only
 * want to query the last error code. If you want to query the error description as well, you have to go on.
 * \code
 *   // Error handling is omitted
 *   char errorDescription[size];
 *   PEAK_Library_GetLastError(&returnCode, errorDescription, &size);
 * \endcode
 *
 * This two-step procedure may not be necessary if you just pass a buffer big enough for holding the description at the
 * first function call.
 *
 * \param[out] lastErrorCode The last function error code.
 * \param[out] lastErrorDescription The description for the last error.
 * \param[in,out] lastErrorDescriptionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT lastErrorCode and/or lastErrorCodeDescriptionSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL lastErrorDescriptionSize is too small
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Library_GetLastError(
    PEAK_RETURN_CODE* lastErrorCode, char* lastErrorDescription, size_t* lastErrorDescriptionSize);

/*!
 * \brief Collects the paths to the CTIs installed on the system.
 *
 * This function depends on the architecture the library was compiled for.
 * Therefore, this function collects 64-bit CTIs for a 64-bit library and 32-bit CTIs for a 32-bit library.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 * \return PEAK_RETURN_CODE_NOT_FOUND The environment variable GENICAM_GENTL32_PATH / GENICAM_GENTL64_PATH was not
 *                                       found or was empty.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnvironmentInspector_UpdateCTIPaths();
/*!
 * \brief Asks for the number of CTI paths.
 *
 * \param[out] numCtiPaths The number of CTI paths.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numCtiPaths is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnvironmentInspector_GetNumCTIPaths(size_t* numCtiPaths);
/*!
 * \brief Asks for the CTI path with the given index.
 *
 * \param[in] index The index to work with.
 * \param[out] ctiPath The CTI path.
 * \param[in,out] ctiPathSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT ctiPathSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL ctiPathSize is too small
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnvironmentInspector_GetCTIPath(size_t index, char* ctiPath, size_t* ctiPathSize);

/*!
 * \brief Creates a producer library.
 *
 * \param[in] ctiPath The CTI to create the producer library from.
 * \param[in] ctiPathSize The size of the given string.
 * \param[out] producerLibraryHandle The handle associated with the producer library.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT ctiPath and/or producerLibraryHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_ProducerLibrary_Construct(
    const char* ctiPath, size_t ctiPathSize, PEAK_PRODUCER_LIBRARY_HANDLE* producerLibraryHandle);
/*!
 * \brief Asks the given producer library for its key.
 *
 * \param[in] producerLibraryHandle The producer library.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE producerLibraryHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_ProducerLibrary_GetKey(
    PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given producer library for its system descriptor.
 *
 * \param[in] producerLibraryHandle The producer library.
 * \param[out] systemDescriptorHandle The system descriptor.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE producerLibraryHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT systemDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_ProducerLibrary_GetSystem(
    PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, PEAK_SYSTEM_DESCRIPTOR_HANDLE* systemDescriptorHandle);
/*!
 * \brief Destroys the given producer library.
 *
 * \param[in] producerLibraryHandle The producer library.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE producerLibraryHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_ProducerLibrary_Destruct(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle);

/*!
 * \brief Casts the given system descriptor to a module descriptor.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] moduleDescriptorHandle The module descriptor.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_ToModuleDescriptor(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE* moduleDescriptorHandle);
/*!
 * \brief Asks the given system descriptor for its key.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetKey(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given system descriptor for an information based on a specific command.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetInfo(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle,
    int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given system descriptor for its display name.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetDisplayName(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given system descriptor for its vendor name.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] vendorName The vendor name.
 * \param[in,out] vendorNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT vendorNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL vendorNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetVendorName(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* vendorName, size_t* vendorNameSize);
/*!
 * \brief Asks the given system descriptor for its model name.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] modelName The model name.
 * \param[in,out] modelNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT modelNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL modelNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetModelName(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* modelName, size_t* modelNameSize);
/*!
 * \brief Asks the given system descriptor for its version.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] version The version.
 * \param[in,out] versionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetVersion(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* version, size_t* versionSize);
/*!
 * \brief Asks the given system descriptor for its TL type.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetTLType(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given system descriptor for its parent CTI's file name.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] ctiFileName The parent CTI's file name.
 * \param[in,out] ctiFileNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT ctiFileNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL ctiFileNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetCTIFileName(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* ctiFileName, size_t* ctiFileNameSize);
/*!
 * \brief Asks the given system descriptor for its parent CTI's full path.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] ctiFullPath The parent CTI's full path.
 * \param[in,out] ctiFullPathSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT ctiFullPathSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL ctiFullPathSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetCTIFullPath(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char* ctiFullPath, size_t* ctiFullPathSize);
/*!
 * \brief Asks the given system descriptor for the major component of its GenTL version.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] gentlVersionMajor The major component of the GenTL version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT gentlVersionMajor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetGenTLVersionMajor(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t* gentlVersionMajor);
/*!
 * \brief Asks the given system descriptor for the minor component of its GenTL version.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] gentlVersionMinor The minor component of the GenTL version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT gentlVersionMinor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetGenTLVersionMinor(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t* gentlVersionMinor);
/*!
 * \brief Asks the given system descriptor for its character encoding.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] characterEncoding The character encoding.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT characterEncoding is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetCharacterEncoding(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_CHARACTER_ENCODING* characterEncoding);
/*!
 * \brief Asks the given system descriptor for its parent library.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] producerLibraryHandle The parent library.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT producerLibraryHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_GetParentLibrary(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_PRODUCER_LIBRARY_HANDLE* producerLibraryHandle);
/*!
 * \brief Opens the system being associated with the given system descriptor.
 *
 * \param[in] systemDescriptorHandle The system descriptor.
 * \param[out] systemHandle The system.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT systemHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_SystemDescriptor_OpenSystem(
    PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_SYSTEM_HANDLE* systemHandle);

/*!
 * \brief Casts the given system to a module.
 *
 * \param[in] systemHandle The system.
 * \param[out] moduleHandle The module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_ToModule(PEAK_SYSTEM_HANDLE systemHandle, PEAK_MODULE_HANDLE* moduleHandle);
/*!
 * \brief Casts the given system to an event-supporting module.
 *
 * \param[in] systemHandle The system.
 * \param[out] eventSupportingModuleHandle The event-supporting module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventSupportingModuleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_ToEventSupportingModule(
    PEAK_SYSTEM_HANDLE systemHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE* eventSupportingModuleHandle);
/*!
 * \brief Asks the given system for its key.
 *
 * \param[in] systemHandle The system.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetKey(PEAK_SYSTEM_HANDLE systemHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given system for an information based on a specific command.
 *
 * \param[in] systemHandle The system.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetInfo(
    PEAK_SYSTEM_HANDLE systemHandle, int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given system for its ID.
 *
 * \param[in] systemHandle The system.
 * \param[out] id The ID.
 * \param[in,out] idSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT idSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL idSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetID(PEAK_SYSTEM_HANDLE systemHandle, char* id, size_t* idSize);
/*!
 * \brief Asks the given system for its display name.
 *
 * \param[in] systemHandle The system.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetDisplayName(
    PEAK_SYSTEM_HANDLE systemHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given system for its vendor name.
 *
 * \param[in] systemHandle The system.
 * \param[out] vendorName The vendor name.
 * \param[in,out] vendorNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT vendorNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL vendorNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetVendorName(
    PEAK_SYSTEM_HANDLE systemHandle, char* vendorName, size_t* vendorNameSize);
/*!
 * \brief Asks the given system for its model name.
 *
 * \param[in] systemHandle The system.
 * \param[out] modelName The model name.
 * \param[in,out] modelNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT modelNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL modelNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetModelName(PEAK_SYSTEM_HANDLE systemHandle, char* modelName, size_t* modelNameSize);
/*!
 * \brief Asks the given system for its version.
 *
 * \param[in] systemHandle The system.
 * \param[out] version The version.
 * \param[in,out] versionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetVersion(PEAK_SYSTEM_HANDLE systemHandle, char* version, size_t* versionSize);
/*!
 * \brief Asks the given system for its TL type.
 *
 * \param[in] systemHandle The system.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetTLType(PEAK_SYSTEM_HANDLE systemHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given system for its parent CTI's file name.
 *
 * \param[in] systemHandle The system.
 * \param[out] ctiFileName The parent CTI's file name.
 * \param[in,out] ctiFileNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT ctiFileNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL ctiFileNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetCTIFileName(
    PEAK_SYSTEM_HANDLE systemHandle, char* ctiFileName, size_t* ctiFileNameSize);
/*!
 * \brief Asks the given system for its parent CTI's full path.
 *
 * \param[in] systemHandle The system.
 * \param[out] ctiFullPath The parent CTI's full path.
 * \param[in,out] ctiFullPathSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT ctiFullPathSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL ctiFullPathSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetCTIFullPath(
    PEAK_SYSTEM_HANDLE systemHandle, char* ctiFullPath, size_t* ctiFullPathSize);
/*!
 * \brief Asks the given system for the major component of its GenTL version.
 *
 * \param[in] systemHandle The system.
 * \param[out] gentlVersionMajor The major component of the GenTL version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT gentlVersionMajor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetGenTLVersionMajor(PEAK_SYSTEM_HANDLE systemHandle, uint32_t* gentlVersionMajor);
/*!
 * \brief Asks the given system for the minor component of its GenTL version.
 *
 * \param[in] systemHandle The system.
 * \param[out] gentlVersionMinor The minor component of the GenTL version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT gentlVersionMinor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetGenTLVersionMinor(PEAK_SYSTEM_HANDLE systemHandle, uint32_t* gentlVersionMinor);
/*!
 * \brief Asks the given system for its character encoding.
 *
 * \param[in] systemHandle The system.
 * \param[out] characterEncoding The character encoding.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT characterEncoding is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetCharacterEncoding(
    PEAK_SYSTEM_HANDLE systemHandle, PEAK_CHARACTER_ENCODING* characterEncoding);
/*!
 * \brief Asks the given system for its parent library.
 *
 * \param[in] systemHandle The system.
 * \param[out] producerLibraryHandle The parent library.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT producerLibraryHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetParentLibrary(
    PEAK_SYSTEM_HANDLE systemHandle, PEAK_PRODUCER_LIBRARY_HANDLE* producerLibraryHandle);
/*!
 * \brief Tells the given system to update its interfaces with the given timeout.
 *
 * \param[in] systemHandle The system.
 * \param[in] timeout_ms The time to wait for discovery.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_UpdateInterfaces(PEAK_SYSTEM_HANDLE systemHandle, uint64_t timeout_ms);
/*!
 * \brief Asks the given system for its number of interfaces.
 *
 * \param[in] systemHandle The system.
 * \param[out] numInterfaces The number of interfaces.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numInterfaces is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetNumInterfaces(PEAK_SYSTEM_HANDLE systemHandle, size_t* numInterfaces);
/*!
 * \brief Asks the given system for the interface descriptor with the given index.
 *
 * \param[in] systemHandle The system.
 * \param[in] index The index.
 * \param[out] interfaceDescriptorHandle The interface descriptor being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT interfaceDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_GetInterface(
    PEAK_SYSTEM_HANDLE systemHandle, size_t index, PEAK_INTERFACE_DESCRIPTOR_HANDLE* interfaceDescriptorHandle);
/*!
 * \brief Registers a callback signaling a found interface at the given system.
 *
 * \param[in] systemHandle The system.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_RegisterInterfaceFoundCallback(PEAK_SYSTEM_HANDLE systemHandle,
    PEAK_INTERFACE_FOUND_CALLBACK callback, void* callbackContext,
    PEAK_INTERFACE_FOUND_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the interface-found callback with the given handle from the given system.
 *
 * \param[in] systemHandle The system.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_UnregisterInterfaceFoundCallback(
    PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a lost interface at the given system.
 *
 * \param[in] systemHandle The system.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_RegisterInterfaceLostCallback(PEAK_SYSTEM_HANDLE systemHandle,
    PEAK_INTERFACE_LOST_CALLBACK callback, void* callbackContext,
    PEAK_INTERFACE_LOST_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the interface-lost callback with the given handle from the given system.
 *
 * \param[in] systemHandle The system.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_UnregisterInterfaceLostCallback(
    PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Destroys the given system.
 *
 * \param[in] systemHandle The system.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_System_Destruct(PEAK_SYSTEM_HANDLE systemHandle);

/*!
 * \brief Casts the given interface descriptor to a module descriptor.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[out] moduleDescriptorHandle The module descriptor.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_ToModuleDescriptor(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle,
    PEAK_MODULE_DESCRIPTOR_HANDLE* moduleDescriptorHandle);
/*!
 * \brief Asks the given interface descriptor for its key.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_GetKey(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given interface descriptor for an information based on a specific command.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_GetInfo(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle,
    int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given interface descriptor for its display name.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_GetDisplayName(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given interface descriptor for its TL type.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_GetTLType(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given interface descriptor for its parent system.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[out] systemHandle The parent system.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT systemHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_GetParentSystem(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_SYSTEM_HANDLE* systemHandle);
/*!
 * \brief Opens the interface being associated with the given interface descriptor.
 *
 * \param[in] interfaceDescriptorHandle The interface descriptor.
 * \param[out] interfaceHandle The interface.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT interfaceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_InterfaceDescriptor_OpenInterface(
    PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_INTERFACE_HANDLE* interfaceHandle);

/*!
 * \brief Casts the given interface to a module.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] moduleHandle The module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_ToModule(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_MODULE_HANDLE* moduleHandle);
/*!
 * \brief Casts the given interface to an event-supporting module.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] eventSupportingModuleHandle The event-supporting module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventSupportingModuleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_ToEventSupportingModule(
    PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE* eventSupportingModuleHandle);
/*!
 * \brief Asks the given interface for its key.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetKey(PEAK_INTERFACE_HANDLE interfaceHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given interface for an information based on a specific command.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetInfo(PEAK_INTERFACE_HANDLE interfaceHandle, int32_t infoCommand,
    int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given interface for its ID.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] id The ID.
 * \param[in,out] idSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT idSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL idSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetID(PEAK_INTERFACE_HANDLE interfaceHandle, char* id, size_t* idSize);
/*!
 * \brief Asks the given interface for its display name.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetDisplayName(
    PEAK_INTERFACE_HANDLE interfaceHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given interface for its TL type.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetTLType(PEAK_INTERFACE_HANDLE interfaceHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given interface for its parent system.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] systemHandle The parent system.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT systemHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetParentSystem(
    PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_SYSTEM_HANDLE* systemHandle);
/*!
 * \brief Tells the given interface to update its devices with the given timeout.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] timeout_ms The time to wait for discovery.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE systemHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_UpdateDevices(PEAK_INTERFACE_HANDLE interfaceHandle, uint64_t timeout_ms);
/*!
 * \brief Asks the given interface for its number of devices.
 *
 * \param[in] interfaceHandle The interface.
 * \param[out] numDevices The number of devices.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numDevices is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetNumDevices(PEAK_INTERFACE_HANDLE interfaceHandle, size_t* numDevices);
/*!
 * \brief Asks the given interface for the device descriptor with the given index.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] index The index.
 * \param[out] deviceDescriptorHandle The device descriptor being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deviceDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_GetDevice(
    PEAK_INTERFACE_HANDLE interfaceHandle, size_t index, PEAK_DEVICE_DESCRIPTOR_HANDLE* deviceDescriptorHandle);
/*!
 * \brief Registers a callback signaling a found device at the given interface.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_RegisterDeviceFoundCallback(PEAK_INTERFACE_HANDLE interfaceHandle,
    PEAK_DEVICE_FOUND_CALLBACK callback, void* callbackContext,
    PEAK_DEVICE_FOUND_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the device-found callback with the given handle from the given interface.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_UnregisterDeviceFoundCallback(
    PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a lost device at the given interface.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_RegisterDeviceLostCallback(PEAK_INTERFACE_HANDLE interfaceHandle,
    PEAK_DEVICE_LOST_CALLBACK callback, void* callbackContext, PEAK_DEVICE_LOST_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the device-lost callback with the given handle from the given interface.
 *
 * \param[in] interfaceHandle The interface.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_UnregisterDeviceLostCallback(
    PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Destroys the given interface.
 *
 * \param[in] interfaceHandle The interface.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE interfaceHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Interface_Destruct(PEAK_INTERFACE_HANDLE interfaceHandle);

/*!
 * \brief Casts the given device descriptor to a module descriptor.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] moduleDescriptorHandle The module descriptor.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_ToModuleDescriptor(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE* moduleDescriptorHandle);
/*!
 * \brief Asks the given device descriptor for its key.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetKey(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given device descriptor for an information based on a specific command.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetInfo(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given device descriptor for its display name.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetDisplayName(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given device descriptor for its vendor name.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] vendorName The vendor name.
 * \param[in,out] vendorNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT vendorNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL vendorNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetVendorName(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* vendorName, size_t* vendorNameSize);
/*!
 * \brief Asks the given device descriptor for its model name.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] modelName The model name.
 * \param[in,out] modelNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT modelNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL modelNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetModelName(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* modelName, size_t* modelNameSize);
/*!
 * \brief Asks the given device descriptor for its version.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] version The version.
 * \param[in,out] versionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetVersion(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* version, size_t* versionSize);
/*!
 * \brief Asks the given device descriptor for its TL type.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetTLType(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given device descriptor for its user-defined name.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] userDefinedName The user-defined name.
 * \param[in,out] userDefinedNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT userDefinedNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL userDefinedNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetUserDefinedName(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* userDefinedName, size_t* userDefinedNameSize);
/*!
 * \brief Asks the given device descriptor for its serial number.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] serialNumber The serial number.
 * \param[in,out] serialNumberSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT serialNumberSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL serialNumberSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetSerialNumber(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char* serialNumber, size_t* serialNumberSize);
/*!
 * \brief Asks the given device descriptor for its access status.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] accessStatus The access status.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT accessStatus is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetAccessStatus(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_STATUS* accessStatus);
/*!
 * \brief Asks the given device descriptor for its timestamp tick frequency.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] timestampTickFrequency The timestamp tick frequency.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT timestampTickFrequency is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetTimestampTickFrequency(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t* timestampTickFrequency);
/*!
 * \brief Asks the given device descriptor whether the device can be opened with exclusive access.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] isOpenable A flag telling whether the device descriptor is openable.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isOpenable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetIsOpenableExclusive(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_BOOL8* isOpenable);
/*!
 * \brief Asks the given device descriptor whether the device can be opened with this specific access type.
 *
 * If the device can be opened with a higher access type, it can also be opened with a lower access type.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] accessType The access type.
 * \param[out] isOpenable A flag telling whether the given access type is available.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isOpenable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetIsOpenable(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_BOOL8* isOpenable);
/*!
 * \brief Opens the device being associated with the given interface descriptor for the given access type.
 *
 * \param[in]  deviceDescriptorHandle The device descriptor.
 * \param[in]  accessType The access type to open the device with.
 * \param[out] deviceHandle The device.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deviceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_OpenDevice(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_DEVICE_HANDLE* deviceHandle);
/*!
 * \brief Asks the given device descriptor for its parent interface.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] interfaceHandle The parent interface.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT interfaceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetParentInterface(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_INTERFACE_HANDLE* interfaceHandle);
/*!
 * \brief Asks the given device descriptor for its monitoring update interval.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[out] monitoringUpdateInterval_ms The monitoring update interval.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT monitoringUpdateInterval_ms is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_GetMonitoringUpdateInterval(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t* monitoringUpdateInterval_ms);
/*!
 * \brief Tells the given device descriptor to set the given monitoring update interval.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] monitoringUpdateInterval_ms The monitoring update interval.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_SetMonitoringUpdateInterval(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t monitoringUpdateInterval_ms);
/*!
 * \brief Asks the given device descriptor whether the given information role is monitored.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] informationRole The information role.
 * \param[out] isInformationRoleMonitored A flag telling whether the given information role is monitored.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isInformationRoleMonitored is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_IsInformationRoleMonitored(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole,
    PEAK_BOOL8* isInformationRoleMonitored);
/*!
 * \brief Tells the given device descriptor to add the given information role to the monitoring.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] informationRole The information role.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_AddInformationRoleToMonitoring(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole);
/*!
 * \brief Tells the given device descriptor to remove the given information role from the monitoring.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] informationRole The information role.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole);
/*!
 * \brief Registers a callback signaling changed information at the given device descriptor.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_RegisterInformationChangedCallback(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK callback, void* callbackContext,
    PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the information changed callback with the given handle from the given device descriptor.
 *
 * \param[in] deviceDescriptorHandle The device descriptor.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DeviceDescriptor_UnregisterInformationChangedCallback(
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE callbackHandle);

/*!
 * \brief Casts the given device to a module.
 *
 * \param[in] deviceHandle The device.
 * \param[out] moduleHandle The module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_ToModule(PEAK_DEVICE_HANDLE deviceHandle, PEAK_MODULE_HANDLE* moduleHandle);
/*!
 * \brief Casts the given device to an event-supporting module.
 *
 * \param[in] deviceHandle The device.
 * \param[out] eventSupportingModuleHandle The event-supporting module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventSupportingModuleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_ToEventSupportingModule(
    PEAK_DEVICE_HANDLE deviceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE* eventSupportingModuleHandle);
/*!
 * \brief Asks the given device for its key.
 *
 * \param[in] deviceHandle The device.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetKey(PEAK_DEVICE_HANDLE deviceHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given device for an information based on a specific command.
 *
 * \param[in] deviceHandle The device.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetInfo(
    PEAK_DEVICE_HANDLE deviceHandle, int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given device for its ID.
 *
 * \param[in] deviceHandle The device.
 * \param[out] id The ID.
 * \param[in,out] idSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT idSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL idSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetID(PEAK_DEVICE_HANDLE deviceHandle, char* id, size_t* idSize);
/*!
 * \brief Asks the given device for its display name.
 *
 * \param[in] deviceHandle The device.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetDisplayName(
    PEAK_DEVICE_HANDLE deviceHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given device for its vendor name.
 *
 * \param[in] deviceHandle The device.
 * \param[out] vendorName The vendor name.
 * \param[in,out] vendorNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT vendorNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL vendorNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetVendorName(
    PEAK_DEVICE_HANDLE deviceHandle, char* vendorName, size_t* vendorNameSize);
/*!
 * \brief Asks the given device for its model name.
 *
 * \param[in] deviceHandle The device.
 * \param[out] modelName The model name.
 * \param[in,out] modelNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT modelNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL modelNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetModelName(PEAK_DEVICE_HANDLE deviceHandle, char* modelName, size_t* modelNameSize);
/*!
 * \brief Asks the given device for its version.
 *
 * \param[in] deviceHandle The device.
 * \param[out] version The version.
 * \param[in,out] versionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetVersion(PEAK_DEVICE_HANDLE deviceHandle, char* version, size_t* versionSize);
/*!
 * \brief Asks the given device for its TL type.
 *
 * \param[in] deviceHandle The device.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetTLType(PEAK_DEVICE_HANDLE deviceHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given device for its user-defined name.
 *
 * \param[in] deviceHandle The device.
 * \param[out] userDefinedName The user-defined name.
 * \param[in,out] userDefinedNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT userDefinedNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL userDefinedNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetUserDefinedName(
    PEAK_DEVICE_HANDLE deviceHandle, char* userDefinedName, size_t* userDefinedNameSize);
/*!
 * \brief Asks the given device for its serial number.
 *
 * \param[in] deviceHandle The device.
 * \param[out] serialNumber The serial number.
 * \param[in,out] serialNumberSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT serialNumberSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL serialNumberSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetSerialNumber(
    PEAK_DEVICE_HANDLE deviceHandle, char* serialNumber, size_t* serialNumberSize);
/*!
 * \brief Asks the given device for its access status.
 *
 * \param[in] deviceHandle The device.
 * \param[out] accessStatus The access status.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT accessStatus is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetAccessStatus(
    PEAK_DEVICE_HANDLE deviceHandle, PEAK_DEVICE_ACCESS_STATUS* accessStatus);
/*!
 * \brief Asks the given device for its timestamp tick frequency.
 *
 * \param[in] deviceHandle The device.
 * \param[out] timestampTickFrequency The timestamp tick frequency.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT timestampTickFrequency is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetTimestampTickFrequency(
    PEAK_DEVICE_HANDLE deviceHandle, uint64_t* timestampTickFrequency);
/*!
 * \brief Asks the given device for its parent interface.
 *
 * \param[in] deviceHandle The device.
 * \param[out] interfaceHandle The parent interface.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT interfaceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetParentInterface(
    PEAK_DEVICE_HANDLE deviceHandle, PEAK_INTERFACE_HANDLE* interfaceHandle);
/*!
 * \brief Asks the given device for its remote device.
 *
 * \param[in] deviceHandle The device.
 * \param[out] remoteDeviceHandle The remote device.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT remoteDeviceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetRemoteDevice(
    PEAK_DEVICE_HANDLE deviceHandle, PEAK_REMOTE_DEVICE_HANDLE* remoteDeviceHandle);
/*!
 * \brief Asks the given device for its number of data streams.
 *
 * \param[in] deviceHandle The device.
 * \param[out] numDataStreams The number of data streams.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numDataStreams is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetNumDataStreams(PEAK_DEVICE_HANDLE deviceHandle, size_t* numDataStreams);
/*!
 * \brief Asks the given device for the data stream descriptor with the given index.
 *
 * \param[in] deviceHandle The device.
 * \param[in] index The index.
 * \param[out] dataStreamDescriptorHandle The data stream descriptor being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT dataStreamDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_GetDataStream(PEAK_DEVICE_HANDLE deviceHandle, size_t index,
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE* dataStreamDescriptorHandle);
/*!
 * \brief Delete the given device.
 *
 * \param[in] deviceHandle The device.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE deviceHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Device_Destruct(PEAK_DEVICE_HANDLE deviceHandle);

/*!
 * \brief Casts the given remote device to a module.
 *
 * \param[in] remoteDeviceHandle The remote device.
 * \param[out] moduleHandle The module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE remoteDeviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RemoteDevice_ToModule(
    PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_MODULE_HANDLE* moduleHandle);
/*!
 * \brief Asks the given remote device for its local device.
 *
 * \param[in] remoteDeviceHandle The remote device.
 * \param[out] deviceHandle The local device.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE remoteDeviceHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deviceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RemoteDevice_GetLocalDevice(
    PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_DEVICE_HANDLE* deviceHandle);

/*!
 * \brief Casts the given data stream descriptor to a module descriptor.
 *
 * \param[in] dataStreamDescriptorHandle The data stream descriptor.
 * \param[out] moduleDescriptorHandle The module descriptor.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleDescriptorHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStreamDescriptor_ToModuleDescriptor(
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle,
    PEAK_MODULE_DESCRIPTOR_HANDLE* moduleDescriptorHandle);
/*!
 * \brief Asks the given data stream descriptor for its key.
 *
 * \param[in] dataStreamDescriptorHandle The data stream descriptor.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStreamDescriptor_GetKey(
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given data stream descriptor for its parent device.
 *
 * \param[in] dataStreamDescriptorHandle The data stream descriptor.
 * \param[out] deviceHandle The parent device.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deviceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStreamDescriptor_GetParentDevice(
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DEVICE_HANDLE* deviceHandle);
/*!
 * \brief Opens the data stream being associated with the given data stream descriptor.
 *
 * \param[in] dataStreamDescriptorHandle The data stream descriptor.
 * \param[out] dataStreamHandle The data stream.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamDescriptorHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT dataStreamHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStreamDescriptor_OpenDataStream(
    PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DATA_STREAM_HANDLE* dataStreamHandle);

/*!
 * \brief Casts the given data stream to a module.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] moduleHandle The module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_ToModule(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_MODULE_HANDLE* moduleHandle);
/*!
 * \brief Casts the given data stream to an event-supporting module.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] eventSupportingModuleHandle The event-supporting module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventSupportingModuleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_ToEventSupportingModule(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE* eventSupportingModuleHandle);
/*!
 * \brief Asks the given data stream for its key.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] key The key.
 * \param[in,out] keySize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT keySize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL keySize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetKey(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char* key, size_t* keySize);
/*!
 * \brief Asks the given data stream for an information based on a specific command.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetInfo(PEAK_DATA_STREAM_HANDLE dataStreamHandle, int32_t infoCommand,
    int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given data stream for its ID.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] id The ID.
 * \param[in,out] idSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT idSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL idSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetID(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char* id, size_t* idSize);
/*!
 * \brief Asks the given data stream for its TL type.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetTLType(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given data stream for its minimum required number of announced buffers.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numBuffersAnnouncedMinRequired The minimum required number of announced buffers.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numBuffersAnnouncedMinRequired is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumBuffersAnnouncedMinRequired(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* numBuffersAnnouncedMinRequired);
/*!
 * \brief Asks the given data stream for its number of announced buffers.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numBuffersAnnounced The number of announced buffers.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numBuffersAnnounced is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumBuffersAnnounced(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* numBuffersAnnounced);
/*!
 * \brief Asks the given data stream for its number of queued buffers.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numBuffersQueued The number of queued buffers.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numBuffersQueued is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumBuffersQueued(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* numBuffersQueued);
/*!
 * \brief Asks the given data stream for its number of buffers awaiting delivery.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numBuffersAwaitDelivery The number of buffers awaiting delivery.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numBuffersAwaitDelivery is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumBuffersAwaitDelivery(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* numBuffersAwaitDelivery);
/*!
 * \brief Asks the given data stream for its number of delivered buffers.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numBuffersDelivered The number of delivered buffers.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numBuffersDelivered is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumBuffersDelivered(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t* numBuffersDelivered);
/*!
 * \brief Asks the given data stream for its number of started buffers.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numBuffersStarted The number of started buffers.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numBuffersStarted is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumBuffersStarted(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t* numBuffersStarted);
/*!
 * \brief Asks the given data stream for its number of underruns.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numUnderruns The number of underruns.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numUnderruns is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumUnderruns(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t* numUnderruns);
/*!
 * \brief Asks the given data stream for its maximum number of chunks per buffer.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] numChunksPerBufferMax The maximum number of chunks per buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numChunksPerBufferMax is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetNumChunksPerBufferMax(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* numChunksPerBufferMax);
/*!
 * \brief Asks the given data stream for its buffer alignment.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] bufferAlignment The buffer alignment in bytes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bufferAlignment is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetBufferAlignment(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* bufferAlignment);
/*!
 * \brief Asks the given data stream for its payload size.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] payloadSize The payload size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT payloadSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetPayloadSize(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t* payloadSize);
/*!
 * \brief Asks the given data stream whether it defines the payload size.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] definesPayloadSize A flag telling whether the data stream defines the payload size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT definesPayloadSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetDefinesPayloadSize(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8* definesPayloadSize);
/*!
 * \brief Asks the given data stream whether it is grabbing.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] isGrabbing A flag telling whether the data stream is grabbing.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isGrabbing is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetIsGrabbing(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8* isGrabbing);
/*!
 * \brief Asks the given data stream for its parent device.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[out] deviceHandle The parent device.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deviceHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_GetParentDevice(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DEVICE_HANDLE* deviceHandle);
/*!
 * \brief Announces the given buffer at the given data stream.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] buffer The buffer.
 * \param[in] bufferSize The size of the buffer in bytes.
 * \param[in] userPtr A pointer to user defined data, for identifying the buffer or attaching custom data to the
 *                    buffer. Optional.
 * \param[in] revocationCallback The callback to call on revocation of the buffer. Optional.
 * \param[in] callbackContext The callback context. Optional.
 * \param[out] bufferHandle The handle being associated with the given buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT buffer and/or revocationCallback and/or bufferHandle are/is a null
 *                                          pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_AnnounceBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, void* buffer,
    size_t bufferSize, void* userPtr, PEAK_BUFFER_REVOCATION_CALLBACK revocationCallback, void* callbackContext,
    PEAK_BUFFER_HANDLE* bufferHandle);
/*!
 * \brief Tells the given data stream to allocate and announce a buffer with the given size.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] bufferSize The size of the buffer in bytes.
 * \param[in] userPtr The user ptr.
 * \param[out] bufferHandle The handle being associated with the given buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bufferHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_AllocAndAnnounceBuffer(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t bufferSize, void* userPtr, PEAK_BUFFER_HANDLE* bufferHandle);
/*!
 * \brief Tells the given data stream to queue the given buffer.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] bufferHandle The buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle and/or bufferHandle are/is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_QueueBuffer(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle);
/*!
 * \brief Tells the given data stream to revoke the given buffer.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] bufferHandle The buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle and/or bufferHandle are/is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_RevokeBuffer(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle);
/*!
 * \brief Tells the given data stream to wait for a finished buffer with the given timeout.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] timeout_ms The timeout.
 * \param[out] bufferHandle The finished buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bufferHandle is a null pointer
 * \return PEAK_RETURN_CODE_ABORTED The wait was aborted
 * \return PEAK_RETURN_CODE_TIMEOUT The wait timed out
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_WaitForFinishedBuffer(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t timeout_ms, PEAK_BUFFER_HANDLE* bufferHandle);
/*!
 * \brief Tells the given data stream to kill a wait for a finished buffer.
 *
 * If there are no wait operations at the point the function gets called, the kill request gets stored. In this case
 * the next wait operation gets killed immediately.
 *
 * \param[in] dataStreamHandle The data stream.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_KillWait(PEAK_DATA_STREAM_HANDLE dataStreamHandle);
/*!
 * \brief Tells the given data stream to flush its queues using the given mode.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] flushMode The flush mode.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_Flush(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DATA_STREAM_FLUSH_MODE flushMode);
/*!
 * \brief Tells the given data stream to start the acquisition using the given mode.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] startMode The acquisition start mode.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_StartAcquisitionInfinite(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode);
/*!
 * \brief Tells the given data stream to start the acquisition using the given mode and to acquire the given number of
 *        buffers.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] startMode The acquisition start mode.
 * \param[in] numToAcquire The number of frames to be acquired.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_StartAcquisition(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode, uint64_t numToAcquire);
/*!
 * \brief Tells the given data stream to stop the acquisition using the given mode.
 *
 * \param[in] dataStreamHandle The data stream.
 * \param[in] stopMode The acquisition stop mode.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_StopAcquisition(
    PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_STOP_MODE stopMode);
/*!
 * \brief Destroys the given data stream.
 *
 * \param[in] dataStreamHandle The data stream.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE dataStreamHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_DataStream_Destruct(PEAK_DATA_STREAM_HANDLE dataStreamHandle);

/*!
 * \brief Casts the given buffer to a module.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] moduleHandle The module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_ToModule(PEAK_BUFFER_HANDLE bufferHandle, PEAK_MODULE_HANDLE* moduleHandle);
/*!
 * \brief Casts the given buffer to an event-supporting module.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] eventSupportingModuleHandle The event-supporting module.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventSupportingModuleHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_ToEventSupportingModule(
    PEAK_BUFFER_HANDLE bufferHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE* eventSupportingModuleHandle);
/*!
 * \brief Asks the given buffer for an information based on a specific command.
 *
 * \param[in] bufferHandle The buffer.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetInfo(
    PEAK_BUFFER_HANDLE bufferHandle, int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given buffer for its TL type.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetTLType(PEAK_BUFFER_HANDLE bufferHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given buffer for its base pointer.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] basePtr The base pointer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT basePtr is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetBasePtr(PEAK_BUFFER_HANDLE bufferHandle, void** basePtr);
/*!
 * \brief Asks the given buffer for its size.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] size The size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT size is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetSize(PEAK_BUFFER_HANDLE bufferHandle, size_t* size);
/*!
 * \brief Asks the given buffer for its user pointer.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] userPtr The user pointer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT userPtr is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetUserPtr(PEAK_BUFFER_HANDLE bufferHandle, void** userPtr);
/*!
 * \brief Asks the given buffer for its payload type.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] payloadType The payload type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT payloadType is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetPayloadType(
    PEAK_BUFFER_HANDLE bufferHandle, PEAK_BUFFER_PAYLOAD_TYPE* payloadType);
/*!
 * \brief Asks the given buffer for its pixel format.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] pixelFormat The pixel format.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT pixelFormat is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetPixelFormat(PEAK_BUFFER_HANDLE bufferHandle, uint64_t* pixelFormat);
/*!
 * \brief Asks the given buffer for its pixel format namespace.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] pixelFormatNamespace The pixel format namespace.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT pixelFormatNamespace is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetPixelFormatNamespace(
    PEAK_BUFFER_HANDLE bufferHandle, PEAK_PIXEL_FORMAT_NAMESPACE* pixelFormatNamespace);
/*!
 * \brief Asks the given buffer for its pixel endianness.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] pixelEndianness The pixel endianness.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT pixelEndianness is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetPixelEndianness(
    PEAK_BUFFER_HANDLE bufferHandle, PEAK_ENDIANNESS* pixelEndianness);
/*!
 * \brief Asks the given buffer for its expected data size.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] expectedDataSize The expected data size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT expectedDataSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetExpectedDataSize(PEAK_BUFFER_HANDLE bufferHandle, size_t* expectedDataSize);
/*!
 * \brief Asks the given buffer for its delivered data size.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] deliveredDataSize The delivered data size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deliveredDataSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetDeliveredDataSize(PEAK_BUFFER_HANDLE bufferHandle, size_t* deliveredDataSize);
/*!
 * \brief Asks the given buffer for its frame ID.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] frameId The frame ID.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT frameId is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetFrameID(PEAK_BUFFER_HANDLE bufferHandle, uint64_t* frameId);
/*!
 * \brief Asks the given buffer for its image offset.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] imageOffset The image offset.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT imageOffset is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetImageOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t* imageOffset);
/*!
 * \brief Asks the given buffer for its delivered image height.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] deliveredImageHeight The delivered image height.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deliveredImageHeight is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetDeliveredImageHeight(PEAK_BUFFER_HANDLE bufferHandle, size_t* deliveredImageHeight);
/*!
 * \brief Asks the given buffer for its delivered chunk payload size.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] deliveredChunkPayloadSize The delivered chunk payload size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deliveredChunkPayloadSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetDeliveredChunkPayloadSize(
    PEAK_BUFFER_HANDLE bufferHandle, size_t* deliveredChunkPayloadSize);
/*!
 * \brief Asks the given buffer for its chunk layout ID.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] chunkLayoutId The chunk layout ID.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT chunkLayoutId is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetChunkLayoutID(PEAK_BUFFER_HANDLE bufferHandle, uint64_t* chunkLayoutId);
/*!
 * \brief Asks the given buffer for its file name.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] fileName The file name.
 * \param[in,out] fileNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL fileNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetFileName(PEAK_BUFFER_HANDLE bufferHandle, char* fileName, size_t* fileNameSize);
/*!
 * \brief Asks the given buffer for its width.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] width The width.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT width is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetWidth(PEAK_BUFFER_HANDLE bufferHandle, size_t* width);
/*!
 * \brief Asks the given buffer for its height.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] height The height.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT height is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetHeight(PEAK_BUFFER_HANDLE bufferHandle, size_t* height);
/*!
 * \brief Asks the given buffer for its X offset.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] xOffset The X offset.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT xOffset is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetXOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t* xOffset);
/*!
 * \brief Asks the given buffer for its Y offset.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] yOffset The Y offset.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT yOffset is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetYOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t* yOffset);
/*!
 * \brief Asks the given buffer for its X padding.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] xPadding The X padding.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT xPadding is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetXPadding(PEAK_BUFFER_HANDLE bufferHandle, size_t* xPadding);
/*!
 * \brief Asks the given buffer for its Y padding.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] yPadding The Y padding.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT yPadding is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetYPadding(PEAK_BUFFER_HANDLE bufferHandle, size_t* yPadding);
/*!
 * \brief Asks the given buffer for its timestamp in ticks.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] timestamp_ticks The timestamp in ticks.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT timestamp_ticks is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetTimestamp_ticks(PEAK_BUFFER_HANDLE bufferHandle, uint64_t* timestamp_ticks);
/*!
 * \brief Asks the given buffer for its timestamp in nanoseconds.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] timestamp_ns The timestamp in nanoseconds.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT timestamp_ns is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetTimestamp_ns(PEAK_BUFFER_HANDLE bufferHandle, uint64_t* timestamp_ns);
/*!
 * \brief Asks the given buffer whether it is queued.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] isQueued A flag telling whether the buffer is queued.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isQueued is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetIsQueued(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* isQueued);
/*!
 * \brief Asks the given buffer whether it is acquiring.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] isAcquiring A flag telling whether the buffer is acquiring.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isAcquiring is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetIsAcquiring(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* isAcquiring);
/*!
 * \brief Asks the given buffer whether it is incomplete.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] isIncomplete A flag telling whether the buffer is incomplete.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isIncomplete is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetIsIncomplete(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* isIncomplete);
/*!
 * \brief Asks the given buffer whether it has new data since the last delivery.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] hasNewData A flag telling whether the buffer has new data.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT hasNewData is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetHasNewData(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* hasNewData);
/*!
 * \brief Asks the given buffer whether it is has an image.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] hasImage A flag telling whether the buffer has an image.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT hasImage is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetHasImage(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* hasImage);
/*!
 * \brief Asks the given buffer whether it has chunks.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] hasChunks A flag telling whether the buffer has chunks.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT hasChunks is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetHasChunks(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* hasChunks);
/*!
 * \brief Tells the given buffer to update its chunks.
 *
 * \param[in] bufferHandle The buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_UpdateChunks(PEAK_BUFFER_HANDLE bufferHandle);
/*!
 * \brief Asks the given buffer for its number of chunks.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] numChunks The number of chunks.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numChunks is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetNumChunks(PEAK_BUFFER_HANDLE bufferHandle, size_t* numChunks);
/*!
 * \brief Asks the given buffer for the chunk with the given index.
 *
 * \param[in] bufferHandle The buffer.
 * \param[in] index The index.
 * \param[out] bufferChunkHandle The chunk being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bufferChunkHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetChunk(
    PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_CHUNK_HANDLE* bufferChunkHandle);
/*!
 * \brief Tells the given buffer to update its parts.
 *
 * \param[in] bufferHandle The buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_UpdateParts(PEAK_BUFFER_HANDLE bufferHandle);
/*!
 * \brief Asks the given buffer for its number of parts.
 *
 * \param[in] bufferHandle The buffer.
 * \param[out] numParts The number of parts.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numParts is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetNumParts(PEAK_BUFFER_HANDLE bufferHandle, size_t* numParts);
/*!
 * \brief Asks the given buffer for the part with the given index.
 *
 * \param[in] bufferHandle The buffer.
 * \param[in] index The index.
 * \param[out] bufferPartHandle The part being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bufferPartHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Buffer_GetPart(
    PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_PART_HANDLE* bufferPartHandle);

/*!
 * \brief Asks the given buffer chunk for its ID.
 *
 * \param[in] bufferChunkHandle The buffer chunk.
 * \param[out] id The ID.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferChunkHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT id is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferChunk_GetID(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, uint64_t* id);
/*!
 * \brief Asks the given buffer chunk for its base pointer.
 *
 * \param[in] bufferChunkHandle The buffer chunk.
 * \param[out] basePtr The base pointer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferChunkHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT basePtr is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferChunk_GetBasePtr(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, void** basePtr);
/*!
 * \brief Asks the given buffer chunk for its size.
 *
 * \param[in] bufferChunkHandle The buffer chunk.
 * \param[out] size The size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferChunkHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT size is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferChunk_GetSize(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, size_t* size);
/*!
 * \brief Asks the given buffer chunk for its parent buffer.
 *
 * \param[in] bufferChunkHandle The buffer chunk.
 * \param[out] bufferHandle The parent buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferChunkHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bufferHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferChunk_GetParentBuffer(
    PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, PEAK_BUFFER_HANDLE* bufferHandle);

/*!
 * \brief Asks the given buffer part for an information based on a specific command.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetInfo(PEAK_BUFFER_PART_HANDLE bufferPartHandle, int32_t infoCommand,
    int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given buffer part for its source ID.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] sourceId The source ID.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT sourceId is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetSourceID(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t* sourceId);
/*!
 * \brief Asks the given buffer part for its base pointer.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] basePtr The base pointer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT basePtr is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetBasePtr(PEAK_BUFFER_PART_HANDLE bufferPartHandle, void** basePtr);
/*!
 * \brief Asks the given buffer part for its size.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] size The size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT size is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetSize(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* size);
/*!
 * \brief Asks the given buffer part for its type.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] type The type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT type is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetType(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_PART_TYPE* type);
/*!
 * \brief Asks the given buffer part for its format.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] format The format.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT format is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetFormat(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t* format);
/*!
 * \brief Asks the given buffer part for its format namespace.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] formatNamespace The formatNamespace.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT formatNamespace is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetFormatNamespace(
    PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t* formatNamespace);
/*!
 * \brief Asks the given buffer part for its width.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] width The width.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT width is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetWidth(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* width);
/*!
 * \brief Asks the given buffer part for its height.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] height The height.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT height is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetHeight(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* height);
/*!
 * \brief Asks the given buffer part for its X offset.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] xOffset The X offset.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT xOffset is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetXOffset(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* xOffset);
/*!
 * \brief Asks the given buffer part for its Y offset.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] yOffset The Y offset.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT yOffset is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetYOffset(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* yOffset);
/*!
 * \brief Asks the given buffer part for its X padding.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] xPadding The X padding.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT xPadding is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetXPadding(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* xPadding);
/*!
 * \brief Asks the given buffer part for its delivered image height.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] deliveredImageHeight The delivered image height.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT deliveredImageHeight is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetDeliveredImageHeight(
    PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t* deliveredImageHeight);
/*!
 * \brief Asks the given buffer part for its parent buffer.
 *
 * \param[in] bufferPartHandle The buffer part.
 * \param[out] bufferHandle The parent buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferPartHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT parentBuffer is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BufferPart_GetParentBuffer(
    PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_HANDLE* bufferHandle);

/*!
 * \brief Asks the given module descriptor for its ID.
 *
 * \param[in] moduleDescriptorHandle The module descriptor.
 * \param[out] id The ID.
 * \param[in,out] idSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT idSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL idSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_ModuleDescriptor_GetID(
    PEAK_MODULE_DESCRIPTOR_HANDLE moduleDescriptorHandle, char* id, size_t* idSize);

/*!
 * \brief Asks the given module for its number of node maps.
 *
 * \param[in] moduleHandle The module.
 * \param[out] numNodeMaps The number of node maps.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE moduleHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numNodeMaps is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Module_GetNumNodeMaps(PEAK_MODULE_HANDLE moduleHandle, size_t* numNodeMaps);
/*!
 * \brief Asks the given module for the node map with the given index.
 *
 * \param[in] moduleHandle The module.
 * \param[in] index The index.
 * \param[out] nodeMapHandle The node map being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE moduleHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeMapHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Module_GetNodeMap(
    PEAK_MODULE_HANDLE moduleHandle, size_t index, PEAK_NODE_MAP_HANDLE* nodeMapHandle);
/*!
 * \brief Asks the given module for its port.
 *
 * \param[in] moduleHandle The module.
 * \param[out] portHandle The port.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE moduleHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT portHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Module_GetPort(PEAK_MODULE_HANDLE moduleHandle, PEAK_PORT_HANDLE* portHandle);

/*!
 * \brief Checks whether the given node map contains a node with the given name.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] nodeName The name of the node to find.
 * \param[in] nodeNameSize The size of the name.
 * \param[out] hasNode A flag telling whether the given node map contains a node with the given name.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle or eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeName and/or hasNode are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_NodeMap_GetHasNode(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, const char* nodeName, size_t nodeNameSize, PEAK_BOOL8* hasNode);
/*!
 * \brief Tells the given node map to find a node with the given name.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] nodeName The name of the node to find.
 * \param[in] nodeNameSize The size of the name.
 * \param[out] nodeHandle The found node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeName and/or nodeHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no node with the given name
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_NodeMap_FindNode(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, const char* nodeName, size_t nodeNameSize, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Tells the given node map to invalidate its nodes.
 *
 * \param[in] nodeMapHandle The node map.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_NodeMap_InvalidateNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle);
/*!
 * \brief Tells the given node map to poll its nodes.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] elapsedTime_ms The elapsed time since the last call in milliseconds.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_NodeMap_PollNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, int64_t elapsedTime_ms);
/*!
 * \brief Asks the given node map for its number of nodes.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[out] numNodes The number of nodes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numNodes is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_NodeMap_GetNumNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t* numNodes);
/*!
 * \brief Asks the given node map for the node with the given index.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] index The index.
 * \param[out] nodeHandle The node being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_NodeMap_GetNode(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t index, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Checks if the buffer contains chunks corresponding to the node map.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] bufferHandle The new buffer to be parsed.
 * \param[out] hasSupportedChunks A flag telling whether the buffer can be parsed.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle or bufferHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT hasSupportedChunks is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.1
 */
PEAK_C_API PEAK_NodeMap_GetHasBufferSupportedChunks(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8* hasSupportedChunks);
/*!
 * \brief Updates chunk information in the node map.
 *
 * When chunks are active, pass each new buffer to this method to parse the chunks and update the corresponding
 * chunk nodes in the NodeMap.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] bufferHandle The new buffer to be parsed.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle or bufferHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.1
 */
PEAK_C_API PEAK_NodeMap_UpdateChunkNodes(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle);
/*!
 * \brief Checks if the event contains data corresponding to the node map.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] eventHandle The new event to be parsed.
 * \param[out] hasSupportedData A flag telling whether the event can be parsed.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle or eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT hasSupportedData is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_NodeMap_GetHasEventSupportedData(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle, PEAK_BOOL8* hasSupportedData);
/*!
 * \brief Updates event information in the node map.
 *
 * When events are active, pass each new event to this method to parse the event data and update the corresponding
 * nodes in the NodeMap.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] eventHandle The new event to be parsed.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle or eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT Event does not have supported data
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_NodeMap_UpdateEventNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle);
/*!
 * \brief Tells the given node map to store the values of its streamable nodes to a file at the given file path.
 *
 * The stored file uses the GenApi persistence file format. It is not recommended to edit files using this format
 * manually unless you are familiar with the GenApi persistence functionality.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] filePath The path of the file to store the node values.
 * \param[in] filePathSize The size of the file path.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT filePath is a null pointer or invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.1
 */
PEAK_C_API PEAK_NodeMap_StoreToFile(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, const char* filePath, size_t filePathSize);
/*!
 * \brief Tells the given node map to load the values of its streamable nodes from a file at the given file path.
 *
 * The file to load has to use the GenApi persistence file format. It is not recommended to edit files using this
 * format manually unless you are familiar with the GenApi persistence functionality.
 *
 * \param[in] nodeMapHandle The node map.
 * \param[in] filePath The path of the file to load the node values from.
 * \param[in] filePathSize The size of the file path.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT filePath is a null pointer or invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.1
 */
PEAK_C_API PEAK_NodeMap_LoadFromFile(
    PEAK_NODE_MAP_HANDLE nodeMapHandle, const char* filePath, size_t filePathSize);
/*!
 * \brief Locks a recursive mutex on the NodeMap.
 *
 * Use this to synchronize NodeMap access from multiple threads.
 *
 * \note Each individual access is already protected by this mutex, so the nodemap doesn't need to
 *       be locked for those. But often, nodemap access consists of multiple calls, e.g.: First,
 *       (1.) change a selector value (e.g. set "GainSelector" node to "DigitalRed"), then (2.)
 *       access a selected node (e.g. set value in "Gain" node). If a different thread changes the
 *       selector value between (1.) and (2.), (2.) will access the wrong selected
 *       node. These kinds of calls should be protected by this lock.
 *
 * \param[in] nodeMapHandle The node map.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_NodeMap_Lock(PEAK_NODE_MAP_HANDLE nodeMapHandle);
/*!
 * \brief Unlocks the recursive mutex on the NodeMap.
 *
 * \param[in] nodeMapHandle The node map.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeMapHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_NodeMap_Unlock(PEAK_NODE_MAP_HANDLE nodeMapHandle);
/*!
 * \brief Casts the given node to an integer node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] integerNodeHandle The integer node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT integerNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to an integer node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToIntegerNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_INTEGER_NODE_HANDLE* integerNodeHandle);
/*!
 * \brief Casts the given node to a boolean node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] booleanNodeHandle The boolean node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT booleanNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to a boolean node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToBooleanNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_BOOLEAN_NODE_HANDLE* booleanNodeHandle);
/*!
 * \brief Casts the given node to a command node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] commandNodeHandle The command node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT commandNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to a command node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToCommandNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_COMMAND_NODE_HANDLE* commandNodeHandle);
/*!
 * \brief Casts the given node to a float node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] floatNodeHandle The command node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT floatNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to a float node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToFloatNode(PEAK_NODE_HANDLE nodeHandle, PEAK_FLOAT_NODE_HANDLE* floatNodeHandle);
/*!
 * \brief Casts the given node to a string node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] stringNodeHandle The string node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT stringNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to a string node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToStringNode(PEAK_NODE_HANDLE nodeHandle, PEAK_STRING_NODE_HANDLE* stringNodeHandle);
/*!
 * \brief Casts the given node to a register node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] registerNodeHandle The register node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT registerNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to a register node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToRegisterNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_REGISTER_NODE_HANDLE* registerNodeHandle);
/*!
 * \brief Casts the given node to a category node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] categoryNodeHandle The category node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT categoryNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to a category node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToCategoryNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_CATEGORY_NODE_HANDLE* categoryNodeHandle);
/*!
 * \brief Casts the given node to an enumeration node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] enumerationNodeHandle The enumeration node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT enumerationNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to an enumeration node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToEnumerationNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_NODE_HANDLE* enumerationNodeHandle);
/*!
 * \brief Casts the given node to an enumeration entry node.
 *
 * \param[in] nodeHandle The node.
 * \param[out] enumerationEntryNodeHandle The enumeration entry node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT enumerationEntryNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_CAST Node cannot be cast to an enumeration entry node
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_ToEnumerationEntryNode(
    PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE* enumerationEntryNodeHandle);
/*!
 * \brief Asks the given node for its name.
 *
 * \param[in] nodeHandle The node.
 * \param[out] name The name.
 * \param[in,out] nameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL nameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetName(PEAK_NODE_HANDLE nodeHandle, char* name, size_t* nameSize);
/*!
 * \brief Asks the given node for its display name.
 *
 * \param[in] nodeHandle The node.
 * \param[out] displayName The display name.
 * \param[in,out] displayNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL displayNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetDisplayName(PEAK_NODE_HANDLE nodeHandle, char* displayName, size_t* displayNameSize);
/*!
 * \brief Asks the given node for its namespace.
 *
 * \param[in] nodeHandle The node.
 * \param[out] _namespace The namespace.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT _namespace is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetNamespace(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_NAMESPACE* _namespace);
/*!
 * \brief Asks the given node for its visibility.
 *
 * \param[in] nodeHandle The node.
 * \param[out] visibility The visibility.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT visibility is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetVisibility(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_VISIBILITY* visibility);
/*!
 * \brief Asks the given node for its access status.
 *
 * \param[in] nodeHandle The node.
 * \param[out] accessStatus The access status.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT accessStatus is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetAccessStatus(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_ACCESS_STATUS* accessStatus);
/*!
 * \brief Asks the given node whether it is cacheable.
 *
 * \param[in] nodeHandle The node.
 * \param[out] isCacheable A flag telling whether the node is cacheable.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isCacheable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetIsCacheable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8* isCacheable);
/*!
 * \brief Asks the given node whether its access status is cacheable.
 *
 * \param[in] nodeHandle The node.
 * \param[out] isAccessStatusCacheable A flag telling whether the node's access status is cacheable.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isAccessStatusCacheable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetIsAccessStatusCacheable(
    PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8* isAccessStatusCacheable);
/*!
 * \brief Asks the given node whether it is streamable.
 *
 * \param[in] nodeHandle The node.
 * \param[out] isStreamable A flag telling whether the node is streamable.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isStreamable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetIsStreamable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8* isStreamable);
/*!
 * \brief Asks the given node whether it is deprecated.
 *
 * \param[in] nodeHandle The node.
 * \param[out] isDeprecated A flag telling whether the node is deprecated.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isDeprecated is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetIsDeprecated(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8* isDeprecated);
/*!
 * \brief Asks the given node whether it is a feature, i.e. it can be reached via
 * category nodes from a category node named "Root".
 *
 * \param[in] nodeHandle The node.
 * \param[out] isFeature A flag telling whether the node is a feature.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isFeature is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_Node_GetIsFeature(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8* isFeature);
/*!
 * \brief Asks the given node for its caching mode.
 *
 * \param[in] nodeHandle The node.
 * \param[out] cachingMode The caching mode.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT cachingMode is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetCachingMode(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CACHING_MODE* cachingMode);
/*!
 * \brief Asks the given node for its polling time.
 *
 * \param[in] nodeHandle The node.
 * \param[out] pollingTime_ms The polling time in milliseconds.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT pollingTime_ms is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetPollingTime(PEAK_NODE_HANDLE nodeHandle, int64_t* pollingTime_ms);
/*!
 * \brief Asks the given node for its tooltip.
 *
 * \param[in] nodeHandle The node.
 * \param[out] toolTip The tooltip.
 * \param[in,out] toolTipSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT toolTipSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL toolTipSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetToolTip(PEAK_NODE_HANDLE nodeHandle, char* toolTip, size_t* toolTipSize);
/*!
 * \brief Asks the given node for its description.
 *
 * \param[in] nodeHandle The node.
 * \param[out] description The description.
 * \param[in,out] descriptionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT descriptionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL descriptionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetDescription(PEAK_NODE_HANDLE nodeHandle, char* description, size_t* descriptionSize);
/*!
 * \brief Asks the given node for its type.
 *
 * \param[in] nodeHandle The node.
 * \param[out] type The type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT type is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetType(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_TYPE* type);
/*!
 * \brief Asks the given node for its parent node map.
 *
 * \param[in] nodeHandle The node.
 * \param[out] nodeMapHandle The type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeMapHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetParentNodeMap(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_MAP_HANDLE* nodeMapHandle);
/*!
 * \brief Tells the given node to find an invalidated node with the given name.
 *
 * \param[in] nodeHandle The node.
 * \param[in] name The name of the node to find.
 * \param[in] nameSize The size of the name.
 * \param[out] invalidatedNodeHandle The found node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT name and/or nodeHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no invalidated node with the given name
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_FindInvalidatedNode(
    PEAK_NODE_HANDLE nodeHandle, const char* name, size_t nameSize, PEAK_NODE_HANDLE* invalidatedNodeHandle);
/*!
 * \brief Asks the given node for its number of invalidated nodes.
 *
 * \param[in] nodeHandle The node.
 * \param[out] numInvalidatedNodes The number of invalidated nodes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numInvalidatedNodes is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetNumInvalidatedNodes(PEAK_NODE_HANDLE nodeHandle, size_t* numInvalidatedNodes);
/*!
 * \brief Asks the given node for the invalidated node with the given index.
 *
 * \param[in] nodeHandle The node.
 * \param[in] index The index.
 * \param[out] invalidatedNodeHandle The invalidated node being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT invalidatedNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetInvalidatedNode(
    PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE* invalidatedNodeHandle);
/*!
 * \brief Tells the given node to find an invalidating node with the given name.
 *
 * \param[in] nodeHandle The node.
 * \param[in] name The name of the node to find.
 * \param[in] nameSize The size of the name.
 * \param[out] invalidatingNodeHandle The found node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT name and/or nodeHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no invalidating node with the given name
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_FindInvalidatingNode(
    PEAK_NODE_HANDLE nodeHandle, const char* name, size_t nameSize, PEAK_NODE_HANDLE* invalidatingNodeHandle);
/*!
 * \brief Asks the given node for its number of invalidating nodes.
 *
 * \param[in] nodeHandle The node.
 * \param[out] numInvalidatingNodes The number of invalidating nodes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numInvalidatingNodes is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetNumInvalidatingNodes(PEAK_NODE_HANDLE nodeHandle, size_t* numInvalidatingNodes);
/*!
 * \brief Asks the given node for the invalidating node with the given index.
 *
 * \param[in] nodeHandle The node.
 * \param[in] index The index.
 * \param[out] invalidatingNodeHandle The invalidating node being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT invalidatingNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetInvalidatingNode(
    PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE* invalidatingNodeHandle);
/*!
 * \brief Tells the given node to find a selected node with the given name.
 *
 * \param[in] nodeHandle The node.
 * \param[in] name The name of the node to find.
 * \param[in] nameSize The size of the name.
 * \param[out] selectedNodeHandle The found node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT name and/or nodeHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no selected node with the given name
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_FindSelectedNode(
    PEAK_NODE_HANDLE nodeHandle, const char* name, size_t nameSize, PEAK_NODE_HANDLE* selectedNodeHandle);
/*!
 * \brief Asks the given node for its number of selected nodes.
 *
 * \param[in] nodeHandle The node.
 * \param[out] numSelectedNodes The number of selected nodes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numSelectedNodes is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetNumSelectedNodes(PEAK_NODE_HANDLE nodeHandle, size_t* numSelectedNodes);
/*!
 * \brief Asks the given node for the selected node with the given index.
 *
 * \param[in] nodeHandle The node.
 * \param[in] index The index.
 * \param[out] selectedNodeHandle The selected node being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT selectedNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetSelectedNode(
    PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE* selectedNodeHandle);
/*!
 * \brief Tells the given node to find a selecting node with the given name.
 *
 * \param[in] nodeHandle The node.
 * \param[in] name The name of the node to find.
 * \param[in] nameSize The size of the name.
 * \param[out] selectingNodeHandle The found node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT name and/or nodeHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no selecting node with the given name
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_FindSelectingNode(
    PEAK_NODE_HANDLE nodeHandle, const char* name, size_t nameSize, PEAK_NODE_HANDLE* selectingNodeHandle);
/*!
 * \brief Asks the given node for its number of selecting nodes.
 *
 * \param[in] nodeHandle The node.
 * \param[out] numSelectingNodes The number of selecting nodes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numSelectingNodes is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetNumSelectingNodes(PEAK_NODE_HANDLE nodeHandle, size_t* numSelectingNodes);
/*!
 * \brief Asks the given node for the selecting node with the given index.
 *
 * \param[in] nodeHandle The node.
 * \param[in] index The index.
 * \param[out] selectingNodeHandle The selecting node being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT selectingNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_GetSelectingNode(
    PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE* selectingNodeHandle);
/*!
 * \brief Registers a callback signaling a change at the given node.
 *
 * \param[in] nodeHandle The node.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_RegisterChangedCallback(PEAK_NODE_HANDLE nodeHandle,
    PEAK_NODE_CHANGED_CALLBACK callback, void* callbackContext,
    PEAK_NODE_CHANGED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the changed callback with the given handle from the given node.
 *
 * \param[in] nodeHandle The node.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE nodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Node_UnregisterChangedCallback(
    PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK_HANDLE callbackHandle);

/*!
 * \brief Casts the given integer node to a node.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_ToNode(
    PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given integer node for its minimum.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] minimum The minimum.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT minimum is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetMinimum(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t* minimum);
/*!
 * \brief Asks the given integer node for its maximum.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] maximum The maximum.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT maximum is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetMaximum(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t* maximum);
/*!
 * \brief Asks the given integer node for its increment.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] increment The increment.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT increment is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetIncrement(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t* increment);
/*!
 * \brief Asks the given integer node for its increment type.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] incrementType The increment type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT incrementType is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetIncrementType(
    PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_INCREMENT_TYPE* incrementType);
/*!
 * \brief Asks the given integer node for its valid values.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] validValues The valid values.
 * \param[in,out] validValuesSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT validValuesSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL validValuesSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetValidValues(
    PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t* validValues, size_t* validValuesSize);
/*!
 * \brief Asks the given integer node for its representation.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] representation The representation.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT representation is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetRepresentation(
    PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_REPRESENTATION* representation);
/*!
 * \brief Asks the given integer node for its unit.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[out] unit The unit.
 * \param[in,out] unitSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT unitSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetUnit(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, char* unit, size_t* unitSize);
/*!
 * \brief Asks the given integer node for its value.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[in] cacheUsePolicy The cache use policy.
 * \param[out] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT value is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_GetValue(
    PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, int64_t* value);
/*!
 * \brief Tells the given integer node to set the given value.
 *
 * \param[in] integerNodeHandle The integer node.
 * \param[in] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE integerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_IntegerNode_SetValue(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t value);

/*!
 * \brief Casts the given boolean node to a node.
 *
 * \param[in] booleanNodeHandle The boolean node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE booleanNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BooleanNode_ToNode(
    PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given boolean node for its value.
 *
 * \param[in] booleanNodeHandle The boolean node.
 * \param[in] cacheUsePolicy The cache use policy.
 * \param[out] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE booleanNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT value is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BooleanNode_GetValue(
    PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_BOOL8* value);
/*!
 * \brief Tells the given boolean node to set the given value.
 *
 * \param[in] booleanNodeHandle The boolean node.
 * \param[in] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE booleanNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_BooleanNode_SetValue(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_BOOL8 value);

/*!
 * \brief Casts the given command node to a node.
 *
 * \param[in] commandNodeHandle The command node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE commandNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CommandNode_ToNode(
    PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given command node whether its associated command is done.
 *
 * \param[in] commandNodeHandle The command node.
 * \param[out] isDone A flag telling whether the command is done.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE commandNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isDone is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CommandNode_GetIsDone(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_BOOL8* isDone);

/*!
 * \brief Tells the given command node to execute its associated command.
 *
 * \param[in] commandNodeHandle The command node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE commandNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CommandNode_Execute(PEAK_COMMAND_NODE_HANDLE commandNodeHandle);
/*!
 * \brief Waits (blocking) without timeout until the previously executed command is finished, i.e.
 * PEAK_CommandNode_GetIsDone() returns true.
 *
 * \param[in] commandNodeHandle The command node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE commandNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CommandNode_WaitUntilDoneInfinite(PEAK_COMMAND_NODE_HANDLE commandNodeHandle);
/*!
 * \brief Waits (blocking) until the previously executed command is finished,
 *        i.e. PEAK_CommandNode_GetIsDone() returns true.
 *
 * \param[in] commandNodeHandle The command node.
 * \param[in] waitTimeout_ms The waiting time for the stop criterion.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE commandNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CommandNode_WaitUntilDone(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, uint64_t waitTimeout_ms);

/*!
 * \brief Casts the given float node to a node.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_ToNode(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given float node for its minimum.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] minimum The minimum.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT minimum is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetMinimum(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double* minimum);
/*!
 * \brief Asks the given float node for its maximum.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] maximum The maximum.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT maximum is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetMaximum(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double* maximum);
/*!
 * \brief Asks the given float node for its increment.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] increment The increment.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT increment is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetIncrement(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double* increment);
/*!
 * \brief Asks the given float node for its increment type.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] incrementType The increment type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT incrementType is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetIncrementType(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_INCREMENT_TYPE* incrementType);
/*!
 * \brief Asks the given float node for its valid values.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] validValues The valid values.
 * \param[in,out] validValuesSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT validValuesSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL validValuesSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetValidValues(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double* validValues, size_t* validValuesSize);
/*!
 * \brief Asks the given float node for its representation.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] representation The representation.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT representation is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetRepresentation(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_REPRESENTATION* representation);
/*!
 * \brief Asks the given float node for its unit.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] unit The unit.
 * \param[in,out] unitSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT unitSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetUnit(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, char* unit, size_t* unitSize);
/*!
 * \brief Asks the given float node for its display notation.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] displayNotation The display notation.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayNotation is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetDisplayNotation(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_DISPLAY_NOTATION* displayNotation);
/*!
 * \brief Asks the given float node for its display precision.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] displayPrecision The display precision.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT displayPrecision is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetDisplayPrecision(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, int64_t* displayPrecision);
/*!
 * \brief Asks the given float node whether it has a constant increment.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[out] hasConstantIncrement A flag telling whether the float node has a constant increment.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT hasConstantIncrement is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetHasConstantIncrement(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_BOOL8* hasConstantIncrement);
/*!
 * \brief Asks the given float node for its value.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[in] cacheUsePolicy The cache use policy.
 * \param[out] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT value is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_GetValue(
    PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, double* value);
/*!
 * \brief Tells the given float node to set the given value.
 *
 * \param[in] floatNodeHandle The float node.
 * \param[in] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE floatNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FloatNode_SetValue(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double value);

/*!
 * \brief Casts the given string node to a node.
 *
 * \param[in] stringNodeHandle The string node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE stringNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_StringNode_ToNode(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given string node for its maximum length.
 *
 * \param[in] stringNodeHandle The string node.
 * \param[out] maximumLength The maximum length.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE stringNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT maximumLength is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_StringNode_GetMaximumLength(PEAK_STRING_NODE_HANDLE stringNodeHandle, int64_t* maximumLength);
/*!
 * \brief Asks the given string node for its value.
 *
 * \param[in] stringNodeHandle The string node.
 * \param[in] cacheUsePolicy The cache use policy.
 * \param[out] value The value.
 * \param[in,out] valueSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE stringNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT valueSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_StringNode_GetValue(PEAK_STRING_NODE_HANDLE stringNodeHandle,
    PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, char* value, size_t* valueSize);
/*!
 * \brief Tells the given string node to set the given value.
 *
 * \param[in] stringNodeHandle The string node.
 * \param[in] value The value.
 * \param[in] valueSize The size of the value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE stringNodeHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_StringNode_SetValue(
    PEAK_STRING_NODE_HANDLE stringNodeHandle, const char* value, size_t valueSize);

/*!
 * \brief Casts the given register node to a node.
 *
 * \param[in] registerNodeHandle The register node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE registerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RegisterNode_ToNode(
    PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given register node for its address.
 *
 * \param[in] registerNodeHandle The register node.
 * \param[out] address The address.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE registerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT address is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RegisterNode_GetAddress(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, uint64_t* address);
/*!
 * \brief Asks the given register node for its length.
 *
 * \param[in] registerNodeHandle The register node.
 * \param[out] length The length.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE registerNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT length is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RegisterNode_GetLength(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, size_t* length);
/*!
 * \brief Tells the given register node to read the given amount of bytes.
 *
 * \param[in] registerNodeHandle The register node.
 * \param[in] cacheUsePolicy The cache use policy.
 * \param[out] bytesToRead The bytes to read.
 * \param[in] bytesToReadSize The amount of bytes to read.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE stringNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bytesToRead is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RegisterNode_Read(PEAK_REGISTER_NODE_HANDLE registerNodeHandle,
    PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, uint8_t* bytesToRead, size_t bytesToReadSize);
/*!
 * \brief Tells the given register node to write the given bytes.
 *
 * \param[in] registerNodeHandle The register node.
 * \param[out] bytesToWrite The bytes to write.
 * \param[in] bytesToWriteSize The amount of bytes to write.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE stringNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bytesToWrite is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_RegisterNode_Write(
    PEAK_REGISTER_NODE_HANDLE registerNodeHandle, const uint8_t* bytesToWrite, size_t bytesToWriteSize);

/*!
 * \brief Casts the given category node to a node.
 *
 * \param[in] categoryNodeHandle The category node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE categoryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CategoryNode_ToNode(
    PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given category node for its number of sub nodes.
 *
 * \param[in] categoryNodeHandle The category node.
 * \param[out] numSubNodes The number of sub nodes.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE categoryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numSubNodes is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CategoryNode_GetNumSubNodes(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t* numSubNodes);
/*!
 * \brief Asks the given category node for the sub node with the given index.
 *
 * \param[in] categoryNodeHandle The category node.
 * \param[in] index The index.
 * \param[out] nodeHandle The sub node being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE categoryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_CategoryNode_GetSubNode(
    PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t index, PEAK_NODE_HANDLE* nodeHandle);

/*!
 * \brief Casts the given enumeration node to a node.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_ToNode(
    PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given enumeration node for its current entry.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] cacheUsePolicy The cache use policy.
 * \param[out] enumerationEntryNodeHandle The current entry.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT enumerationEntryNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_GetCurrentEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle,
    PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_ENUMERATION_ENTRY_NODE_HANDLE* enumerationEntryNodeHandle);
/*!
 * \brief Tells the given enumeration node to set its current entry to the given entry.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] enumerationEntryNodeHandle The entry.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle and/or enumerationEntryNodeHandle are/is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT There is no matching entry in this enumeration node.
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_SetCurrentEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle,
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle);
/*!
 * \brief Tells the given enumeration node to set its current entry by the given symbolic value.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] symbolicValue The symbolic value.
 * \param[in] symbolicValueSize Size of the given buffer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT symbolicValue is a null pointer
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT There is no entry with this symbolicValue in this enumeration node.
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue(
    PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char* symbolicValue, size_t symbolicValueSize);
/*!
 * \brief Tells the given enumeration node to set its current entry by the given value.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT There is no entry with this value in this enumeration node.
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_SetCurrentEntryByValue(
    PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value);
/*!
 * \brief Tells the given enumeration node to find a entry with the given symbolic value.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] symbolicValue The symbolic value.
 * \param[in] symbolicValueSize The size of the symbolic value.
 * \param[out] enumerationEntryNodeHandle The found entry.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT symbolicValue and/or enumerationEntryNodeHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no entry with the given symbolic value
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_FindEntryBySymbolicValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle,
    const char* symbolicValue, size_t symbolicValueSize,
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE* enumerationEntryNodeHandle);
/*!
 * \brief Tells the given enumeration node to find a entry with the given value.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] value The value.
 * \param[out] enumerationEntryNodeHandle The found entry.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT enumerationEntryNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_NOT_FOUND There is no entry with the given value
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_FindEntryByValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle,
    int64_t value, PEAK_ENUMERATION_ENTRY_NODE_HANDLE* enumerationEntryNodeHandle);
/*!
 * \brief Asks the given enumeration node for its number of entries.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[out] numEntries The number of entries.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numEntries is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_GetNumEntries(
    PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t* numEntries);
/*!
 * \brief Asks the given enumeration node for the entry with the given index.
 *
 * \param[in] enumerationNodeHandle The enumeration node.
 * \param[in] index The index.
 * \param[out] enumerationEntryNodeHandle The entry being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT enumerationEntryNodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationNode_GetEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t index,
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE* enumerationEntryNodeHandle);

/*!
 * \brief Casts the given enumeration entry node to a node.
 *
 * \param[in] enumerationEntryNodeHandle The enumeration entry node.
 * \param[out] nodeHandle The node.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationEntryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nodeHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationEntryNode_ToNode(
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_NODE_HANDLE* nodeHandle);
/*!
 * \brief Asks the given enumeration entry node whether it is self clearing.
 *
 * \param[in] enumerationEntryNodeHandle The enumeration entry node.
 * \param[out] isSelfClearing A flag telling whether the enumeration entry node is self clearing.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationEntryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isSelfClearing is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationEntryNode_GetIsSelfClearing(
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_BOOL8* isSelfClearing);
/*!
 * \brief Asks the given enumeration entry node for its symbolic value.
 *
 * \param[in] enumerationEntryNodeHandle The enumeration entry node.
 * \param[out] symbolicValue The symbolic value.
 * \param[in,out] symbolicValueSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationEntryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT symbolicValueSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL symbolicValueSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationEntryNode_GetSymbolicValue(
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, char* symbolicValue, size_t* symbolicValueSize);
/*!
 * \brief Asks the given enumeration entry node for its value.
 *
 * \param[in] enumerationEntryNodeHandle The enumeration entry node.
 * \param[out] value The value.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE enumerationEntryNodeHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT value is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EnumerationEntryNode_GetValue(
    PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, int64_t* value);

/*!
 * \brief Asks the given port for an information based on a specific command.
 *
 * \param[in] portHandle The port.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetInfo(
    PEAK_PORT_HANDLE portHandle, int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given port for its ID.
 *
 * \param[in] portHandle The port.
 * \param[out] id The ID.
 * \param[in,out] idSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT idSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL idSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetID(PEAK_PORT_HANDLE portHandle, char* id, size_t* idSize);
/*!
 * \brief Asks the given port for its name.
 *
 * \param[in] portHandle The port.
 * \param[out] name The name.
 * \param[in,out] nameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT nameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL nameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetName(PEAK_PORT_HANDLE portHandle, char* name, size_t* nameSize);
/*!
 * \brief Asks the given port for its vendor name.
 *
 * \param[in] portHandle The port.
 * \param[out] vendorName The vendor name.
 * \param[in,out] vendorNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT vendorNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL vendorNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetVendorName(PEAK_PORT_HANDLE portHandle, char* vendorName, size_t* vendorNameSize);
/*!
 * \brief Asks the given port for its model name.
 *
 * \param[in] portHandle The port.
 * \param[out] modelName The model name.
 * \param[in,out] modelNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT modelNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL modelNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetModelName(PEAK_PORT_HANDLE portHandle, char* modelName, size_t* modelNameSize);
/*!
 * \brief Asks the given port for its version.
 *
 * \param[in] portHandle The port.
 * \param[out] version The version.
 * \param[in,out] versionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetVersion(PEAK_PORT_HANDLE portHandle, char* version, size_t* versionSize);
/*!
 * \brief Asks the given port for its TL type.
 *
 * \param[in] portHandle The port.
 * \param[out] tlType The TL type.
 * \param[in,out] tlTypeSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT tlTypeSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL tlTypeSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetTLType(PEAK_PORT_HANDLE portHandle, char* tlType, size_t* tlTypeSize);
/*!
 * \brief Asks the given port for its module name.
 *
 * \param[in] portHandle The port.
 * \param[out] moduleName The module name.
 * \param[in,out] moduleNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT moduleNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL moduleNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetModuleName(PEAK_PORT_HANDLE portHandle, char* moduleName, size_t* moduleNameSize);
/*!
 * \brief Asks the given port for its data endianness.
 *
 * \param[in] portHandle The port.
 * \param[out] dataEndianness The data endianness.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT dataEndianness is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetDataEndianness(PEAK_PORT_HANDLE portHandle, PEAK_ENDIANNESS* dataEndianness);
/*!
 * \brief Asks the given port whether it is readable.
 *
 * \param[in] portHandle The port.
 * \param[out] isReadable A flag telling whether the port is readable.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isReadable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetIsReadable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8* isReadable);
/*!
 * \brief Asks the given port whether it is writable.
 *
 * \param[in] portHandle The port.
 * \param[out] isWritable A flag telling whether the port is writable.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isWritable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetIsWritable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8* isWritable);
/*!
 * \brief Asks the given port whether it is available.
 *
 * \param[in] portHandle The port.
 * \param[out] isAvailable A flag telling whether the port is available.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isAvailable is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetIsAvailable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8* isAvailable);
/*!
 * \brief Asks the given port whether it is implemented.
 *
 * \param[in] portHandle The port.
 * \param[out] isImplemented A flag telling whether the port is implemented.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isImplemented is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetIsImplemented(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8* isImplemented);
/*!
 * \brief Tells the given port to read the given amount of bytes at the given address.
 *
 * \param[in] portHandle The port.
 * \param[in] address The address.
 * \param[out] bytesToRead The buffer for the bytes to read.
 * \param[in] bytesToReadSize The amount of bytes to read.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ADDRESS address is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bytesToRead is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_Read(
    PEAK_PORT_HANDLE portHandle, uint64_t address, uint8_t* bytesToRead, size_t bytesToReadSize);
/*!
 * \brief Tells the given port to write the given amount of bytes at the given address.
 *
 * \param[in] portHandle The port.
 * \param[in] address The address.
 * \param[in] bytesToWrite The bytes to write.
 * \param[in] bytesToWriteSize The amount of bytes to write.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ADDRESS address is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT bytesToWrite is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE The given value is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_Write(
    PEAK_PORT_HANDLE portHandle, uint64_t address, const uint8_t* bytesToWrite, size_t bytesToWriteSize);
/*!
 * \brief Asks the given port for its number of URLs.
 *
 * \param[in] portHandle The port.
 * \param[out] numUrls The number of URLs.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numUrls is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetNumURLs(PEAK_PORT_HANDLE portHandle, size_t* numUrls);
/*!
 * \brief Asks the given port for the URL with the given index.
 *
 * \param[in] portHandle The port.
 * \param[in] index The index.
 * \param[out] portUrlHandle The URL being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT portUrlHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Port_GetURL(PEAK_PORT_HANDLE portHandle, size_t index, PEAK_PORT_URL_HANDLE* portUrlHandle);

/*!
 * \brief Asks the given port URL for an information based on a specific command.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetInfo(
    PEAK_PORT_URL_HANDLE portUrlHandle, int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given port URL for its URL string.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] url The URL string.
 * \param[in,out] urlSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT urlSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL urlSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetURL(PEAK_PORT_URL_HANDLE portUrlHandle, char* url, size_t* urlSize);
/*!
 * \brief Asks the given port URL for its scheme.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] scheme The scheme.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT scheme is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetScheme(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_URL_SCHEME* scheme);
/*!
 * \brief Asks the given port URL for its file name.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileName The file name.
 * \param[in,out] fileNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL fileNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileName(PEAK_PORT_URL_HANDLE portUrlHandle, char* fileName, size_t* fileNameSize);
/*!
 * \brief Asks the given port URL for its file register address.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileRegisterAddress The file register address.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileRegisterAddress is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileRegisterAddress(
    PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t* fileRegisterAddress);
/*!
 * \brief Asks the given port URL for its file size.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileSize The file size.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileSize(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t* fileSize);
/*!
 * \brief Asks the given port URL for its file SHA1 hash.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileSha1Hash The file SHA1 hash.
 * \param[in,out] fileSha1HashSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileSha1HashSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileSHA1Hash(
    PEAK_PORT_URL_HANDLE portUrlHandle, uint8_t* fileSha1Hash, size_t* fileSha1HashSize);
/*!
 * \brief Asks the given port URL for the major component of its file version.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileVersionMajor The major component of the file version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileVersionMajor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileVersionMajor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t* fileVersionMajor);
/*!
 * \brief Asks the given port URL for the minor component of its file version.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileVersionMinor The minor component of the file version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileVersionMinor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileVersionMinor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t* fileVersionMinor);
/*!
 * \brief Asks the given port URL for the subminor component of its file version.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileVersionSubminor The subminor component of the file version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileVersionSubminor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileVersionSubminor(
    PEAK_PORT_URL_HANDLE portUrlHandle, int32_t* fileVersionSubminor);
/*!
 * \brief Asks the given port URL for the major component of its file schema version.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileSchemaVersionMajor The major component of the file schema version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileSchemaVersionMajor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileSchemaVersionMajor(
    PEAK_PORT_URL_HANDLE portUrlHandle, int32_t* fileSchemaVersionMajor);
/*!
 * \brief Asks the given port URL for the minor component of its file schema version.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] fileSchemaVersionMinor The minor component of the file schema version.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileSchemaVersionMinor is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetFileSchemaVersionMinor(
    PEAK_PORT_URL_HANDLE portUrlHandle, int32_t* fileSchemaVersionMinor);
/*!
 * \brief Asks the given port URL for its parent port.
 *
 * \param[in] portUrlHandle The port URL.
 * \param[out] portHandle The parent port.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE portUrlHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT portHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_PortURL_GetParentPort(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_HANDLE* portHandle);

/*!
 * \brief Tells the given event-supporting module to enable events of the given type.
 *
 * \param[in] eventSupportingModuleHandle The event-supporting module.
 * \param[in] eventType The event type to enable.
 * \param[out] eventControllerHandle The event controller being associated with the given event type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventSupportingModuleHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventControllerHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventSupportingModule_EnableEvents(
    PEAK_EVENT_SUPPORTING_MODULE_HANDLE eventSupportingModuleHandle, PEAK_EVENT_TYPE eventType,
    PEAK_EVENT_CONTROLLER_HANDLE* eventControllerHandle);

/*!
 * \brief Asks the given event controller for an information based on a specific command.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_GetInfo(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle,
    int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the event controller for its number of events in queue.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[out] numEventsInQueue The number of events in queue.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numEventsInQueue is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_GetNumEventsInQueue(
    PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t* numEventsInQueue);
/*!
 * \brief Asks the event controller for its number of fired events.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[out] numEventsFired The number of fired events.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numEventsFired is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_GetNumEventsFired(
    PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t* numEventsFired);
/*!
 * \brief Asks the event controller for the maximum size of an event.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[out] eventMaxSize The maximum size of an event.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventMaxSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_GetEventMaxSize(
    PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t* eventMaxSize);
/*!
 * \brief Asks the event controller for the maximum size of an event's data.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[out] eventDataMaxSize The maximum size of an event's data.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventDataMaxSize is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_GetEventDataMaxSize(
    PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t* eventDataMaxSize);
/*!
 * \brief Asks the event controller for its controlled event type.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[out] controlledEventType The controlled event type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT controlledEventType is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_GetControlledEventType(
    PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, PEAK_EVENT_TYPE* controlledEventType);
/*!
 * \brief Tells the given event controller to wait for an event with the given timeout.
 *
 * \param[in] eventControllerHandle The event controller.
 * \param[in] timeout_ms The timeout.
 * \param[out] eventHandle The event.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT eventHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_WaitForEvent(
    PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t timeout_ms, PEAK_EVENT_HANDLE* eventHandle);
/*!
 * \brief Tells the given event controller to kill a wait for an event.
 *
 * If there are no wait operations at the point the function gets called, the kill request gets stored. In this case
 * the next wait operation gets killed immediately.
 *
 * \param[in] eventControllerHandle The event controller.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_KillWait(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
/*!
 * \brief Tells the given event controller to flush all events in queue.
 *
 * \param[in] eventControllerHandle The event controller.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_FlushEvents(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
/*!
 * \brief Destroys the given event controller.
 *
 * \param[in] eventControllerHandle The event controller.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventControllerHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_EventController_Destruct(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);

/*!
 * \brief Asks the given event for an information based on a specific command.
 *
 * \param[in] eventHandle The event.
 * \param[in] infoCommand The command of interest.
 * \param[out] infoDataType The data type of the returned information data.
 * \param[out] info The returned information data.
 * \param[in,out] infoSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT infoDataType and/or infoSize are/is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL infoSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Event_GetInfo(
    PEAK_EVENT_HANDLE eventHandle, int32_t infoCommand, int32_t* infoDataType, uint8_t* info, size_t* infoSize);
/*!
 * \brief Asks the given event for its ID.
 *
 * \param[in] eventHandle The event.
 * \param[out] id The ID.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT id is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Event_GetID(PEAK_EVENT_HANDLE eventHandle, uint64_t* id);
/*!
 * \brief Asks the given event for its data.
 *
 * The delivered data depend on the event type.
 *
 * \param[in] eventHandle The event.
 * \param[out] data The data.
 * \param[in,out] dataSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT dataSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL dataSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Event_GetData(PEAK_EVENT_HANDLE eventHandle, uint8_t* data, size_t* dataSize);
/*!
 * \brief Asks the given event for its type.
 *
 * \param[in] eventHandle The event.
 * \param[out] type The type.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT type is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Event_GetType(PEAK_EVENT_HANDLE eventHandle, PEAK_EVENT_TYPE* type);
/*!
 * \brief Asks the given event for its raw data.
 *
 * The delivered data depend on the underlying transport layer
 * (GEV, USB3, ...) and the event type.
 * (e.g. If the underlying CTI implements GEV and the event is a remote device event,
 * the delivered data will be the event raw data of a GEV event)
 *
 * \param[in] eventHandle The event.
 * \param[out] rawData The raw data.
 * \param[in,out] rawDataSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT rawDataSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL rawDataSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_Event_GetRawData(PEAK_EVENT_HANDLE eventHandle, uint8_t* rawData, size_t* rawDataSize);
/*!
 * \brief Destroys the given event.
 *
 * \param[in] eventHandle The event.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE eventHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_Event_Destruct(PEAK_EVENT_HANDLE eventHandle);

/*!
 * \brief Creates a firmware updater.
 *
 * \param[out] firmwareUpdaterHandle The firmware updater.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT firmwareUpdaterHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdater_Construct(PEAK_FIRMWARE_UPDATER_HANDLE* firmwareUpdaterHandle);
/*!
 * \brief Tells the given firmware updater to collect all update information from the .guf file.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 * \param[in] gufPath The path to the *.guf file.
 * \param[in] gufPathSize The size of the given *.guf file path.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle and/or deviceDescriptorHandle are/is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT gufPath is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation(
    PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char* gufPath, size_t gufPathSize);
/*!
 * \brief Tells the given firmware updater to collect all update information fitting to the given device.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 * \param[in] gufPath The path to the *.guf file.
 * \param[in] gufPathSize The size of the given *.guf file path.
 * \param[in] deviceDescriptorHandle The device the update information have to fit to.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle and/or deviceDescriptorHandle are/is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT gufPath is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation(
    PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char* gufPath, size_t gufPathSize,
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle);
/*!
 * \brief Asks the given firmware updater for its number of update information.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 * \param[out] numFirmwareUpdateInformation The number of update information.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT numFirmwareUpdateInformation is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation(
    PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t* numFirmwareUpdateInformation);
/*!
 * \brief Asks the given firmware updater for the update information with the given index.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 * \param[in] index The index.
 * \param[out] firmwareUpdateInformationHandle The update information being associated with the given index.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT firmwareUpdateInformationHandle is a null pointer
 * \return PEAK_RETURN_CODE_OUT_OF_RANGE index is out of range
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdater_GetFirmwareUpdateInformation(
    PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t index,
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE* firmwareUpdateInformationHandle);
/*!
 * \brief Tells the given firmware updater to update the given device with the given update information.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 * \param[in] deviceDescriptorHandle The device.
 * \param[in] firmwareUpdateInformationHandle The update information.
 * \param[in] firmwareUpdateProgressObserverHandle Used to observe the update process. Can be a null pointer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle and/or deviceDescriptorHandle and/or
 *                                        firmwareUpdateInformationHandle are/is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 * \return PEAK_RETURN_CODE_TIMEOUT The device can't be found after a reboot within 60 seconds.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdater_UpdateDevice(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle,
    PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle,
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle);
/*!
 * \brief Tells the given firmware updater to update the given device with the given update information.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 * \param[in] deviceDescriptorHandle The device.
 * \param[in] firmwareUpdateInformationHandle The update information.
 * \param[in] firmwareUpdateProgressObserverHandle Used to observe the update process. Can be a null pointer.
 * \param[in] deviceResetDiscoveryTimeout_ms Time to wait for a device to reboot during the update.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle and/or deviceDescriptorHandle and/or
 *                                        firmwareUpdateInformationHandle are/is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 * \return PEAK_RETURN_CODE_TIMEOUT The device can't be found after a reboot within the deviceResetDiscoveryTimeout.
 *
 * \since 1.2
 */
PEAK_C_API PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout(
    PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle,
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle,
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    uint64_t deviceResetDiscoveryTimeout_ms);
/*!
 * \brief Destroys the given firmware updater.
 *
 * \param[in] firmwareUpdaterHandle The firmware updater.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdaterHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdater_Destruct(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle);

/*!
 * \brief Asks the given firmware update information whether it is valid.
 *
 * \param[in] firmwareUpdateInformationHandle The firmware update information.
 * \param[out] isValid A flag telling whether the firmware update information is valid.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT isValid is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetIsValid(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_BOOL8* isValid);
/*!
 * \brief Asks the given firmware update information for its file name.
 *
 * \param[in] firmwareUpdateInformationHandle The port.
 * \param[out] fileName The file name.
 * \param[in,out] fileNameSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT fileNameSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL fileNameSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetFileName(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char* fileName, size_t* fileNameSize);
/*!
 * \brief Asks the given firmware update information for its description.
 *
 * \param[in] firmwareUpdateInformationHandle The port.
 * \param[out] description The description.
 * \param[in,out] descriptionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT descriptionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL descriptionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetDescription(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char* description,
    size_t* descriptionSize);
/*!
 * \brief Asks the given firmware update information for its version.
 *
 * \param[in] firmwareUpdateInformationHandle The port.
 * \param[out] version The version.
 * \param[in,out] versionSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetVersion(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char* version, size_t* versionSize);
/*!
 * \brief Asks the given firmware update information for its version extraction pattern.
 *
 * \param[in] firmwareUpdateInformationHandle The port.
 * \param[out] versionExtractionPattern The version extraction pattern.
 * \param[in,out] versionExtractionPatternSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionExtractionPatternSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL versionExtractionPatternSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char* versionExtractionPattern,
    size_t* versionExtractionPatternSize);
/*!
 * \brief Asks the given firmware update information for its version style.
 *
 * \param[in] firmwareUpdateInformationHandle The firmware update information.
 * \param[out] versionStyle The version style.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT versionStyle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetVersionStyle(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle,
    PEAK_FIRMWARE_UPDATE_VERSION_STYLE* versionStyle);
/*!
 * \brief Asks the given firmware update information for its release notes.
 *
 * \param[in] firmwareUpdateInformationHandle The port.
 * \param[out] releaseNotes The release notes.
 * \param[in,out] releaseNotesSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT releaseNotesSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL releaseNotesSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetReleaseNotes(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char* releaseNotes,
    size_t* releaseNotesSize);
/*!
 * \brief Asks the given firmware update information for its release notes URL.
 *
 * \param[in] firmwareUpdateInformationHandle The port.
 * \param[out] releaseNotesUrl The release notes URL.
 * \param[in,out] releaseNotesUrlSize IN: Size of the given buffer - OUT: Size of the returned data
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT releaseNotesUrlSize is a null pointer
 * \return PEAK_RETURN_CODE_BUFFER_TOO_SMALL releaseNotesUrlSize is too small
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetReleaseNotesURL(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char* releaseNotesUrl,
    size_t* releaseNotesUrlSize);
/*!
 * \brief Asks the given firmware update information for its user set persistence.
 *
 * \param[in] firmwareUpdateInformationHandle The firmware update information.
 * \param[out] userSetPersistence The user set persistence.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT userSetPersistence is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetUserSetPersistence(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle,
    PEAK_FIRMWARE_UPDATE_PERSISTENCE* userSetPersistence);
/*!
 * \brief Asks the given firmware update information for its sequencer set persistence.
 *
 * \param[in] firmwareUpdateInformationHandle The firmware update information.
 * \param[out] sequencerSetPersistence The sequencer set persistence.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateInformationHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT sequencerSetPersistence is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence(
    PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle,
    PEAK_FIRMWARE_UPDATE_PERSISTENCE* sequencerSetPersistence);

/*!
 * \brief Creates a firmware update progress observer.
 *
 * \param[out] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT firmwareUpdateProgressObserverHandle is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_Construct(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE* firmwareUpdateProgressObserverHandle);
/*!
 * \brief Registers a callback signaling a started update at the given firmware update progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK callback, void* callbackContext,
    PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the update started callback with the given handle from the given firmware update progress
 *        observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a started update step at the given firmware update progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK callback, void* callbackContext,
    PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the update step started callback with the given handle from the given firmware update progress
 *        observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a progress change within an update step at the given firmware update progress
 *        observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK callback, void* callbackContext,
    PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the update step progress changed callback with the given handle from the given firmware update
 *        progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a finished update step at the given firmware update progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK callback, void* callbackContext,
    PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the update step finished callback with the given handle from the given firmware update
 *        progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a finished update at the given firmware update progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK callback, void* callbackContext,
    PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the update finished callback with the given handle from the given firmware update progress
 *        observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Registers a callback signaling a failed update at the given firmware update progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callback The callback.
 * \param[in] callbackContext The callback context.
 * \param[out] callbackHandle The handle for the registered callback.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_INVALID_ARGUMENT callback and/or callbackHandle are/is a null pointer
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK callback, void* callbackContext,
    PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE* callbackHandle);
/*!
 * \brief Unregisters the update failed callback with the given handle from the given firmware update progress
 *        observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 * \param[in] callbackHandle The handle of the callback to unregister.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle,
    PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE callbackHandle);
/*!
 * \brief Destroys the given firmware update progress observer.
 *
 * \param[in] firmwareUpdateProgressObserverHandle The firmware update progress observer.
 *
 * \return PEAK_RETURN_CODE_SUCCESS Everything was fine
 * \return PEAK_RETURN_CODE_INVALID_HANDLE firmwareUpdateProgressObserverHandle is invalid
 * \return PEAK_RETURN_CODE_ERROR An internal error has occurred.
 *
 * \since 1.0
 */
PEAK_C_API PEAK_FirmwareUpdateProgressObserver_Destruct(
    PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle);

#    ifdef __cplusplus
} // extern "C"
#    endif

#else
#    ifdef __cplusplus
} // extern "C"
#    endif

#    include <peak/backend/peak_dynamic_loader.h>
#endif // !PEAK_DYNAMIC_LOADING
