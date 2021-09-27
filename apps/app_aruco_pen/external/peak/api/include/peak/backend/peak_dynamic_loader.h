/*!
 * \file    peak_dynamic_loader.h
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */
#pragma once

#include <peak/backend/peak_backend.h>
#include <string>
#include <cstdint>

#ifdef __linux__
    #include <dlfcn.h>
#else
    #include <vector>
    #include <windows.h>
    #include <tchar.h>
#endif
 
#include <stdexcept>

namespace peak
{
namespace dynamic
{

typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_Initialize)();
typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_Close)();
typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_IsInitialized)(PEAK_BOOL8 * isInitialized);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_GetVersionMajor)(uint32_t * libraryVersionMajor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_GetVersionMinor)(uint32_t * libraryVersionMinor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_GetVersionSubminor)(uint32_t * libraryVersionSubminor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Library_GetLastError)(PEAK_RETURN_CODE * lastErrorCode, char * lastErrorDescription, size_t * lastErrorDescriptionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnvironmentInspector_UpdateCTIPaths)();
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnvironmentInspector_GetNumCTIPaths)(size_t * numCtiPaths);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnvironmentInspector_GetCTIPath)(size_t index, char * ctiPath, size_t * ctiPathSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_ProducerLibrary_Construct)(const char * ctiPath, size_t ctiPathSize, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_ProducerLibrary_GetKey)(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_ProducerLibrary_GetSystem)(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, PEAK_SYSTEM_DESCRIPTOR_HANDLE * systemDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_ProducerLibrary_Destruct)(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_ToModuleDescriptor)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetKey)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetInfo)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetDisplayName)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetVendorName)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * vendorName, size_t * vendorNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetModelName)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * modelName, size_t * modelNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetVersion)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * version, size_t * versionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetTLType)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetCTIFileName)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * ctiFileName, size_t * ctiFileNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetCTIFullPath)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * ctiFullPath, size_t * ctiFullPathSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetGenTLVersionMajor)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t * gentlVersionMajor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetGenTLVersionMinor)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t * gentlVersionMinor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetCharacterEncoding)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_CHARACTER_ENCODING * characterEncoding);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_GetParentLibrary)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_SystemDescriptor_OpenSystem)(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_SYSTEM_HANDLE * systemHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_ToModule)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_MODULE_HANDLE * moduleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_ToEventSupportingModule)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetKey)(PEAK_SYSTEM_HANDLE systemHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetInfo)(PEAK_SYSTEM_HANDLE systemHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetID)(PEAK_SYSTEM_HANDLE systemHandle, char * id, size_t * idSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetDisplayName)(PEAK_SYSTEM_HANDLE systemHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetVendorName)(PEAK_SYSTEM_HANDLE systemHandle, char * vendorName, size_t * vendorNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetModelName)(PEAK_SYSTEM_HANDLE systemHandle, char * modelName, size_t * modelNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetVersion)(PEAK_SYSTEM_HANDLE systemHandle, char * version, size_t * versionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetTLType)(PEAK_SYSTEM_HANDLE systemHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetCTIFileName)(PEAK_SYSTEM_HANDLE systemHandle, char * ctiFileName, size_t * ctiFileNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetCTIFullPath)(PEAK_SYSTEM_HANDLE systemHandle, char * ctiFullPath, size_t * ctiFullPathSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetGenTLVersionMajor)(PEAK_SYSTEM_HANDLE systemHandle, uint32_t * gentlVersionMajor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetGenTLVersionMinor)(PEAK_SYSTEM_HANDLE systemHandle, uint32_t * gentlVersionMinor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetCharacterEncoding)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_CHARACTER_ENCODING * characterEncoding);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetParentLibrary)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_UpdateInterfaces)(PEAK_SYSTEM_HANDLE systemHandle, uint64_t timeout_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetNumInterfaces)(PEAK_SYSTEM_HANDLE systemHandle, size_t * numInterfaces);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_GetInterface)(PEAK_SYSTEM_HANDLE systemHandle, size_t index, PEAK_INTERFACE_DESCRIPTOR_HANDLE * interfaceDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_RegisterInterfaceFoundCallback)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK callback, void * callbackContext, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_UnregisterInterfaceFoundCallback)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_RegisterInterfaceLostCallback)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK callback, void * callbackContext, PEAK_INTERFACE_LOST_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_UnregisterInterfaceLostCallback)(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_System_Destruct)(PEAK_SYSTEM_HANDLE systemHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_ToModuleDescriptor)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_GetKey)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_GetInfo)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_GetDisplayName)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_GetTLType)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_GetParentSystem)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_SYSTEM_HANDLE * systemHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_InterfaceDescriptor_OpenInterface)(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_INTERFACE_HANDLE * interfaceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_ToModule)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_MODULE_HANDLE * moduleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_ToEventSupportingModule)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetKey)(PEAK_INTERFACE_HANDLE interfaceHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetInfo)(PEAK_INTERFACE_HANDLE interfaceHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetID)(PEAK_INTERFACE_HANDLE interfaceHandle, char * id, size_t * idSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetDisplayName)(PEAK_INTERFACE_HANDLE interfaceHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetTLType)(PEAK_INTERFACE_HANDLE interfaceHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetParentSystem)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_SYSTEM_HANDLE * systemHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_UpdateDevices)(PEAK_INTERFACE_HANDLE interfaceHandle, uint64_t timeout_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetNumDevices)(PEAK_INTERFACE_HANDLE interfaceHandle, size_t * numDevices);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_GetDevice)(PEAK_INTERFACE_HANDLE interfaceHandle, size_t index, PEAK_DEVICE_DESCRIPTOR_HANDLE * deviceDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_RegisterDeviceFoundCallback)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK callback, void * callbackContext, PEAK_DEVICE_FOUND_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_UnregisterDeviceFoundCallback)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_RegisterDeviceLostCallback)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK callback, void * callbackContext, PEAK_DEVICE_LOST_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_UnregisterDeviceLostCallback)(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Interface_Destruct)(PEAK_INTERFACE_HANDLE interfaceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_ToModuleDescriptor)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetKey)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetInfo)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetDisplayName)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetVendorName)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * vendorName, size_t * vendorNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetModelName)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * modelName, size_t * modelNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetVersion)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * version, size_t * versionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetTLType)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetUserDefinedName)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * userDefinedName, size_t * userDefinedNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetSerialNumber)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * serialNumber, size_t * serialNumberSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetAccessStatus)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_STATUS * accessStatus);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetTimestampTickFrequency)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t * timestampTickFrequency);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetIsOpenableExclusive)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_BOOL8 * isOpenable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetIsOpenable)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_BOOL8 * isOpenable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_OpenDevice)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_DEVICE_HANDLE * deviceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetParentInterface)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_INTERFACE_HANDLE * interfaceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t * monitoringUpdateInterval_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t monitoringUpdateInterval_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_IsInformationRoleMonitored)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole, PEAK_BOOL8 * isInformationRoleMonitored);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_RegisterInformationChangedCallback)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK callback, void * callbackContext, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback)(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_ToModule)(PEAK_DEVICE_HANDLE deviceHandle, PEAK_MODULE_HANDLE * moduleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_ToEventSupportingModule)(PEAK_DEVICE_HANDLE deviceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetKey)(PEAK_DEVICE_HANDLE deviceHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetInfo)(PEAK_DEVICE_HANDLE deviceHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetID)(PEAK_DEVICE_HANDLE deviceHandle, char * id, size_t * idSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetDisplayName)(PEAK_DEVICE_HANDLE deviceHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetVendorName)(PEAK_DEVICE_HANDLE deviceHandle, char * vendorName, size_t * vendorNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetModelName)(PEAK_DEVICE_HANDLE deviceHandle, char * modelName, size_t * modelNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetVersion)(PEAK_DEVICE_HANDLE deviceHandle, char * version, size_t * versionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetTLType)(PEAK_DEVICE_HANDLE deviceHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetUserDefinedName)(PEAK_DEVICE_HANDLE deviceHandle, char * userDefinedName, size_t * userDefinedNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetSerialNumber)(PEAK_DEVICE_HANDLE deviceHandle, char * serialNumber, size_t * serialNumberSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetAccessStatus)(PEAK_DEVICE_HANDLE deviceHandle, PEAK_DEVICE_ACCESS_STATUS * accessStatus);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetTimestampTickFrequency)(PEAK_DEVICE_HANDLE deviceHandle, uint64_t * timestampTickFrequency);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetParentInterface)(PEAK_DEVICE_HANDLE deviceHandle, PEAK_INTERFACE_HANDLE * interfaceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetRemoteDevice)(PEAK_DEVICE_HANDLE deviceHandle, PEAK_REMOTE_DEVICE_HANDLE * remoteDeviceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetNumDataStreams)(PEAK_DEVICE_HANDLE deviceHandle, size_t * numDataStreams);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_GetDataStream)(PEAK_DEVICE_HANDLE deviceHandle, size_t index, PEAK_DATA_STREAM_DESCRIPTOR_HANDLE * dataStreamDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Device_Destruct)(PEAK_DEVICE_HANDLE deviceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RemoteDevice_ToModule)(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_MODULE_HANDLE * moduleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RemoteDevice_GetLocalDevice)(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_DEVICE_HANDLE * deviceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStreamDescriptor_ToModuleDescriptor)(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStreamDescriptor_GetKey)(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStreamDescriptor_GetParentDevice)(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DEVICE_HANDLE * deviceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStreamDescriptor_OpenDataStream)(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DATA_STREAM_HANDLE * dataStreamHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_ToModule)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_MODULE_HANDLE * moduleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_ToEventSupportingModule)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetKey)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * key, size_t * keySize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetInfo)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetID)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * id, size_t * idSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetTLType)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAnnouncedMinRequired);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumBuffersAnnounced)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAnnounced);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumBuffersQueued)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersQueued);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumBuffersAwaitDelivery)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAwaitDelivery);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumBuffersDelivered)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numBuffersDelivered);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumBuffersStarted)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numBuffersStarted);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumUnderruns)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numUnderruns);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetNumChunksPerBufferMax)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numChunksPerBufferMax);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetBufferAlignment)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * bufferAlignment);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetPayloadSize)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * payloadSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetDefinesPayloadSize)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8 * definesPayloadSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetIsGrabbing)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8 * isGrabbing);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_GetParentDevice)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DEVICE_HANDLE * deviceHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_AnnounceBuffer)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, void * buffer, size_t bufferSize, void * userPtr, PEAK_BUFFER_REVOCATION_CALLBACK revocationCallback, void * callbackContext, PEAK_BUFFER_HANDLE * bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_AllocAndAnnounceBuffer)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t bufferSize, void * userPtr, PEAK_BUFFER_HANDLE * bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_QueueBuffer)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_RevokeBuffer)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_WaitForFinishedBuffer)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t timeout_ms, PEAK_BUFFER_HANDLE * bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_KillWait)(PEAK_DATA_STREAM_HANDLE dataStreamHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_Flush)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DATA_STREAM_FLUSH_MODE flushMode);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_StartAcquisitionInfinite)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_StartAcquisition)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode, uint64_t numToAcquire);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_StopAcquisition)(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_STOP_MODE stopMode);
typedef PEAK_RETURN_CODE (*dyn_PEAK_DataStream_Destruct)(PEAK_DATA_STREAM_HANDLE dataStreamHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_ToModule)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_MODULE_HANDLE * moduleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_ToEventSupportingModule)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetInfo)(PEAK_BUFFER_HANDLE bufferHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetTLType)(PEAK_BUFFER_HANDLE bufferHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetBasePtr)(PEAK_BUFFER_HANDLE bufferHandle, void * * basePtr);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetSize)(PEAK_BUFFER_HANDLE bufferHandle, size_t * size);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetUserPtr)(PEAK_BUFFER_HANDLE bufferHandle, void * * userPtr);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetPayloadType)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BUFFER_PAYLOAD_TYPE * payloadType);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetPixelFormat)(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * pixelFormat);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetPixelFormatNamespace)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_PIXEL_FORMAT_NAMESPACE * pixelFormatNamespace);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetPixelEndianness)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_ENDIANNESS * pixelEndianness);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetExpectedDataSize)(PEAK_BUFFER_HANDLE bufferHandle, size_t * expectedDataSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetDeliveredDataSize)(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredDataSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetFrameID)(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * frameId);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetImageOffset)(PEAK_BUFFER_HANDLE bufferHandle, size_t * imageOffset);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetDeliveredImageHeight)(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredImageHeight);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetDeliveredChunkPayloadSize)(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredChunkPayloadSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetChunkLayoutID)(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * chunkLayoutId);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetFileName)(PEAK_BUFFER_HANDLE bufferHandle, char * fileName, size_t * fileNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetWidth)(PEAK_BUFFER_HANDLE bufferHandle, size_t * width);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetHeight)(PEAK_BUFFER_HANDLE bufferHandle, size_t * height);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetXOffset)(PEAK_BUFFER_HANDLE bufferHandle, size_t * xOffset);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetYOffset)(PEAK_BUFFER_HANDLE bufferHandle, size_t * yOffset);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetXPadding)(PEAK_BUFFER_HANDLE bufferHandle, size_t * xPadding);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetYPadding)(PEAK_BUFFER_HANDLE bufferHandle, size_t * yPadding);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetTimestamp_ticks)(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * timestamp_ticks);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetTimestamp_ns)(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * timestamp_ns);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetIsQueued)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isQueued);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetIsAcquiring)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isAcquiring);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetIsIncomplete)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isIncomplete);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetHasNewData)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasNewData);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetHasImage)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasImage);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetHasChunks)(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasChunks);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_UpdateChunks)(PEAK_BUFFER_HANDLE bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetNumChunks)(PEAK_BUFFER_HANDLE bufferHandle, size_t * numChunks);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetChunk)(PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_CHUNK_HANDLE * bufferChunkHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_UpdateParts)(PEAK_BUFFER_HANDLE bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetNumParts)(PEAK_BUFFER_HANDLE bufferHandle, size_t * numParts);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Buffer_GetPart)(PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_PART_HANDLE * bufferPartHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferChunk_GetID)(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, uint64_t * id);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferChunk_GetBasePtr)(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, void * * basePtr);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferChunk_GetSize)(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, size_t * size);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferChunk_GetParentBuffer)(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, PEAK_BUFFER_HANDLE * bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetInfo)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetSourceID)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * sourceId);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetBasePtr)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, void * * basePtr);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetSize)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * size);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetType)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_PART_TYPE * type);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetFormat)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * format);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetFormatNamespace)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * formatNamespace);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetWidth)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * width);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetHeight)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * height);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetXOffset)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * xOffset);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetYOffset)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * yOffset);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetXPadding)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * xPadding);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetDeliveredImageHeight)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * deliveredImageHeight);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BufferPart_GetParentBuffer)(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_HANDLE * bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_ModuleDescriptor_GetID)(PEAK_MODULE_DESCRIPTOR_HANDLE moduleDescriptorHandle, char * id, size_t * idSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Module_GetNumNodeMaps)(PEAK_MODULE_HANDLE moduleHandle, size_t * numNodeMaps);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Module_GetNodeMap)(PEAK_MODULE_HANDLE moduleHandle, size_t index, PEAK_NODE_MAP_HANDLE * nodeMapHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Module_GetPort)(PEAK_MODULE_HANDLE moduleHandle, PEAK_PORT_HANDLE * portHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_GetHasNode)(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * nodeName, size_t nodeNameSize, PEAK_BOOL8 * hasNode);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_FindNode)(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * nodeName, size_t nodeNameSize, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_InvalidateNodes)(PEAK_NODE_MAP_HANDLE nodeMapHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_PollNodes)(PEAK_NODE_MAP_HANDLE nodeMapHandle, int64_t elapsedTime_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_GetNumNodes)(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t * numNodes);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_GetNode)(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t index, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_GetHasBufferSupportedChunks)(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasSupportedChunks);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_UpdateChunkNodes)(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_GetHasEventSupportedData)(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle, PEAK_BOOL8 * hasSupportedData);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_UpdateEventNodes)(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_StoreToFile)(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * filePath, size_t filePathSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_LoadFromFile)(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * filePath, size_t filePathSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_Lock)(PEAK_NODE_MAP_HANDLE nodeMapHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_NodeMap_Unlock)(PEAK_NODE_MAP_HANDLE nodeMapHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToIntegerNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_INTEGER_NODE_HANDLE * integerNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToBooleanNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOLEAN_NODE_HANDLE * booleanNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToCommandNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_COMMAND_NODE_HANDLE * commandNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToFloatNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_FLOAT_NODE_HANDLE * floatNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToStringNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_STRING_NODE_HANDLE * stringNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToRegisterNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_REGISTER_NODE_HANDLE * registerNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToCategoryNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_CATEGORY_NODE_HANDLE * categoryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToEnumerationNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_NODE_HANDLE * enumerationNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_ToEnumerationEntryNode)(PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetName)(PEAK_NODE_HANDLE nodeHandle, char * name, size_t * nameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetDisplayName)(PEAK_NODE_HANDLE nodeHandle, char * displayName, size_t * displayNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetNamespace)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_NAMESPACE * _namespace);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetVisibility)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_VISIBILITY * visibility);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetAccessStatus)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_ACCESS_STATUS * accessStatus);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetIsCacheable)(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isCacheable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetIsAccessStatusCacheable)(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isAccessStatusCacheable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetIsStreamable)(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isStreamable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetIsDeprecated)(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isDeprecated);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetIsFeature)(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isFeature);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetCachingMode)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CACHING_MODE * cachingMode);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetPollingTime)(PEAK_NODE_HANDLE nodeHandle, int64_t * pollingTime_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetToolTip)(PEAK_NODE_HANDLE nodeHandle, char * toolTip, size_t * toolTipSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetDescription)(PEAK_NODE_HANDLE nodeHandle, char * description, size_t * descriptionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetType)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_TYPE * type);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetParentNodeMap)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_MAP_HANDLE * nodeMapHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_FindInvalidatedNode)(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * invalidatedNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetNumInvalidatedNodes)(PEAK_NODE_HANDLE nodeHandle, size_t * numInvalidatedNodes);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetInvalidatedNode)(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * invalidatedNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_FindInvalidatingNode)(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * invalidatingNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetNumInvalidatingNodes)(PEAK_NODE_HANDLE nodeHandle, size_t * numInvalidatingNodes);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetInvalidatingNode)(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * invalidatingNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_FindSelectedNode)(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * selectedNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetNumSelectedNodes)(PEAK_NODE_HANDLE nodeHandle, size_t * numSelectedNodes);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetSelectedNode)(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * selectedNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_FindSelectingNode)(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * selectingNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetNumSelectingNodes)(PEAK_NODE_HANDLE nodeHandle, size_t * numSelectingNodes);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_GetSelectingNode)(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * selectingNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_RegisterChangedCallback)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK callback, void * callbackContext, PEAK_NODE_CHANGED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Node_UnregisterChangedCallback)(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_ToNode)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetMinimum)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * minimum);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetMaximum)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * maximum);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetIncrement)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * increment);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetIncrementType)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_INCREMENT_TYPE * incrementType);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetValidValues)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * validValues, size_t * validValuesSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetRepresentation)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_REPRESENTATION * representation);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetUnit)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, char * unit, size_t * unitSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_GetValue)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, int64_t * value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_IntegerNode_SetValue)(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BooleanNode_ToNode)(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BooleanNode_GetValue)(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_BOOL8 * value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_BooleanNode_SetValue)(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_BOOL8 value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CommandNode_ToNode)(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CommandNode_GetIsDone)(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_BOOL8 * isDone);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CommandNode_Execute)(PEAK_COMMAND_NODE_HANDLE commandNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CommandNode_WaitUntilDoneInfinite)(PEAK_COMMAND_NODE_HANDLE commandNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CommandNode_WaitUntilDone)(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, uint64_t waitTimeout_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_ToNode)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetMinimum)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * minimum);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetMaximum)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * maximum);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetIncrement)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * increment);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetIncrementType)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_INCREMENT_TYPE * incrementType);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetValidValues)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * validValues, size_t * validValuesSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetRepresentation)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_REPRESENTATION * representation);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetUnit)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, char * unit, size_t * unitSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetDisplayNotation)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_DISPLAY_NOTATION * displayNotation);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetDisplayPrecision)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, int64_t * displayPrecision);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetHasConstantIncrement)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_BOOL8 * hasConstantIncrement);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_GetValue)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, double * value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FloatNode_SetValue)(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_StringNode_ToNode)(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_StringNode_GetMaximumLength)(PEAK_STRING_NODE_HANDLE stringNodeHandle, int64_t * maximumLength);
typedef PEAK_RETURN_CODE (*dyn_PEAK_StringNode_GetValue)(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, char * value, size_t * valueSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_StringNode_SetValue)(PEAK_STRING_NODE_HANDLE stringNodeHandle, const char * value, size_t valueSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RegisterNode_ToNode)(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RegisterNode_GetAddress)(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, uint64_t * address);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RegisterNode_GetLength)(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, size_t * length);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RegisterNode_Read)(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, uint8_t * bytesToRead, size_t bytesToReadSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_RegisterNode_Write)(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, const uint8_t * bytesToWrite, size_t bytesToWriteSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CategoryNode_ToNode)(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CategoryNode_GetNumSubNodes)(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t * numSubNodes);
typedef PEAK_RETURN_CODE (*dyn_PEAK_CategoryNode_GetSubNode)(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t index, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_ToNode)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_GetCurrentEntry)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_SetCurrentEntry)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char * symbolicValue, size_t symbolicValueSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_SetCurrentEntryByValue)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_FindEntryBySymbolicValue)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char * symbolicValue, size_t symbolicValueSize, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_FindEntryByValue)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_GetNumEntries)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t * numEntries);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationNode_GetEntry)(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t index, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationEntryNode_ToNode)(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationEntryNode_GetIsSelfClearing)(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_BOOL8 * isSelfClearing);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationEntryNode_GetSymbolicValue)(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, char * symbolicValue, size_t * symbolicValueSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EnumerationEntryNode_GetValue)(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, int64_t * value);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetInfo)(PEAK_PORT_HANDLE portHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetID)(PEAK_PORT_HANDLE portHandle, char * id, size_t * idSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetName)(PEAK_PORT_HANDLE portHandle, char * name, size_t * nameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetVendorName)(PEAK_PORT_HANDLE portHandle, char * vendorName, size_t * vendorNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetModelName)(PEAK_PORT_HANDLE portHandle, char * modelName, size_t * modelNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetVersion)(PEAK_PORT_HANDLE portHandle, char * version, size_t * versionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetTLType)(PEAK_PORT_HANDLE portHandle, char * tlType, size_t * tlTypeSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetModuleName)(PEAK_PORT_HANDLE portHandle, char * moduleName, size_t * moduleNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetDataEndianness)(PEAK_PORT_HANDLE portHandle, PEAK_ENDIANNESS * dataEndianness);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetIsReadable)(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isReadable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetIsWritable)(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isWritable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetIsAvailable)(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isAvailable);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetIsImplemented)(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isImplemented);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_Read)(PEAK_PORT_HANDLE portHandle, uint64_t address, uint8_t * bytesToRead, size_t bytesToReadSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_Write)(PEAK_PORT_HANDLE portHandle, uint64_t address, const uint8_t * bytesToWrite, size_t bytesToWriteSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetNumURLs)(PEAK_PORT_HANDLE portHandle, size_t * numUrls);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Port_GetURL)(PEAK_PORT_HANDLE portHandle, size_t index, PEAK_PORT_URL_HANDLE * portUrlHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetInfo)(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetURL)(PEAK_PORT_URL_HANDLE portUrlHandle, char * url, size_t * urlSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetScheme)(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_URL_SCHEME * scheme);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileName)(PEAK_PORT_URL_HANDLE portUrlHandle, char * fileName, size_t * fileNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileRegisterAddress)(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t * fileRegisterAddress);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileSize)(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t * fileSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileSHA1Hash)(PEAK_PORT_URL_HANDLE portUrlHandle, uint8_t * fileSha1Hash, size_t * fileSha1HashSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileVersionMajor)(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionMajor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileVersionMinor)(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionMinor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileVersionSubminor)(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionSubminor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileSchemaVersionMajor)(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileSchemaVersionMajor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetFileSchemaVersionMinor)(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileSchemaVersionMinor);
typedef PEAK_RETURN_CODE (*dyn_PEAK_PortURL_GetParentPort)(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_HANDLE * portHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventSupportingModule_EnableEvents)(PEAK_EVENT_SUPPORTING_MODULE_HANDLE eventSupportingModuleHandle, PEAK_EVENT_TYPE eventType, PEAK_EVENT_CONTROLLER_HANDLE * eventControllerHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_GetInfo)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_GetNumEventsInQueue)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * numEventsInQueue);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_GetNumEventsFired)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t * numEventsFired);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_GetEventMaxSize)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * eventMaxSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_GetEventDataMaxSize)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * eventDataMaxSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_GetControlledEventType)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, PEAK_EVENT_TYPE * controlledEventType);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_WaitForEvent)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t timeout_ms, PEAK_EVENT_HANDLE * eventHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_KillWait)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_FlushEvents)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_EventController_Destruct)(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Event_GetInfo)(PEAK_EVENT_HANDLE eventHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Event_GetID)(PEAK_EVENT_HANDLE eventHandle, uint64_t * id);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Event_GetData)(PEAK_EVENT_HANDLE eventHandle, uint8_t * data, size_t * dataSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Event_GetType)(PEAK_EVENT_HANDLE eventHandle, PEAK_EVENT_TYPE * type);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Event_GetRawData)(PEAK_EVENT_HANDLE eventHandle, uint8_t * rawData, size_t * rawDataSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_Event_Destruct)(PEAK_EVENT_HANDLE eventHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_Construct)(PEAK_FIRMWARE_UPDATER_HANDLE * firmwareUpdaterHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char * gufPath, size_t gufPathSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char * gufPath, size_t gufPathSize, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t * numFirmwareUpdateInformation);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t index, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE * firmwareUpdateInformationHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_UpdateDevice)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, uint64_t deviceResetDiscoveryTimeout_ms);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdater_Destruct)(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetIsValid)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_BOOL8 * isValid);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetFileName)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * fileName, size_t * fileNameSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetDescription)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * description, size_t * descriptionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetVersion)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * version, size_t * versionSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * versionExtractionPattern, size_t * versionExtractionPatternSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetVersionStyle)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_VERSION_STYLE * versionStyle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetReleaseNotes)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * releaseNotes, size_t * releaseNotesSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * releaseNotesUrl, size_t * releaseNotesUrlSize);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetUserSetPersistence)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PERSISTENCE * userSetPersistence);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence)(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PERSISTENCE * sequencerSetPersistence);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_Construct)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE * firmwareUpdateProgressObserverHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE * callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE callbackHandle);
typedef PEAK_RETURN_CODE (*dyn_PEAK_FirmwareUpdateProgressObserver_Destruct)(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle);

                        
class DynamicLoader
{
private:
    DynamicLoader();
    
    static DynamicLoader& instance()
    {
        static DynamicLoader dynamicLoader{};
        return dynamicLoader;
    }
    bool loadLib(const char* file);
    void unload();
    bool setPointers(bool load);

public:
    ~DynamicLoader();
    
    static bool isLoaded();
    
    static PEAK_RETURN_CODE PEAK_Library_Initialize();
    static PEAK_RETURN_CODE PEAK_Library_Close();
    static PEAK_RETURN_CODE PEAK_Library_IsInitialized(PEAK_BOOL8 * isInitialized);
    static PEAK_RETURN_CODE PEAK_Library_GetVersionMajor(uint32_t * libraryVersionMajor);
    static PEAK_RETURN_CODE PEAK_Library_GetVersionMinor(uint32_t * libraryVersionMinor);
    static PEAK_RETURN_CODE PEAK_Library_GetVersionSubminor(uint32_t * libraryVersionSubminor);
    static PEAK_RETURN_CODE PEAK_Library_GetLastError(PEAK_RETURN_CODE * lastErrorCode, char * lastErrorDescription, size_t * lastErrorDescriptionSize);
    static PEAK_RETURN_CODE PEAK_EnvironmentInspector_UpdateCTIPaths();
    static PEAK_RETURN_CODE PEAK_EnvironmentInspector_GetNumCTIPaths(size_t * numCtiPaths);
    static PEAK_RETURN_CODE PEAK_EnvironmentInspector_GetCTIPath(size_t index, char * ctiPath, size_t * ctiPathSize);
    static PEAK_RETURN_CODE PEAK_ProducerLibrary_Construct(const char * ctiPath, size_t ctiPathSize, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle);
    static PEAK_RETURN_CODE PEAK_ProducerLibrary_GetKey(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_ProducerLibrary_GetSystem(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, PEAK_SYSTEM_DESCRIPTOR_HANDLE * systemDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_ProducerLibrary_Destruct(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_ToModuleDescriptor(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetKey(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetInfo(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetDisplayName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetVendorName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * vendorName, size_t * vendorNameSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetModelName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * modelName, size_t * modelNameSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetVersion(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * version, size_t * versionSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetTLType(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetCTIFileName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * ctiFileName, size_t * ctiFileNameSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetCTIFullPath(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * ctiFullPath, size_t * ctiFullPathSize);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetGenTLVersionMajor(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t * gentlVersionMajor);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetGenTLVersionMinor(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t * gentlVersionMinor);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetCharacterEncoding(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_CHARACTER_ENCODING * characterEncoding);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_GetParentLibrary(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle);
    static PEAK_RETURN_CODE PEAK_SystemDescriptor_OpenSystem(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_SYSTEM_HANDLE * systemHandle);
    static PEAK_RETURN_CODE PEAK_System_ToModule(PEAK_SYSTEM_HANDLE systemHandle, PEAK_MODULE_HANDLE * moduleHandle);
    static PEAK_RETURN_CODE PEAK_System_ToEventSupportingModule(PEAK_SYSTEM_HANDLE systemHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
    static PEAK_RETURN_CODE PEAK_System_GetKey(PEAK_SYSTEM_HANDLE systemHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_System_GetInfo(PEAK_SYSTEM_HANDLE systemHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_System_GetID(PEAK_SYSTEM_HANDLE systemHandle, char * id, size_t * idSize);
    static PEAK_RETURN_CODE PEAK_System_GetDisplayName(PEAK_SYSTEM_HANDLE systemHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_System_GetVendorName(PEAK_SYSTEM_HANDLE systemHandle, char * vendorName, size_t * vendorNameSize);
    static PEAK_RETURN_CODE PEAK_System_GetModelName(PEAK_SYSTEM_HANDLE systemHandle, char * modelName, size_t * modelNameSize);
    static PEAK_RETURN_CODE PEAK_System_GetVersion(PEAK_SYSTEM_HANDLE systemHandle, char * version, size_t * versionSize);
    static PEAK_RETURN_CODE PEAK_System_GetTLType(PEAK_SYSTEM_HANDLE systemHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_System_GetCTIFileName(PEAK_SYSTEM_HANDLE systemHandle, char * ctiFileName, size_t * ctiFileNameSize);
    static PEAK_RETURN_CODE PEAK_System_GetCTIFullPath(PEAK_SYSTEM_HANDLE systemHandle, char * ctiFullPath, size_t * ctiFullPathSize);
    static PEAK_RETURN_CODE PEAK_System_GetGenTLVersionMajor(PEAK_SYSTEM_HANDLE systemHandle, uint32_t * gentlVersionMajor);
    static PEAK_RETURN_CODE PEAK_System_GetGenTLVersionMinor(PEAK_SYSTEM_HANDLE systemHandle, uint32_t * gentlVersionMinor);
    static PEAK_RETURN_CODE PEAK_System_GetCharacterEncoding(PEAK_SYSTEM_HANDLE systemHandle, PEAK_CHARACTER_ENCODING * characterEncoding);
    static PEAK_RETURN_CODE PEAK_System_GetParentLibrary(PEAK_SYSTEM_HANDLE systemHandle, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle);
    static PEAK_RETURN_CODE PEAK_System_UpdateInterfaces(PEAK_SYSTEM_HANDLE systemHandle, uint64_t timeout_ms);
    static PEAK_RETURN_CODE PEAK_System_GetNumInterfaces(PEAK_SYSTEM_HANDLE systemHandle, size_t * numInterfaces);
    static PEAK_RETURN_CODE PEAK_System_GetInterface(PEAK_SYSTEM_HANDLE systemHandle, size_t index, PEAK_INTERFACE_DESCRIPTOR_HANDLE * interfaceDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_System_RegisterInterfaceFoundCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK callback, void * callbackContext, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_System_UnregisterInterfaceFoundCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_System_RegisterInterfaceLostCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK callback, void * callbackContext, PEAK_INTERFACE_LOST_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_System_UnregisterInterfaceLostCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_System_Destruct(PEAK_SYSTEM_HANDLE systemHandle);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_ToModuleDescriptor(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_GetKey(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_GetInfo(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_GetDisplayName(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_GetTLType(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_GetParentSystem(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_SYSTEM_HANDLE * systemHandle);
    static PEAK_RETURN_CODE PEAK_InterfaceDescriptor_OpenInterface(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_INTERFACE_HANDLE * interfaceHandle);
    static PEAK_RETURN_CODE PEAK_Interface_ToModule(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_MODULE_HANDLE * moduleHandle);
    static PEAK_RETURN_CODE PEAK_Interface_ToEventSupportingModule(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
    static PEAK_RETURN_CODE PEAK_Interface_GetKey(PEAK_INTERFACE_HANDLE interfaceHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_Interface_GetInfo(PEAK_INTERFACE_HANDLE interfaceHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_Interface_GetID(PEAK_INTERFACE_HANDLE interfaceHandle, char * id, size_t * idSize);
    static PEAK_RETURN_CODE PEAK_Interface_GetDisplayName(PEAK_INTERFACE_HANDLE interfaceHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_Interface_GetTLType(PEAK_INTERFACE_HANDLE interfaceHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_Interface_GetParentSystem(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_SYSTEM_HANDLE * systemHandle);
    static PEAK_RETURN_CODE PEAK_Interface_UpdateDevices(PEAK_INTERFACE_HANDLE interfaceHandle, uint64_t timeout_ms);
    static PEAK_RETURN_CODE PEAK_Interface_GetNumDevices(PEAK_INTERFACE_HANDLE interfaceHandle, size_t * numDevices);
    static PEAK_RETURN_CODE PEAK_Interface_GetDevice(PEAK_INTERFACE_HANDLE interfaceHandle, size_t index, PEAK_DEVICE_DESCRIPTOR_HANDLE * deviceDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_Interface_RegisterDeviceFoundCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK callback, void * callbackContext, PEAK_DEVICE_FOUND_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_Interface_UnregisterDeviceFoundCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_Interface_RegisterDeviceLostCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK callback, void * callbackContext, PEAK_DEVICE_LOST_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_Interface_UnregisterDeviceLostCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_Interface_Destruct(PEAK_INTERFACE_HANDLE interfaceHandle);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_ToModuleDescriptor(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetKey(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetInfo(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetDisplayName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetVendorName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * vendorName, size_t * vendorNameSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetModelName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * modelName, size_t * modelNameSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetVersion(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * version, size_t * versionSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetTLType(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetUserDefinedName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * userDefinedName, size_t * userDefinedNameSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetSerialNumber(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * serialNumber, size_t * serialNumberSize);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetAccessStatus(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_STATUS * accessStatus);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetTimestampTickFrequency(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t * timestampTickFrequency);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetIsOpenableExclusive(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_BOOL8 * isOpenable);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetIsOpenable(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_BOOL8 * isOpenable);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_OpenDevice(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_DEVICE_HANDLE * deviceHandle);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetParentInterface(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_INTERFACE_HANDLE * interfaceHandle);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_GetMonitoringUpdateInterval(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t * monitoringUpdateInterval_ms);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_SetMonitoringUpdateInterval(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t monitoringUpdateInterval_ms);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_IsInformationRoleMonitored(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole, PEAK_BOOL8 * isInformationRoleMonitored);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_AddInformationRoleToMonitoring(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_RegisterInformationChangedCallback(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK callback, void * callbackContext, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_DeviceDescriptor_UnregisterInformationChangedCallback(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_Device_ToModule(PEAK_DEVICE_HANDLE deviceHandle, PEAK_MODULE_HANDLE * moduleHandle);
    static PEAK_RETURN_CODE PEAK_Device_ToEventSupportingModule(PEAK_DEVICE_HANDLE deviceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
    static PEAK_RETURN_CODE PEAK_Device_GetKey(PEAK_DEVICE_HANDLE deviceHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_Device_GetInfo(PEAK_DEVICE_HANDLE deviceHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_Device_GetID(PEAK_DEVICE_HANDLE deviceHandle, char * id, size_t * idSize);
    static PEAK_RETURN_CODE PEAK_Device_GetDisplayName(PEAK_DEVICE_HANDLE deviceHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_Device_GetVendorName(PEAK_DEVICE_HANDLE deviceHandle, char * vendorName, size_t * vendorNameSize);
    static PEAK_RETURN_CODE PEAK_Device_GetModelName(PEAK_DEVICE_HANDLE deviceHandle, char * modelName, size_t * modelNameSize);
    static PEAK_RETURN_CODE PEAK_Device_GetVersion(PEAK_DEVICE_HANDLE deviceHandle, char * version, size_t * versionSize);
    static PEAK_RETURN_CODE PEAK_Device_GetTLType(PEAK_DEVICE_HANDLE deviceHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_Device_GetUserDefinedName(PEAK_DEVICE_HANDLE deviceHandle, char * userDefinedName, size_t * userDefinedNameSize);
    static PEAK_RETURN_CODE PEAK_Device_GetSerialNumber(PEAK_DEVICE_HANDLE deviceHandle, char * serialNumber, size_t * serialNumberSize);
    static PEAK_RETURN_CODE PEAK_Device_GetAccessStatus(PEAK_DEVICE_HANDLE deviceHandle, PEAK_DEVICE_ACCESS_STATUS * accessStatus);
    static PEAK_RETURN_CODE PEAK_Device_GetTimestampTickFrequency(PEAK_DEVICE_HANDLE deviceHandle, uint64_t * timestampTickFrequency);
    static PEAK_RETURN_CODE PEAK_Device_GetParentInterface(PEAK_DEVICE_HANDLE deviceHandle, PEAK_INTERFACE_HANDLE * interfaceHandle);
    static PEAK_RETURN_CODE PEAK_Device_GetRemoteDevice(PEAK_DEVICE_HANDLE deviceHandle, PEAK_REMOTE_DEVICE_HANDLE * remoteDeviceHandle);
    static PEAK_RETURN_CODE PEAK_Device_GetNumDataStreams(PEAK_DEVICE_HANDLE deviceHandle, size_t * numDataStreams);
    static PEAK_RETURN_CODE PEAK_Device_GetDataStream(PEAK_DEVICE_HANDLE deviceHandle, size_t index, PEAK_DATA_STREAM_DESCRIPTOR_HANDLE * dataStreamDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_Device_Destruct(PEAK_DEVICE_HANDLE deviceHandle);
    static PEAK_RETURN_CODE PEAK_RemoteDevice_ToModule(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_MODULE_HANDLE * moduleHandle);
    static PEAK_RETURN_CODE PEAK_RemoteDevice_GetLocalDevice(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_DEVICE_HANDLE * deviceHandle);
    static PEAK_RETURN_CODE PEAK_DataStreamDescriptor_ToModuleDescriptor(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_DataStreamDescriptor_GetKey(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_DataStreamDescriptor_GetParentDevice(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DEVICE_HANDLE * deviceHandle);
    static PEAK_RETURN_CODE PEAK_DataStreamDescriptor_OpenDataStream(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DATA_STREAM_HANDLE * dataStreamHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_ToModule(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_MODULE_HANDLE * moduleHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_ToEventSupportingModule(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_GetKey(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * key, size_t * keySize);
    static PEAK_RETURN_CODE PEAK_DataStream_GetInfo(PEAK_DATA_STREAM_HANDLE dataStreamHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_DataStream_GetID(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * id, size_t * idSize);
    static PEAK_RETURN_CODE PEAK_DataStream_GetTLType(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumBuffersAnnouncedMinRequired(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAnnouncedMinRequired);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumBuffersAnnounced(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAnnounced);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumBuffersQueued(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersQueued);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumBuffersAwaitDelivery(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAwaitDelivery);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumBuffersDelivered(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numBuffersDelivered);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumBuffersStarted(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numBuffersStarted);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumUnderruns(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numUnderruns);
    static PEAK_RETURN_CODE PEAK_DataStream_GetNumChunksPerBufferMax(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numChunksPerBufferMax);
    static PEAK_RETURN_CODE PEAK_DataStream_GetBufferAlignment(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * bufferAlignment);
    static PEAK_RETURN_CODE PEAK_DataStream_GetPayloadSize(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * payloadSize);
    static PEAK_RETURN_CODE PEAK_DataStream_GetDefinesPayloadSize(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8 * definesPayloadSize);
    static PEAK_RETURN_CODE PEAK_DataStream_GetIsGrabbing(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8 * isGrabbing);
    static PEAK_RETURN_CODE PEAK_DataStream_GetParentDevice(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DEVICE_HANDLE * deviceHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_AnnounceBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, void * buffer, size_t bufferSize, void * userPtr, PEAK_BUFFER_REVOCATION_CALLBACK revocationCallback, void * callbackContext, PEAK_BUFFER_HANDLE * bufferHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_AllocAndAnnounceBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t bufferSize, void * userPtr, PEAK_BUFFER_HANDLE * bufferHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_QueueBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_RevokeBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_WaitForFinishedBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t timeout_ms, PEAK_BUFFER_HANDLE * bufferHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_KillWait(PEAK_DATA_STREAM_HANDLE dataStreamHandle);
    static PEAK_RETURN_CODE PEAK_DataStream_Flush(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DATA_STREAM_FLUSH_MODE flushMode);
    static PEAK_RETURN_CODE PEAK_DataStream_StartAcquisitionInfinite(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode);
    static PEAK_RETURN_CODE PEAK_DataStream_StartAcquisition(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode, uint64_t numToAcquire);
    static PEAK_RETURN_CODE PEAK_DataStream_StopAcquisition(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_STOP_MODE stopMode);
    static PEAK_RETURN_CODE PEAK_DataStream_Destruct(PEAK_DATA_STREAM_HANDLE dataStreamHandle);
    static PEAK_RETURN_CODE PEAK_Buffer_ToModule(PEAK_BUFFER_HANDLE bufferHandle, PEAK_MODULE_HANDLE * moduleHandle);
    static PEAK_RETURN_CODE PEAK_Buffer_ToEventSupportingModule(PEAK_BUFFER_HANDLE bufferHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle);
    static PEAK_RETURN_CODE PEAK_Buffer_GetInfo(PEAK_BUFFER_HANDLE bufferHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_Buffer_GetTLType(PEAK_BUFFER_HANDLE bufferHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_Buffer_GetBasePtr(PEAK_BUFFER_HANDLE bufferHandle, void * * basePtr);
    static PEAK_RETURN_CODE PEAK_Buffer_GetSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * size);
    static PEAK_RETURN_CODE PEAK_Buffer_GetUserPtr(PEAK_BUFFER_HANDLE bufferHandle, void * * userPtr);
    static PEAK_RETURN_CODE PEAK_Buffer_GetPayloadType(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BUFFER_PAYLOAD_TYPE * payloadType);
    static PEAK_RETURN_CODE PEAK_Buffer_GetPixelFormat(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * pixelFormat);
    static PEAK_RETURN_CODE PEAK_Buffer_GetPixelFormatNamespace(PEAK_BUFFER_HANDLE bufferHandle, PEAK_PIXEL_FORMAT_NAMESPACE * pixelFormatNamespace);
    static PEAK_RETURN_CODE PEAK_Buffer_GetPixelEndianness(PEAK_BUFFER_HANDLE bufferHandle, PEAK_ENDIANNESS * pixelEndianness);
    static PEAK_RETURN_CODE PEAK_Buffer_GetExpectedDataSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * expectedDataSize);
    static PEAK_RETURN_CODE PEAK_Buffer_GetDeliveredDataSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredDataSize);
    static PEAK_RETURN_CODE PEAK_Buffer_GetFrameID(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * frameId);
    static PEAK_RETURN_CODE PEAK_Buffer_GetImageOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t * imageOffset);
    static PEAK_RETURN_CODE PEAK_Buffer_GetDeliveredImageHeight(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredImageHeight);
    static PEAK_RETURN_CODE PEAK_Buffer_GetDeliveredChunkPayloadSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredChunkPayloadSize);
    static PEAK_RETURN_CODE PEAK_Buffer_GetChunkLayoutID(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * chunkLayoutId);
    static PEAK_RETURN_CODE PEAK_Buffer_GetFileName(PEAK_BUFFER_HANDLE bufferHandle, char * fileName, size_t * fileNameSize);
    static PEAK_RETURN_CODE PEAK_Buffer_GetWidth(PEAK_BUFFER_HANDLE bufferHandle, size_t * width);
    static PEAK_RETURN_CODE PEAK_Buffer_GetHeight(PEAK_BUFFER_HANDLE bufferHandle, size_t * height);
    static PEAK_RETURN_CODE PEAK_Buffer_GetXOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t * xOffset);
    static PEAK_RETURN_CODE PEAK_Buffer_GetYOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t * yOffset);
    static PEAK_RETURN_CODE PEAK_Buffer_GetXPadding(PEAK_BUFFER_HANDLE bufferHandle, size_t * xPadding);
    static PEAK_RETURN_CODE PEAK_Buffer_GetYPadding(PEAK_BUFFER_HANDLE bufferHandle, size_t * yPadding);
    static PEAK_RETURN_CODE PEAK_Buffer_GetTimestamp_ticks(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * timestamp_ticks);
    static PEAK_RETURN_CODE PEAK_Buffer_GetTimestamp_ns(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * timestamp_ns);
    static PEAK_RETURN_CODE PEAK_Buffer_GetIsQueued(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isQueued);
    static PEAK_RETURN_CODE PEAK_Buffer_GetIsAcquiring(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isAcquiring);
    static PEAK_RETURN_CODE PEAK_Buffer_GetIsIncomplete(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isIncomplete);
    static PEAK_RETURN_CODE PEAK_Buffer_GetHasNewData(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasNewData);
    static PEAK_RETURN_CODE PEAK_Buffer_GetHasImage(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasImage);
    static PEAK_RETURN_CODE PEAK_Buffer_GetHasChunks(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasChunks);
    static PEAK_RETURN_CODE PEAK_Buffer_UpdateChunks(PEAK_BUFFER_HANDLE bufferHandle);
    static PEAK_RETURN_CODE PEAK_Buffer_GetNumChunks(PEAK_BUFFER_HANDLE bufferHandle, size_t * numChunks);
    static PEAK_RETURN_CODE PEAK_Buffer_GetChunk(PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_CHUNK_HANDLE * bufferChunkHandle);
    static PEAK_RETURN_CODE PEAK_Buffer_UpdateParts(PEAK_BUFFER_HANDLE bufferHandle);
    static PEAK_RETURN_CODE PEAK_Buffer_GetNumParts(PEAK_BUFFER_HANDLE bufferHandle, size_t * numParts);
    static PEAK_RETURN_CODE PEAK_Buffer_GetPart(PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_PART_HANDLE * bufferPartHandle);
    static PEAK_RETURN_CODE PEAK_BufferChunk_GetID(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, uint64_t * id);
    static PEAK_RETURN_CODE PEAK_BufferChunk_GetBasePtr(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, void * * basePtr);
    static PEAK_RETURN_CODE PEAK_BufferChunk_GetSize(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, size_t * size);
    static PEAK_RETURN_CODE PEAK_BufferChunk_GetParentBuffer(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, PEAK_BUFFER_HANDLE * bufferHandle);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetInfo(PEAK_BUFFER_PART_HANDLE bufferPartHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetSourceID(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * sourceId);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetBasePtr(PEAK_BUFFER_PART_HANDLE bufferPartHandle, void * * basePtr);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetSize(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * size);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetType(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_PART_TYPE * type);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetFormat(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * format);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetFormatNamespace(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * formatNamespace);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetWidth(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * width);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetHeight(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * height);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetXOffset(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * xOffset);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetYOffset(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * yOffset);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetXPadding(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * xPadding);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetDeliveredImageHeight(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * deliveredImageHeight);
    static PEAK_RETURN_CODE PEAK_BufferPart_GetParentBuffer(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_HANDLE * bufferHandle);
    static PEAK_RETURN_CODE PEAK_ModuleDescriptor_GetID(PEAK_MODULE_DESCRIPTOR_HANDLE moduleDescriptorHandle, char * id, size_t * idSize);
    static PEAK_RETURN_CODE PEAK_Module_GetNumNodeMaps(PEAK_MODULE_HANDLE moduleHandle, size_t * numNodeMaps);
    static PEAK_RETURN_CODE PEAK_Module_GetNodeMap(PEAK_MODULE_HANDLE moduleHandle, size_t index, PEAK_NODE_MAP_HANDLE * nodeMapHandle);
    static PEAK_RETURN_CODE PEAK_Module_GetPort(PEAK_MODULE_HANDLE moduleHandle, PEAK_PORT_HANDLE * portHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_GetHasNode(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * nodeName, size_t nodeNameSize, PEAK_BOOL8 * hasNode);
    static PEAK_RETURN_CODE PEAK_NodeMap_FindNode(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * nodeName, size_t nodeNameSize, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_InvalidateNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_PollNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, int64_t elapsedTime_ms);
    static PEAK_RETURN_CODE PEAK_NodeMap_GetNumNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t * numNodes);
    static PEAK_RETURN_CODE PEAK_NodeMap_GetNode(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t index, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_GetHasBufferSupportedChunks(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasSupportedChunks);
    static PEAK_RETURN_CODE PEAK_NodeMap_UpdateChunkNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_GetHasEventSupportedData(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle, PEAK_BOOL8 * hasSupportedData);
    static PEAK_RETURN_CODE PEAK_NodeMap_UpdateEventNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_StoreToFile(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * filePath, size_t filePathSize);
    static PEAK_RETURN_CODE PEAK_NodeMap_LoadFromFile(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * filePath, size_t filePathSize);
    static PEAK_RETURN_CODE PEAK_NodeMap_Lock(PEAK_NODE_MAP_HANDLE nodeMapHandle);
    static PEAK_RETURN_CODE PEAK_NodeMap_Unlock(PEAK_NODE_MAP_HANDLE nodeMapHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToIntegerNode(PEAK_NODE_HANDLE nodeHandle, PEAK_INTEGER_NODE_HANDLE * integerNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToBooleanNode(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOLEAN_NODE_HANDLE * booleanNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToCommandNode(PEAK_NODE_HANDLE nodeHandle, PEAK_COMMAND_NODE_HANDLE * commandNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToFloatNode(PEAK_NODE_HANDLE nodeHandle, PEAK_FLOAT_NODE_HANDLE * floatNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToStringNode(PEAK_NODE_HANDLE nodeHandle, PEAK_STRING_NODE_HANDLE * stringNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToRegisterNode(PEAK_NODE_HANDLE nodeHandle, PEAK_REGISTER_NODE_HANDLE * registerNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToCategoryNode(PEAK_NODE_HANDLE nodeHandle, PEAK_CATEGORY_NODE_HANDLE * categoryNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToEnumerationNode(PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_NODE_HANDLE * enumerationNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_ToEnumerationEntryNode(PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_GetName(PEAK_NODE_HANDLE nodeHandle, char * name, size_t * nameSize);
    static PEAK_RETURN_CODE PEAK_Node_GetDisplayName(PEAK_NODE_HANDLE nodeHandle, char * displayName, size_t * displayNameSize);
    static PEAK_RETURN_CODE PEAK_Node_GetNamespace(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_NAMESPACE * _namespace);
    static PEAK_RETURN_CODE PEAK_Node_GetVisibility(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_VISIBILITY * visibility);
    static PEAK_RETURN_CODE PEAK_Node_GetAccessStatus(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_ACCESS_STATUS * accessStatus);
    static PEAK_RETURN_CODE PEAK_Node_GetIsCacheable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isCacheable);
    static PEAK_RETURN_CODE PEAK_Node_GetIsAccessStatusCacheable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isAccessStatusCacheable);
    static PEAK_RETURN_CODE PEAK_Node_GetIsStreamable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isStreamable);
    static PEAK_RETURN_CODE PEAK_Node_GetIsDeprecated(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isDeprecated);
    static PEAK_RETURN_CODE PEAK_Node_GetIsFeature(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isFeature);
    static PEAK_RETURN_CODE PEAK_Node_GetCachingMode(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CACHING_MODE * cachingMode);
    static PEAK_RETURN_CODE PEAK_Node_GetPollingTime(PEAK_NODE_HANDLE nodeHandle, int64_t * pollingTime_ms);
    static PEAK_RETURN_CODE PEAK_Node_GetToolTip(PEAK_NODE_HANDLE nodeHandle, char * toolTip, size_t * toolTipSize);
    static PEAK_RETURN_CODE PEAK_Node_GetDescription(PEAK_NODE_HANDLE nodeHandle, char * description, size_t * descriptionSize);
    static PEAK_RETURN_CODE PEAK_Node_GetType(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_TYPE * type);
    static PEAK_RETURN_CODE PEAK_Node_GetParentNodeMap(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_MAP_HANDLE * nodeMapHandle);
    static PEAK_RETURN_CODE PEAK_Node_FindInvalidatedNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * invalidatedNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_GetNumInvalidatedNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numInvalidatedNodes);
    static PEAK_RETURN_CODE PEAK_Node_GetInvalidatedNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * invalidatedNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_FindInvalidatingNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * invalidatingNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_GetNumInvalidatingNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numInvalidatingNodes);
    static PEAK_RETURN_CODE PEAK_Node_GetInvalidatingNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * invalidatingNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_FindSelectedNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * selectedNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_GetNumSelectedNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numSelectedNodes);
    static PEAK_RETURN_CODE PEAK_Node_GetSelectedNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * selectedNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_FindSelectingNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * selectingNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_GetNumSelectingNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numSelectingNodes);
    static PEAK_RETURN_CODE PEAK_Node_GetSelectingNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * selectingNodeHandle);
    static PEAK_RETURN_CODE PEAK_Node_RegisterChangedCallback(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK callback, void * callbackContext, PEAK_NODE_CHANGED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_Node_UnregisterChangedCallback(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_IntegerNode_ToNode(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetMinimum(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * minimum);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetMaximum(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * maximum);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetIncrement(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * increment);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetIncrementType(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_INCREMENT_TYPE * incrementType);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetValidValues(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * validValues, size_t * validValuesSize);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetRepresentation(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_REPRESENTATION * representation);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetUnit(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, char * unit, size_t * unitSize);
    static PEAK_RETURN_CODE PEAK_IntegerNode_GetValue(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, int64_t * value);
    static PEAK_RETURN_CODE PEAK_IntegerNode_SetValue(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t value);
    static PEAK_RETURN_CODE PEAK_BooleanNode_ToNode(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_BooleanNode_GetValue(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_BOOL8 * value);
    static PEAK_RETURN_CODE PEAK_BooleanNode_SetValue(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_BOOL8 value);
    static PEAK_RETURN_CODE PEAK_CommandNode_ToNode(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_CommandNode_GetIsDone(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_BOOL8 * isDone);
    static PEAK_RETURN_CODE PEAK_CommandNode_Execute(PEAK_COMMAND_NODE_HANDLE commandNodeHandle);
    static PEAK_RETURN_CODE PEAK_CommandNode_WaitUntilDoneInfinite(PEAK_COMMAND_NODE_HANDLE commandNodeHandle);
    static PEAK_RETURN_CODE PEAK_CommandNode_WaitUntilDone(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, uint64_t waitTimeout_ms);
    static PEAK_RETURN_CODE PEAK_FloatNode_ToNode(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetMinimum(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * minimum);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetMaximum(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * maximum);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetIncrement(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * increment);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetIncrementType(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_INCREMENT_TYPE * incrementType);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetValidValues(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * validValues, size_t * validValuesSize);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetRepresentation(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_REPRESENTATION * representation);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetUnit(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, char * unit, size_t * unitSize);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetDisplayNotation(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_DISPLAY_NOTATION * displayNotation);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetDisplayPrecision(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, int64_t * displayPrecision);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetHasConstantIncrement(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_BOOL8 * hasConstantIncrement);
    static PEAK_RETURN_CODE PEAK_FloatNode_GetValue(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, double * value);
    static PEAK_RETURN_CODE PEAK_FloatNode_SetValue(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double value);
    static PEAK_RETURN_CODE PEAK_StringNode_ToNode(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_StringNode_GetMaximumLength(PEAK_STRING_NODE_HANDLE stringNodeHandle, int64_t * maximumLength);
    static PEAK_RETURN_CODE PEAK_StringNode_GetValue(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, char * value, size_t * valueSize);
    static PEAK_RETURN_CODE PEAK_StringNode_SetValue(PEAK_STRING_NODE_HANDLE stringNodeHandle, const char * value, size_t valueSize);
    static PEAK_RETURN_CODE PEAK_RegisterNode_ToNode(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_RegisterNode_GetAddress(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, uint64_t * address);
    static PEAK_RETURN_CODE PEAK_RegisterNode_GetLength(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, size_t * length);
    static PEAK_RETURN_CODE PEAK_RegisterNode_Read(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, uint8_t * bytesToRead, size_t bytesToReadSize);
    static PEAK_RETURN_CODE PEAK_RegisterNode_Write(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, const uint8_t * bytesToWrite, size_t bytesToWriteSize);
    static PEAK_RETURN_CODE PEAK_CategoryNode_ToNode(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_CategoryNode_GetNumSubNodes(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t * numSubNodes);
    static PEAK_RETURN_CODE PEAK_CategoryNode_GetSubNode(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t index, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_ToNode(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_GetCurrentEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_SetCurrentEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char * symbolicValue, size_t symbolicValueSize);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_SetCurrentEntryByValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_FindEntryBySymbolicValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char * symbolicValue, size_t symbolicValueSize, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_FindEntryByValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_GetNumEntries(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t * numEntries);
    static PEAK_RETURN_CODE PEAK_EnumerationNode_GetEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t index, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationEntryNode_ToNode(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_NODE_HANDLE * nodeHandle);
    static PEAK_RETURN_CODE PEAK_EnumerationEntryNode_GetIsSelfClearing(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_BOOL8 * isSelfClearing);
    static PEAK_RETURN_CODE PEAK_EnumerationEntryNode_GetSymbolicValue(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, char * symbolicValue, size_t * symbolicValueSize);
    static PEAK_RETURN_CODE PEAK_EnumerationEntryNode_GetValue(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, int64_t * value);
    static PEAK_RETURN_CODE PEAK_Port_GetInfo(PEAK_PORT_HANDLE portHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_Port_GetID(PEAK_PORT_HANDLE portHandle, char * id, size_t * idSize);
    static PEAK_RETURN_CODE PEAK_Port_GetName(PEAK_PORT_HANDLE portHandle, char * name, size_t * nameSize);
    static PEAK_RETURN_CODE PEAK_Port_GetVendorName(PEAK_PORT_HANDLE portHandle, char * vendorName, size_t * vendorNameSize);
    static PEAK_RETURN_CODE PEAK_Port_GetModelName(PEAK_PORT_HANDLE portHandle, char * modelName, size_t * modelNameSize);
    static PEAK_RETURN_CODE PEAK_Port_GetVersion(PEAK_PORT_HANDLE portHandle, char * version, size_t * versionSize);
    static PEAK_RETURN_CODE PEAK_Port_GetTLType(PEAK_PORT_HANDLE portHandle, char * tlType, size_t * tlTypeSize);
    static PEAK_RETURN_CODE PEAK_Port_GetModuleName(PEAK_PORT_HANDLE portHandle, char * moduleName, size_t * moduleNameSize);
    static PEAK_RETURN_CODE PEAK_Port_GetDataEndianness(PEAK_PORT_HANDLE portHandle, PEAK_ENDIANNESS * dataEndianness);
    static PEAK_RETURN_CODE PEAK_Port_GetIsReadable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isReadable);
    static PEAK_RETURN_CODE PEAK_Port_GetIsWritable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isWritable);
    static PEAK_RETURN_CODE PEAK_Port_GetIsAvailable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isAvailable);
    static PEAK_RETURN_CODE PEAK_Port_GetIsImplemented(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isImplemented);
    static PEAK_RETURN_CODE PEAK_Port_Read(PEAK_PORT_HANDLE portHandle, uint64_t address, uint8_t * bytesToRead, size_t bytesToReadSize);
    static PEAK_RETURN_CODE PEAK_Port_Write(PEAK_PORT_HANDLE portHandle, uint64_t address, const uint8_t * bytesToWrite, size_t bytesToWriteSize);
    static PEAK_RETURN_CODE PEAK_Port_GetNumURLs(PEAK_PORT_HANDLE portHandle, size_t * numUrls);
    static PEAK_RETURN_CODE PEAK_Port_GetURL(PEAK_PORT_HANDLE portHandle, size_t index, PEAK_PORT_URL_HANDLE * portUrlHandle);
    static PEAK_RETURN_CODE PEAK_PortURL_GetInfo(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_PortURL_GetURL(PEAK_PORT_URL_HANDLE portUrlHandle, char * url, size_t * urlSize);
    static PEAK_RETURN_CODE PEAK_PortURL_GetScheme(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_URL_SCHEME * scheme);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileName(PEAK_PORT_URL_HANDLE portUrlHandle, char * fileName, size_t * fileNameSize);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileRegisterAddress(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t * fileRegisterAddress);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileSize(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t * fileSize);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileSHA1Hash(PEAK_PORT_URL_HANDLE portUrlHandle, uint8_t * fileSha1Hash, size_t * fileSha1HashSize);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileVersionMajor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionMajor);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileVersionMinor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionMinor);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileVersionSubminor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionSubminor);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileSchemaVersionMajor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileSchemaVersionMajor);
    static PEAK_RETURN_CODE PEAK_PortURL_GetFileSchemaVersionMinor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileSchemaVersionMinor);
    static PEAK_RETURN_CODE PEAK_PortURL_GetParentPort(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_HANDLE * portHandle);
    static PEAK_RETURN_CODE PEAK_EventSupportingModule_EnableEvents(PEAK_EVENT_SUPPORTING_MODULE_HANDLE eventSupportingModuleHandle, PEAK_EVENT_TYPE eventType, PEAK_EVENT_CONTROLLER_HANDLE * eventControllerHandle);
    static PEAK_RETURN_CODE PEAK_EventController_GetInfo(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_EventController_GetNumEventsInQueue(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * numEventsInQueue);
    static PEAK_RETURN_CODE PEAK_EventController_GetNumEventsFired(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t * numEventsFired);
    static PEAK_RETURN_CODE PEAK_EventController_GetEventMaxSize(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * eventMaxSize);
    static PEAK_RETURN_CODE PEAK_EventController_GetEventDataMaxSize(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * eventDataMaxSize);
    static PEAK_RETURN_CODE PEAK_EventController_GetControlledEventType(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, PEAK_EVENT_TYPE * controlledEventType);
    static PEAK_RETURN_CODE PEAK_EventController_WaitForEvent(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t timeout_ms, PEAK_EVENT_HANDLE * eventHandle);
    static PEAK_RETURN_CODE PEAK_EventController_KillWait(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
    static PEAK_RETURN_CODE PEAK_EventController_FlushEvents(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
    static PEAK_RETURN_CODE PEAK_EventController_Destruct(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
    static PEAK_RETURN_CODE PEAK_Event_GetInfo(PEAK_EVENT_HANDLE eventHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize);
    static PEAK_RETURN_CODE PEAK_Event_GetID(PEAK_EVENT_HANDLE eventHandle, uint64_t * id);
    static PEAK_RETURN_CODE PEAK_Event_GetData(PEAK_EVENT_HANDLE eventHandle, uint8_t * data, size_t * dataSize);
    static PEAK_RETURN_CODE PEAK_Event_GetType(PEAK_EVENT_HANDLE eventHandle, PEAK_EVENT_TYPE * type);
    static PEAK_RETURN_CODE PEAK_Event_GetRawData(PEAK_EVENT_HANDLE eventHandle, uint8_t * rawData, size_t * rawDataSize);
    static PEAK_RETURN_CODE PEAK_Event_Destruct(PEAK_EVENT_HANDLE eventHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_Construct(PEAK_FIRMWARE_UPDATER_HANDLE * firmwareUpdaterHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char * gufPath, size_t gufPathSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char * gufPath, size_t gufPathSize, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t * numFirmwareUpdateInformation);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_GetFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t index, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE * firmwareUpdateInformationHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_UpdateDevice(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, uint64_t deviceResetDiscoveryTimeout_ms);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdater_Destruct(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetIsValid(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_BOOL8 * isValid);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetFileName(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * fileName, size_t * fileNameSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetDescription(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * description, size_t * descriptionSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetVersion(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * version, size_t * versionSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * versionExtractionPattern, size_t * versionExtractionPatternSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetVersionStyle(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_VERSION_STYLE * versionStyle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetReleaseNotes(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * releaseNotes, size_t * releaseNotesSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetReleaseNotesURL(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * releaseNotesUrl, size_t * releaseNotesUrlSize);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetUserSetPersistence(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PERSISTENCE * userSetPersistence);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PERSISTENCE * sequencerSetPersistence);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_Construct(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE * firmwareUpdateProgressObserverHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE * callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE callbackHandle);
    static PEAK_RETURN_CODE PEAK_FirmwareUpdateProgressObserver_Destruct(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle);
       
private:
    void* m_handle = nullptr;
    dyn_PEAK_Library_Initialize m_PEAK_Library_Initialize{};
    dyn_PEAK_Library_Close m_PEAK_Library_Close{};
    dyn_PEAK_Library_IsInitialized m_PEAK_Library_IsInitialized{};
    dyn_PEAK_Library_GetVersionMajor m_PEAK_Library_GetVersionMajor{};
    dyn_PEAK_Library_GetVersionMinor m_PEAK_Library_GetVersionMinor{};
    dyn_PEAK_Library_GetVersionSubminor m_PEAK_Library_GetVersionSubminor{};
    dyn_PEAK_Library_GetLastError m_PEAK_Library_GetLastError{};
    dyn_PEAK_EnvironmentInspector_UpdateCTIPaths m_PEAK_EnvironmentInspector_UpdateCTIPaths{};
    dyn_PEAK_EnvironmentInspector_GetNumCTIPaths m_PEAK_EnvironmentInspector_GetNumCTIPaths{};
    dyn_PEAK_EnvironmentInspector_GetCTIPath m_PEAK_EnvironmentInspector_GetCTIPath{};
    dyn_PEAK_ProducerLibrary_Construct m_PEAK_ProducerLibrary_Construct{};
    dyn_PEAK_ProducerLibrary_GetKey m_PEAK_ProducerLibrary_GetKey{};
    dyn_PEAK_ProducerLibrary_GetSystem m_PEAK_ProducerLibrary_GetSystem{};
    dyn_PEAK_ProducerLibrary_Destruct m_PEAK_ProducerLibrary_Destruct{};
    dyn_PEAK_SystemDescriptor_ToModuleDescriptor m_PEAK_SystemDescriptor_ToModuleDescriptor{};
    dyn_PEAK_SystemDescriptor_GetKey m_PEAK_SystemDescriptor_GetKey{};
    dyn_PEAK_SystemDescriptor_GetInfo m_PEAK_SystemDescriptor_GetInfo{};
    dyn_PEAK_SystemDescriptor_GetDisplayName m_PEAK_SystemDescriptor_GetDisplayName{};
    dyn_PEAK_SystemDescriptor_GetVendorName m_PEAK_SystemDescriptor_GetVendorName{};
    dyn_PEAK_SystemDescriptor_GetModelName m_PEAK_SystemDescriptor_GetModelName{};
    dyn_PEAK_SystemDescriptor_GetVersion m_PEAK_SystemDescriptor_GetVersion{};
    dyn_PEAK_SystemDescriptor_GetTLType m_PEAK_SystemDescriptor_GetTLType{};
    dyn_PEAK_SystemDescriptor_GetCTIFileName m_PEAK_SystemDescriptor_GetCTIFileName{};
    dyn_PEAK_SystemDescriptor_GetCTIFullPath m_PEAK_SystemDescriptor_GetCTIFullPath{};
    dyn_PEAK_SystemDescriptor_GetGenTLVersionMajor m_PEAK_SystemDescriptor_GetGenTLVersionMajor{};
    dyn_PEAK_SystemDescriptor_GetGenTLVersionMinor m_PEAK_SystemDescriptor_GetGenTLVersionMinor{};
    dyn_PEAK_SystemDescriptor_GetCharacterEncoding m_PEAK_SystemDescriptor_GetCharacterEncoding{};
    dyn_PEAK_SystemDescriptor_GetParentLibrary m_PEAK_SystemDescriptor_GetParentLibrary{};
    dyn_PEAK_SystemDescriptor_OpenSystem m_PEAK_SystemDescriptor_OpenSystem{};
    dyn_PEAK_System_ToModule m_PEAK_System_ToModule{};
    dyn_PEAK_System_ToEventSupportingModule m_PEAK_System_ToEventSupportingModule{};
    dyn_PEAK_System_GetKey m_PEAK_System_GetKey{};
    dyn_PEAK_System_GetInfo m_PEAK_System_GetInfo{};
    dyn_PEAK_System_GetID m_PEAK_System_GetID{};
    dyn_PEAK_System_GetDisplayName m_PEAK_System_GetDisplayName{};
    dyn_PEAK_System_GetVendorName m_PEAK_System_GetVendorName{};
    dyn_PEAK_System_GetModelName m_PEAK_System_GetModelName{};
    dyn_PEAK_System_GetVersion m_PEAK_System_GetVersion{};
    dyn_PEAK_System_GetTLType m_PEAK_System_GetTLType{};
    dyn_PEAK_System_GetCTIFileName m_PEAK_System_GetCTIFileName{};
    dyn_PEAK_System_GetCTIFullPath m_PEAK_System_GetCTIFullPath{};
    dyn_PEAK_System_GetGenTLVersionMajor m_PEAK_System_GetGenTLVersionMajor{};
    dyn_PEAK_System_GetGenTLVersionMinor m_PEAK_System_GetGenTLVersionMinor{};
    dyn_PEAK_System_GetCharacterEncoding m_PEAK_System_GetCharacterEncoding{};
    dyn_PEAK_System_GetParentLibrary m_PEAK_System_GetParentLibrary{};
    dyn_PEAK_System_UpdateInterfaces m_PEAK_System_UpdateInterfaces{};
    dyn_PEAK_System_GetNumInterfaces m_PEAK_System_GetNumInterfaces{};
    dyn_PEAK_System_GetInterface m_PEAK_System_GetInterface{};
    dyn_PEAK_System_RegisterInterfaceFoundCallback m_PEAK_System_RegisterInterfaceFoundCallback{};
    dyn_PEAK_System_UnregisterInterfaceFoundCallback m_PEAK_System_UnregisterInterfaceFoundCallback{};
    dyn_PEAK_System_RegisterInterfaceLostCallback m_PEAK_System_RegisterInterfaceLostCallback{};
    dyn_PEAK_System_UnregisterInterfaceLostCallback m_PEAK_System_UnregisterInterfaceLostCallback{};
    dyn_PEAK_System_Destruct m_PEAK_System_Destruct{};
    dyn_PEAK_InterfaceDescriptor_ToModuleDescriptor m_PEAK_InterfaceDescriptor_ToModuleDescriptor{};
    dyn_PEAK_InterfaceDescriptor_GetKey m_PEAK_InterfaceDescriptor_GetKey{};
    dyn_PEAK_InterfaceDescriptor_GetInfo m_PEAK_InterfaceDescriptor_GetInfo{};
    dyn_PEAK_InterfaceDescriptor_GetDisplayName m_PEAK_InterfaceDescriptor_GetDisplayName{};
    dyn_PEAK_InterfaceDescriptor_GetTLType m_PEAK_InterfaceDescriptor_GetTLType{};
    dyn_PEAK_InterfaceDescriptor_GetParentSystem m_PEAK_InterfaceDescriptor_GetParentSystem{};
    dyn_PEAK_InterfaceDescriptor_OpenInterface m_PEAK_InterfaceDescriptor_OpenInterface{};
    dyn_PEAK_Interface_ToModule m_PEAK_Interface_ToModule{};
    dyn_PEAK_Interface_ToEventSupportingModule m_PEAK_Interface_ToEventSupportingModule{};
    dyn_PEAK_Interface_GetKey m_PEAK_Interface_GetKey{};
    dyn_PEAK_Interface_GetInfo m_PEAK_Interface_GetInfo{};
    dyn_PEAK_Interface_GetID m_PEAK_Interface_GetID{};
    dyn_PEAK_Interface_GetDisplayName m_PEAK_Interface_GetDisplayName{};
    dyn_PEAK_Interface_GetTLType m_PEAK_Interface_GetTLType{};
    dyn_PEAK_Interface_GetParentSystem m_PEAK_Interface_GetParentSystem{};
    dyn_PEAK_Interface_UpdateDevices m_PEAK_Interface_UpdateDevices{};
    dyn_PEAK_Interface_GetNumDevices m_PEAK_Interface_GetNumDevices{};
    dyn_PEAK_Interface_GetDevice m_PEAK_Interface_GetDevice{};
    dyn_PEAK_Interface_RegisterDeviceFoundCallback m_PEAK_Interface_RegisterDeviceFoundCallback{};
    dyn_PEAK_Interface_UnregisterDeviceFoundCallback m_PEAK_Interface_UnregisterDeviceFoundCallback{};
    dyn_PEAK_Interface_RegisterDeviceLostCallback m_PEAK_Interface_RegisterDeviceLostCallback{};
    dyn_PEAK_Interface_UnregisterDeviceLostCallback m_PEAK_Interface_UnregisterDeviceLostCallback{};
    dyn_PEAK_Interface_Destruct m_PEAK_Interface_Destruct{};
    dyn_PEAK_DeviceDescriptor_ToModuleDescriptor m_PEAK_DeviceDescriptor_ToModuleDescriptor{};
    dyn_PEAK_DeviceDescriptor_GetKey m_PEAK_DeviceDescriptor_GetKey{};
    dyn_PEAK_DeviceDescriptor_GetInfo m_PEAK_DeviceDescriptor_GetInfo{};
    dyn_PEAK_DeviceDescriptor_GetDisplayName m_PEAK_DeviceDescriptor_GetDisplayName{};
    dyn_PEAK_DeviceDescriptor_GetVendorName m_PEAK_DeviceDescriptor_GetVendorName{};
    dyn_PEAK_DeviceDescriptor_GetModelName m_PEAK_DeviceDescriptor_GetModelName{};
    dyn_PEAK_DeviceDescriptor_GetVersion m_PEAK_DeviceDescriptor_GetVersion{};
    dyn_PEAK_DeviceDescriptor_GetTLType m_PEAK_DeviceDescriptor_GetTLType{};
    dyn_PEAK_DeviceDescriptor_GetUserDefinedName m_PEAK_DeviceDescriptor_GetUserDefinedName{};
    dyn_PEAK_DeviceDescriptor_GetSerialNumber m_PEAK_DeviceDescriptor_GetSerialNumber{};
    dyn_PEAK_DeviceDescriptor_GetAccessStatus m_PEAK_DeviceDescriptor_GetAccessStatus{};
    dyn_PEAK_DeviceDescriptor_GetTimestampTickFrequency m_PEAK_DeviceDescriptor_GetTimestampTickFrequency{};
    dyn_PEAK_DeviceDescriptor_GetIsOpenableExclusive m_PEAK_DeviceDescriptor_GetIsOpenableExclusive{};
    dyn_PEAK_DeviceDescriptor_GetIsOpenable m_PEAK_DeviceDescriptor_GetIsOpenable{};
    dyn_PEAK_DeviceDescriptor_OpenDevice m_PEAK_DeviceDescriptor_OpenDevice{};
    dyn_PEAK_DeviceDescriptor_GetParentInterface m_PEAK_DeviceDescriptor_GetParentInterface{};
    dyn_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval m_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval{};
    dyn_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval m_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval{};
    dyn_PEAK_DeviceDescriptor_IsInformationRoleMonitored m_PEAK_DeviceDescriptor_IsInformationRoleMonitored{};
    dyn_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring m_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring{};
    dyn_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring m_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring{};
    dyn_PEAK_DeviceDescriptor_RegisterInformationChangedCallback m_PEAK_DeviceDescriptor_RegisterInformationChangedCallback{};
    dyn_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback m_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback{};
    dyn_PEAK_Device_ToModule m_PEAK_Device_ToModule{};
    dyn_PEAK_Device_ToEventSupportingModule m_PEAK_Device_ToEventSupportingModule{};
    dyn_PEAK_Device_GetKey m_PEAK_Device_GetKey{};
    dyn_PEAK_Device_GetInfo m_PEAK_Device_GetInfo{};
    dyn_PEAK_Device_GetID m_PEAK_Device_GetID{};
    dyn_PEAK_Device_GetDisplayName m_PEAK_Device_GetDisplayName{};
    dyn_PEAK_Device_GetVendorName m_PEAK_Device_GetVendorName{};
    dyn_PEAK_Device_GetModelName m_PEAK_Device_GetModelName{};
    dyn_PEAK_Device_GetVersion m_PEAK_Device_GetVersion{};
    dyn_PEAK_Device_GetTLType m_PEAK_Device_GetTLType{};
    dyn_PEAK_Device_GetUserDefinedName m_PEAK_Device_GetUserDefinedName{};
    dyn_PEAK_Device_GetSerialNumber m_PEAK_Device_GetSerialNumber{};
    dyn_PEAK_Device_GetAccessStatus m_PEAK_Device_GetAccessStatus{};
    dyn_PEAK_Device_GetTimestampTickFrequency m_PEAK_Device_GetTimestampTickFrequency{};
    dyn_PEAK_Device_GetParentInterface m_PEAK_Device_GetParentInterface{};
    dyn_PEAK_Device_GetRemoteDevice m_PEAK_Device_GetRemoteDevice{};
    dyn_PEAK_Device_GetNumDataStreams m_PEAK_Device_GetNumDataStreams{};
    dyn_PEAK_Device_GetDataStream m_PEAK_Device_GetDataStream{};
    dyn_PEAK_Device_Destruct m_PEAK_Device_Destruct{};
    dyn_PEAK_RemoteDevice_ToModule m_PEAK_RemoteDevice_ToModule{};
    dyn_PEAK_RemoteDevice_GetLocalDevice m_PEAK_RemoteDevice_GetLocalDevice{};
    dyn_PEAK_DataStreamDescriptor_ToModuleDescriptor m_PEAK_DataStreamDescriptor_ToModuleDescriptor{};
    dyn_PEAK_DataStreamDescriptor_GetKey m_PEAK_DataStreamDescriptor_GetKey{};
    dyn_PEAK_DataStreamDescriptor_GetParentDevice m_PEAK_DataStreamDescriptor_GetParentDevice{};
    dyn_PEAK_DataStreamDescriptor_OpenDataStream m_PEAK_DataStreamDescriptor_OpenDataStream{};
    dyn_PEAK_DataStream_ToModule m_PEAK_DataStream_ToModule{};
    dyn_PEAK_DataStream_ToEventSupportingModule m_PEAK_DataStream_ToEventSupportingModule{};
    dyn_PEAK_DataStream_GetKey m_PEAK_DataStream_GetKey{};
    dyn_PEAK_DataStream_GetInfo m_PEAK_DataStream_GetInfo{};
    dyn_PEAK_DataStream_GetID m_PEAK_DataStream_GetID{};
    dyn_PEAK_DataStream_GetTLType m_PEAK_DataStream_GetTLType{};
    dyn_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired m_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired{};
    dyn_PEAK_DataStream_GetNumBuffersAnnounced m_PEAK_DataStream_GetNumBuffersAnnounced{};
    dyn_PEAK_DataStream_GetNumBuffersQueued m_PEAK_DataStream_GetNumBuffersQueued{};
    dyn_PEAK_DataStream_GetNumBuffersAwaitDelivery m_PEAK_DataStream_GetNumBuffersAwaitDelivery{};
    dyn_PEAK_DataStream_GetNumBuffersDelivered m_PEAK_DataStream_GetNumBuffersDelivered{};
    dyn_PEAK_DataStream_GetNumBuffersStarted m_PEAK_DataStream_GetNumBuffersStarted{};
    dyn_PEAK_DataStream_GetNumUnderruns m_PEAK_DataStream_GetNumUnderruns{};
    dyn_PEAK_DataStream_GetNumChunksPerBufferMax m_PEAK_DataStream_GetNumChunksPerBufferMax{};
    dyn_PEAK_DataStream_GetBufferAlignment m_PEAK_DataStream_GetBufferAlignment{};
    dyn_PEAK_DataStream_GetPayloadSize m_PEAK_DataStream_GetPayloadSize{};
    dyn_PEAK_DataStream_GetDefinesPayloadSize m_PEAK_DataStream_GetDefinesPayloadSize{};
    dyn_PEAK_DataStream_GetIsGrabbing m_PEAK_DataStream_GetIsGrabbing{};
    dyn_PEAK_DataStream_GetParentDevice m_PEAK_DataStream_GetParentDevice{};
    dyn_PEAK_DataStream_AnnounceBuffer m_PEAK_DataStream_AnnounceBuffer{};
    dyn_PEAK_DataStream_AllocAndAnnounceBuffer m_PEAK_DataStream_AllocAndAnnounceBuffer{};
    dyn_PEAK_DataStream_QueueBuffer m_PEAK_DataStream_QueueBuffer{};
    dyn_PEAK_DataStream_RevokeBuffer m_PEAK_DataStream_RevokeBuffer{};
    dyn_PEAK_DataStream_WaitForFinishedBuffer m_PEAK_DataStream_WaitForFinishedBuffer{};
    dyn_PEAK_DataStream_KillWait m_PEAK_DataStream_KillWait{};
    dyn_PEAK_DataStream_Flush m_PEAK_DataStream_Flush{};
    dyn_PEAK_DataStream_StartAcquisitionInfinite m_PEAK_DataStream_StartAcquisitionInfinite{};
    dyn_PEAK_DataStream_StartAcquisition m_PEAK_DataStream_StartAcquisition{};
    dyn_PEAK_DataStream_StopAcquisition m_PEAK_DataStream_StopAcquisition{};
    dyn_PEAK_DataStream_Destruct m_PEAK_DataStream_Destruct{};
    dyn_PEAK_Buffer_ToModule m_PEAK_Buffer_ToModule{};
    dyn_PEAK_Buffer_ToEventSupportingModule m_PEAK_Buffer_ToEventSupportingModule{};
    dyn_PEAK_Buffer_GetInfo m_PEAK_Buffer_GetInfo{};
    dyn_PEAK_Buffer_GetTLType m_PEAK_Buffer_GetTLType{};
    dyn_PEAK_Buffer_GetBasePtr m_PEAK_Buffer_GetBasePtr{};
    dyn_PEAK_Buffer_GetSize m_PEAK_Buffer_GetSize{};
    dyn_PEAK_Buffer_GetUserPtr m_PEAK_Buffer_GetUserPtr{};
    dyn_PEAK_Buffer_GetPayloadType m_PEAK_Buffer_GetPayloadType{};
    dyn_PEAK_Buffer_GetPixelFormat m_PEAK_Buffer_GetPixelFormat{};
    dyn_PEAK_Buffer_GetPixelFormatNamespace m_PEAK_Buffer_GetPixelFormatNamespace{};
    dyn_PEAK_Buffer_GetPixelEndianness m_PEAK_Buffer_GetPixelEndianness{};
    dyn_PEAK_Buffer_GetExpectedDataSize m_PEAK_Buffer_GetExpectedDataSize{};
    dyn_PEAK_Buffer_GetDeliveredDataSize m_PEAK_Buffer_GetDeliveredDataSize{};
    dyn_PEAK_Buffer_GetFrameID m_PEAK_Buffer_GetFrameID{};
    dyn_PEAK_Buffer_GetImageOffset m_PEAK_Buffer_GetImageOffset{};
    dyn_PEAK_Buffer_GetDeliveredImageHeight m_PEAK_Buffer_GetDeliveredImageHeight{};
    dyn_PEAK_Buffer_GetDeliveredChunkPayloadSize m_PEAK_Buffer_GetDeliveredChunkPayloadSize{};
    dyn_PEAK_Buffer_GetChunkLayoutID m_PEAK_Buffer_GetChunkLayoutID{};
    dyn_PEAK_Buffer_GetFileName m_PEAK_Buffer_GetFileName{};
    dyn_PEAK_Buffer_GetWidth m_PEAK_Buffer_GetWidth{};
    dyn_PEAK_Buffer_GetHeight m_PEAK_Buffer_GetHeight{};
    dyn_PEAK_Buffer_GetXOffset m_PEAK_Buffer_GetXOffset{};
    dyn_PEAK_Buffer_GetYOffset m_PEAK_Buffer_GetYOffset{};
    dyn_PEAK_Buffer_GetXPadding m_PEAK_Buffer_GetXPadding{};
    dyn_PEAK_Buffer_GetYPadding m_PEAK_Buffer_GetYPadding{};
    dyn_PEAK_Buffer_GetTimestamp_ticks m_PEAK_Buffer_GetTimestamp_ticks{};
    dyn_PEAK_Buffer_GetTimestamp_ns m_PEAK_Buffer_GetTimestamp_ns{};
    dyn_PEAK_Buffer_GetIsQueued m_PEAK_Buffer_GetIsQueued{};
    dyn_PEAK_Buffer_GetIsAcquiring m_PEAK_Buffer_GetIsAcquiring{};
    dyn_PEAK_Buffer_GetIsIncomplete m_PEAK_Buffer_GetIsIncomplete{};
    dyn_PEAK_Buffer_GetHasNewData m_PEAK_Buffer_GetHasNewData{};
    dyn_PEAK_Buffer_GetHasImage m_PEAK_Buffer_GetHasImage{};
    dyn_PEAK_Buffer_GetHasChunks m_PEAK_Buffer_GetHasChunks{};
    dyn_PEAK_Buffer_UpdateChunks m_PEAK_Buffer_UpdateChunks{};
    dyn_PEAK_Buffer_GetNumChunks m_PEAK_Buffer_GetNumChunks{};
    dyn_PEAK_Buffer_GetChunk m_PEAK_Buffer_GetChunk{};
    dyn_PEAK_Buffer_UpdateParts m_PEAK_Buffer_UpdateParts{};
    dyn_PEAK_Buffer_GetNumParts m_PEAK_Buffer_GetNumParts{};
    dyn_PEAK_Buffer_GetPart m_PEAK_Buffer_GetPart{};
    dyn_PEAK_BufferChunk_GetID m_PEAK_BufferChunk_GetID{};
    dyn_PEAK_BufferChunk_GetBasePtr m_PEAK_BufferChunk_GetBasePtr{};
    dyn_PEAK_BufferChunk_GetSize m_PEAK_BufferChunk_GetSize{};
    dyn_PEAK_BufferChunk_GetParentBuffer m_PEAK_BufferChunk_GetParentBuffer{};
    dyn_PEAK_BufferPart_GetInfo m_PEAK_BufferPart_GetInfo{};
    dyn_PEAK_BufferPart_GetSourceID m_PEAK_BufferPart_GetSourceID{};
    dyn_PEAK_BufferPart_GetBasePtr m_PEAK_BufferPart_GetBasePtr{};
    dyn_PEAK_BufferPart_GetSize m_PEAK_BufferPart_GetSize{};
    dyn_PEAK_BufferPart_GetType m_PEAK_BufferPart_GetType{};
    dyn_PEAK_BufferPart_GetFormat m_PEAK_BufferPart_GetFormat{};
    dyn_PEAK_BufferPart_GetFormatNamespace m_PEAK_BufferPart_GetFormatNamespace{};
    dyn_PEAK_BufferPart_GetWidth m_PEAK_BufferPart_GetWidth{};
    dyn_PEAK_BufferPart_GetHeight m_PEAK_BufferPart_GetHeight{};
    dyn_PEAK_BufferPart_GetXOffset m_PEAK_BufferPart_GetXOffset{};
    dyn_PEAK_BufferPart_GetYOffset m_PEAK_BufferPart_GetYOffset{};
    dyn_PEAK_BufferPart_GetXPadding m_PEAK_BufferPart_GetXPadding{};
    dyn_PEAK_BufferPart_GetDeliveredImageHeight m_PEAK_BufferPart_GetDeliveredImageHeight{};
    dyn_PEAK_BufferPart_GetParentBuffer m_PEAK_BufferPart_GetParentBuffer{};
    dyn_PEAK_ModuleDescriptor_GetID m_PEAK_ModuleDescriptor_GetID{};
    dyn_PEAK_Module_GetNumNodeMaps m_PEAK_Module_GetNumNodeMaps{};
    dyn_PEAK_Module_GetNodeMap m_PEAK_Module_GetNodeMap{};
    dyn_PEAK_Module_GetPort m_PEAK_Module_GetPort{};
    dyn_PEAK_NodeMap_GetHasNode m_PEAK_NodeMap_GetHasNode{};
    dyn_PEAK_NodeMap_FindNode m_PEAK_NodeMap_FindNode{};
    dyn_PEAK_NodeMap_InvalidateNodes m_PEAK_NodeMap_InvalidateNodes{};
    dyn_PEAK_NodeMap_PollNodes m_PEAK_NodeMap_PollNodes{};
    dyn_PEAK_NodeMap_GetNumNodes m_PEAK_NodeMap_GetNumNodes{};
    dyn_PEAK_NodeMap_GetNode m_PEAK_NodeMap_GetNode{};
    dyn_PEAK_NodeMap_GetHasBufferSupportedChunks m_PEAK_NodeMap_GetHasBufferSupportedChunks{};
    dyn_PEAK_NodeMap_UpdateChunkNodes m_PEAK_NodeMap_UpdateChunkNodes{};
    dyn_PEAK_NodeMap_GetHasEventSupportedData m_PEAK_NodeMap_GetHasEventSupportedData{};
    dyn_PEAK_NodeMap_UpdateEventNodes m_PEAK_NodeMap_UpdateEventNodes{};
    dyn_PEAK_NodeMap_StoreToFile m_PEAK_NodeMap_StoreToFile{};
    dyn_PEAK_NodeMap_LoadFromFile m_PEAK_NodeMap_LoadFromFile{};
    dyn_PEAK_NodeMap_Lock m_PEAK_NodeMap_Lock{};
    dyn_PEAK_NodeMap_Unlock m_PEAK_NodeMap_Unlock{};
    dyn_PEAK_Node_ToIntegerNode m_PEAK_Node_ToIntegerNode{};
    dyn_PEAK_Node_ToBooleanNode m_PEAK_Node_ToBooleanNode{};
    dyn_PEAK_Node_ToCommandNode m_PEAK_Node_ToCommandNode{};
    dyn_PEAK_Node_ToFloatNode m_PEAK_Node_ToFloatNode{};
    dyn_PEAK_Node_ToStringNode m_PEAK_Node_ToStringNode{};
    dyn_PEAK_Node_ToRegisterNode m_PEAK_Node_ToRegisterNode{};
    dyn_PEAK_Node_ToCategoryNode m_PEAK_Node_ToCategoryNode{};
    dyn_PEAK_Node_ToEnumerationNode m_PEAK_Node_ToEnumerationNode{};
    dyn_PEAK_Node_ToEnumerationEntryNode m_PEAK_Node_ToEnumerationEntryNode{};
    dyn_PEAK_Node_GetName m_PEAK_Node_GetName{};
    dyn_PEAK_Node_GetDisplayName m_PEAK_Node_GetDisplayName{};
    dyn_PEAK_Node_GetNamespace m_PEAK_Node_GetNamespace{};
    dyn_PEAK_Node_GetVisibility m_PEAK_Node_GetVisibility{};
    dyn_PEAK_Node_GetAccessStatus m_PEAK_Node_GetAccessStatus{};
    dyn_PEAK_Node_GetIsCacheable m_PEAK_Node_GetIsCacheable{};
    dyn_PEAK_Node_GetIsAccessStatusCacheable m_PEAK_Node_GetIsAccessStatusCacheable{};
    dyn_PEAK_Node_GetIsStreamable m_PEAK_Node_GetIsStreamable{};
    dyn_PEAK_Node_GetIsDeprecated m_PEAK_Node_GetIsDeprecated{};
    dyn_PEAK_Node_GetIsFeature m_PEAK_Node_GetIsFeature{};
    dyn_PEAK_Node_GetCachingMode m_PEAK_Node_GetCachingMode{};
    dyn_PEAK_Node_GetPollingTime m_PEAK_Node_GetPollingTime{};
    dyn_PEAK_Node_GetToolTip m_PEAK_Node_GetToolTip{};
    dyn_PEAK_Node_GetDescription m_PEAK_Node_GetDescription{};
    dyn_PEAK_Node_GetType m_PEAK_Node_GetType{};
    dyn_PEAK_Node_GetParentNodeMap m_PEAK_Node_GetParentNodeMap{};
    dyn_PEAK_Node_FindInvalidatedNode m_PEAK_Node_FindInvalidatedNode{};
    dyn_PEAK_Node_GetNumInvalidatedNodes m_PEAK_Node_GetNumInvalidatedNodes{};
    dyn_PEAK_Node_GetInvalidatedNode m_PEAK_Node_GetInvalidatedNode{};
    dyn_PEAK_Node_FindInvalidatingNode m_PEAK_Node_FindInvalidatingNode{};
    dyn_PEAK_Node_GetNumInvalidatingNodes m_PEAK_Node_GetNumInvalidatingNodes{};
    dyn_PEAK_Node_GetInvalidatingNode m_PEAK_Node_GetInvalidatingNode{};
    dyn_PEAK_Node_FindSelectedNode m_PEAK_Node_FindSelectedNode{};
    dyn_PEAK_Node_GetNumSelectedNodes m_PEAK_Node_GetNumSelectedNodes{};
    dyn_PEAK_Node_GetSelectedNode m_PEAK_Node_GetSelectedNode{};
    dyn_PEAK_Node_FindSelectingNode m_PEAK_Node_FindSelectingNode{};
    dyn_PEAK_Node_GetNumSelectingNodes m_PEAK_Node_GetNumSelectingNodes{};
    dyn_PEAK_Node_GetSelectingNode m_PEAK_Node_GetSelectingNode{};
    dyn_PEAK_Node_RegisterChangedCallback m_PEAK_Node_RegisterChangedCallback{};
    dyn_PEAK_Node_UnregisterChangedCallback m_PEAK_Node_UnregisterChangedCallback{};
    dyn_PEAK_IntegerNode_ToNode m_PEAK_IntegerNode_ToNode{};
    dyn_PEAK_IntegerNode_GetMinimum m_PEAK_IntegerNode_GetMinimum{};
    dyn_PEAK_IntegerNode_GetMaximum m_PEAK_IntegerNode_GetMaximum{};
    dyn_PEAK_IntegerNode_GetIncrement m_PEAK_IntegerNode_GetIncrement{};
    dyn_PEAK_IntegerNode_GetIncrementType m_PEAK_IntegerNode_GetIncrementType{};
    dyn_PEAK_IntegerNode_GetValidValues m_PEAK_IntegerNode_GetValidValues{};
    dyn_PEAK_IntegerNode_GetRepresentation m_PEAK_IntegerNode_GetRepresentation{};
    dyn_PEAK_IntegerNode_GetUnit m_PEAK_IntegerNode_GetUnit{};
    dyn_PEAK_IntegerNode_GetValue m_PEAK_IntegerNode_GetValue{};
    dyn_PEAK_IntegerNode_SetValue m_PEAK_IntegerNode_SetValue{};
    dyn_PEAK_BooleanNode_ToNode m_PEAK_BooleanNode_ToNode{};
    dyn_PEAK_BooleanNode_GetValue m_PEAK_BooleanNode_GetValue{};
    dyn_PEAK_BooleanNode_SetValue m_PEAK_BooleanNode_SetValue{};
    dyn_PEAK_CommandNode_ToNode m_PEAK_CommandNode_ToNode{};
    dyn_PEAK_CommandNode_GetIsDone m_PEAK_CommandNode_GetIsDone{};
    dyn_PEAK_CommandNode_Execute m_PEAK_CommandNode_Execute{};
    dyn_PEAK_CommandNode_WaitUntilDoneInfinite m_PEAK_CommandNode_WaitUntilDoneInfinite{};
    dyn_PEAK_CommandNode_WaitUntilDone m_PEAK_CommandNode_WaitUntilDone{};
    dyn_PEAK_FloatNode_ToNode m_PEAK_FloatNode_ToNode{};
    dyn_PEAK_FloatNode_GetMinimum m_PEAK_FloatNode_GetMinimum{};
    dyn_PEAK_FloatNode_GetMaximum m_PEAK_FloatNode_GetMaximum{};
    dyn_PEAK_FloatNode_GetIncrement m_PEAK_FloatNode_GetIncrement{};
    dyn_PEAK_FloatNode_GetIncrementType m_PEAK_FloatNode_GetIncrementType{};
    dyn_PEAK_FloatNode_GetValidValues m_PEAK_FloatNode_GetValidValues{};
    dyn_PEAK_FloatNode_GetRepresentation m_PEAK_FloatNode_GetRepresentation{};
    dyn_PEAK_FloatNode_GetUnit m_PEAK_FloatNode_GetUnit{};
    dyn_PEAK_FloatNode_GetDisplayNotation m_PEAK_FloatNode_GetDisplayNotation{};
    dyn_PEAK_FloatNode_GetDisplayPrecision m_PEAK_FloatNode_GetDisplayPrecision{};
    dyn_PEAK_FloatNode_GetHasConstantIncrement m_PEAK_FloatNode_GetHasConstantIncrement{};
    dyn_PEAK_FloatNode_GetValue m_PEAK_FloatNode_GetValue{};
    dyn_PEAK_FloatNode_SetValue m_PEAK_FloatNode_SetValue{};
    dyn_PEAK_StringNode_ToNode m_PEAK_StringNode_ToNode{};
    dyn_PEAK_StringNode_GetMaximumLength m_PEAK_StringNode_GetMaximumLength{};
    dyn_PEAK_StringNode_GetValue m_PEAK_StringNode_GetValue{};
    dyn_PEAK_StringNode_SetValue m_PEAK_StringNode_SetValue{};
    dyn_PEAK_RegisterNode_ToNode m_PEAK_RegisterNode_ToNode{};
    dyn_PEAK_RegisterNode_GetAddress m_PEAK_RegisterNode_GetAddress{};
    dyn_PEAK_RegisterNode_GetLength m_PEAK_RegisterNode_GetLength{};
    dyn_PEAK_RegisterNode_Read m_PEAK_RegisterNode_Read{};
    dyn_PEAK_RegisterNode_Write m_PEAK_RegisterNode_Write{};
    dyn_PEAK_CategoryNode_ToNode m_PEAK_CategoryNode_ToNode{};
    dyn_PEAK_CategoryNode_GetNumSubNodes m_PEAK_CategoryNode_GetNumSubNodes{};
    dyn_PEAK_CategoryNode_GetSubNode m_PEAK_CategoryNode_GetSubNode{};
    dyn_PEAK_EnumerationNode_ToNode m_PEAK_EnumerationNode_ToNode{};
    dyn_PEAK_EnumerationNode_GetCurrentEntry m_PEAK_EnumerationNode_GetCurrentEntry{};
    dyn_PEAK_EnumerationNode_SetCurrentEntry m_PEAK_EnumerationNode_SetCurrentEntry{};
    dyn_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue m_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue{};
    dyn_PEAK_EnumerationNode_SetCurrentEntryByValue m_PEAK_EnumerationNode_SetCurrentEntryByValue{};
    dyn_PEAK_EnumerationNode_FindEntryBySymbolicValue m_PEAK_EnumerationNode_FindEntryBySymbolicValue{};
    dyn_PEAK_EnumerationNode_FindEntryByValue m_PEAK_EnumerationNode_FindEntryByValue{};
    dyn_PEAK_EnumerationNode_GetNumEntries m_PEAK_EnumerationNode_GetNumEntries{};
    dyn_PEAK_EnumerationNode_GetEntry m_PEAK_EnumerationNode_GetEntry{};
    dyn_PEAK_EnumerationEntryNode_ToNode m_PEAK_EnumerationEntryNode_ToNode{};
    dyn_PEAK_EnumerationEntryNode_GetIsSelfClearing m_PEAK_EnumerationEntryNode_GetIsSelfClearing{};
    dyn_PEAK_EnumerationEntryNode_GetSymbolicValue m_PEAK_EnumerationEntryNode_GetSymbolicValue{};
    dyn_PEAK_EnumerationEntryNode_GetValue m_PEAK_EnumerationEntryNode_GetValue{};
    dyn_PEAK_Port_GetInfo m_PEAK_Port_GetInfo{};
    dyn_PEAK_Port_GetID m_PEAK_Port_GetID{};
    dyn_PEAK_Port_GetName m_PEAK_Port_GetName{};
    dyn_PEAK_Port_GetVendorName m_PEAK_Port_GetVendorName{};
    dyn_PEAK_Port_GetModelName m_PEAK_Port_GetModelName{};
    dyn_PEAK_Port_GetVersion m_PEAK_Port_GetVersion{};
    dyn_PEAK_Port_GetTLType m_PEAK_Port_GetTLType{};
    dyn_PEAK_Port_GetModuleName m_PEAK_Port_GetModuleName{};
    dyn_PEAK_Port_GetDataEndianness m_PEAK_Port_GetDataEndianness{};
    dyn_PEAK_Port_GetIsReadable m_PEAK_Port_GetIsReadable{};
    dyn_PEAK_Port_GetIsWritable m_PEAK_Port_GetIsWritable{};
    dyn_PEAK_Port_GetIsAvailable m_PEAK_Port_GetIsAvailable{};
    dyn_PEAK_Port_GetIsImplemented m_PEAK_Port_GetIsImplemented{};
    dyn_PEAK_Port_Read m_PEAK_Port_Read{};
    dyn_PEAK_Port_Write m_PEAK_Port_Write{};
    dyn_PEAK_Port_GetNumURLs m_PEAK_Port_GetNumURLs{};
    dyn_PEAK_Port_GetURL m_PEAK_Port_GetURL{};
    dyn_PEAK_PortURL_GetInfo m_PEAK_PortURL_GetInfo{};
    dyn_PEAK_PortURL_GetURL m_PEAK_PortURL_GetURL{};
    dyn_PEAK_PortURL_GetScheme m_PEAK_PortURL_GetScheme{};
    dyn_PEAK_PortURL_GetFileName m_PEAK_PortURL_GetFileName{};
    dyn_PEAK_PortURL_GetFileRegisterAddress m_PEAK_PortURL_GetFileRegisterAddress{};
    dyn_PEAK_PortURL_GetFileSize m_PEAK_PortURL_GetFileSize{};
    dyn_PEAK_PortURL_GetFileSHA1Hash m_PEAK_PortURL_GetFileSHA1Hash{};
    dyn_PEAK_PortURL_GetFileVersionMajor m_PEAK_PortURL_GetFileVersionMajor{};
    dyn_PEAK_PortURL_GetFileVersionMinor m_PEAK_PortURL_GetFileVersionMinor{};
    dyn_PEAK_PortURL_GetFileVersionSubminor m_PEAK_PortURL_GetFileVersionSubminor{};
    dyn_PEAK_PortURL_GetFileSchemaVersionMajor m_PEAK_PortURL_GetFileSchemaVersionMajor{};
    dyn_PEAK_PortURL_GetFileSchemaVersionMinor m_PEAK_PortURL_GetFileSchemaVersionMinor{};
    dyn_PEAK_PortURL_GetParentPort m_PEAK_PortURL_GetParentPort{};
    dyn_PEAK_EventSupportingModule_EnableEvents m_PEAK_EventSupportingModule_EnableEvents{};
    dyn_PEAK_EventController_GetInfo m_PEAK_EventController_GetInfo{};
    dyn_PEAK_EventController_GetNumEventsInQueue m_PEAK_EventController_GetNumEventsInQueue{};
    dyn_PEAK_EventController_GetNumEventsFired m_PEAK_EventController_GetNumEventsFired{};
    dyn_PEAK_EventController_GetEventMaxSize m_PEAK_EventController_GetEventMaxSize{};
    dyn_PEAK_EventController_GetEventDataMaxSize m_PEAK_EventController_GetEventDataMaxSize{};
    dyn_PEAK_EventController_GetControlledEventType m_PEAK_EventController_GetControlledEventType{};
    dyn_PEAK_EventController_WaitForEvent m_PEAK_EventController_WaitForEvent{};
    dyn_PEAK_EventController_KillWait m_PEAK_EventController_KillWait{};
    dyn_PEAK_EventController_FlushEvents m_PEAK_EventController_FlushEvents{};
    dyn_PEAK_EventController_Destruct m_PEAK_EventController_Destruct{};
    dyn_PEAK_Event_GetInfo m_PEAK_Event_GetInfo{};
    dyn_PEAK_Event_GetID m_PEAK_Event_GetID{};
    dyn_PEAK_Event_GetData m_PEAK_Event_GetData{};
    dyn_PEAK_Event_GetType m_PEAK_Event_GetType{};
    dyn_PEAK_Event_GetRawData m_PEAK_Event_GetRawData{};
    dyn_PEAK_Event_Destruct m_PEAK_Event_Destruct{};
    dyn_PEAK_FirmwareUpdater_Construct m_PEAK_FirmwareUpdater_Construct{};
    dyn_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation m_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation{};
    dyn_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation m_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation{};
    dyn_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation m_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation{};
    dyn_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation m_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation{};
    dyn_PEAK_FirmwareUpdater_UpdateDevice m_PEAK_FirmwareUpdater_UpdateDevice{};
    dyn_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout m_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout{};
    dyn_PEAK_FirmwareUpdater_Destruct m_PEAK_FirmwareUpdater_Destruct{};
    dyn_PEAK_FirmwareUpdateInformation_GetIsValid m_PEAK_FirmwareUpdateInformation_GetIsValid{};
    dyn_PEAK_FirmwareUpdateInformation_GetFileName m_PEAK_FirmwareUpdateInformation_GetFileName{};
    dyn_PEAK_FirmwareUpdateInformation_GetDescription m_PEAK_FirmwareUpdateInformation_GetDescription{};
    dyn_PEAK_FirmwareUpdateInformation_GetVersion m_PEAK_FirmwareUpdateInformation_GetVersion{};
    dyn_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern m_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern{};
    dyn_PEAK_FirmwareUpdateInformation_GetVersionStyle m_PEAK_FirmwareUpdateInformation_GetVersionStyle{};
    dyn_PEAK_FirmwareUpdateInformation_GetReleaseNotes m_PEAK_FirmwareUpdateInformation_GetReleaseNotes{};
    dyn_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL m_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL{};
    dyn_PEAK_FirmwareUpdateInformation_GetUserSetPersistence m_PEAK_FirmwareUpdateInformation_GetUserSetPersistence{};
    dyn_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence m_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence{};
    dyn_PEAK_FirmwareUpdateProgressObserver_Construct m_PEAK_FirmwareUpdateProgressObserver_Construct{};
    dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback{};
    dyn_PEAK_FirmwareUpdateProgressObserver_Destruct m_PEAK_FirmwareUpdateProgressObserver_Destruct{};

};

inline void* import_function(void *module, const char* proc_name)
{
#ifdef __linux__
    return dlsym(module, proc_name);
#else
    return GetProcAddress(static_cast<HMODULE>(module), proc_name);
#endif
}
            
inline DynamicLoader::DynamicLoader()
{
#if defined _WIN32 || defined _WIN64
    size_t sz = 0;
    if (_wgetenv_s(&sz, NULL, 0, L"IDS_PEAK_SDK_PATH") == 0)
    {
        std::vector<wchar_t> env_ids_peak(sz);
        if (_wgetenv_s(&sz, env_ids_peak.data(), sz, L"IDS_PEAK_SDK_PATH") == 0)
        {
            if (_wgetenv_s(&sz, NULL, 0, L"PATH") == 0)
            {
                std::vector<wchar_t> env_path(sz);
                if (_wgetenv_s(&sz, env_path.data(), sz, L"PATH") == 0)
                {
                    std::wstring ids_peak_path(env_ids_peak.data());
#ifdef _WIN64
                    ids_peak_path.append(L"\\api\\lib\\x86_64");
#else
                    ids_peak_path.append(L"\\api\\lib\\x86_32");
#endif
                    std::wstring path_var(env_path.data());
                    path_var.append(L";").append(ids_peak_path);
                    _wputenv_s(L"PATH", path_var.c_str());
                }
            }
        }
    }
    
    loadLib("ids_peak.dll");
#else
    loadLib("libids_peak.so");
#endif
}

inline DynamicLoader::~DynamicLoader()
{
    if(m_handle != nullptr)
    {
        unload();
    }
}

inline bool DynamicLoader::isLoaded()
{
    auto&& inst = instance();
    return inst.m_handle != nullptr;
}

inline void DynamicLoader::unload()
{
    setPointers(false);
    
    if (m_handle != nullptr)
    {
#ifdef __linux__
        dlclose(m_handle);
#else
        FreeLibrary(static_cast<HMODULE>(m_handle));
#endif
    }
    m_handle = nullptr;
}


inline bool DynamicLoader::loadLib(const char* file)
{
    bool ret = false;
    
    if (file)
    {
#ifdef __linux__
        m_handle = dlopen(file, RTLD_NOW);
#else
        m_handle = LoadLibraryA(file);
#endif
        if (m_handle != nullptr)
        {
            try {
                setPointers(true);
                ret = true;
            } catch (const std::exception&) {
                unload();
                throw;
            }
        }
        else
        {
            throw std::runtime_error(std::string("Lib load failed: ") + file);
        }
    }
    else
    {
        throw std::runtime_error("Filename empty");
    }

    return ret;
}

inline bool DynamicLoader::setPointers(bool load)
{

    m_PEAK_Library_Initialize = (dyn_PEAK_Library_Initialize) (load ?  import_function(m_handle, "PEAK_Library_Initialize") : nullptr);
    if(m_PEAK_Library_Initialize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_Initialize");
    }        

    m_PEAK_Library_Close = (dyn_PEAK_Library_Close) (load ?  import_function(m_handle, "PEAK_Library_Close") : nullptr);
    if(m_PEAK_Library_Close == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_Close");
    }        

    m_PEAK_Library_IsInitialized = (dyn_PEAK_Library_IsInitialized) (load ?  import_function(m_handle, "PEAK_Library_IsInitialized") : nullptr);
    if(m_PEAK_Library_IsInitialized == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_IsInitialized");
    }        

    m_PEAK_Library_GetVersionMajor = (dyn_PEAK_Library_GetVersionMajor) (load ?  import_function(m_handle, "PEAK_Library_GetVersionMajor") : nullptr);
    if(m_PEAK_Library_GetVersionMajor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_GetVersionMajor");
    }        

    m_PEAK_Library_GetVersionMinor = (dyn_PEAK_Library_GetVersionMinor) (load ?  import_function(m_handle, "PEAK_Library_GetVersionMinor") : nullptr);
    if(m_PEAK_Library_GetVersionMinor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_GetVersionMinor");
    }        

    m_PEAK_Library_GetVersionSubminor = (dyn_PEAK_Library_GetVersionSubminor) (load ?  import_function(m_handle, "PEAK_Library_GetVersionSubminor") : nullptr);
    if(m_PEAK_Library_GetVersionSubminor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_GetVersionSubminor");
    }        

    m_PEAK_Library_GetLastError = (dyn_PEAK_Library_GetLastError) (load ?  import_function(m_handle, "PEAK_Library_GetLastError") : nullptr);
    if(m_PEAK_Library_GetLastError == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Library_GetLastError");
    }        

    m_PEAK_EnvironmentInspector_UpdateCTIPaths = (dyn_PEAK_EnvironmentInspector_UpdateCTIPaths) (load ?  import_function(m_handle, "PEAK_EnvironmentInspector_UpdateCTIPaths") : nullptr);
    if(m_PEAK_EnvironmentInspector_UpdateCTIPaths == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnvironmentInspector_UpdateCTIPaths");
    }        

    m_PEAK_EnvironmentInspector_GetNumCTIPaths = (dyn_PEAK_EnvironmentInspector_GetNumCTIPaths) (load ?  import_function(m_handle, "PEAK_EnvironmentInspector_GetNumCTIPaths") : nullptr);
    if(m_PEAK_EnvironmentInspector_GetNumCTIPaths == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnvironmentInspector_GetNumCTIPaths");
    }        

    m_PEAK_EnvironmentInspector_GetCTIPath = (dyn_PEAK_EnvironmentInspector_GetCTIPath) (load ?  import_function(m_handle, "PEAK_EnvironmentInspector_GetCTIPath") : nullptr);
    if(m_PEAK_EnvironmentInspector_GetCTIPath == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnvironmentInspector_GetCTIPath");
    }        

    m_PEAK_ProducerLibrary_Construct = (dyn_PEAK_ProducerLibrary_Construct) (load ?  import_function(m_handle, "PEAK_ProducerLibrary_Construct") : nullptr);
    if(m_PEAK_ProducerLibrary_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_ProducerLibrary_Construct");
    }        

    m_PEAK_ProducerLibrary_GetKey = (dyn_PEAK_ProducerLibrary_GetKey) (load ?  import_function(m_handle, "PEAK_ProducerLibrary_GetKey") : nullptr);
    if(m_PEAK_ProducerLibrary_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_ProducerLibrary_GetKey");
    }        

    m_PEAK_ProducerLibrary_GetSystem = (dyn_PEAK_ProducerLibrary_GetSystem) (load ?  import_function(m_handle, "PEAK_ProducerLibrary_GetSystem") : nullptr);
    if(m_PEAK_ProducerLibrary_GetSystem == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_ProducerLibrary_GetSystem");
    }        

    m_PEAK_ProducerLibrary_Destruct = (dyn_PEAK_ProducerLibrary_Destruct) (load ?  import_function(m_handle, "PEAK_ProducerLibrary_Destruct") : nullptr);
    if(m_PEAK_ProducerLibrary_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_ProducerLibrary_Destruct");
    }        

    m_PEAK_SystemDescriptor_ToModuleDescriptor = (dyn_PEAK_SystemDescriptor_ToModuleDescriptor) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_ToModuleDescriptor") : nullptr);
    if(m_PEAK_SystemDescriptor_ToModuleDescriptor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_ToModuleDescriptor");
    }        

    m_PEAK_SystemDescriptor_GetKey = (dyn_PEAK_SystemDescriptor_GetKey) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetKey") : nullptr);
    if(m_PEAK_SystemDescriptor_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetKey");
    }        

    m_PEAK_SystemDescriptor_GetInfo = (dyn_PEAK_SystemDescriptor_GetInfo) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetInfo") : nullptr);
    if(m_PEAK_SystemDescriptor_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetInfo");
    }        

    m_PEAK_SystemDescriptor_GetDisplayName = (dyn_PEAK_SystemDescriptor_GetDisplayName) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetDisplayName") : nullptr);
    if(m_PEAK_SystemDescriptor_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetDisplayName");
    }        

    m_PEAK_SystemDescriptor_GetVendorName = (dyn_PEAK_SystemDescriptor_GetVendorName) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetVendorName") : nullptr);
    if(m_PEAK_SystemDescriptor_GetVendorName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetVendorName");
    }        

    m_PEAK_SystemDescriptor_GetModelName = (dyn_PEAK_SystemDescriptor_GetModelName) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetModelName") : nullptr);
    if(m_PEAK_SystemDescriptor_GetModelName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetModelName");
    }        

    m_PEAK_SystemDescriptor_GetVersion = (dyn_PEAK_SystemDescriptor_GetVersion) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetVersion") : nullptr);
    if(m_PEAK_SystemDescriptor_GetVersion == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetVersion");
    }        

    m_PEAK_SystemDescriptor_GetTLType = (dyn_PEAK_SystemDescriptor_GetTLType) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetTLType") : nullptr);
    if(m_PEAK_SystemDescriptor_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetTLType");
    }        

    m_PEAK_SystemDescriptor_GetCTIFileName = (dyn_PEAK_SystemDescriptor_GetCTIFileName) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetCTIFileName") : nullptr);
    if(m_PEAK_SystemDescriptor_GetCTIFileName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetCTIFileName");
    }        

    m_PEAK_SystemDescriptor_GetCTIFullPath = (dyn_PEAK_SystemDescriptor_GetCTIFullPath) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetCTIFullPath") : nullptr);
    if(m_PEAK_SystemDescriptor_GetCTIFullPath == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetCTIFullPath");
    }        

    m_PEAK_SystemDescriptor_GetGenTLVersionMajor = (dyn_PEAK_SystemDescriptor_GetGenTLVersionMajor) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetGenTLVersionMajor") : nullptr);
    if(m_PEAK_SystemDescriptor_GetGenTLVersionMajor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetGenTLVersionMajor");
    }        

    m_PEAK_SystemDescriptor_GetGenTLVersionMinor = (dyn_PEAK_SystemDescriptor_GetGenTLVersionMinor) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetGenTLVersionMinor") : nullptr);
    if(m_PEAK_SystemDescriptor_GetGenTLVersionMinor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetGenTLVersionMinor");
    }        

    m_PEAK_SystemDescriptor_GetCharacterEncoding = (dyn_PEAK_SystemDescriptor_GetCharacterEncoding) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetCharacterEncoding") : nullptr);
    if(m_PEAK_SystemDescriptor_GetCharacterEncoding == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetCharacterEncoding");
    }        

    m_PEAK_SystemDescriptor_GetParentLibrary = (dyn_PEAK_SystemDescriptor_GetParentLibrary) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_GetParentLibrary") : nullptr);
    if(m_PEAK_SystemDescriptor_GetParentLibrary == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_GetParentLibrary");
    }        

    m_PEAK_SystemDescriptor_OpenSystem = (dyn_PEAK_SystemDescriptor_OpenSystem) (load ?  import_function(m_handle, "PEAK_SystemDescriptor_OpenSystem") : nullptr);
    if(m_PEAK_SystemDescriptor_OpenSystem == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_SystemDescriptor_OpenSystem");
    }        

    m_PEAK_System_ToModule = (dyn_PEAK_System_ToModule) (load ?  import_function(m_handle, "PEAK_System_ToModule") : nullptr);
    if(m_PEAK_System_ToModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_ToModule");
    }        

    m_PEAK_System_ToEventSupportingModule = (dyn_PEAK_System_ToEventSupportingModule) (load ?  import_function(m_handle, "PEAK_System_ToEventSupportingModule") : nullptr);
    if(m_PEAK_System_ToEventSupportingModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_ToEventSupportingModule");
    }        

    m_PEAK_System_GetKey = (dyn_PEAK_System_GetKey) (load ?  import_function(m_handle, "PEAK_System_GetKey") : nullptr);
    if(m_PEAK_System_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetKey");
    }        

    m_PEAK_System_GetInfo = (dyn_PEAK_System_GetInfo) (load ?  import_function(m_handle, "PEAK_System_GetInfo") : nullptr);
    if(m_PEAK_System_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetInfo");
    }        

    m_PEAK_System_GetID = (dyn_PEAK_System_GetID) (load ?  import_function(m_handle, "PEAK_System_GetID") : nullptr);
    if(m_PEAK_System_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetID");
    }        

    m_PEAK_System_GetDisplayName = (dyn_PEAK_System_GetDisplayName) (load ?  import_function(m_handle, "PEAK_System_GetDisplayName") : nullptr);
    if(m_PEAK_System_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetDisplayName");
    }        

    m_PEAK_System_GetVendorName = (dyn_PEAK_System_GetVendorName) (load ?  import_function(m_handle, "PEAK_System_GetVendorName") : nullptr);
    if(m_PEAK_System_GetVendorName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetVendorName");
    }        

    m_PEAK_System_GetModelName = (dyn_PEAK_System_GetModelName) (load ?  import_function(m_handle, "PEAK_System_GetModelName") : nullptr);
    if(m_PEAK_System_GetModelName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetModelName");
    }        

    m_PEAK_System_GetVersion = (dyn_PEAK_System_GetVersion) (load ?  import_function(m_handle, "PEAK_System_GetVersion") : nullptr);
    if(m_PEAK_System_GetVersion == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetVersion");
    }        

    m_PEAK_System_GetTLType = (dyn_PEAK_System_GetTLType) (load ?  import_function(m_handle, "PEAK_System_GetTLType") : nullptr);
    if(m_PEAK_System_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetTLType");
    }        

    m_PEAK_System_GetCTIFileName = (dyn_PEAK_System_GetCTIFileName) (load ?  import_function(m_handle, "PEAK_System_GetCTIFileName") : nullptr);
    if(m_PEAK_System_GetCTIFileName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetCTIFileName");
    }        

    m_PEAK_System_GetCTIFullPath = (dyn_PEAK_System_GetCTIFullPath) (load ?  import_function(m_handle, "PEAK_System_GetCTIFullPath") : nullptr);
    if(m_PEAK_System_GetCTIFullPath == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetCTIFullPath");
    }        

    m_PEAK_System_GetGenTLVersionMajor = (dyn_PEAK_System_GetGenTLVersionMajor) (load ?  import_function(m_handle, "PEAK_System_GetGenTLVersionMajor") : nullptr);
    if(m_PEAK_System_GetGenTLVersionMajor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetGenTLVersionMajor");
    }        

    m_PEAK_System_GetGenTLVersionMinor = (dyn_PEAK_System_GetGenTLVersionMinor) (load ?  import_function(m_handle, "PEAK_System_GetGenTLVersionMinor") : nullptr);
    if(m_PEAK_System_GetGenTLVersionMinor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetGenTLVersionMinor");
    }        

    m_PEAK_System_GetCharacterEncoding = (dyn_PEAK_System_GetCharacterEncoding) (load ?  import_function(m_handle, "PEAK_System_GetCharacterEncoding") : nullptr);
    if(m_PEAK_System_GetCharacterEncoding == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetCharacterEncoding");
    }        

    m_PEAK_System_GetParentLibrary = (dyn_PEAK_System_GetParentLibrary) (load ?  import_function(m_handle, "PEAK_System_GetParentLibrary") : nullptr);
    if(m_PEAK_System_GetParentLibrary == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetParentLibrary");
    }        

    m_PEAK_System_UpdateInterfaces = (dyn_PEAK_System_UpdateInterfaces) (load ?  import_function(m_handle, "PEAK_System_UpdateInterfaces") : nullptr);
    if(m_PEAK_System_UpdateInterfaces == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_UpdateInterfaces");
    }        

    m_PEAK_System_GetNumInterfaces = (dyn_PEAK_System_GetNumInterfaces) (load ?  import_function(m_handle, "PEAK_System_GetNumInterfaces") : nullptr);
    if(m_PEAK_System_GetNumInterfaces == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetNumInterfaces");
    }        

    m_PEAK_System_GetInterface = (dyn_PEAK_System_GetInterface) (load ?  import_function(m_handle, "PEAK_System_GetInterface") : nullptr);
    if(m_PEAK_System_GetInterface == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_GetInterface");
    }        

    m_PEAK_System_RegisterInterfaceFoundCallback = (dyn_PEAK_System_RegisterInterfaceFoundCallback) (load ?  import_function(m_handle, "PEAK_System_RegisterInterfaceFoundCallback") : nullptr);
    if(m_PEAK_System_RegisterInterfaceFoundCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_RegisterInterfaceFoundCallback");
    }        

    m_PEAK_System_UnregisterInterfaceFoundCallback = (dyn_PEAK_System_UnregisterInterfaceFoundCallback) (load ?  import_function(m_handle, "PEAK_System_UnregisterInterfaceFoundCallback") : nullptr);
    if(m_PEAK_System_UnregisterInterfaceFoundCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_UnregisterInterfaceFoundCallback");
    }        

    m_PEAK_System_RegisterInterfaceLostCallback = (dyn_PEAK_System_RegisterInterfaceLostCallback) (load ?  import_function(m_handle, "PEAK_System_RegisterInterfaceLostCallback") : nullptr);
    if(m_PEAK_System_RegisterInterfaceLostCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_RegisterInterfaceLostCallback");
    }        

    m_PEAK_System_UnregisterInterfaceLostCallback = (dyn_PEAK_System_UnregisterInterfaceLostCallback) (load ?  import_function(m_handle, "PEAK_System_UnregisterInterfaceLostCallback") : nullptr);
    if(m_PEAK_System_UnregisterInterfaceLostCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_UnregisterInterfaceLostCallback");
    }        

    m_PEAK_System_Destruct = (dyn_PEAK_System_Destruct) (load ?  import_function(m_handle, "PEAK_System_Destruct") : nullptr);
    if(m_PEAK_System_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_System_Destruct");
    }        

    m_PEAK_InterfaceDescriptor_ToModuleDescriptor = (dyn_PEAK_InterfaceDescriptor_ToModuleDescriptor) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_ToModuleDescriptor") : nullptr);
    if(m_PEAK_InterfaceDescriptor_ToModuleDescriptor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_ToModuleDescriptor");
    }        

    m_PEAK_InterfaceDescriptor_GetKey = (dyn_PEAK_InterfaceDescriptor_GetKey) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_GetKey") : nullptr);
    if(m_PEAK_InterfaceDescriptor_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_GetKey");
    }        

    m_PEAK_InterfaceDescriptor_GetInfo = (dyn_PEAK_InterfaceDescriptor_GetInfo) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_GetInfo") : nullptr);
    if(m_PEAK_InterfaceDescriptor_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_GetInfo");
    }        

    m_PEAK_InterfaceDescriptor_GetDisplayName = (dyn_PEAK_InterfaceDescriptor_GetDisplayName) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_GetDisplayName") : nullptr);
    if(m_PEAK_InterfaceDescriptor_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_GetDisplayName");
    }        

    m_PEAK_InterfaceDescriptor_GetTLType = (dyn_PEAK_InterfaceDescriptor_GetTLType) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_GetTLType") : nullptr);
    if(m_PEAK_InterfaceDescriptor_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_GetTLType");
    }        

    m_PEAK_InterfaceDescriptor_GetParentSystem = (dyn_PEAK_InterfaceDescriptor_GetParentSystem) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_GetParentSystem") : nullptr);
    if(m_PEAK_InterfaceDescriptor_GetParentSystem == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_GetParentSystem");
    }        

    m_PEAK_InterfaceDescriptor_OpenInterface = (dyn_PEAK_InterfaceDescriptor_OpenInterface) (load ?  import_function(m_handle, "PEAK_InterfaceDescriptor_OpenInterface") : nullptr);
    if(m_PEAK_InterfaceDescriptor_OpenInterface == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_InterfaceDescriptor_OpenInterface");
    }        

    m_PEAK_Interface_ToModule = (dyn_PEAK_Interface_ToModule) (load ?  import_function(m_handle, "PEAK_Interface_ToModule") : nullptr);
    if(m_PEAK_Interface_ToModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_ToModule");
    }        

    m_PEAK_Interface_ToEventSupportingModule = (dyn_PEAK_Interface_ToEventSupportingModule) (load ?  import_function(m_handle, "PEAK_Interface_ToEventSupportingModule") : nullptr);
    if(m_PEAK_Interface_ToEventSupportingModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_ToEventSupportingModule");
    }        

    m_PEAK_Interface_GetKey = (dyn_PEAK_Interface_GetKey) (load ?  import_function(m_handle, "PEAK_Interface_GetKey") : nullptr);
    if(m_PEAK_Interface_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetKey");
    }        

    m_PEAK_Interface_GetInfo = (dyn_PEAK_Interface_GetInfo) (load ?  import_function(m_handle, "PEAK_Interface_GetInfo") : nullptr);
    if(m_PEAK_Interface_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetInfo");
    }        

    m_PEAK_Interface_GetID = (dyn_PEAK_Interface_GetID) (load ?  import_function(m_handle, "PEAK_Interface_GetID") : nullptr);
    if(m_PEAK_Interface_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetID");
    }        

    m_PEAK_Interface_GetDisplayName = (dyn_PEAK_Interface_GetDisplayName) (load ?  import_function(m_handle, "PEAK_Interface_GetDisplayName") : nullptr);
    if(m_PEAK_Interface_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetDisplayName");
    }        

    m_PEAK_Interface_GetTLType = (dyn_PEAK_Interface_GetTLType) (load ?  import_function(m_handle, "PEAK_Interface_GetTLType") : nullptr);
    if(m_PEAK_Interface_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetTLType");
    }        

    m_PEAK_Interface_GetParentSystem = (dyn_PEAK_Interface_GetParentSystem) (load ?  import_function(m_handle, "PEAK_Interface_GetParentSystem") : nullptr);
    if(m_PEAK_Interface_GetParentSystem == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetParentSystem");
    }        

    m_PEAK_Interface_UpdateDevices = (dyn_PEAK_Interface_UpdateDevices) (load ?  import_function(m_handle, "PEAK_Interface_UpdateDevices") : nullptr);
    if(m_PEAK_Interface_UpdateDevices == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_UpdateDevices");
    }        

    m_PEAK_Interface_GetNumDevices = (dyn_PEAK_Interface_GetNumDevices) (load ?  import_function(m_handle, "PEAK_Interface_GetNumDevices") : nullptr);
    if(m_PEAK_Interface_GetNumDevices == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetNumDevices");
    }        

    m_PEAK_Interface_GetDevice = (dyn_PEAK_Interface_GetDevice) (load ?  import_function(m_handle, "PEAK_Interface_GetDevice") : nullptr);
    if(m_PEAK_Interface_GetDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_GetDevice");
    }        

    m_PEAK_Interface_RegisterDeviceFoundCallback = (dyn_PEAK_Interface_RegisterDeviceFoundCallback) (load ?  import_function(m_handle, "PEAK_Interface_RegisterDeviceFoundCallback") : nullptr);
    if(m_PEAK_Interface_RegisterDeviceFoundCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_RegisterDeviceFoundCallback");
    }        

    m_PEAK_Interface_UnregisterDeviceFoundCallback = (dyn_PEAK_Interface_UnregisterDeviceFoundCallback) (load ?  import_function(m_handle, "PEAK_Interface_UnregisterDeviceFoundCallback") : nullptr);
    if(m_PEAK_Interface_UnregisterDeviceFoundCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_UnregisterDeviceFoundCallback");
    }        

    m_PEAK_Interface_RegisterDeviceLostCallback = (dyn_PEAK_Interface_RegisterDeviceLostCallback) (load ?  import_function(m_handle, "PEAK_Interface_RegisterDeviceLostCallback") : nullptr);
    if(m_PEAK_Interface_RegisterDeviceLostCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_RegisterDeviceLostCallback");
    }        

    m_PEAK_Interface_UnregisterDeviceLostCallback = (dyn_PEAK_Interface_UnregisterDeviceLostCallback) (load ?  import_function(m_handle, "PEAK_Interface_UnregisterDeviceLostCallback") : nullptr);
    if(m_PEAK_Interface_UnregisterDeviceLostCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_UnregisterDeviceLostCallback");
    }        

    m_PEAK_Interface_Destruct = (dyn_PEAK_Interface_Destruct) (load ?  import_function(m_handle, "PEAK_Interface_Destruct") : nullptr);
    if(m_PEAK_Interface_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Interface_Destruct");
    }        

    m_PEAK_DeviceDescriptor_ToModuleDescriptor = (dyn_PEAK_DeviceDescriptor_ToModuleDescriptor) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_ToModuleDescriptor") : nullptr);
    if(m_PEAK_DeviceDescriptor_ToModuleDescriptor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_ToModuleDescriptor");
    }        

    m_PEAK_DeviceDescriptor_GetKey = (dyn_PEAK_DeviceDescriptor_GetKey) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetKey") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetKey");
    }        

    m_PEAK_DeviceDescriptor_GetInfo = (dyn_PEAK_DeviceDescriptor_GetInfo) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetInfo") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetInfo");
    }        

    m_PEAK_DeviceDescriptor_GetDisplayName = (dyn_PEAK_DeviceDescriptor_GetDisplayName) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetDisplayName") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetDisplayName");
    }        

    m_PEAK_DeviceDescriptor_GetVendorName = (dyn_PEAK_DeviceDescriptor_GetVendorName) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetVendorName") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetVendorName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetVendorName");
    }        

    m_PEAK_DeviceDescriptor_GetModelName = (dyn_PEAK_DeviceDescriptor_GetModelName) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetModelName") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetModelName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetModelName");
    }        

    m_PEAK_DeviceDescriptor_GetVersion = (dyn_PEAK_DeviceDescriptor_GetVersion) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetVersion") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetVersion == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetVersion");
    }        

    m_PEAK_DeviceDescriptor_GetTLType = (dyn_PEAK_DeviceDescriptor_GetTLType) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetTLType") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetTLType");
    }        

    m_PEAK_DeviceDescriptor_GetUserDefinedName = (dyn_PEAK_DeviceDescriptor_GetUserDefinedName) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetUserDefinedName") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetUserDefinedName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetUserDefinedName");
    }        

    m_PEAK_DeviceDescriptor_GetSerialNumber = (dyn_PEAK_DeviceDescriptor_GetSerialNumber) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetSerialNumber") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetSerialNumber == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetSerialNumber");
    }        

    m_PEAK_DeviceDescriptor_GetAccessStatus = (dyn_PEAK_DeviceDescriptor_GetAccessStatus) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetAccessStatus") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetAccessStatus == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetAccessStatus");
    }        

    m_PEAK_DeviceDescriptor_GetTimestampTickFrequency = (dyn_PEAK_DeviceDescriptor_GetTimestampTickFrequency) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetTimestampTickFrequency") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetTimestampTickFrequency == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetTimestampTickFrequency");
    }        

    m_PEAK_DeviceDescriptor_GetIsOpenableExclusive = (dyn_PEAK_DeviceDescriptor_GetIsOpenableExclusive) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetIsOpenableExclusive") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetIsOpenableExclusive == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetIsOpenableExclusive");
    }        

    m_PEAK_DeviceDescriptor_GetIsOpenable = (dyn_PEAK_DeviceDescriptor_GetIsOpenable) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetIsOpenable") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetIsOpenable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetIsOpenable");
    }        

    m_PEAK_DeviceDescriptor_OpenDevice = (dyn_PEAK_DeviceDescriptor_OpenDevice) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_OpenDevice") : nullptr);
    if(m_PEAK_DeviceDescriptor_OpenDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_OpenDevice");
    }        

    m_PEAK_DeviceDescriptor_GetParentInterface = (dyn_PEAK_DeviceDescriptor_GetParentInterface) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetParentInterface") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetParentInterface == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetParentInterface");
    }        

    m_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval = (dyn_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_GetMonitoringUpdateInterval") : nullptr);
    if(m_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_GetMonitoringUpdateInterval");
    }        

    m_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval = (dyn_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_SetMonitoringUpdateInterval") : nullptr);
    if(m_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_SetMonitoringUpdateInterval");
    }        

    m_PEAK_DeviceDescriptor_IsInformationRoleMonitored = (dyn_PEAK_DeviceDescriptor_IsInformationRoleMonitored) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_IsInformationRoleMonitored") : nullptr);
    if(m_PEAK_DeviceDescriptor_IsInformationRoleMonitored == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_IsInformationRoleMonitored");
    }        

    m_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring = (dyn_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_AddInformationRoleToMonitoring") : nullptr);
    if(m_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_AddInformationRoleToMonitoring");
    }        

    m_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring = (dyn_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring") : nullptr);
    if(m_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring");
    }        

    m_PEAK_DeviceDescriptor_RegisterInformationChangedCallback = (dyn_PEAK_DeviceDescriptor_RegisterInformationChangedCallback) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_RegisterInformationChangedCallback") : nullptr);
    if(m_PEAK_DeviceDescriptor_RegisterInformationChangedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_RegisterInformationChangedCallback");
    }        

    m_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback = (dyn_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback) (load ?  import_function(m_handle, "PEAK_DeviceDescriptor_UnregisterInformationChangedCallback") : nullptr);
    if(m_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DeviceDescriptor_UnregisterInformationChangedCallback");
    }        

    m_PEAK_Device_ToModule = (dyn_PEAK_Device_ToModule) (load ?  import_function(m_handle, "PEAK_Device_ToModule") : nullptr);
    if(m_PEAK_Device_ToModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_ToModule");
    }        

    m_PEAK_Device_ToEventSupportingModule = (dyn_PEAK_Device_ToEventSupportingModule) (load ?  import_function(m_handle, "PEAK_Device_ToEventSupportingModule") : nullptr);
    if(m_PEAK_Device_ToEventSupportingModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_ToEventSupportingModule");
    }        

    m_PEAK_Device_GetKey = (dyn_PEAK_Device_GetKey) (load ?  import_function(m_handle, "PEAK_Device_GetKey") : nullptr);
    if(m_PEAK_Device_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetKey");
    }        

    m_PEAK_Device_GetInfo = (dyn_PEAK_Device_GetInfo) (load ?  import_function(m_handle, "PEAK_Device_GetInfo") : nullptr);
    if(m_PEAK_Device_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetInfo");
    }        

    m_PEAK_Device_GetID = (dyn_PEAK_Device_GetID) (load ?  import_function(m_handle, "PEAK_Device_GetID") : nullptr);
    if(m_PEAK_Device_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetID");
    }        

    m_PEAK_Device_GetDisplayName = (dyn_PEAK_Device_GetDisplayName) (load ?  import_function(m_handle, "PEAK_Device_GetDisplayName") : nullptr);
    if(m_PEAK_Device_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetDisplayName");
    }        

    m_PEAK_Device_GetVendorName = (dyn_PEAK_Device_GetVendorName) (load ?  import_function(m_handle, "PEAK_Device_GetVendorName") : nullptr);
    if(m_PEAK_Device_GetVendorName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetVendorName");
    }        

    m_PEAK_Device_GetModelName = (dyn_PEAK_Device_GetModelName) (load ?  import_function(m_handle, "PEAK_Device_GetModelName") : nullptr);
    if(m_PEAK_Device_GetModelName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetModelName");
    }        

    m_PEAK_Device_GetVersion = (dyn_PEAK_Device_GetVersion) (load ?  import_function(m_handle, "PEAK_Device_GetVersion") : nullptr);
    if(m_PEAK_Device_GetVersion == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetVersion");
    }        

    m_PEAK_Device_GetTLType = (dyn_PEAK_Device_GetTLType) (load ?  import_function(m_handle, "PEAK_Device_GetTLType") : nullptr);
    if(m_PEAK_Device_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetTLType");
    }        

    m_PEAK_Device_GetUserDefinedName = (dyn_PEAK_Device_GetUserDefinedName) (load ?  import_function(m_handle, "PEAK_Device_GetUserDefinedName") : nullptr);
    if(m_PEAK_Device_GetUserDefinedName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetUserDefinedName");
    }        

    m_PEAK_Device_GetSerialNumber = (dyn_PEAK_Device_GetSerialNumber) (load ?  import_function(m_handle, "PEAK_Device_GetSerialNumber") : nullptr);
    if(m_PEAK_Device_GetSerialNumber == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetSerialNumber");
    }        

    m_PEAK_Device_GetAccessStatus = (dyn_PEAK_Device_GetAccessStatus) (load ?  import_function(m_handle, "PEAK_Device_GetAccessStatus") : nullptr);
    if(m_PEAK_Device_GetAccessStatus == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetAccessStatus");
    }        

    m_PEAK_Device_GetTimestampTickFrequency = (dyn_PEAK_Device_GetTimestampTickFrequency) (load ?  import_function(m_handle, "PEAK_Device_GetTimestampTickFrequency") : nullptr);
    if(m_PEAK_Device_GetTimestampTickFrequency == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetTimestampTickFrequency");
    }        

    m_PEAK_Device_GetParentInterface = (dyn_PEAK_Device_GetParentInterface) (load ?  import_function(m_handle, "PEAK_Device_GetParentInterface") : nullptr);
    if(m_PEAK_Device_GetParentInterface == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetParentInterface");
    }        

    m_PEAK_Device_GetRemoteDevice = (dyn_PEAK_Device_GetRemoteDevice) (load ?  import_function(m_handle, "PEAK_Device_GetRemoteDevice") : nullptr);
    if(m_PEAK_Device_GetRemoteDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetRemoteDevice");
    }        

    m_PEAK_Device_GetNumDataStreams = (dyn_PEAK_Device_GetNumDataStreams) (load ?  import_function(m_handle, "PEAK_Device_GetNumDataStreams") : nullptr);
    if(m_PEAK_Device_GetNumDataStreams == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetNumDataStreams");
    }        

    m_PEAK_Device_GetDataStream = (dyn_PEAK_Device_GetDataStream) (load ?  import_function(m_handle, "PEAK_Device_GetDataStream") : nullptr);
    if(m_PEAK_Device_GetDataStream == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_GetDataStream");
    }        

    m_PEAK_Device_Destruct = (dyn_PEAK_Device_Destruct) (load ?  import_function(m_handle, "PEAK_Device_Destruct") : nullptr);
    if(m_PEAK_Device_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Device_Destruct");
    }        

    m_PEAK_RemoteDevice_ToModule = (dyn_PEAK_RemoteDevice_ToModule) (load ?  import_function(m_handle, "PEAK_RemoteDevice_ToModule") : nullptr);
    if(m_PEAK_RemoteDevice_ToModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RemoteDevice_ToModule");
    }        

    m_PEAK_RemoteDevice_GetLocalDevice = (dyn_PEAK_RemoteDevice_GetLocalDevice) (load ?  import_function(m_handle, "PEAK_RemoteDevice_GetLocalDevice") : nullptr);
    if(m_PEAK_RemoteDevice_GetLocalDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RemoteDevice_GetLocalDevice");
    }        

    m_PEAK_DataStreamDescriptor_ToModuleDescriptor = (dyn_PEAK_DataStreamDescriptor_ToModuleDescriptor) (load ?  import_function(m_handle, "PEAK_DataStreamDescriptor_ToModuleDescriptor") : nullptr);
    if(m_PEAK_DataStreamDescriptor_ToModuleDescriptor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStreamDescriptor_ToModuleDescriptor");
    }        

    m_PEAK_DataStreamDescriptor_GetKey = (dyn_PEAK_DataStreamDescriptor_GetKey) (load ?  import_function(m_handle, "PEAK_DataStreamDescriptor_GetKey") : nullptr);
    if(m_PEAK_DataStreamDescriptor_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStreamDescriptor_GetKey");
    }        

    m_PEAK_DataStreamDescriptor_GetParentDevice = (dyn_PEAK_DataStreamDescriptor_GetParentDevice) (load ?  import_function(m_handle, "PEAK_DataStreamDescriptor_GetParentDevice") : nullptr);
    if(m_PEAK_DataStreamDescriptor_GetParentDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStreamDescriptor_GetParentDevice");
    }        

    m_PEAK_DataStreamDescriptor_OpenDataStream = (dyn_PEAK_DataStreamDescriptor_OpenDataStream) (load ?  import_function(m_handle, "PEAK_DataStreamDescriptor_OpenDataStream") : nullptr);
    if(m_PEAK_DataStreamDescriptor_OpenDataStream == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStreamDescriptor_OpenDataStream");
    }        

    m_PEAK_DataStream_ToModule = (dyn_PEAK_DataStream_ToModule) (load ?  import_function(m_handle, "PEAK_DataStream_ToModule") : nullptr);
    if(m_PEAK_DataStream_ToModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_ToModule");
    }        

    m_PEAK_DataStream_ToEventSupportingModule = (dyn_PEAK_DataStream_ToEventSupportingModule) (load ?  import_function(m_handle, "PEAK_DataStream_ToEventSupportingModule") : nullptr);
    if(m_PEAK_DataStream_ToEventSupportingModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_ToEventSupportingModule");
    }        

    m_PEAK_DataStream_GetKey = (dyn_PEAK_DataStream_GetKey) (load ?  import_function(m_handle, "PEAK_DataStream_GetKey") : nullptr);
    if(m_PEAK_DataStream_GetKey == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetKey");
    }        

    m_PEAK_DataStream_GetInfo = (dyn_PEAK_DataStream_GetInfo) (load ?  import_function(m_handle, "PEAK_DataStream_GetInfo") : nullptr);
    if(m_PEAK_DataStream_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetInfo");
    }        

    m_PEAK_DataStream_GetID = (dyn_PEAK_DataStream_GetID) (load ?  import_function(m_handle, "PEAK_DataStream_GetID") : nullptr);
    if(m_PEAK_DataStream_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetID");
    }        

    m_PEAK_DataStream_GetTLType = (dyn_PEAK_DataStream_GetTLType) (load ?  import_function(m_handle, "PEAK_DataStream_GetTLType") : nullptr);
    if(m_PEAK_DataStream_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetTLType");
    }        

    m_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired = (dyn_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumBuffersAnnouncedMinRequired") : nullptr);
    if(m_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumBuffersAnnouncedMinRequired");
    }        

    m_PEAK_DataStream_GetNumBuffersAnnounced = (dyn_PEAK_DataStream_GetNumBuffersAnnounced) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumBuffersAnnounced") : nullptr);
    if(m_PEAK_DataStream_GetNumBuffersAnnounced == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumBuffersAnnounced");
    }        

    m_PEAK_DataStream_GetNumBuffersQueued = (dyn_PEAK_DataStream_GetNumBuffersQueued) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumBuffersQueued") : nullptr);
    if(m_PEAK_DataStream_GetNumBuffersQueued == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumBuffersQueued");
    }        

    m_PEAK_DataStream_GetNumBuffersAwaitDelivery = (dyn_PEAK_DataStream_GetNumBuffersAwaitDelivery) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumBuffersAwaitDelivery") : nullptr);
    if(m_PEAK_DataStream_GetNumBuffersAwaitDelivery == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumBuffersAwaitDelivery");
    }        

    m_PEAK_DataStream_GetNumBuffersDelivered = (dyn_PEAK_DataStream_GetNumBuffersDelivered) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumBuffersDelivered") : nullptr);
    if(m_PEAK_DataStream_GetNumBuffersDelivered == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumBuffersDelivered");
    }        

    m_PEAK_DataStream_GetNumBuffersStarted = (dyn_PEAK_DataStream_GetNumBuffersStarted) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumBuffersStarted") : nullptr);
    if(m_PEAK_DataStream_GetNumBuffersStarted == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumBuffersStarted");
    }        

    m_PEAK_DataStream_GetNumUnderruns = (dyn_PEAK_DataStream_GetNumUnderruns) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumUnderruns") : nullptr);
    if(m_PEAK_DataStream_GetNumUnderruns == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumUnderruns");
    }        

    m_PEAK_DataStream_GetNumChunksPerBufferMax = (dyn_PEAK_DataStream_GetNumChunksPerBufferMax) (load ?  import_function(m_handle, "PEAK_DataStream_GetNumChunksPerBufferMax") : nullptr);
    if(m_PEAK_DataStream_GetNumChunksPerBufferMax == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetNumChunksPerBufferMax");
    }        

    m_PEAK_DataStream_GetBufferAlignment = (dyn_PEAK_DataStream_GetBufferAlignment) (load ?  import_function(m_handle, "PEAK_DataStream_GetBufferAlignment") : nullptr);
    if(m_PEAK_DataStream_GetBufferAlignment == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetBufferAlignment");
    }        

    m_PEAK_DataStream_GetPayloadSize = (dyn_PEAK_DataStream_GetPayloadSize) (load ?  import_function(m_handle, "PEAK_DataStream_GetPayloadSize") : nullptr);
    if(m_PEAK_DataStream_GetPayloadSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetPayloadSize");
    }        

    m_PEAK_DataStream_GetDefinesPayloadSize = (dyn_PEAK_DataStream_GetDefinesPayloadSize) (load ?  import_function(m_handle, "PEAK_DataStream_GetDefinesPayloadSize") : nullptr);
    if(m_PEAK_DataStream_GetDefinesPayloadSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetDefinesPayloadSize");
    }        

    m_PEAK_DataStream_GetIsGrabbing = (dyn_PEAK_DataStream_GetIsGrabbing) (load ?  import_function(m_handle, "PEAK_DataStream_GetIsGrabbing") : nullptr);
    if(m_PEAK_DataStream_GetIsGrabbing == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetIsGrabbing");
    }        

    m_PEAK_DataStream_GetParentDevice = (dyn_PEAK_DataStream_GetParentDevice) (load ?  import_function(m_handle, "PEAK_DataStream_GetParentDevice") : nullptr);
    if(m_PEAK_DataStream_GetParentDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_GetParentDevice");
    }        

    m_PEAK_DataStream_AnnounceBuffer = (dyn_PEAK_DataStream_AnnounceBuffer) (load ?  import_function(m_handle, "PEAK_DataStream_AnnounceBuffer") : nullptr);
    if(m_PEAK_DataStream_AnnounceBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_AnnounceBuffer");
    }        

    m_PEAK_DataStream_AllocAndAnnounceBuffer = (dyn_PEAK_DataStream_AllocAndAnnounceBuffer) (load ?  import_function(m_handle, "PEAK_DataStream_AllocAndAnnounceBuffer") : nullptr);
    if(m_PEAK_DataStream_AllocAndAnnounceBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_AllocAndAnnounceBuffer");
    }        

    m_PEAK_DataStream_QueueBuffer = (dyn_PEAK_DataStream_QueueBuffer) (load ?  import_function(m_handle, "PEAK_DataStream_QueueBuffer") : nullptr);
    if(m_PEAK_DataStream_QueueBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_QueueBuffer");
    }        

    m_PEAK_DataStream_RevokeBuffer = (dyn_PEAK_DataStream_RevokeBuffer) (load ?  import_function(m_handle, "PEAK_DataStream_RevokeBuffer") : nullptr);
    if(m_PEAK_DataStream_RevokeBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_RevokeBuffer");
    }        

    m_PEAK_DataStream_WaitForFinishedBuffer = (dyn_PEAK_DataStream_WaitForFinishedBuffer) (load ?  import_function(m_handle, "PEAK_DataStream_WaitForFinishedBuffer") : nullptr);
    if(m_PEAK_DataStream_WaitForFinishedBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_WaitForFinishedBuffer");
    }        

    m_PEAK_DataStream_KillWait = (dyn_PEAK_DataStream_KillWait) (load ?  import_function(m_handle, "PEAK_DataStream_KillWait") : nullptr);
    if(m_PEAK_DataStream_KillWait == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_KillWait");
    }        

    m_PEAK_DataStream_Flush = (dyn_PEAK_DataStream_Flush) (load ?  import_function(m_handle, "PEAK_DataStream_Flush") : nullptr);
    if(m_PEAK_DataStream_Flush == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_Flush");
    }        

    m_PEAK_DataStream_StartAcquisitionInfinite = (dyn_PEAK_DataStream_StartAcquisitionInfinite) (load ?  import_function(m_handle, "PEAK_DataStream_StartAcquisitionInfinite") : nullptr);
    if(m_PEAK_DataStream_StartAcquisitionInfinite == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_StartAcquisitionInfinite");
    }        

    m_PEAK_DataStream_StartAcquisition = (dyn_PEAK_DataStream_StartAcquisition) (load ?  import_function(m_handle, "PEAK_DataStream_StartAcquisition") : nullptr);
    if(m_PEAK_DataStream_StartAcquisition == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_StartAcquisition");
    }        

    m_PEAK_DataStream_StopAcquisition = (dyn_PEAK_DataStream_StopAcquisition) (load ?  import_function(m_handle, "PEAK_DataStream_StopAcquisition") : nullptr);
    if(m_PEAK_DataStream_StopAcquisition == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_StopAcquisition");
    }        

    m_PEAK_DataStream_Destruct = (dyn_PEAK_DataStream_Destruct) (load ?  import_function(m_handle, "PEAK_DataStream_Destruct") : nullptr);
    if(m_PEAK_DataStream_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_DataStream_Destruct");
    }        

    m_PEAK_Buffer_ToModule = (dyn_PEAK_Buffer_ToModule) (load ?  import_function(m_handle, "PEAK_Buffer_ToModule") : nullptr);
    if(m_PEAK_Buffer_ToModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_ToModule");
    }        

    m_PEAK_Buffer_ToEventSupportingModule = (dyn_PEAK_Buffer_ToEventSupportingModule) (load ?  import_function(m_handle, "PEAK_Buffer_ToEventSupportingModule") : nullptr);
    if(m_PEAK_Buffer_ToEventSupportingModule == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_ToEventSupportingModule");
    }        

    m_PEAK_Buffer_GetInfo = (dyn_PEAK_Buffer_GetInfo) (load ?  import_function(m_handle, "PEAK_Buffer_GetInfo") : nullptr);
    if(m_PEAK_Buffer_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetInfo");
    }        

    m_PEAK_Buffer_GetTLType = (dyn_PEAK_Buffer_GetTLType) (load ?  import_function(m_handle, "PEAK_Buffer_GetTLType") : nullptr);
    if(m_PEAK_Buffer_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetTLType");
    }        

    m_PEAK_Buffer_GetBasePtr = (dyn_PEAK_Buffer_GetBasePtr) (load ?  import_function(m_handle, "PEAK_Buffer_GetBasePtr") : nullptr);
    if(m_PEAK_Buffer_GetBasePtr == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetBasePtr");
    }        

    m_PEAK_Buffer_GetSize = (dyn_PEAK_Buffer_GetSize) (load ?  import_function(m_handle, "PEAK_Buffer_GetSize") : nullptr);
    if(m_PEAK_Buffer_GetSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetSize");
    }        

    m_PEAK_Buffer_GetUserPtr = (dyn_PEAK_Buffer_GetUserPtr) (load ?  import_function(m_handle, "PEAK_Buffer_GetUserPtr") : nullptr);
    if(m_PEAK_Buffer_GetUserPtr == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetUserPtr");
    }        

    m_PEAK_Buffer_GetPayloadType = (dyn_PEAK_Buffer_GetPayloadType) (load ?  import_function(m_handle, "PEAK_Buffer_GetPayloadType") : nullptr);
    if(m_PEAK_Buffer_GetPayloadType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetPayloadType");
    }        

    m_PEAK_Buffer_GetPixelFormat = (dyn_PEAK_Buffer_GetPixelFormat) (load ?  import_function(m_handle, "PEAK_Buffer_GetPixelFormat") : nullptr);
    if(m_PEAK_Buffer_GetPixelFormat == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetPixelFormat");
    }        

    m_PEAK_Buffer_GetPixelFormatNamespace = (dyn_PEAK_Buffer_GetPixelFormatNamespace) (load ?  import_function(m_handle, "PEAK_Buffer_GetPixelFormatNamespace") : nullptr);
    if(m_PEAK_Buffer_GetPixelFormatNamespace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetPixelFormatNamespace");
    }        

    m_PEAK_Buffer_GetPixelEndianness = (dyn_PEAK_Buffer_GetPixelEndianness) (load ?  import_function(m_handle, "PEAK_Buffer_GetPixelEndianness") : nullptr);
    if(m_PEAK_Buffer_GetPixelEndianness == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetPixelEndianness");
    }        

    m_PEAK_Buffer_GetExpectedDataSize = (dyn_PEAK_Buffer_GetExpectedDataSize) (load ?  import_function(m_handle, "PEAK_Buffer_GetExpectedDataSize") : nullptr);
    if(m_PEAK_Buffer_GetExpectedDataSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetExpectedDataSize");
    }        

    m_PEAK_Buffer_GetDeliveredDataSize = (dyn_PEAK_Buffer_GetDeliveredDataSize) (load ?  import_function(m_handle, "PEAK_Buffer_GetDeliveredDataSize") : nullptr);
    if(m_PEAK_Buffer_GetDeliveredDataSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetDeliveredDataSize");
    }        

    m_PEAK_Buffer_GetFrameID = (dyn_PEAK_Buffer_GetFrameID) (load ?  import_function(m_handle, "PEAK_Buffer_GetFrameID") : nullptr);
    if(m_PEAK_Buffer_GetFrameID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetFrameID");
    }        

    m_PEAK_Buffer_GetImageOffset = (dyn_PEAK_Buffer_GetImageOffset) (load ?  import_function(m_handle, "PEAK_Buffer_GetImageOffset") : nullptr);
    if(m_PEAK_Buffer_GetImageOffset == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetImageOffset");
    }        

    m_PEAK_Buffer_GetDeliveredImageHeight = (dyn_PEAK_Buffer_GetDeliveredImageHeight) (load ?  import_function(m_handle, "PEAK_Buffer_GetDeliveredImageHeight") : nullptr);
    if(m_PEAK_Buffer_GetDeliveredImageHeight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetDeliveredImageHeight");
    }        

    m_PEAK_Buffer_GetDeliveredChunkPayloadSize = (dyn_PEAK_Buffer_GetDeliveredChunkPayloadSize) (load ?  import_function(m_handle, "PEAK_Buffer_GetDeliveredChunkPayloadSize") : nullptr);
    if(m_PEAK_Buffer_GetDeliveredChunkPayloadSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetDeliveredChunkPayloadSize");
    }        

    m_PEAK_Buffer_GetChunkLayoutID = (dyn_PEAK_Buffer_GetChunkLayoutID) (load ?  import_function(m_handle, "PEAK_Buffer_GetChunkLayoutID") : nullptr);
    if(m_PEAK_Buffer_GetChunkLayoutID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetChunkLayoutID");
    }        

    m_PEAK_Buffer_GetFileName = (dyn_PEAK_Buffer_GetFileName) (load ?  import_function(m_handle, "PEAK_Buffer_GetFileName") : nullptr);
    if(m_PEAK_Buffer_GetFileName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetFileName");
    }        

    m_PEAK_Buffer_GetWidth = (dyn_PEAK_Buffer_GetWidth) (load ?  import_function(m_handle, "PEAK_Buffer_GetWidth") : nullptr);
    if(m_PEAK_Buffer_GetWidth == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetWidth");
    }        

    m_PEAK_Buffer_GetHeight = (dyn_PEAK_Buffer_GetHeight) (load ?  import_function(m_handle, "PEAK_Buffer_GetHeight") : nullptr);
    if(m_PEAK_Buffer_GetHeight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetHeight");
    }        

    m_PEAK_Buffer_GetXOffset = (dyn_PEAK_Buffer_GetXOffset) (load ?  import_function(m_handle, "PEAK_Buffer_GetXOffset") : nullptr);
    if(m_PEAK_Buffer_GetXOffset == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetXOffset");
    }        

    m_PEAK_Buffer_GetYOffset = (dyn_PEAK_Buffer_GetYOffset) (load ?  import_function(m_handle, "PEAK_Buffer_GetYOffset") : nullptr);
    if(m_PEAK_Buffer_GetYOffset == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetYOffset");
    }        

    m_PEAK_Buffer_GetXPadding = (dyn_PEAK_Buffer_GetXPadding) (load ?  import_function(m_handle, "PEAK_Buffer_GetXPadding") : nullptr);
    if(m_PEAK_Buffer_GetXPadding == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetXPadding");
    }        

    m_PEAK_Buffer_GetYPadding = (dyn_PEAK_Buffer_GetYPadding) (load ?  import_function(m_handle, "PEAK_Buffer_GetYPadding") : nullptr);
    if(m_PEAK_Buffer_GetYPadding == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetYPadding");
    }        

    m_PEAK_Buffer_GetTimestamp_ticks = (dyn_PEAK_Buffer_GetTimestamp_ticks) (load ?  import_function(m_handle, "PEAK_Buffer_GetTimestamp_ticks") : nullptr);
    if(m_PEAK_Buffer_GetTimestamp_ticks == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetTimestamp_ticks");
    }        

    m_PEAK_Buffer_GetTimestamp_ns = (dyn_PEAK_Buffer_GetTimestamp_ns) (load ?  import_function(m_handle, "PEAK_Buffer_GetTimestamp_ns") : nullptr);
    if(m_PEAK_Buffer_GetTimestamp_ns == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetTimestamp_ns");
    }        

    m_PEAK_Buffer_GetIsQueued = (dyn_PEAK_Buffer_GetIsQueued) (load ?  import_function(m_handle, "PEAK_Buffer_GetIsQueued") : nullptr);
    if(m_PEAK_Buffer_GetIsQueued == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetIsQueued");
    }        

    m_PEAK_Buffer_GetIsAcquiring = (dyn_PEAK_Buffer_GetIsAcquiring) (load ?  import_function(m_handle, "PEAK_Buffer_GetIsAcquiring") : nullptr);
    if(m_PEAK_Buffer_GetIsAcquiring == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetIsAcquiring");
    }        

    m_PEAK_Buffer_GetIsIncomplete = (dyn_PEAK_Buffer_GetIsIncomplete) (load ?  import_function(m_handle, "PEAK_Buffer_GetIsIncomplete") : nullptr);
    if(m_PEAK_Buffer_GetIsIncomplete == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetIsIncomplete");
    }        

    m_PEAK_Buffer_GetHasNewData = (dyn_PEAK_Buffer_GetHasNewData) (load ?  import_function(m_handle, "PEAK_Buffer_GetHasNewData") : nullptr);
    if(m_PEAK_Buffer_GetHasNewData == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetHasNewData");
    }        

    m_PEAK_Buffer_GetHasImage = (dyn_PEAK_Buffer_GetHasImage) (load ?  import_function(m_handle, "PEAK_Buffer_GetHasImage") : nullptr);
    if(m_PEAK_Buffer_GetHasImage == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetHasImage");
    }        

    m_PEAK_Buffer_GetHasChunks = (dyn_PEAK_Buffer_GetHasChunks) (load ?  import_function(m_handle, "PEAK_Buffer_GetHasChunks") : nullptr);
    if(m_PEAK_Buffer_GetHasChunks == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetHasChunks");
    }        

    m_PEAK_Buffer_UpdateChunks = (dyn_PEAK_Buffer_UpdateChunks) (load ?  import_function(m_handle, "PEAK_Buffer_UpdateChunks") : nullptr);
    if(m_PEAK_Buffer_UpdateChunks == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_UpdateChunks");
    }        

    m_PEAK_Buffer_GetNumChunks = (dyn_PEAK_Buffer_GetNumChunks) (load ?  import_function(m_handle, "PEAK_Buffer_GetNumChunks") : nullptr);
    if(m_PEAK_Buffer_GetNumChunks == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetNumChunks");
    }        

    m_PEAK_Buffer_GetChunk = (dyn_PEAK_Buffer_GetChunk) (load ?  import_function(m_handle, "PEAK_Buffer_GetChunk") : nullptr);
    if(m_PEAK_Buffer_GetChunk == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetChunk");
    }        

    m_PEAK_Buffer_UpdateParts = (dyn_PEAK_Buffer_UpdateParts) (load ?  import_function(m_handle, "PEAK_Buffer_UpdateParts") : nullptr);
    if(m_PEAK_Buffer_UpdateParts == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_UpdateParts");
    }        

    m_PEAK_Buffer_GetNumParts = (dyn_PEAK_Buffer_GetNumParts) (load ?  import_function(m_handle, "PEAK_Buffer_GetNumParts") : nullptr);
    if(m_PEAK_Buffer_GetNumParts == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetNumParts");
    }        

    m_PEAK_Buffer_GetPart = (dyn_PEAK_Buffer_GetPart) (load ?  import_function(m_handle, "PEAK_Buffer_GetPart") : nullptr);
    if(m_PEAK_Buffer_GetPart == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Buffer_GetPart");
    }        

    m_PEAK_BufferChunk_GetID = (dyn_PEAK_BufferChunk_GetID) (load ?  import_function(m_handle, "PEAK_BufferChunk_GetID") : nullptr);
    if(m_PEAK_BufferChunk_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferChunk_GetID");
    }        

    m_PEAK_BufferChunk_GetBasePtr = (dyn_PEAK_BufferChunk_GetBasePtr) (load ?  import_function(m_handle, "PEAK_BufferChunk_GetBasePtr") : nullptr);
    if(m_PEAK_BufferChunk_GetBasePtr == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferChunk_GetBasePtr");
    }        

    m_PEAK_BufferChunk_GetSize = (dyn_PEAK_BufferChunk_GetSize) (load ?  import_function(m_handle, "PEAK_BufferChunk_GetSize") : nullptr);
    if(m_PEAK_BufferChunk_GetSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferChunk_GetSize");
    }        

    m_PEAK_BufferChunk_GetParentBuffer = (dyn_PEAK_BufferChunk_GetParentBuffer) (load ?  import_function(m_handle, "PEAK_BufferChunk_GetParentBuffer") : nullptr);
    if(m_PEAK_BufferChunk_GetParentBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferChunk_GetParentBuffer");
    }        

    m_PEAK_BufferPart_GetInfo = (dyn_PEAK_BufferPart_GetInfo) (load ?  import_function(m_handle, "PEAK_BufferPart_GetInfo") : nullptr);
    if(m_PEAK_BufferPart_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetInfo");
    }        

    m_PEAK_BufferPart_GetSourceID = (dyn_PEAK_BufferPart_GetSourceID) (load ?  import_function(m_handle, "PEAK_BufferPart_GetSourceID") : nullptr);
    if(m_PEAK_BufferPart_GetSourceID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetSourceID");
    }        

    m_PEAK_BufferPart_GetBasePtr = (dyn_PEAK_BufferPart_GetBasePtr) (load ?  import_function(m_handle, "PEAK_BufferPart_GetBasePtr") : nullptr);
    if(m_PEAK_BufferPart_GetBasePtr == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetBasePtr");
    }        

    m_PEAK_BufferPart_GetSize = (dyn_PEAK_BufferPart_GetSize) (load ?  import_function(m_handle, "PEAK_BufferPart_GetSize") : nullptr);
    if(m_PEAK_BufferPart_GetSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetSize");
    }        

    m_PEAK_BufferPart_GetType = (dyn_PEAK_BufferPart_GetType) (load ?  import_function(m_handle, "PEAK_BufferPart_GetType") : nullptr);
    if(m_PEAK_BufferPart_GetType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetType");
    }        

    m_PEAK_BufferPart_GetFormat = (dyn_PEAK_BufferPart_GetFormat) (load ?  import_function(m_handle, "PEAK_BufferPart_GetFormat") : nullptr);
    if(m_PEAK_BufferPart_GetFormat == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetFormat");
    }        

    m_PEAK_BufferPart_GetFormatNamespace = (dyn_PEAK_BufferPart_GetFormatNamespace) (load ?  import_function(m_handle, "PEAK_BufferPart_GetFormatNamespace") : nullptr);
    if(m_PEAK_BufferPart_GetFormatNamespace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetFormatNamespace");
    }        

    m_PEAK_BufferPart_GetWidth = (dyn_PEAK_BufferPart_GetWidth) (load ?  import_function(m_handle, "PEAK_BufferPart_GetWidth") : nullptr);
    if(m_PEAK_BufferPart_GetWidth == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetWidth");
    }        

    m_PEAK_BufferPart_GetHeight = (dyn_PEAK_BufferPart_GetHeight) (load ?  import_function(m_handle, "PEAK_BufferPart_GetHeight") : nullptr);
    if(m_PEAK_BufferPart_GetHeight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetHeight");
    }        

    m_PEAK_BufferPart_GetXOffset = (dyn_PEAK_BufferPart_GetXOffset) (load ?  import_function(m_handle, "PEAK_BufferPart_GetXOffset") : nullptr);
    if(m_PEAK_BufferPart_GetXOffset == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetXOffset");
    }        

    m_PEAK_BufferPart_GetYOffset = (dyn_PEAK_BufferPart_GetYOffset) (load ?  import_function(m_handle, "PEAK_BufferPart_GetYOffset") : nullptr);
    if(m_PEAK_BufferPart_GetYOffset == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetYOffset");
    }        

    m_PEAK_BufferPart_GetXPadding = (dyn_PEAK_BufferPart_GetXPadding) (load ?  import_function(m_handle, "PEAK_BufferPart_GetXPadding") : nullptr);
    if(m_PEAK_BufferPart_GetXPadding == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetXPadding");
    }        

    m_PEAK_BufferPart_GetDeliveredImageHeight = (dyn_PEAK_BufferPart_GetDeliveredImageHeight) (load ?  import_function(m_handle, "PEAK_BufferPart_GetDeliveredImageHeight") : nullptr);
    if(m_PEAK_BufferPart_GetDeliveredImageHeight == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetDeliveredImageHeight");
    }        

    m_PEAK_BufferPart_GetParentBuffer = (dyn_PEAK_BufferPart_GetParentBuffer) (load ?  import_function(m_handle, "PEAK_BufferPart_GetParentBuffer") : nullptr);
    if(m_PEAK_BufferPart_GetParentBuffer == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BufferPart_GetParentBuffer");
    }        

    m_PEAK_ModuleDescriptor_GetID = (dyn_PEAK_ModuleDescriptor_GetID) (load ?  import_function(m_handle, "PEAK_ModuleDescriptor_GetID") : nullptr);
    if(m_PEAK_ModuleDescriptor_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_ModuleDescriptor_GetID");
    }        

    m_PEAK_Module_GetNumNodeMaps = (dyn_PEAK_Module_GetNumNodeMaps) (load ?  import_function(m_handle, "PEAK_Module_GetNumNodeMaps") : nullptr);
    if(m_PEAK_Module_GetNumNodeMaps == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Module_GetNumNodeMaps");
    }        

    m_PEAK_Module_GetNodeMap = (dyn_PEAK_Module_GetNodeMap) (load ?  import_function(m_handle, "PEAK_Module_GetNodeMap") : nullptr);
    if(m_PEAK_Module_GetNodeMap == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Module_GetNodeMap");
    }        

    m_PEAK_Module_GetPort = (dyn_PEAK_Module_GetPort) (load ?  import_function(m_handle, "PEAK_Module_GetPort") : nullptr);
    if(m_PEAK_Module_GetPort == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Module_GetPort");
    }        

    m_PEAK_NodeMap_GetHasNode = (dyn_PEAK_NodeMap_GetHasNode) (load ?  import_function(m_handle, "PEAK_NodeMap_GetHasNode") : nullptr);
    if(m_PEAK_NodeMap_GetHasNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_GetHasNode");
    }        

    m_PEAK_NodeMap_FindNode = (dyn_PEAK_NodeMap_FindNode) (load ?  import_function(m_handle, "PEAK_NodeMap_FindNode") : nullptr);
    if(m_PEAK_NodeMap_FindNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_FindNode");
    }        

    m_PEAK_NodeMap_InvalidateNodes = (dyn_PEAK_NodeMap_InvalidateNodes) (load ?  import_function(m_handle, "PEAK_NodeMap_InvalidateNodes") : nullptr);
    if(m_PEAK_NodeMap_InvalidateNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_InvalidateNodes");
    }        

    m_PEAK_NodeMap_PollNodes = (dyn_PEAK_NodeMap_PollNodes) (load ?  import_function(m_handle, "PEAK_NodeMap_PollNodes") : nullptr);
    if(m_PEAK_NodeMap_PollNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_PollNodes");
    }        

    m_PEAK_NodeMap_GetNumNodes = (dyn_PEAK_NodeMap_GetNumNodes) (load ?  import_function(m_handle, "PEAK_NodeMap_GetNumNodes") : nullptr);
    if(m_PEAK_NodeMap_GetNumNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_GetNumNodes");
    }        

    m_PEAK_NodeMap_GetNode = (dyn_PEAK_NodeMap_GetNode) (load ?  import_function(m_handle, "PEAK_NodeMap_GetNode") : nullptr);
    if(m_PEAK_NodeMap_GetNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_GetNode");
    }        

    m_PEAK_NodeMap_GetHasBufferSupportedChunks = (dyn_PEAK_NodeMap_GetHasBufferSupportedChunks) (load ?  import_function(m_handle, "PEAK_NodeMap_GetHasBufferSupportedChunks") : nullptr);
    if(m_PEAK_NodeMap_GetHasBufferSupportedChunks == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_GetHasBufferSupportedChunks");
    }        

    m_PEAK_NodeMap_UpdateChunkNodes = (dyn_PEAK_NodeMap_UpdateChunkNodes) (load ?  import_function(m_handle, "PEAK_NodeMap_UpdateChunkNodes") : nullptr);
    if(m_PEAK_NodeMap_UpdateChunkNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_UpdateChunkNodes");
    }        

    m_PEAK_NodeMap_GetHasEventSupportedData = (dyn_PEAK_NodeMap_GetHasEventSupportedData) (load ?  import_function(m_handle, "PEAK_NodeMap_GetHasEventSupportedData") : nullptr);
    if(m_PEAK_NodeMap_GetHasEventSupportedData == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_GetHasEventSupportedData");
    }        

    m_PEAK_NodeMap_UpdateEventNodes = (dyn_PEAK_NodeMap_UpdateEventNodes) (load ?  import_function(m_handle, "PEAK_NodeMap_UpdateEventNodes") : nullptr);
    if(m_PEAK_NodeMap_UpdateEventNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_UpdateEventNodes");
    }        

    m_PEAK_NodeMap_StoreToFile = (dyn_PEAK_NodeMap_StoreToFile) (load ?  import_function(m_handle, "PEAK_NodeMap_StoreToFile") : nullptr);
    if(m_PEAK_NodeMap_StoreToFile == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_StoreToFile");
    }        

    m_PEAK_NodeMap_LoadFromFile = (dyn_PEAK_NodeMap_LoadFromFile) (load ?  import_function(m_handle, "PEAK_NodeMap_LoadFromFile") : nullptr);
    if(m_PEAK_NodeMap_LoadFromFile == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_LoadFromFile");
    }        

    m_PEAK_NodeMap_Lock = (dyn_PEAK_NodeMap_Lock) (load ?  import_function(m_handle, "PEAK_NodeMap_Lock") : nullptr);
    if(m_PEAK_NodeMap_Lock == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_Lock");
    }        

    m_PEAK_NodeMap_Unlock = (dyn_PEAK_NodeMap_Unlock) (load ?  import_function(m_handle, "PEAK_NodeMap_Unlock") : nullptr);
    if(m_PEAK_NodeMap_Unlock == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_NodeMap_Unlock");
    }        

    m_PEAK_Node_ToIntegerNode = (dyn_PEAK_Node_ToIntegerNode) (load ?  import_function(m_handle, "PEAK_Node_ToIntegerNode") : nullptr);
    if(m_PEAK_Node_ToIntegerNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToIntegerNode");
    }        

    m_PEAK_Node_ToBooleanNode = (dyn_PEAK_Node_ToBooleanNode) (load ?  import_function(m_handle, "PEAK_Node_ToBooleanNode") : nullptr);
    if(m_PEAK_Node_ToBooleanNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToBooleanNode");
    }        

    m_PEAK_Node_ToCommandNode = (dyn_PEAK_Node_ToCommandNode) (load ?  import_function(m_handle, "PEAK_Node_ToCommandNode") : nullptr);
    if(m_PEAK_Node_ToCommandNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToCommandNode");
    }        

    m_PEAK_Node_ToFloatNode = (dyn_PEAK_Node_ToFloatNode) (load ?  import_function(m_handle, "PEAK_Node_ToFloatNode") : nullptr);
    if(m_PEAK_Node_ToFloatNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToFloatNode");
    }        

    m_PEAK_Node_ToStringNode = (dyn_PEAK_Node_ToStringNode) (load ?  import_function(m_handle, "PEAK_Node_ToStringNode") : nullptr);
    if(m_PEAK_Node_ToStringNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToStringNode");
    }        

    m_PEAK_Node_ToRegisterNode = (dyn_PEAK_Node_ToRegisterNode) (load ?  import_function(m_handle, "PEAK_Node_ToRegisterNode") : nullptr);
    if(m_PEAK_Node_ToRegisterNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToRegisterNode");
    }        

    m_PEAK_Node_ToCategoryNode = (dyn_PEAK_Node_ToCategoryNode) (load ?  import_function(m_handle, "PEAK_Node_ToCategoryNode") : nullptr);
    if(m_PEAK_Node_ToCategoryNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToCategoryNode");
    }        

    m_PEAK_Node_ToEnumerationNode = (dyn_PEAK_Node_ToEnumerationNode) (load ?  import_function(m_handle, "PEAK_Node_ToEnumerationNode") : nullptr);
    if(m_PEAK_Node_ToEnumerationNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToEnumerationNode");
    }        

    m_PEAK_Node_ToEnumerationEntryNode = (dyn_PEAK_Node_ToEnumerationEntryNode) (load ?  import_function(m_handle, "PEAK_Node_ToEnumerationEntryNode") : nullptr);
    if(m_PEAK_Node_ToEnumerationEntryNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_ToEnumerationEntryNode");
    }        

    m_PEAK_Node_GetName = (dyn_PEAK_Node_GetName) (load ?  import_function(m_handle, "PEAK_Node_GetName") : nullptr);
    if(m_PEAK_Node_GetName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetName");
    }        

    m_PEAK_Node_GetDisplayName = (dyn_PEAK_Node_GetDisplayName) (load ?  import_function(m_handle, "PEAK_Node_GetDisplayName") : nullptr);
    if(m_PEAK_Node_GetDisplayName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetDisplayName");
    }        

    m_PEAK_Node_GetNamespace = (dyn_PEAK_Node_GetNamespace) (load ?  import_function(m_handle, "PEAK_Node_GetNamespace") : nullptr);
    if(m_PEAK_Node_GetNamespace == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetNamespace");
    }        

    m_PEAK_Node_GetVisibility = (dyn_PEAK_Node_GetVisibility) (load ?  import_function(m_handle, "PEAK_Node_GetVisibility") : nullptr);
    if(m_PEAK_Node_GetVisibility == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetVisibility");
    }        

    m_PEAK_Node_GetAccessStatus = (dyn_PEAK_Node_GetAccessStatus) (load ?  import_function(m_handle, "PEAK_Node_GetAccessStatus") : nullptr);
    if(m_PEAK_Node_GetAccessStatus == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetAccessStatus");
    }        

    m_PEAK_Node_GetIsCacheable = (dyn_PEAK_Node_GetIsCacheable) (load ?  import_function(m_handle, "PEAK_Node_GetIsCacheable") : nullptr);
    if(m_PEAK_Node_GetIsCacheable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetIsCacheable");
    }        

    m_PEAK_Node_GetIsAccessStatusCacheable = (dyn_PEAK_Node_GetIsAccessStatusCacheable) (load ?  import_function(m_handle, "PEAK_Node_GetIsAccessStatusCacheable") : nullptr);
    if(m_PEAK_Node_GetIsAccessStatusCacheable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetIsAccessStatusCacheable");
    }        

    m_PEAK_Node_GetIsStreamable = (dyn_PEAK_Node_GetIsStreamable) (load ?  import_function(m_handle, "PEAK_Node_GetIsStreamable") : nullptr);
    if(m_PEAK_Node_GetIsStreamable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetIsStreamable");
    }        

    m_PEAK_Node_GetIsDeprecated = (dyn_PEAK_Node_GetIsDeprecated) (load ?  import_function(m_handle, "PEAK_Node_GetIsDeprecated") : nullptr);
    if(m_PEAK_Node_GetIsDeprecated == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetIsDeprecated");
    }        

    m_PEAK_Node_GetIsFeature = (dyn_PEAK_Node_GetIsFeature) (load ?  import_function(m_handle, "PEAK_Node_GetIsFeature") : nullptr);
    if(m_PEAK_Node_GetIsFeature == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetIsFeature");
    }        

    m_PEAK_Node_GetCachingMode = (dyn_PEAK_Node_GetCachingMode) (load ?  import_function(m_handle, "PEAK_Node_GetCachingMode") : nullptr);
    if(m_PEAK_Node_GetCachingMode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetCachingMode");
    }        

    m_PEAK_Node_GetPollingTime = (dyn_PEAK_Node_GetPollingTime) (load ?  import_function(m_handle, "PEAK_Node_GetPollingTime") : nullptr);
    if(m_PEAK_Node_GetPollingTime == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetPollingTime");
    }        

    m_PEAK_Node_GetToolTip = (dyn_PEAK_Node_GetToolTip) (load ?  import_function(m_handle, "PEAK_Node_GetToolTip") : nullptr);
    if(m_PEAK_Node_GetToolTip == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetToolTip");
    }        

    m_PEAK_Node_GetDescription = (dyn_PEAK_Node_GetDescription) (load ?  import_function(m_handle, "PEAK_Node_GetDescription") : nullptr);
    if(m_PEAK_Node_GetDescription == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetDescription");
    }        

    m_PEAK_Node_GetType = (dyn_PEAK_Node_GetType) (load ?  import_function(m_handle, "PEAK_Node_GetType") : nullptr);
    if(m_PEAK_Node_GetType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetType");
    }        

    m_PEAK_Node_GetParentNodeMap = (dyn_PEAK_Node_GetParentNodeMap) (load ?  import_function(m_handle, "PEAK_Node_GetParentNodeMap") : nullptr);
    if(m_PEAK_Node_GetParentNodeMap == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetParentNodeMap");
    }        

    m_PEAK_Node_FindInvalidatedNode = (dyn_PEAK_Node_FindInvalidatedNode) (load ?  import_function(m_handle, "PEAK_Node_FindInvalidatedNode") : nullptr);
    if(m_PEAK_Node_FindInvalidatedNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_FindInvalidatedNode");
    }        

    m_PEAK_Node_GetNumInvalidatedNodes = (dyn_PEAK_Node_GetNumInvalidatedNodes) (load ?  import_function(m_handle, "PEAK_Node_GetNumInvalidatedNodes") : nullptr);
    if(m_PEAK_Node_GetNumInvalidatedNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetNumInvalidatedNodes");
    }        

    m_PEAK_Node_GetInvalidatedNode = (dyn_PEAK_Node_GetInvalidatedNode) (load ?  import_function(m_handle, "PEAK_Node_GetInvalidatedNode") : nullptr);
    if(m_PEAK_Node_GetInvalidatedNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetInvalidatedNode");
    }        

    m_PEAK_Node_FindInvalidatingNode = (dyn_PEAK_Node_FindInvalidatingNode) (load ?  import_function(m_handle, "PEAK_Node_FindInvalidatingNode") : nullptr);
    if(m_PEAK_Node_FindInvalidatingNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_FindInvalidatingNode");
    }        

    m_PEAK_Node_GetNumInvalidatingNodes = (dyn_PEAK_Node_GetNumInvalidatingNodes) (load ?  import_function(m_handle, "PEAK_Node_GetNumInvalidatingNodes") : nullptr);
    if(m_PEAK_Node_GetNumInvalidatingNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetNumInvalidatingNodes");
    }        

    m_PEAK_Node_GetInvalidatingNode = (dyn_PEAK_Node_GetInvalidatingNode) (load ?  import_function(m_handle, "PEAK_Node_GetInvalidatingNode") : nullptr);
    if(m_PEAK_Node_GetInvalidatingNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetInvalidatingNode");
    }        

    m_PEAK_Node_FindSelectedNode = (dyn_PEAK_Node_FindSelectedNode) (load ?  import_function(m_handle, "PEAK_Node_FindSelectedNode") : nullptr);
    if(m_PEAK_Node_FindSelectedNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_FindSelectedNode");
    }        

    m_PEAK_Node_GetNumSelectedNodes = (dyn_PEAK_Node_GetNumSelectedNodes) (load ?  import_function(m_handle, "PEAK_Node_GetNumSelectedNodes") : nullptr);
    if(m_PEAK_Node_GetNumSelectedNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetNumSelectedNodes");
    }        

    m_PEAK_Node_GetSelectedNode = (dyn_PEAK_Node_GetSelectedNode) (load ?  import_function(m_handle, "PEAK_Node_GetSelectedNode") : nullptr);
    if(m_PEAK_Node_GetSelectedNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetSelectedNode");
    }        

    m_PEAK_Node_FindSelectingNode = (dyn_PEAK_Node_FindSelectingNode) (load ?  import_function(m_handle, "PEAK_Node_FindSelectingNode") : nullptr);
    if(m_PEAK_Node_FindSelectingNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_FindSelectingNode");
    }        

    m_PEAK_Node_GetNumSelectingNodes = (dyn_PEAK_Node_GetNumSelectingNodes) (load ?  import_function(m_handle, "PEAK_Node_GetNumSelectingNodes") : nullptr);
    if(m_PEAK_Node_GetNumSelectingNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetNumSelectingNodes");
    }        

    m_PEAK_Node_GetSelectingNode = (dyn_PEAK_Node_GetSelectingNode) (load ?  import_function(m_handle, "PEAK_Node_GetSelectingNode") : nullptr);
    if(m_PEAK_Node_GetSelectingNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_GetSelectingNode");
    }        

    m_PEAK_Node_RegisterChangedCallback = (dyn_PEAK_Node_RegisterChangedCallback) (load ?  import_function(m_handle, "PEAK_Node_RegisterChangedCallback") : nullptr);
    if(m_PEAK_Node_RegisterChangedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_RegisterChangedCallback");
    }        

    m_PEAK_Node_UnregisterChangedCallback = (dyn_PEAK_Node_UnregisterChangedCallback) (load ?  import_function(m_handle, "PEAK_Node_UnregisterChangedCallback") : nullptr);
    if(m_PEAK_Node_UnregisterChangedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Node_UnregisterChangedCallback");
    }        

    m_PEAK_IntegerNode_ToNode = (dyn_PEAK_IntegerNode_ToNode) (load ?  import_function(m_handle, "PEAK_IntegerNode_ToNode") : nullptr);
    if(m_PEAK_IntegerNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_ToNode");
    }        

    m_PEAK_IntegerNode_GetMinimum = (dyn_PEAK_IntegerNode_GetMinimum) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetMinimum") : nullptr);
    if(m_PEAK_IntegerNode_GetMinimum == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetMinimum");
    }        

    m_PEAK_IntegerNode_GetMaximum = (dyn_PEAK_IntegerNode_GetMaximum) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetMaximum") : nullptr);
    if(m_PEAK_IntegerNode_GetMaximum == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetMaximum");
    }        

    m_PEAK_IntegerNode_GetIncrement = (dyn_PEAK_IntegerNode_GetIncrement) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetIncrement") : nullptr);
    if(m_PEAK_IntegerNode_GetIncrement == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetIncrement");
    }        

    m_PEAK_IntegerNode_GetIncrementType = (dyn_PEAK_IntegerNode_GetIncrementType) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetIncrementType") : nullptr);
    if(m_PEAK_IntegerNode_GetIncrementType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetIncrementType");
    }        

    m_PEAK_IntegerNode_GetValidValues = (dyn_PEAK_IntegerNode_GetValidValues) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetValidValues") : nullptr);
    if(m_PEAK_IntegerNode_GetValidValues == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetValidValues");
    }        

    m_PEAK_IntegerNode_GetRepresentation = (dyn_PEAK_IntegerNode_GetRepresentation) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetRepresentation") : nullptr);
    if(m_PEAK_IntegerNode_GetRepresentation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetRepresentation");
    }        

    m_PEAK_IntegerNode_GetUnit = (dyn_PEAK_IntegerNode_GetUnit) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetUnit") : nullptr);
    if(m_PEAK_IntegerNode_GetUnit == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetUnit");
    }        

    m_PEAK_IntegerNode_GetValue = (dyn_PEAK_IntegerNode_GetValue) (load ?  import_function(m_handle, "PEAK_IntegerNode_GetValue") : nullptr);
    if(m_PEAK_IntegerNode_GetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_GetValue");
    }        

    m_PEAK_IntegerNode_SetValue = (dyn_PEAK_IntegerNode_SetValue) (load ?  import_function(m_handle, "PEAK_IntegerNode_SetValue") : nullptr);
    if(m_PEAK_IntegerNode_SetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_IntegerNode_SetValue");
    }        

    m_PEAK_BooleanNode_ToNode = (dyn_PEAK_BooleanNode_ToNode) (load ?  import_function(m_handle, "PEAK_BooleanNode_ToNode") : nullptr);
    if(m_PEAK_BooleanNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BooleanNode_ToNode");
    }        

    m_PEAK_BooleanNode_GetValue = (dyn_PEAK_BooleanNode_GetValue) (load ?  import_function(m_handle, "PEAK_BooleanNode_GetValue") : nullptr);
    if(m_PEAK_BooleanNode_GetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BooleanNode_GetValue");
    }        

    m_PEAK_BooleanNode_SetValue = (dyn_PEAK_BooleanNode_SetValue) (load ?  import_function(m_handle, "PEAK_BooleanNode_SetValue") : nullptr);
    if(m_PEAK_BooleanNode_SetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_BooleanNode_SetValue");
    }        

    m_PEAK_CommandNode_ToNode = (dyn_PEAK_CommandNode_ToNode) (load ?  import_function(m_handle, "PEAK_CommandNode_ToNode") : nullptr);
    if(m_PEAK_CommandNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CommandNode_ToNode");
    }        

    m_PEAK_CommandNode_GetIsDone = (dyn_PEAK_CommandNode_GetIsDone) (load ?  import_function(m_handle, "PEAK_CommandNode_GetIsDone") : nullptr);
    if(m_PEAK_CommandNode_GetIsDone == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CommandNode_GetIsDone");
    }        

    m_PEAK_CommandNode_Execute = (dyn_PEAK_CommandNode_Execute) (load ?  import_function(m_handle, "PEAK_CommandNode_Execute") : nullptr);
    if(m_PEAK_CommandNode_Execute == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CommandNode_Execute");
    }        

    m_PEAK_CommandNode_WaitUntilDoneInfinite = (dyn_PEAK_CommandNode_WaitUntilDoneInfinite) (load ?  import_function(m_handle, "PEAK_CommandNode_WaitUntilDoneInfinite") : nullptr);
    if(m_PEAK_CommandNode_WaitUntilDoneInfinite == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CommandNode_WaitUntilDoneInfinite");
    }        

    m_PEAK_CommandNode_WaitUntilDone = (dyn_PEAK_CommandNode_WaitUntilDone) (load ?  import_function(m_handle, "PEAK_CommandNode_WaitUntilDone") : nullptr);
    if(m_PEAK_CommandNode_WaitUntilDone == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CommandNode_WaitUntilDone");
    }        

    m_PEAK_FloatNode_ToNode = (dyn_PEAK_FloatNode_ToNode) (load ?  import_function(m_handle, "PEAK_FloatNode_ToNode") : nullptr);
    if(m_PEAK_FloatNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_ToNode");
    }        

    m_PEAK_FloatNode_GetMinimum = (dyn_PEAK_FloatNode_GetMinimum) (load ?  import_function(m_handle, "PEAK_FloatNode_GetMinimum") : nullptr);
    if(m_PEAK_FloatNode_GetMinimum == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetMinimum");
    }        

    m_PEAK_FloatNode_GetMaximum = (dyn_PEAK_FloatNode_GetMaximum) (load ?  import_function(m_handle, "PEAK_FloatNode_GetMaximum") : nullptr);
    if(m_PEAK_FloatNode_GetMaximum == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetMaximum");
    }        

    m_PEAK_FloatNode_GetIncrement = (dyn_PEAK_FloatNode_GetIncrement) (load ?  import_function(m_handle, "PEAK_FloatNode_GetIncrement") : nullptr);
    if(m_PEAK_FloatNode_GetIncrement == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetIncrement");
    }        

    m_PEAK_FloatNode_GetIncrementType = (dyn_PEAK_FloatNode_GetIncrementType) (load ?  import_function(m_handle, "PEAK_FloatNode_GetIncrementType") : nullptr);
    if(m_PEAK_FloatNode_GetIncrementType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetIncrementType");
    }        

    m_PEAK_FloatNode_GetValidValues = (dyn_PEAK_FloatNode_GetValidValues) (load ?  import_function(m_handle, "PEAK_FloatNode_GetValidValues") : nullptr);
    if(m_PEAK_FloatNode_GetValidValues == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetValidValues");
    }        

    m_PEAK_FloatNode_GetRepresentation = (dyn_PEAK_FloatNode_GetRepresentation) (load ?  import_function(m_handle, "PEAK_FloatNode_GetRepresentation") : nullptr);
    if(m_PEAK_FloatNode_GetRepresentation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetRepresentation");
    }        

    m_PEAK_FloatNode_GetUnit = (dyn_PEAK_FloatNode_GetUnit) (load ?  import_function(m_handle, "PEAK_FloatNode_GetUnit") : nullptr);
    if(m_PEAK_FloatNode_GetUnit == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetUnit");
    }        

    m_PEAK_FloatNode_GetDisplayNotation = (dyn_PEAK_FloatNode_GetDisplayNotation) (load ?  import_function(m_handle, "PEAK_FloatNode_GetDisplayNotation") : nullptr);
    if(m_PEAK_FloatNode_GetDisplayNotation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetDisplayNotation");
    }        

    m_PEAK_FloatNode_GetDisplayPrecision = (dyn_PEAK_FloatNode_GetDisplayPrecision) (load ?  import_function(m_handle, "PEAK_FloatNode_GetDisplayPrecision") : nullptr);
    if(m_PEAK_FloatNode_GetDisplayPrecision == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetDisplayPrecision");
    }        

    m_PEAK_FloatNode_GetHasConstantIncrement = (dyn_PEAK_FloatNode_GetHasConstantIncrement) (load ?  import_function(m_handle, "PEAK_FloatNode_GetHasConstantIncrement") : nullptr);
    if(m_PEAK_FloatNode_GetHasConstantIncrement == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetHasConstantIncrement");
    }        

    m_PEAK_FloatNode_GetValue = (dyn_PEAK_FloatNode_GetValue) (load ?  import_function(m_handle, "PEAK_FloatNode_GetValue") : nullptr);
    if(m_PEAK_FloatNode_GetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_GetValue");
    }        

    m_PEAK_FloatNode_SetValue = (dyn_PEAK_FloatNode_SetValue) (load ?  import_function(m_handle, "PEAK_FloatNode_SetValue") : nullptr);
    if(m_PEAK_FloatNode_SetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FloatNode_SetValue");
    }        

    m_PEAK_StringNode_ToNode = (dyn_PEAK_StringNode_ToNode) (load ?  import_function(m_handle, "PEAK_StringNode_ToNode") : nullptr);
    if(m_PEAK_StringNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_StringNode_ToNode");
    }        

    m_PEAK_StringNode_GetMaximumLength = (dyn_PEAK_StringNode_GetMaximumLength) (load ?  import_function(m_handle, "PEAK_StringNode_GetMaximumLength") : nullptr);
    if(m_PEAK_StringNode_GetMaximumLength == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_StringNode_GetMaximumLength");
    }        

    m_PEAK_StringNode_GetValue = (dyn_PEAK_StringNode_GetValue) (load ?  import_function(m_handle, "PEAK_StringNode_GetValue") : nullptr);
    if(m_PEAK_StringNode_GetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_StringNode_GetValue");
    }        

    m_PEAK_StringNode_SetValue = (dyn_PEAK_StringNode_SetValue) (load ?  import_function(m_handle, "PEAK_StringNode_SetValue") : nullptr);
    if(m_PEAK_StringNode_SetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_StringNode_SetValue");
    }        

    m_PEAK_RegisterNode_ToNode = (dyn_PEAK_RegisterNode_ToNode) (load ?  import_function(m_handle, "PEAK_RegisterNode_ToNode") : nullptr);
    if(m_PEAK_RegisterNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RegisterNode_ToNode");
    }        

    m_PEAK_RegisterNode_GetAddress = (dyn_PEAK_RegisterNode_GetAddress) (load ?  import_function(m_handle, "PEAK_RegisterNode_GetAddress") : nullptr);
    if(m_PEAK_RegisterNode_GetAddress == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RegisterNode_GetAddress");
    }        

    m_PEAK_RegisterNode_GetLength = (dyn_PEAK_RegisterNode_GetLength) (load ?  import_function(m_handle, "PEAK_RegisterNode_GetLength") : nullptr);
    if(m_PEAK_RegisterNode_GetLength == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RegisterNode_GetLength");
    }        

    m_PEAK_RegisterNode_Read = (dyn_PEAK_RegisterNode_Read) (load ?  import_function(m_handle, "PEAK_RegisterNode_Read") : nullptr);
    if(m_PEAK_RegisterNode_Read == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RegisterNode_Read");
    }        

    m_PEAK_RegisterNode_Write = (dyn_PEAK_RegisterNode_Write) (load ?  import_function(m_handle, "PEAK_RegisterNode_Write") : nullptr);
    if(m_PEAK_RegisterNode_Write == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_RegisterNode_Write");
    }        

    m_PEAK_CategoryNode_ToNode = (dyn_PEAK_CategoryNode_ToNode) (load ?  import_function(m_handle, "PEAK_CategoryNode_ToNode") : nullptr);
    if(m_PEAK_CategoryNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CategoryNode_ToNode");
    }        

    m_PEAK_CategoryNode_GetNumSubNodes = (dyn_PEAK_CategoryNode_GetNumSubNodes) (load ?  import_function(m_handle, "PEAK_CategoryNode_GetNumSubNodes") : nullptr);
    if(m_PEAK_CategoryNode_GetNumSubNodes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CategoryNode_GetNumSubNodes");
    }        

    m_PEAK_CategoryNode_GetSubNode = (dyn_PEAK_CategoryNode_GetSubNode) (load ?  import_function(m_handle, "PEAK_CategoryNode_GetSubNode") : nullptr);
    if(m_PEAK_CategoryNode_GetSubNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_CategoryNode_GetSubNode");
    }        

    m_PEAK_EnumerationNode_ToNode = (dyn_PEAK_EnumerationNode_ToNode) (load ?  import_function(m_handle, "PEAK_EnumerationNode_ToNode") : nullptr);
    if(m_PEAK_EnumerationNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_ToNode");
    }        

    m_PEAK_EnumerationNode_GetCurrentEntry = (dyn_PEAK_EnumerationNode_GetCurrentEntry) (load ?  import_function(m_handle, "PEAK_EnumerationNode_GetCurrentEntry") : nullptr);
    if(m_PEAK_EnumerationNode_GetCurrentEntry == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_GetCurrentEntry");
    }        

    m_PEAK_EnumerationNode_SetCurrentEntry = (dyn_PEAK_EnumerationNode_SetCurrentEntry) (load ?  import_function(m_handle, "PEAK_EnumerationNode_SetCurrentEntry") : nullptr);
    if(m_PEAK_EnumerationNode_SetCurrentEntry == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_SetCurrentEntry");
    }        

    m_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue = (dyn_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue) (load ?  import_function(m_handle, "PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue") : nullptr);
    if(m_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue");
    }        

    m_PEAK_EnumerationNode_SetCurrentEntryByValue = (dyn_PEAK_EnumerationNode_SetCurrentEntryByValue) (load ?  import_function(m_handle, "PEAK_EnumerationNode_SetCurrentEntryByValue") : nullptr);
    if(m_PEAK_EnumerationNode_SetCurrentEntryByValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_SetCurrentEntryByValue");
    }        

    m_PEAK_EnumerationNode_FindEntryBySymbolicValue = (dyn_PEAK_EnumerationNode_FindEntryBySymbolicValue) (load ?  import_function(m_handle, "PEAK_EnumerationNode_FindEntryBySymbolicValue") : nullptr);
    if(m_PEAK_EnumerationNode_FindEntryBySymbolicValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_FindEntryBySymbolicValue");
    }        

    m_PEAK_EnumerationNode_FindEntryByValue = (dyn_PEAK_EnumerationNode_FindEntryByValue) (load ?  import_function(m_handle, "PEAK_EnumerationNode_FindEntryByValue") : nullptr);
    if(m_PEAK_EnumerationNode_FindEntryByValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_FindEntryByValue");
    }        

    m_PEAK_EnumerationNode_GetNumEntries = (dyn_PEAK_EnumerationNode_GetNumEntries) (load ?  import_function(m_handle, "PEAK_EnumerationNode_GetNumEntries") : nullptr);
    if(m_PEAK_EnumerationNode_GetNumEntries == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_GetNumEntries");
    }        

    m_PEAK_EnumerationNode_GetEntry = (dyn_PEAK_EnumerationNode_GetEntry) (load ?  import_function(m_handle, "PEAK_EnumerationNode_GetEntry") : nullptr);
    if(m_PEAK_EnumerationNode_GetEntry == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationNode_GetEntry");
    }        

    m_PEAK_EnumerationEntryNode_ToNode = (dyn_PEAK_EnumerationEntryNode_ToNode) (load ?  import_function(m_handle, "PEAK_EnumerationEntryNode_ToNode") : nullptr);
    if(m_PEAK_EnumerationEntryNode_ToNode == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationEntryNode_ToNode");
    }        

    m_PEAK_EnumerationEntryNode_GetIsSelfClearing = (dyn_PEAK_EnumerationEntryNode_GetIsSelfClearing) (load ?  import_function(m_handle, "PEAK_EnumerationEntryNode_GetIsSelfClearing") : nullptr);
    if(m_PEAK_EnumerationEntryNode_GetIsSelfClearing == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationEntryNode_GetIsSelfClearing");
    }        

    m_PEAK_EnumerationEntryNode_GetSymbolicValue = (dyn_PEAK_EnumerationEntryNode_GetSymbolicValue) (load ?  import_function(m_handle, "PEAK_EnumerationEntryNode_GetSymbolicValue") : nullptr);
    if(m_PEAK_EnumerationEntryNode_GetSymbolicValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationEntryNode_GetSymbolicValue");
    }        

    m_PEAK_EnumerationEntryNode_GetValue = (dyn_PEAK_EnumerationEntryNode_GetValue) (load ?  import_function(m_handle, "PEAK_EnumerationEntryNode_GetValue") : nullptr);
    if(m_PEAK_EnumerationEntryNode_GetValue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EnumerationEntryNode_GetValue");
    }        

    m_PEAK_Port_GetInfo = (dyn_PEAK_Port_GetInfo) (load ?  import_function(m_handle, "PEAK_Port_GetInfo") : nullptr);
    if(m_PEAK_Port_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetInfo");
    }        

    m_PEAK_Port_GetID = (dyn_PEAK_Port_GetID) (load ?  import_function(m_handle, "PEAK_Port_GetID") : nullptr);
    if(m_PEAK_Port_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetID");
    }        

    m_PEAK_Port_GetName = (dyn_PEAK_Port_GetName) (load ?  import_function(m_handle, "PEAK_Port_GetName") : nullptr);
    if(m_PEAK_Port_GetName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetName");
    }        

    m_PEAK_Port_GetVendorName = (dyn_PEAK_Port_GetVendorName) (load ?  import_function(m_handle, "PEAK_Port_GetVendorName") : nullptr);
    if(m_PEAK_Port_GetVendorName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetVendorName");
    }        

    m_PEAK_Port_GetModelName = (dyn_PEAK_Port_GetModelName) (load ?  import_function(m_handle, "PEAK_Port_GetModelName") : nullptr);
    if(m_PEAK_Port_GetModelName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetModelName");
    }        

    m_PEAK_Port_GetVersion = (dyn_PEAK_Port_GetVersion) (load ?  import_function(m_handle, "PEAK_Port_GetVersion") : nullptr);
    if(m_PEAK_Port_GetVersion == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetVersion");
    }        

    m_PEAK_Port_GetTLType = (dyn_PEAK_Port_GetTLType) (load ?  import_function(m_handle, "PEAK_Port_GetTLType") : nullptr);
    if(m_PEAK_Port_GetTLType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetTLType");
    }        

    m_PEAK_Port_GetModuleName = (dyn_PEAK_Port_GetModuleName) (load ?  import_function(m_handle, "PEAK_Port_GetModuleName") : nullptr);
    if(m_PEAK_Port_GetModuleName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetModuleName");
    }        

    m_PEAK_Port_GetDataEndianness = (dyn_PEAK_Port_GetDataEndianness) (load ?  import_function(m_handle, "PEAK_Port_GetDataEndianness") : nullptr);
    if(m_PEAK_Port_GetDataEndianness == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetDataEndianness");
    }        

    m_PEAK_Port_GetIsReadable = (dyn_PEAK_Port_GetIsReadable) (load ?  import_function(m_handle, "PEAK_Port_GetIsReadable") : nullptr);
    if(m_PEAK_Port_GetIsReadable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetIsReadable");
    }        

    m_PEAK_Port_GetIsWritable = (dyn_PEAK_Port_GetIsWritable) (load ?  import_function(m_handle, "PEAK_Port_GetIsWritable") : nullptr);
    if(m_PEAK_Port_GetIsWritable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetIsWritable");
    }        

    m_PEAK_Port_GetIsAvailable = (dyn_PEAK_Port_GetIsAvailable) (load ?  import_function(m_handle, "PEAK_Port_GetIsAvailable") : nullptr);
    if(m_PEAK_Port_GetIsAvailable == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetIsAvailable");
    }        

    m_PEAK_Port_GetIsImplemented = (dyn_PEAK_Port_GetIsImplemented) (load ?  import_function(m_handle, "PEAK_Port_GetIsImplemented") : nullptr);
    if(m_PEAK_Port_GetIsImplemented == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetIsImplemented");
    }        

    m_PEAK_Port_Read = (dyn_PEAK_Port_Read) (load ?  import_function(m_handle, "PEAK_Port_Read") : nullptr);
    if(m_PEAK_Port_Read == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_Read");
    }        

    m_PEAK_Port_Write = (dyn_PEAK_Port_Write) (load ?  import_function(m_handle, "PEAK_Port_Write") : nullptr);
    if(m_PEAK_Port_Write == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_Write");
    }        

    m_PEAK_Port_GetNumURLs = (dyn_PEAK_Port_GetNumURLs) (load ?  import_function(m_handle, "PEAK_Port_GetNumURLs") : nullptr);
    if(m_PEAK_Port_GetNumURLs == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetNumURLs");
    }        

    m_PEAK_Port_GetURL = (dyn_PEAK_Port_GetURL) (load ?  import_function(m_handle, "PEAK_Port_GetURL") : nullptr);
    if(m_PEAK_Port_GetURL == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Port_GetURL");
    }        

    m_PEAK_PortURL_GetInfo = (dyn_PEAK_PortURL_GetInfo) (load ?  import_function(m_handle, "PEAK_PortURL_GetInfo") : nullptr);
    if(m_PEAK_PortURL_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetInfo");
    }        

    m_PEAK_PortURL_GetURL = (dyn_PEAK_PortURL_GetURL) (load ?  import_function(m_handle, "PEAK_PortURL_GetURL") : nullptr);
    if(m_PEAK_PortURL_GetURL == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetURL");
    }        

    m_PEAK_PortURL_GetScheme = (dyn_PEAK_PortURL_GetScheme) (load ?  import_function(m_handle, "PEAK_PortURL_GetScheme") : nullptr);
    if(m_PEAK_PortURL_GetScheme == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetScheme");
    }        

    m_PEAK_PortURL_GetFileName = (dyn_PEAK_PortURL_GetFileName) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileName") : nullptr);
    if(m_PEAK_PortURL_GetFileName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileName");
    }        

    m_PEAK_PortURL_GetFileRegisterAddress = (dyn_PEAK_PortURL_GetFileRegisterAddress) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileRegisterAddress") : nullptr);
    if(m_PEAK_PortURL_GetFileRegisterAddress == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileRegisterAddress");
    }        

    m_PEAK_PortURL_GetFileSize = (dyn_PEAK_PortURL_GetFileSize) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileSize") : nullptr);
    if(m_PEAK_PortURL_GetFileSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileSize");
    }        

    m_PEAK_PortURL_GetFileSHA1Hash = (dyn_PEAK_PortURL_GetFileSHA1Hash) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileSHA1Hash") : nullptr);
    if(m_PEAK_PortURL_GetFileSHA1Hash == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileSHA1Hash");
    }        

    m_PEAK_PortURL_GetFileVersionMajor = (dyn_PEAK_PortURL_GetFileVersionMajor) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileVersionMajor") : nullptr);
    if(m_PEAK_PortURL_GetFileVersionMajor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileVersionMajor");
    }        

    m_PEAK_PortURL_GetFileVersionMinor = (dyn_PEAK_PortURL_GetFileVersionMinor) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileVersionMinor") : nullptr);
    if(m_PEAK_PortURL_GetFileVersionMinor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileVersionMinor");
    }        

    m_PEAK_PortURL_GetFileVersionSubminor = (dyn_PEAK_PortURL_GetFileVersionSubminor) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileVersionSubminor") : nullptr);
    if(m_PEAK_PortURL_GetFileVersionSubminor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileVersionSubminor");
    }        

    m_PEAK_PortURL_GetFileSchemaVersionMajor = (dyn_PEAK_PortURL_GetFileSchemaVersionMajor) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileSchemaVersionMajor") : nullptr);
    if(m_PEAK_PortURL_GetFileSchemaVersionMajor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileSchemaVersionMajor");
    }        

    m_PEAK_PortURL_GetFileSchemaVersionMinor = (dyn_PEAK_PortURL_GetFileSchemaVersionMinor) (load ?  import_function(m_handle, "PEAK_PortURL_GetFileSchemaVersionMinor") : nullptr);
    if(m_PEAK_PortURL_GetFileSchemaVersionMinor == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetFileSchemaVersionMinor");
    }        

    m_PEAK_PortURL_GetParentPort = (dyn_PEAK_PortURL_GetParentPort) (load ?  import_function(m_handle, "PEAK_PortURL_GetParentPort") : nullptr);
    if(m_PEAK_PortURL_GetParentPort == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_PortURL_GetParentPort");
    }        

    m_PEAK_EventSupportingModule_EnableEvents = (dyn_PEAK_EventSupportingModule_EnableEvents) (load ?  import_function(m_handle, "PEAK_EventSupportingModule_EnableEvents") : nullptr);
    if(m_PEAK_EventSupportingModule_EnableEvents == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventSupportingModule_EnableEvents");
    }        

    m_PEAK_EventController_GetInfo = (dyn_PEAK_EventController_GetInfo) (load ?  import_function(m_handle, "PEAK_EventController_GetInfo") : nullptr);
    if(m_PEAK_EventController_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_GetInfo");
    }        

    m_PEAK_EventController_GetNumEventsInQueue = (dyn_PEAK_EventController_GetNumEventsInQueue) (load ?  import_function(m_handle, "PEAK_EventController_GetNumEventsInQueue") : nullptr);
    if(m_PEAK_EventController_GetNumEventsInQueue == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_GetNumEventsInQueue");
    }        

    m_PEAK_EventController_GetNumEventsFired = (dyn_PEAK_EventController_GetNumEventsFired) (load ?  import_function(m_handle, "PEAK_EventController_GetNumEventsFired") : nullptr);
    if(m_PEAK_EventController_GetNumEventsFired == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_GetNumEventsFired");
    }        

    m_PEAK_EventController_GetEventMaxSize = (dyn_PEAK_EventController_GetEventMaxSize) (load ?  import_function(m_handle, "PEAK_EventController_GetEventMaxSize") : nullptr);
    if(m_PEAK_EventController_GetEventMaxSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_GetEventMaxSize");
    }        

    m_PEAK_EventController_GetEventDataMaxSize = (dyn_PEAK_EventController_GetEventDataMaxSize) (load ?  import_function(m_handle, "PEAK_EventController_GetEventDataMaxSize") : nullptr);
    if(m_PEAK_EventController_GetEventDataMaxSize == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_GetEventDataMaxSize");
    }        

    m_PEAK_EventController_GetControlledEventType = (dyn_PEAK_EventController_GetControlledEventType) (load ?  import_function(m_handle, "PEAK_EventController_GetControlledEventType") : nullptr);
    if(m_PEAK_EventController_GetControlledEventType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_GetControlledEventType");
    }        

    m_PEAK_EventController_WaitForEvent = (dyn_PEAK_EventController_WaitForEvent) (load ?  import_function(m_handle, "PEAK_EventController_WaitForEvent") : nullptr);
    if(m_PEAK_EventController_WaitForEvent == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_WaitForEvent");
    }        

    m_PEAK_EventController_KillWait = (dyn_PEAK_EventController_KillWait) (load ?  import_function(m_handle, "PEAK_EventController_KillWait") : nullptr);
    if(m_PEAK_EventController_KillWait == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_KillWait");
    }        

    m_PEAK_EventController_FlushEvents = (dyn_PEAK_EventController_FlushEvents) (load ?  import_function(m_handle, "PEAK_EventController_FlushEvents") : nullptr);
    if(m_PEAK_EventController_FlushEvents == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_FlushEvents");
    }        

    m_PEAK_EventController_Destruct = (dyn_PEAK_EventController_Destruct) (load ?  import_function(m_handle, "PEAK_EventController_Destruct") : nullptr);
    if(m_PEAK_EventController_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_EventController_Destruct");
    }        

    m_PEAK_Event_GetInfo = (dyn_PEAK_Event_GetInfo) (load ?  import_function(m_handle, "PEAK_Event_GetInfo") : nullptr);
    if(m_PEAK_Event_GetInfo == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Event_GetInfo");
    }        

    m_PEAK_Event_GetID = (dyn_PEAK_Event_GetID) (load ?  import_function(m_handle, "PEAK_Event_GetID") : nullptr);
    if(m_PEAK_Event_GetID == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Event_GetID");
    }        

    m_PEAK_Event_GetData = (dyn_PEAK_Event_GetData) (load ?  import_function(m_handle, "PEAK_Event_GetData") : nullptr);
    if(m_PEAK_Event_GetData == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Event_GetData");
    }        

    m_PEAK_Event_GetType = (dyn_PEAK_Event_GetType) (load ?  import_function(m_handle, "PEAK_Event_GetType") : nullptr);
    if(m_PEAK_Event_GetType == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Event_GetType");
    }        

    m_PEAK_Event_GetRawData = (dyn_PEAK_Event_GetRawData) (load ?  import_function(m_handle, "PEAK_Event_GetRawData") : nullptr);
    if(m_PEAK_Event_GetRawData == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Event_GetRawData");
    }        

    m_PEAK_Event_Destruct = (dyn_PEAK_Event_Destruct) (load ?  import_function(m_handle, "PEAK_Event_Destruct") : nullptr);
    if(m_PEAK_Event_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_Event_Destruct");
    }        

    m_PEAK_FirmwareUpdater_Construct = (dyn_PEAK_FirmwareUpdater_Construct) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_Construct") : nullptr);
    if(m_PEAK_FirmwareUpdater_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_Construct");
    }        

    m_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation = (dyn_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation") : nullptr);
    if(m_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation");
    }        

    m_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation = (dyn_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation") : nullptr);
    if(m_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation");
    }        

    m_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation = (dyn_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation") : nullptr);
    if(m_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation");
    }        

    m_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation = (dyn_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_GetFirmwareUpdateInformation") : nullptr);
    if(m_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_GetFirmwareUpdateInformation");
    }        

    m_PEAK_FirmwareUpdater_UpdateDevice = (dyn_PEAK_FirmwareUpdater_UpdateDevice) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_UpdateDevice") : nullptr);
    if(m_PEAK_FirmwareUpdater_UpdateDevice == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_UpdateDevice");
    }        

    m_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout = (dyn_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout") : nullptr);
    if(m_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout");
    }        

    m_PEAK_FirmwareUpdater_Destruct = (dyn_PEAK_FirmwareUpdater_Destruct) (load ?  import_function(m_handle, "PEAK_FirmwareUpdater_Destruct") : nullptr);
    if(m_PEAK_FirmwareUpdater_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdater_Destruct");
    }        

    m_PEAK_FirmwareUpdateInformation_GetIsValid = (dyn_PEAK_FirmwareUpdateInformation_GetIsValid) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetIsValid") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetIsValid == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetIsValid");
    }        

    m_PEAK_FirmwareUpdateInformation_GetFileName = (dyn_PEAK_FirmwareUpdateInformation_GetFileName) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetFileName") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetFileName == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetFileName");
    }        

    m_PEAK_FirmwareUpdateInformation_GetDescription = (dyn_PEAK_FirmwareUpdateInformation_GetDescription) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetDescription") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetDescription == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetDescription");
    }        

    m_PEAK_FirmwareUpdateInformation_GetVersion = (dyn_PEAK_FirmwareUpdateInformation_GetVersion) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetVersion") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetVersion == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetVersion");
    }        

    m_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern = (dyn_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern");
    }        

    m_PEAK_FirmwareUpdateInformation_GetVersionStyle = (dyn_PEAK_FirmwareUpdateInformation_GetVersionStyle) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetVersionStyle") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetVersionStyle == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetVersionStyle");
    }        

    m_PEAK_FirmwareUpdateInformation_GetReleaseNotes = (dyn_PEAK_FirmwareUpdateInformation_GetReleaseNotes) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetReleaseNotes") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetReleaseNotes == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetReleaseNotes");
    }        

    m_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL = (dyn_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetReleaseNotesURL") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetReleaseNotesURL");
    }        

    m_PEAK_FirmwareUpdateInformation_GetUserSetPersistence = (dyn_PEAK_FirmwareUpdateInformation_GetUserSetPersistence) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetUserSetPersistence") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetUserSetPersistence == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetUserSetPersistence");
    }        

    m_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence = (dyn_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence") : nullptr);
    if(m_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_Construct = (dyn_PEAK_FirmwareUpdateProgressObserver_Construct) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_Construct") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_Construct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_Construct");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback = (dyn_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback");
    }        

    m_PEAK_FirmwareUpdateProgressObserver_Destruct = (dyn_PEAK_FirmwareUpdateProgressObserver_Destruct) (load ?  import_function(m_handle, "PEAK_FirmwareUpdateProgressObserver_Destruct") : nullptr);
    if(m_PEAK_FirmwareUpdateProgressObserver_Destruct == nullptr && load)
    {
        throw std::runtime_error("Failed to load PEAK_FirmwareUpdateProgressObserver_Destruct");
    }        

            
            return true;
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_Initialize()
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_Initialize)
    {
        return inst.m_PEAK_Library_Initialize();
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_Close()
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_Close)
    {
        return inst.m_PEAK_Library_Close();
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_IsInitialized(PEAK_BOOL8 * isInitialized)
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_IsInitialized)
    {
        return inst.m_PEAK_Library_IsInitialized(isInitialized);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_GetVersionMajor(uint32_t * libraryVersionMajor)
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_GetVersionMajor)
    {
        return inst.m_PEAK_Library_GetVersionMajor(libraryVersionMajor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_GetVersionMinor(uint32_t * libraryVersionMinor)
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_GetVersionMinor)
    {
        return inst.m_PEAK_Library_GetVersionMinor(libraryVersionMinor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_GetVersionSubminor(uint32_t * libraryVersionSubminor)
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_GetVersionSubminor)
    {
        return inst.m_PEAK_Library_GetVersionSubminor(libraryVersionSubminor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Library_GetLastError(PEAK_RETURN_CODE * lastErrorCode, char * lastErrorDescription, size_t * lastErrorDescriptionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Library_GetLastError)
    {
        return inst.m_PEAK_Library_GetLastError(lastErrorCode, lastErrorDescription, lastErrorDescriptionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnvironmentInspector_UpdateCTIPaths()
{
    auto& inst = instance();
    if(inst.m_PEAK_EnvironmentInspector_UpdateCTIPaths)
    {
        return inst.m_PEAK_EnvironmentInspector_UpdateCTIPaths();
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnvironmentInspector_GetNumCTIPaths(size_t * numCtiPaths)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnvironmentInspector_GetNumCTIPaths)
    {
        return inst.m_PEAK_EnvironmentInspector_GetNumCTIPaths(numCtiPaths);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnvironmentInspector_GetCTIPath(size_t index, char * ctiPath, size_t * ctiPathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnvironmentInspector_GetCTIPath)
    {
        return inst.m_PEAK_EnvironmentInspector_GetCTIPath(index, ctiPath, ctiPathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_ProducerLibrary_Construct(const char * ctiPath, size_t ctiPathSize, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_ProducerLibrary_Construct)
    {
        return inst.m_PEAK_ProducerLibrary_Construct(ctiPath, ctiPathSize, producerLibraryHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_ProducerLibrary_GetKey(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_ProducerLibrary_GetKey)
    {
        return inst.m_PEAK_ProducerLibrary_GetKey(producerLibraryHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_ProducerLibrary_GetSystem(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle, PEAK_SYSTEM_DESCRIPTOR_HANDLE * systemDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_ProducerLibrary_GetSystem)
    {
        return inst.m_PEAK_ProducerLibrary_GetSystem(producerLibraryHandle, systemDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_ProducerLibrary_Destruct(PEAK_PRODUCER_LIBRARY_HANDLE producerLibraryHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_ProducerLibrary_Destruct)
    {
        return inst.m_PEAK_ProducerLibrary_Destruct(producerLibraryHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_ToModuleDescriptor(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_ToModuleDescriptor)
    {
        return inst.m_PEAK_SystemDescriptor_ToModuleDescriptor(systemDescriptorHandle, moduleDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetKey(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetKey)
    {
        return inst.m_PEAK_SystemDescriptor_GetKey(systemDescriptorHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetInfo(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetInfo)
    {
        return inst.m_PEAK_SystemDescriptor_GetInfo(systemDescriptorHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetDisplayName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetDisplayName)
    {
        return inst.m_PEAK_SystemDescriptor_GetDisplayName(systemDescriptorHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetVendorName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * vendorName, size_t * vendorNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetVendorName)
    {
        return inst.m_PEAK_SystemDescriptor_GetVendorName(systemDescriptorHandle, vendorName, vendorNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetModelName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * modelName, size_t * modelNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetModelName)
    {
        return inst.m_PEAK_SystemDescriptor_GetModelName(systemDescriptorHandle, modelName, modelNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetVersion(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * version, size_t * versionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetVersion)
    {
        return inst.m_PEAK_SystemDescriptor_GetVersion(systemDescriptorHandle, version, versionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetTLType(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetTLType)
    {
        return inst.m_PEAK_SystemDescriptor_GetTLType(systemDescriptorHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetCTIFileName(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * ctiFileName, size_t * ctiFileNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetCTIFileName)
    {
        return inst.m_PEAK_SystemDescriptor_GetCTIFileName(systemDescriptorHandle, ctiFileName, ctiFileNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetCTIFullPath(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, char * ctiFullPath, size_t * ctiFullPathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetCTIFullPath)
    {
        return inst.m_PEAK_SystemDescriptor_GetCTIFullPath(systemDescriptorHandle, ctiFullPath, ctiFullPathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetGenTLVersionMajor(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t * gentlVersionMajor)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetGenTLVersionMajor)
    {
        return inst.m_PEAK_SystemDescriptor_GetGenTLVersionMajor(systemDescriptorHandle, gentlVersionMajor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetGenTLVersionMinor(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, uint32_t * gentlVersionMinor)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetGenTLVersionMinor)
    {
        return inst.m_PEAK_SystemDescriptor_GetGenTLVersionMinor(systemDescriptorHandle, gentlVersionMinor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetCharacterEncoding(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_CHARACTER_ENCODING * characterEncoding)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetCharacterEncoding)
    {
        return inst.m_PEAK_SystemDescriptor_GetCharacterEncoding(systemDescriptorHandle, characterEncoding);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_GetParentLibrary(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_GetParentLibrary)
    {
        return inst.m_PEAK_SystemDescriptor_GetParentLibrary(systemDescriptorHandle, producerLibraryHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_SystemDescriptor_OpenSystem(PEAK_SYSTEM_DESCRIPTOR_HANDLE systemDescriptorHandle, PEAK_SYSTEM_HANDLE * systemHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_SystemDescriptor_OpenSystem)
    {
        return inst.m_PEAK_SystemDescriptor_OpenSystem(systemDescriptorHandle, systemHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_ToModule(PEAK_SYSTEM_HANDLE systemHandle, PEAK_MODULE_HANDLE * moduleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_ToModule)
    {
        return inst.m_PEAK_System_ToModule(systemHandle, moduleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_ToEventSupportingModule(PEAK_SYSTEM_HANDLE systemHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_ToEventSupportingModule)
    {
        return inst.m_PEAK_System_ToEventSupportingModule(systemHandle, eventSupportingModuleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetKey(PEAK_SYSTEM_HANDLE systemHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetKey)
    {
        return inst.m_PEAK_System_GetKey(systemHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetInfo(PEAK_SYSTEM_HANDLE systemHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetInfo)
    {
        return inst.m_PEAK_System_GetInfo(systemHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetID(PEAK_SYSTEM_HANDLE systemHandle, char * id, size_t * idSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetID)
    {
        return inst.m_PEAK_System_GetID(systemHandle, id, idSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetDisplayName(PEAK_SYSTEM_HANDLE systemHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetDisplayName)
    {
        return inst.m_PEAK_System_GetDisplayName(systemHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetVendorName(PEAK_SYSTEM_HANDLE systemHandle, char * vendorName, size_t * vendorNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetVendorName)
    {
        return inst.m_PEAK_System_GetVendorName(systemHandle, vendorName, vendorNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetModelName(PEAK_SYSTEM_HANDLE systemHandle, char * modelName, size_t * modelNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetModelName)
    {
        return inst.m_PEAK_System_GetModelName(systemHandle, modelName, modelNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetVersion(PEAK_SYSTEM_HANDLE systemHandle, char * version, size_t * versionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetVersion)
    {
        return inst.m_PEAK_System_GetVersion(systemHandle, version, versionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetTLType(PEAK_SYSTEM_HANDLE systemHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetTLType)
    {
        return inst.m_PEAK_System_GetTLType(systemHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetCTIFileName(PEAK_SYSTEM_HANDLE systemHandle, char * ctiFileName, size_t * ctiFileNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetCTIFileName)
    {
        return inst.m_PEAK_System_GetCTIFileName(systemHandle, ctiFileName, ctiFileNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetCTIFullPath(PEAK_SYSTEM_HANDLE systemHandle, char * ctiFullPath, size_t * ctiFullPathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetCTIFullPath)
    {
        return inst.m_PEAK_System_GetCTIFullPath(systemHandle, ctiFullPath, ctiFullPathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetGenTLVersionMajor(PEAK_SYSTEM_HANDLE systemHandle, uint32_t * gentlVersionMajor)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetGenTLVersionMajor)
    {
        return inst.m_PEAK_System_GetGenTLVersionMajor(systemHandle, gentlVersionMajor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetGenTLVersionMinor(PEAK_SYSTEM_HANDLE systemHandle, uint32_t * gentlVersionMinor)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetGenTLVersionMinor)
    {
        return inst.m_PEAK_System_GetGenTLVersionMinor(systemHandle, gentlVersionMinor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetCharacterEncoding(PEAK_SYSTEM_HANDLE systemHandle, PEAK_CHARACTER_ENCODING * characterEncoding)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetCharacterEncoding)
    {
        return inst.m_PEAK_System_GetCharacterEncoding(systemHandle, characterEncoding);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetParentLibrary(PEAK_SYSTEM_HANDLE systemHandle, PEAK_PRODUCER_LIBRARY_HANDLE * producerLibraryHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetParentLibrary)
    {
        return inst.m_PEAK_System_GetParentLibrary(systemHandle, producerLibraryHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_UpdateInterfaces(PEAK_SYSTEM_HANDLE systemHandle, uint64_t timeout_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_UpdateInterfaces)
    {
        return inst.m_PEAK_System_UpdateInterfaces(systemHandle, timeout_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetNumInterfaces(PEAK_SYSTEM_HANDLE systemHandle, size_t * numInterfaces)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetNumInterfaces)
    {
        return inst.m_PEAK_System_GetNumInterfaces(systemHandle, numInterfaces);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_GetInterface(PEAK_SYSTEM_HANDLE systemHandle, size_t index, PEAK_INTERFACE_DESCRIPTOR_HANDLE * interfaceDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_GetInterface)
    {
        return inst.m_PEAK_System_GetInterface(systemHandle, index, interfaceDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_RegisterInterfaceFoundCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK callback, void * callbackContext, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_RegisterInterfaceFoundCallback)
    {
        return inst.m_PEAK_System_RegisterInterfaceFoundCallback(systemHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_UnregisterInterfaceFoundCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_FOUND_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_UnregisterInterfaceFoundCallback)
    {
        return inst.m_PEAK_System_UnregisterInterfaceFoundCallback(systemHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_RegisterInterfaceLostCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK callback, void * callbackContext, PEAK_INTERFACE_LOST_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_RegisterInterfaceLostCallback)
    {
        return inst.m_PEAK_System_RegisterInterfaceLostCallback(systemHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_UnregisterInterfaceLostCallback(PEAK_SYSTEM_HANDLE systemHandle, PEAK_INTERFACE_LOST_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_UnregisterInterfaceLostCallback)
    {
        return inst.m_PEAK_System_UnregisterInterfaceLostCallback(systemHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_System_Destruct(PEAK_SYSTEM_HANDLE systemHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_System_Destruct)
    {
        return inst.m_PEAK_System_Destruct(systemHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_ToModuleDescriptor(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_ToModuleDescriptor)
    {
        return inst.m_PEAK_InterfaceDescriptor_ToModuleDescriptor(interfaceDescriptorHandle, moduleDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_GetKey(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_GetKey)
    {
        return inst.m_PEAK_InterfaceDescriptor_GetKey(interfaceDescriptorHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_GetInfo(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_GetInfo)
    {
        return inst.m_PEAK_InterfaceDescriptor_GetInfo(interfaceDescriptorHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_GetDisplayName(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_GetDisplayName)
    {
        return inst.m_PEAK_InterfaceDescriptor_GetDisplayName(interfaceDescriptorHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_GetTLType(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_GetTLType)
    {
        return inst.m_PEAK_InterfaceDescriptor_GetTLType(interfaceDescriptorHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_GetParentSystem(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_SYSTEM_HANDLE * systemHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_GetParentSystem)
    {
        return inst.m_PEAK_InterfaceDescriptor_GetParentSystem(interfaceDescriptorHandle, systemHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_InterfaceDescriptor_OpenInterface(PEAK_INTERFACE_DESCRIPTOR_HANDLE interfaceDescriptorHandle, PEAK_INTERFACE_HANDLE * interfaceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_InterfaceDescriptor_OpenInterface)
    {
        return inst.m_PEAK_InterfaceDescriptor_OpenInterface(interfaceDescriptorHandle, interfaceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_ToModule(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_MODULE_HANDLE * moduleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_ToModule)
    {
        return inst.m_PEAK_Interface_ToModule(interfaceHandle, moduleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_ToEventSupportingModule(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_ToEventSupportingModule)
    {
        return inst.m_PEAK_Interface_ToEventSupportingModule(interfaceHandle, eventSupportingModuleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetKey(PEAK_INTERFACE_HANDLE interfaceHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetKey)
    {
        return inst.m_PEAK_Interface_GetKey(interfaceHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetInfo(PEAK_INTERFACE_HANDLE interfaceHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetInfo)
    {
        return inst.m_PEAK_Interface_GetInfo(interfaceHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetID(PEAK_INTERFACE_HANDLE interfaceHandle, char * id, size_t * idSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetID)
    {
        return inst.m_PEAK_Interface_GetID(interfaceHandle, id, idSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetDisplayName(PEAK_INTERFACE_HANDLE interfaceHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetDisplayName)
    {
        return inst.m_PEAK_Interface_GetDisplayName(interfaceHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetTLType(PEAK_INTERFACE_HANDLE interfaceHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetTLType)
    {
        return inst.m_PEAK_Interface_GetTLType(interfaceHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetParentSystem(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_SYSTEM_HANDLE * systemHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetParentSystem)
    {
        return inst.m_PEAK_Interface_GetParentSystem(interfaceHandle, systemHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_UpdateDevices(PEAK_INTERFACE_HANDLE interfaceHandle, uint64_t timeout_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_UpdateDevices)
    {
        return inst.m_PEAK_Interface_UpdateDevices(interfaceHandle, timeout_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetNumDevices(PEAK_INTERFACE_HANDLE interfaceHandle, size_t * numDevices)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetNumDevices)
    {
        return inst.m_PEAK_Interface_GetNumDevices(interfaceHandle, numDevices);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_GetDevice(PEAK_INTERFACE_HANDLE interfaceHandle, size_t index, PEAK_DEVICE_DESCRIPTOR_HANDLE * deviceDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_GetDevice)
    {
        return inst.m_PEAK_Interface_GetDevice(interfaceHandle, index, deviceDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_RegisterDeviceFoundCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK callback, void * callbackContext, PEAK_DEVICE_FOUND_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_RegisterDeviceFoundCallback)
    {
        return inst.m_PEAK_Interface_RegisterDeviceFoundCallback(interfaceHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_UnregisterDeviceFoundCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_FOUND_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_UnregisterDeviceFoundCallback)
    {
        return inst.m_PEAK_Interface_UnregisterDeviceFoundCallback(interfaceHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_RegisterDeviceLostCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK callback, void * callbackContext, PEAK_DEVICE_LOST_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_RegisterDeviceLostCallback)
    {
        return inst.m_PEAK_Interface_RegisterDeviceLostCallback(interfaceHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_UnregisterDeviceLostCallback(PEAK_INTERFACE_HANDLE interfaceHandle, PEAK_DEVICE_LOST_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_UnregisterDeviceLostCallback)
    {
        return inst.m_PEAK_Interface_UnregisterDeviceLostCallback(interfaceHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Interface_Destruct(PEAK_INTERFACE_HANDLE interfaceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Interface_Destruct)
    {
        return inst.m_PEAK_Interface_Destruct(interfaceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_ToModuleDescriptor(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_ToModuleDescriptor)
    {
        return inst.m_PEAK_DeviceDescriptor_ToModuleDescriptor(deviceDescriptorHandle, moduleDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetKey(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetKey)
    {
        return inst.m_PEAK_DeviceDescriptor_GetKey(deviceDescriptorHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetInfo(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetInfo)
    {
        return inst.m_PEAK_DeviceDescriptor_GetInfo(deviceDescriptorHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetDisplayName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetDisplayName)
    {
        return inst.m_PEAK_DeviceDescriptor_GetDisplayName(deviceDescriptorHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetVendorName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * vendorName, size_t * vendorNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetVendorName)
    {
        return inst.m_PEAK_DeviceDescriptor_GetVendorName(deviceDescriptorHandle, vendorName, vendorNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetModelName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * modelName, size_t * modelNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetModelName)
    {
        return inst.m_PEAK_DeviceDescriptor_GetModelName(deviceDescriptorHandle, modelName, modelNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetVersion(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * version, size_t * versionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetVersion)
    {
        return inst.m_PEAK_DeviceDescriptor_GetVersion(deviceDescriptorHandle, version, versionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetTLType(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetTLType)
    {
        return inst.m_PEAK_DeviceDescriptor_GetTLType(deviceDescriptorHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetUserDefinedName(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * userDefinedName, size_t * userDefinedNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetUserDefinedName)
    {
        return inst.m_PEAK_DeviceDescriptor_GetUserDefinedName(deviceDescriptorHandle, userDefinedName, userDefinedNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetSerialNumber(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, char * serialNumber, size_t * serialNumberSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetSerialNumber)
    {
        return inst.m_PEAK_DeviceDescriptor_GetSerialNumber(deviceDescriptorHandle, serialNumber, serialNumberSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetAccessStatus(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_STATUS * accessStatus)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetAccessStatus)
    {
        return inst.m_PEAK_DeviceDescriptor_GetAccessStatus(deviceDescriptorHandle, accessStatus);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetTimestampTickFrequency(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t * timestampTickFrequency)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetTimestampTickFrequency)
    {
        return inst.m_PEAK_DeviceDescriptor_GetTimestampTickFrequency(deviceDescriptorHandle, timestampTickFrequency);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetIsOpenableExclusive(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_BOOL8 * isOpenable)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetIsOpenableExclusive)
    {
        return inst.m_PEAK_DeviceDescriptor_GetIsOpenableExclusive(deviceDescriptorHandle, isOpenable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetIsOpenable(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_BOOL8 * isOpenable)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetIsOpenable)
    {
        return inst.m_PEAK_DeviceDescriptor_GetIsOpenable(deviceDescriptorHandle, accessType, isOpenable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_OpenDevice(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_ACCESS_TYPE accessType, PEAK_DEVICE_HANDLE * deviceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_OpenDevice)
    {
        return inst.m_PEAK_DeviceDescriptor_OpenDevice(deviceDescriptorHandle, accessType, deviceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetParentInterface(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_INTERFACE_HANDLE * interfaceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetParentInterface)
    {
        return inst.m_PEAK_DeviceDescriptor_GetParentInterface(deviceDescriptorHandle, interfaceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_GetMonitoringUpdateInterval(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t * monitoringUpdateInterval_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval)
    {
        return inst.m_PEAK_DeviceDescriptor_GetMonitoringUpdateInterval(deviceDescriptorHandle, monitoringUpdateInterval_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_SetMonitoringUpdateInterval(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, uint64_t monitoringUpdateInterval_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval)
    {
        return inst.m_PEAK_DeviceDescriptor_SetMonitoringUpdateInterval(deviceDescriptorHandle, monitoringUpdateInterval_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_IsInformationRoleMonitored(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole, PEAK_BOOL8 * isInformationRoleMonitored)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_IsInformationRoleMonitored)
    {
        return inst.m_PEAK_DeviceDescriptor_IsInformationRoleMonitored(deviceDescriptorHandle, informationRole, isInformationRoleMonitored);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_AddInformationRoleToMonitoring(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring)
    {
        return inst.m_PEAK_DeviceDescriptor_AddInformationRoleToMonitoring(deviceDescriptorHandle, informationRole);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_INFORMATION_ROLE informationRole)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring)
    {
        return inst.m_PEAK_DeviceDescriptor_RemoveInformationRoleFromMonitoring(deviceDescriptorHandle, informationRole);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_RegisterInformationChangedCallback(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK callback, void * callbackContext, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_RegisterInformationChangedCallback)
    {
        return inst.m_PEAK_DeviceDescriptor_RegisterInformationChangedCallback(deviceDescriptorHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DeviceDescriptor_UnregisterInformationChangedCallback(PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_DEVICE_DESCRIPTOR_INFORMATION_CHANGED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback)
    {
        return inst.m_PEAK_DeviceDescriptor_UnregisterInformationChangedCallback(deviceDescriptorHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_ToModule(PEAK_DEVICE_HANDLE deviceHandle, PEAK_MODULE_HANDLE * moduleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_ToModule)
    {
        return inst.m_PEAK_Device_ToModule(deviceHandle, moduleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_ToEventSupportingModule(PEAK_DEVICE_HANDLE deviceHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_ToEventSupportingModule)
    {
        return inst.m_PEAK_Device_ToEventSupportingModule(deviceHandle, eventSupportingModuleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetKey(PEAK_DEVICE_HANDLE deviceHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetKey)
    {
        return inst.m_PEAK_Device_GetKey(deviceHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetInfo(PEAK_DEVICE_HANDLE deviceHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetInfo)
    {
        return inst.m_PEAK_Device_GetInfo(deviceHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetID(PEAK_DEVICE_HANDLE deviceHandle, char * id, size_t * idSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetID)
    {
        return inst.m_PEAK_Device_GetID(deviceHandle, id, idSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetDisplayName(PEAK_DEVICE_HANDLE deviceHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetDisplayName)
    {
        return inst.m_PEAK_Device_GetDisplayName(deviceHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetVendorName(PEAK_DEVICE_HANDLE deviceHandle, char * vendorName, size_t * vendorNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetVendorName)
    {
        return inst.m_PEAK_Device_GetVendorName(deviceHandle, vendorName, vendorNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetModelName(PEAK_DEVICE_HANDLE deviceHandle, char * modelName, size_t * modelNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetModelName)
    {
        return inst.m_PEAK_Device_GetModelName(deviceHandle, modelName, modelNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetVersion(PEAK_DEVICE_HANDLE deviceHandle, char * version, size_t * versionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetVersion)
    {
        return inst.m_PEAK_Device_GetVersion(deviceHandle, version, versionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetTLType(PEAK_DEVICE_HANDLE deviceHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetTLType)
    {
        return inst.m_PEAK_Device_GetTLType(deviceHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetUserDefinedName(PEAK_DEVICE_HANDLE deviceHandle, char * userDefinedName, size_t * userDefinedNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetUserDefinedName)
    {
        return inst.m_PEAK_Device_GetUserDefinedName(deviceHandle, userDefinedName, userDefinedNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetSerialNumber(PEAK_DEVICE_HANDLE deviceHandle, char * serialNumber, size_t * serialNumberSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetSerialNumber)
    {
        return inst.m_PEAK_Device_GetSerialNumber(deviceHandle, serialNumber, serialNumberSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetAccessStatus(PEAK_DEVICE_HANDLE deviceHandle, PEAK_DEVICE_ACCESS_STATUS * accessStatus)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetAccessStatus)
    {
        return inst.m_PEAK_Device_GetAccessStatus(deviceHandle, accessStatus);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetTimestampTickFrequency(PEAK_DEVICE_HANDLE deviceHandle, uint64_t * timestampTickFrequency)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetTimestampTickFrequency)
    {
        return inst.m_PEAK_Device_GetTimestampTickFrequency(deviceHandle, timestampTickFrequency);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetParentInterface(PEAK_DEVICE_HANDLE deviceHandle, PEAK_INTERFACE_HANDLE * interfaceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetParentInterface)
    {
        return inst.m_PEAK_Device_GetParentInterface(deviceHandle, interfaceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetRemoteDevice(PEAK_DEVICE_HANDLE deviceHandle, PEAK_REMOTE_DEVICE_HANDLE * remoteDeviceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetRemoteDevice)
    {
        return inst.m_PEAK_Device_GetRemoteDevice(deviceHandle, remoteDeviceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetNumDataStreams(PEAK_DEVICE_HANDLE deviceHandle, size_t * numDataStreams)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetNumDataStreams)
    {
        return inst.m_PEAK_Device_GetNumDataStreams(deviceHandle, numDataStreams);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_GetDataStream(PEAK_DEVICE_HANDLE deviceHandle, size_t index, PEAK_DATA_STREAM_DESCRIPTOR_HANDLE * dataStreamDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_GetDataStream)
    {
        return inst.m_PEAK_Device_GetDataStream(deviceHandle, index, dataStreamDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Device_Destruct(PEAK_DEVICE_HANDLE deviceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Device_Destruct)
    {
        return inst.m_PEAK_Device_Destruct(deviceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RemoteDevice_ToModule(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_MODULE_HANDLE * moduleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_RemoteDevice_ToModule)
    {
        return inst.m_PEAK_RemoteDevice_ToModule(remoteDeviceHandle, moduleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RemoteDevice_GetLocalDevice(PEAK_REMOTE_DEVICE_HANDLE remoteDeviceHandle, PEAK_DEVICE_HANDLE * deviceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_RemoteDevice_GetLocalDevice)
    {
        return inst.m_PEAK_RemoteDevice_GetLocalDevice(remoteDeviceHandle, deviceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStreamDescriptor_ToModuleDescriptor(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_MODULE_DESCRIPTOR_HANDLE * moduleDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStreamDescriptor_ToModuleDescriptor)
    {
        return inst.m_PEAK_DataStreamDescriptor_ToModuleDescriptor(dataStreamDescriptorHandle, moduleDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStreamDescriptor_GetKey(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStreamDescriptor_GetKey)
    {
        return inst.m_PEAK_DataStreamDescriptor_GetKey(dataStreamDescriptorHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStreamDescriptor_GetParentDevice(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DEVICE_HANDLE * deviceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStreamDescriptor_GetParentDevice)
    {
        return inst.m_PEAK_DataStreamDescriptor_GetParentDevice(dataStreamDescriptorHandle, deviceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStreamDescriptor_OpenDataStream(PEAK_DATA_STREAM_DESCRIPTOR_HANDLE dataStreamDescriptorHandle, PEAK_DATA_STREAM_HANDLE * dataStreamHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStreamDescriptor_OpenDataStream)
    {
        return inst.m_PEAK_DataStreamDescriptor_OpenDataStream(dataStreamDescriptorHandle, dataStreamHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_ToModule(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_MODULE_HANDLE * moduleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_ToModule)
    {
        return inst.m_PEAK_DataStream_ToModule(dataStreamHandle, moduleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_ToEventSupportingModule(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_ToEventSupportingModule)
    {
        return inst.m_PEAK_DataStream_ToEventSupportingModule(dataStreamHandle, eventSupportingModuleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetKey(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * key, size_t * keySize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetKey)
    {
        return inst.m_PEAK_DataStream_GetKey(dataStreamHandle, key, keySize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetInfo(PEAK_DATA_STREAM_HANDLE dataStreamHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetInfo)
    {
        return inst.m_PEAK_DataStream_GetInfo(dataStreamHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetID(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * id, size_t * idSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetID)
    {
        return inst.m_PEAK_DataStream_GetID(dataStreamHandle, id, idSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetTLType(PEAK_DATA_STREAM_HANDLE dataStreamHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetTLType)
    {
        return inst.m_PEAK_DataStream_GetTLType(dataStreamHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumBuffersAnnouncedMinRequired(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAnnouncedMinRequired)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired)
    {
        return inst.m_PEAK_DataStream_GetNumBuffersAnnouncedMinRequired(dataStreamHandle, numBuffersAnnouncedMinRequired);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumBuffersAnnounced(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAnnounced)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumBuffersAnnounced)
    {
        return inst.m_PEAK_DataStream_GetNumBuffersAnnounced(dataStreamHandle, numBuffersAnnounced);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumBuffersQueued(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersQueued)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumBuffersQueued)
    {
        return inst.m_PEAK_DataStream_GetNumBuffersQueued(dataStreamHandle, numBuffersQueued);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumBuffersAwaitDelivery(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numBuffersAwaitDelivery)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumBuffersAwaitDelivery)
    {
        return inst.m_PEAK_DataStream_GetNumBuffersAwaitDelivery(dataStreamHandle, numBuffersAwaitDelivery);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumBuffersDelivered(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numBuffersDelivered)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumBuffersDelivered)
    {
        return inst.m_PEAK_DataStream_GetNumBuffersDelivered(dataStreamHandle, numBuffersDelivered);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumBuffersStarted(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numBuffersStarted)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumBuffersStarted)
    {
        return inst.m_PEAK_DataStream_GetNumBuffersStarted(dataStreamHandle, numBuffersStarted);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumUnderruns(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t * numUnderruns)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumUnderruns)
    {
        return inst.m_PEAK_DataStream_GetNumUnderruns(dataStreamHandle, numUnderruns);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetNumChunksPerBufferMax(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * numChunksPerBufferMax)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetNumChunksPerBufferMax)
    {
        return inst.m_PEAK_DataStream_GetNumChunksPerBufferMax(dataStreamHandle, numChunksPerBufferMax);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetBufferAlignment(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * bufferAlignment)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetBufferAlignment)
    {
        return inst.m_PEAK_DataStream_GetBufferAlignment(dataStreamHandle, bufferAlignment);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetPayloadSize(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t * payloadSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetPayloadSize)
    {
        return inst.m_PEAK_DataStream_GetPayloadSize(dataStreamHandle, payloadSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetDefinesPayloadSize(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8 * definesPayloadSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetDefinesPayloadSize)
    {
        return inst.m_PEAK_DataStream_GetDefinesPayloadSize(dataStreamHandle, definesPayloadSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetIsGrabbing(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BOOL8 * isGrabbing)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetIsGrabbing)
    {
        return inst.m_PEAK_DataStream_GetIsGrabbing(dataStreamHandle, isGrabbing);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_GetParentDevice(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DEVICE_HANDLE * deviceHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_GetParentDevice)
    {
        return inst.m_PEAK_DataStream_GetParentDevice(dataStreamHandle, deviceHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_AnnounceBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, void * buffer, size_t bufferSize, void * userPtr, PEAK_BUFFER_REVOCATION_CALLBACK revocationCallback, void * callbackContext, PEAK_BUFFER_HANDLE * bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_AnnounceBuffer)
    {
        return inst.m_PEAK_DataStream_AnnounceBuffer(dataStreamHandle, buffer, bufferSize, userPtr, revocationCallback, callbackContext, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_AllocAndAnnounceBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, size_t bufferSize, void * userPtr, PEAK_BUFFER_HANDLE * bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_AllocAndAnnounceBuffer)
    {
        return inst.m_PEAK_DataStream_AllocAndAnnounceBuffer(dataStreamHandle, bufferSize, userPtr, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_QueueBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_QueueBuffer)
    {
        return inst.m_PEAK_DataStream_QueueBuffer(dataStreamHandle, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_RevokeBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_BUFFER_HANDLE bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_RevokeBuffer)
    {
        return inst.m_PEAK_DataStream_RevokeBuffer(dataStreamHandle, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_WaitForFinishedBuffer(PEAK_DATA_STREAM_HANDLE dataStreamHandle, uint64_t timeout_ms, PEAK_BUFFER_HANDLE * bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_WaitForFinishedBuffer)
    {
        return inst.m_PEAK_DataStream_WaitForFinishedBuffer(dataStreamHandle, timeout_ms, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_KillWait(PEAK_DATA_STREAM_HANDLE dataStreamHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_KillWait)
    {
        return inst.m_PEAK_DataStream_KillWait(dataStreamHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_Flush(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_DATA_STREAM_FLUSH_MODE flushMode)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_Flush)
    {
        return inst.m_PEAK_DataStream_Flush(dataStreamHandle, flushMode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_StartAcquisitionInfinite(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_StartAcquisitionInfinite)
    {
        return inst.m_PEAK_DataStream_StartAcquisitionInfinite(dataStreamHandle, startMode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_StartAcquisition(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_START_MODE startMode, uint64_t numToAcquire)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_StartAcquisition)
    {
        return inst.m_PEAK_DataStream_StartAcquisition(dataStreamHandle, startMode, numToAcquire);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_StopAcquisition(PEAK_DATA_STREAM_HANDLE dataStreamHandle, PEAK_ACQUISITION_STOP_MODE stopMode)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_StopAcquisition)
    {
        return inst.m_PEAK_DataStream_StopAcquisition(dataStreamHandle, stopMode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_DataStream_Destruct(PEAK_DATA_STREAM_HANDLE dataStreamHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_DataStream_Destruct)
    {
        return inst.m_PEAK_DataStream_Destruct(dataStreamHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_ToModule(PEAK_BUFFER_HANDLE bufferHandle, PEAK_MODULE_HANDLE * moduleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_ToModule)
    {
        return inst.m_PEAK_Buffer_ToModule(bufferHandle, moduleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_ToEventSupportingModule(PEAK_BUFFER_HANDLE bufferHandle, PEAK_EVENT_SUPPORTING_MODULE_HANDLE * eventSupportingModuleHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_ToEventSupportingModule)
    {
        return inst.m_PEAK_Buffer_ToEventSupportingModule(bufferHandle, eventSupportingModuleHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetInfo(PEAK_BUFFER_HANDLE bufferHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetInfo)
    {
        return inst.m_PEAK_Buffer_GetInfo(bufferHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetTLType(PEAK_BUFFER_HANDLE bufferHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetTLType)
    {
        return inst.m_PEAK_Buffer_GetTLType(bufferHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetBasePtr(PEAK_BUFFER_HANDLE bufferHandle, void * * basePtr)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetBasePtr)
    {
        return inst.m_PEAK_Buffer_GetBasePtr(bufferHandle, basePtr);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * size)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetSize)
    {
        return inst.m_PEAK_Buffer_GetSize(bufferHandle, size);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetUserPtr(PEAK_BUFFER_HANDLE bufferHandle, void * * userPtr)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetUserPtr)
    {
        return inst.m_PEAK_Buffer_GetUserPtr(bufferHandle, userPtr);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetPayloadType(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BUFFER_PAYLOAD_TYPE * payloadType)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetPayloadType)
    {
        return inst.m_PEAK_Buffer_GetPayloadType(bufferHandle, payloadType);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetPixelFormat(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * pixelFormat)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetPixelFormat)
    {
        return inst.m_PEAK_Buffer_GetPixelFormat(bufferHandle, pixelFormat);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetPixelFormatNamespace(PEAK_BUFFER_HANDLE bufferHandle, PEAK_PIXEL_FORMAT_NAMESPACE * pixelFormatNamespace)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetPixelFormatNamespace)
    {
        return inst.m_PEAK_Buffer_GetPixelFormatNamespace(bufferHandle, pixelFormatNamespace);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetPixelEndianness(PEAK_BUFFER_HANDLE bufferHandle, PEAK_ENDIANNESS * pixelEndianness)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetPixelEndianness)
    {
        return inst.m_PEAK_Buffer_GetPixelEndianness(bufferHandle, pixelEndianness);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetExpectedDataSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * expectedDataSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetExpectedDataSize)
    {
        return inst.m_PEAK_Buffer_GetExpectedDataSize(bufferHandle, expectedDataSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetDeliveredDataSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredDataSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetDeliveredDataSize)
    {
        return inst.m_PEAK_Buffer_GetDeliveredDataSize(bufferHandle, deliveredDataSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetFrameID(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * frameId)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetFrameID)
    {
        return inst.m_PEAK_Buffer_GetFrameID(bufferHandle, frameId);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetImageOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t * imageOffset)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetImageOffset)
    {
        return inst.m_PEAK_Buffer_GetImageOffset(bufferHandle, imageOffset);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetDeliveredImageHeight(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredImageHeight)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetDeliveredImageHeight)
    {
        return inst.m_PEAK_Buffer_GetDeliveredImageHeight(bufferHandle, deliveredImageHeight);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetDeliveredChunkPayloadSize(PEAK_BUFFER_HANDLE bufferHandle, size_t * deliveredChunkPayloadSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetDeliveredChunkPayloadSize)
    {
        return inst.m_PEAK_Buffer_GetDeliveredChunkPayloadSize(bufferHandle, deliveredChunkPayloadSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetChunkLayoutID(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * chunkLayoutId)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetChunkLayoutID)
    {
        return inst.m_PEAK_Buffer_GetChunkLayoutID(bufferHandle, chunkLayoutId);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetFileName(PEAK_BUFFER_HANDLE bufferHandle, char * fileName, size_t * fileNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetFileName)
    {
        return inst.m_PEAK_Buffer_GetFileName(bufferHandle, fileName, fileNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetWidth(PEAK_BUFFER_HANDLE bufferHandle, size_t * width)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetWidth)
    {
        return inst.m_PEAK_Buffer_GetWidth(bufferHandle, width);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetHeight(PEAK_BUFFER_HANDLE bufferHandle, size_t * height)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetHeight)
    {
        return inst.m_PEAK_Buffer_GetHeight(bufferHandle, height);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetXOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t * xOffset)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetXOffset)
    {
        return inst.m_PEAK_Buffer_GetXOffset(bufferHandle, xOffset);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetYOffset(PEAK_BUFFER_HANDLE bufferHandle, size_t * yOffset)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetYOffset)
    {
        return inst.m_PEAK_Buffer_GetYOffset(bufferHandle, yOffset);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetXPadding(PEAK_BUFFER_HANDLE bufferHandle, size_t * xPadding)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetXPadding)
    {
        return inst.m_PEAK_Buffer_GetXPadding(bufferHandle, xPadding);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetYPadding(PEAK_BUFFER_HANDLE bufferHandle, size_t * yPadding)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetYPadding)
    {
        return inst.m_PEAK_Buffer_GetYPadding(bufferHandle, yPadding);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetTimestamp_ticks(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * timestamp_ticks)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetTimestamp_ticks)
    {
        return inst.m_PEAK_Buffer_GetTimestamp_ticks(bufferHandle, timestamp_ticks);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetTimestamp_ns(PEAK_BUFFER_HANDLE bufferHandle, uint64_t * timestamp_ns)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetTimestamp_ns)
    {
        return inst.m_PEAK_Buffer_GetTimestamp_ns(bufferHandle, timestamp_ns);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetIsQueued(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isQueued)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetIsQueued)
    {
        return inst.m_PEAK_Buffer_GetIsQueued(bufferHandle, isQueued);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetIsAcquiring(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isAcquiring)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetIsAcquiring)
    {
        return inst.m_PEAK_Buffer_GetIsAcquiring(bufferHandle, isAcquiring);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetIsIncomplete(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * isIncomplete)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetIsIncomplete)
    {
        return inst.m_PEAK_Buffer_GetIsIncomplete(bufferHandle, isIncomplete);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetHasNewData(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasNewData)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetHasNewData)
    {
        return inst.m_PEAK_Buffer_GetHasNewData(bufferHandle, hasNewData);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetHasImage(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasImage)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetHasImage)
    {
        return inst.m_PEAK_Buffer_GetHasImage(bufferHandle, hasImage);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetHasChunks(PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasChunks)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetHasChunks)
    {
        return inst.m_PEAK_Buffer_GetHasChunks(bufferHandle, hasChunks);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_UpdateChunks(PEAK_BUFFER_HANDLE bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_UpdateChunks)
    {
        return inst.m_PEAK_Buffer_UpdateChunks(bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetNumChunks(PEAK_BUFFER_HANDLE bufferHandle, size_t * numChunks)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetNumChunks)
    {
        return inst.m_PEAK_Buffer_GetNumChunks(bufferHandle, numChunks);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetChunk(PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_CHUNK_HANDLE * bufferChunkHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetChunk)
    {
        return inst.m_PEAK_Buffer_GetChunk(bufferHandle, index, bufferChunkHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_UpdateParts(PEAK_BUFFER_HANDLE bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_UpdateParts)
    {
        return inst.m_PEAK_Buffer_UpdateParts(bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetNumParts(PEAK_BUFFER_HANDLE bufferHandle, size_t * numParts)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetNumParts)
    {
        return inst.m_PEAK_Buffer_GetNumParts(bufferHandle, numParts);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Buffer_GetPart(PEAK_BUFFER_HANDLE bufferHandle, size_t index, PEAK_BUFFER_PART_HANDLE * bufferPartHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Buffer_GetPart)
    {
        return inst.m_PEAK_Buffer_GetPart(bufferHandle, index, bufferPartHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferChunk_GetID(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, uint64_t * id)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferChunk_GetID)
    {
        return inst.m_PEAK_BufferChunk_GetID(bufferChunkHandle, id);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferChunk_GetBasePtr(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, void * * basePtr)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferChunk_GetBasePtr)
    {
        return inst.m_PEAK_BufferChunk_GetBasePtr(bufferChunkHandle, basePtr);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferChunk_GetSize(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, size_t * size)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferChunk_GetSize)
    {
        return inst.m_PEAK_BufferChunk_GetSize(bufferChunkHandle, size);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferChunk_GetParentBuffer(PEAK_BUFFER_CHUNK_HANDLE bufferChunkHandle, PEAK_BUFFER_HANDLE * bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferChunk_GetParentBuffer)
    {
        return inst.m_PEAK_BufferChunk_GetParentBuffer(bufferChunkHandle, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetInfo(PEAK_BUFFER_PART_HANDLE bufferPartHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetInfo)
    {
        return inst.m_PEAK_BufferPart_GetInfo(bufferPartHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetSourceID(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * sourceId)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetSourceID)
    {
        return inst.m_PEAK_BufferPart_GetSourceID(bufferPartHandle, sourceId);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetBasePtr(PEAK_BUFFER_PART_HANDLE bufferPartHandle, void * * basePtr)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetBasePtr)
    {
        return inst.m_PEAK_BufferPart_GetBasePtr(bufferPartHandle, basePtr);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetSize(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * size)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetSize)
    {
        return inst.m_PEAK_BufferPart_GetSize(bufferPartHandle, size);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetType(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_PART_TYPE * type)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetType)
    {
        return inst.m_PEAK_BufferPart_GetType(bufferPartHandle, type);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetFormat(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * format)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetFormat)
    {
        return inst.m_PEAK_BufferPart_GetFormat(bufferPartHandle, format);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetFormatNamespace(PEAK_BUFFER_PART_HANDLE bufferPartHandle, uint64_t * formatNamespace)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetFormatNamespace)
    {
        return inst.m_PEAK_BufferPart_GetFormatNamespace(bufferPartHandle, formatNamespace);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetWidth(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * width)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetWidth)
    {
        return inst.m_PEAK_BufferPart_GetWidth(bufferPartHandle, width);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetHeight(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * height)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetHeight)
    {
        return inst.m_PEAK_BufferPart_GetHeight(bufferPartHandle, height);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetXOffset(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * xOffset)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetXOffset)
    {
        return inst.m_PEAK_BufferPart_GetXOffset(bufferPartHandle, xOffset);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetYOffset(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * yOffset)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetYOffset)
    {
        return inst.m_PEAK_BufferPart_GetYOffset(bufferPartHandle, yOffset);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetXPadding(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * xPadding)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetXPadding)
    {
        return inst.m_PEAK_BufferPart_GetXPadding(bufferPartHandle, xPadding);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetDeliveredImageHeight(PEAK_BUFFER_PART_HANDLE bufferPartHandle, size_t * deliveredImageHeight)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetDeliveredImageHeight)
    {
        return inst.m_PEAK_BufferPart_GetDeliveredImageHeight(bufferPartHandle, deliveredImageHeight);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BufferPart_GetParentBuffer(PEAK_BUFFER_PART_HANDLE bufferPartHandle, PEAK_BUFFER_HANDLE * bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_BufferPart_GetParentBuffer)
    {
        return inst.m_PEAK_BufferPart_GetParentBuffer(bufferPartHandle, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_ModuleDescriptor_GetID(PEAK_MODULE_DESCRIPTOR_HANDLE moduleDescriptorHandle, char * id, size_t * idSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_ModuleDescriptor_GetID)
    {
        return inst.m_PEAK_ModuleDescriptor_GetID(moduleDescriptorHandle, id, idSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Module_GetNumNodeMaps(PEAK_MODULE_HANDLE moduleHandle, size_t * numNodeMaps)
{
    auto& inst = instance();
    if(inst.m_PEAK_Module_GetNumNodeMaps)
    {
        return inst.m_PEAK_Module_GetNumNodeMaps(moduleHandle, numNodeMaps);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Module_GetNodeMap(PEAK_MODULE_HANDLE moduleHandle, size_t index, PEAK_NODE_MAP_HANDLE * nodeMapHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Module_GetNodeMap)
    {
        return inst.m_PEAK_Module_GetNodeMap(moduleHandle, index, nodeMapHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Module_GetPort(PEAK_MODULE_HANDLE moduleHandle, PEAK_PORT_HANDLE * portHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Module_GetPort)
    {
        return inst.m_PEAK_Module_GetPort(moduleHandle, portHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_GetHasNode(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * nodeName, size_t nodeNameSize, PEAK_BOOL8 * hasNode)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_GetHasNode)
    {
        return inst.m_PEAK_NodeMap_GetHasNode(nodeMapHandle, nodeName, nodeNameSize, hasNode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_FindNode(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * nodeName, size_t nodeNameSize, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_FindNode)
    {
        return inst.m_PEAK_NodeMap_FindNode(nodeMapHandle, nodeName, nodeNameSize, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_InvalidateNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_InvalidateNodes)
    {
        return inst.m_PEAK_NodeMap_InvalidateNodes(nodeMapHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_PollNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, int64_t elapsedTime_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_PollNodes)
    {
        return inst.m_PEAK_NodeMap_PollNodes(nodeMapHandle, elapsedTime_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_GetNumNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t * numNodes)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_GetNumNodes)
    {
        return inst.m_PEAK_NodeMap_GetNumNodes(nodeMapHandle, numNodes);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_GetNode(PEAK_NODE_MAP_HANDLE nodeMapHandle, size_t index, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_GetNode)
    {
        return inst.m_PEAK_NodeMap_GetNode(nodeMapHandle, index, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_GetHasBufferSupportedChunks(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle, PEAK_BOOL8 * hasSupportedChunks)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_GetHasBufferSupportedChunks)
    {
        return inst.m_PEAK_NodeMap_GetHasBufferSupportedChunks(nodeMapHandle, bufferHandle, hasSupportedChunks);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_UpdateChunkNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_BUFFER_HANDLE bufferHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_UpdateChunkNodes)
    {
        return inst.m_PEAK_NodeMap_UpdateChunkNodes(nodeMapHandle, bufferHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_GetHasEventSupportedData(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle, PEAK_BOOL8 * hasSupportedData)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_GetHasEventSupportedData)
    {
        return inst.m_PEAK_NodeMap_GetHasEventSupportedData(nodeMapHandle, eventHandle, hasSupportedData);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_UpdateEventNodes(PEAK_NODE_MAP_HANDLE nodeMapHandle, PEAK_EVENT_HANDLE eventHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_UpdateEventNodes)
    {
        return inst.m_PEAK_NodeMap_UpdateEventNodes(nodeMapHandle, eventHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_StoreToFile(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_StoreToFile)
    {
        return inst.m_PEAK_NodeMap_StoreToFile(nodeMapHandle, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_LoadFromFile(PEAK_NODE_MAP_HANDLE nodeMapHandle, const char * filePath, size_t filePathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_LoadFromFile)
    {
        return inst.m_PEAK_NodeMap_LoadFromFile(nodeMapHandle, filePath, filePathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_Lock(PEAK_NODE_MAP_HANDLE nodeMapHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_Lock)
    {
        return inst.m_PEAK_NodeMap_Lock(nodeMapHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_NodeMap_Unlock(PEAK_NODE_MAP_HANDLE nodeMapHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_NodeMap_Unlock)
    {
        return inst.m_PEAK_NodeMap_Unlock(nodeMapHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToIntegerNode(PEAK_NODE_HANDLE nodeHandle, PEAK_INTEGER_NODE_HANDLE * integerNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToIntegerNode)
    {
        return inst.m_PEAK_Node_ToIntegerNode(nodeHandle, integerNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToBooleanNode(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOLEAN_NODE_HANDLE * booleanNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToBooleanNode)
    {
        return inst.m_PEAK_Node_ToBooleanNode(nodeHandle, booleanNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToCommandNode(PEAK_NODE_HANDLE nodeHandle, PEAK_COMMAND_NODE_HANDLE * commandNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToCommandNode)
    {
        return inst.m_PEAK_Node_ToCommandNode(nodeHandle, commandNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToFloatNode(PEAK_NODE_HANDLE nodeHandle, PEAK_FLOAT_NODE_HANDLE * floatNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToFloatNode)
    {
        return inst.m_PEAK_Node_ToFloatNode(nodeHandle, floatNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToStringNode(PEAK_NODE_HANDLE nodeHandle, PEAK_STRING_NODE_HANDLE * stringNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToStringNode)
    {
        return inst.m_PEAK_Node_ToStringNode(nodeHandle, stringNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToRegisterNode(PEAK_NODE_HANDLE nodeHandle, PEAK_REGISTER_NODE_HANDLE * registerNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToRegisterNode)
    {
        return inst.m_PEAK_Node_ToRegisterNode(nodeHandle, registerNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToCategoryNode(PEAK_NODE_HANDLE nodeHandle, PEAK_CATEGORY_NODE_HANDLE * categoryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToCategoryNode)
    {
        return inst.m_PEAK_Node_ToCategoryNode(nodeHandle, categoryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToEnumerationNode(PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_NODE_HANDLE * enumerationNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToEnumerationNode)
    {
        return inst.m_PEAK_Node_ToEnumerationNode(nodeHandle, enumerationNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_ToEnumerationEntryNode(PEAK_NODE_HANDLE nodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_ToEnumerationEntryNode)
    {
        return inst.m_PEAK_Node_ToEnumerationEntryNode(nodeHandle, enumerationEntryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetName(PEAK_NODE_HANDLE nodeHandle, char * name, size_t * nameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetName)
    {
        return inst.m_PEAK_Node_GetName(nodeHandle, name, nameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetDisplayName(PEAK_NODE_HANDLE nodeHandle, char * displayName, size_t * displayNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetDisplayName)
    {
        return inst.m_PEAK_Node_GetDisplayName(nodeHandle, displayName, displayNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetNamespace(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_NAMESPACE * _namespace)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetNamespace)
    {
        return inst.m_PEAK_Node_GetNamespace(nodeHandle, _namespace);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetVisibility(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_VISIBILITY * visibility)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetVisibility)
    {
        return inst.m_PEAK_Node_GetVisibility(nodeHandle, visibility);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetAccessStatus(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_ACCESS_STATUS * accessStatus)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetAccessStatus)
    {
        return inst.m_PEAK_Node_GetAccessStatus(nodeHandle, accessStatus);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetIsCacheable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isCacheable)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetIsCacheable)
    {
        return inst.m_PEAK_Node_GetIsCacheable(nodeHandle, isCacheable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetIsAccessStatusCacheable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isAccessStatusCacheable)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetIsAccessStatusCacheable)
    {
        return inst.m_PEAK_Node_GetIsAccessStatusCacheable(nodeHandle, isAccessStatusCacheable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetIsStreamable(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isStreamable)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetIsStreamable)
    {
        return inst.m_PEAK_Node_GetIsStreamable(nodeHandle, isStreamable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetIsDeprecated(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isDeprecated)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetIsDeprecated)
    {
        return inst.m_PEAK_Node_GetIsDeprecated(nodeHandle, isDeprecated);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetIsFeature(PEAK_NODE_HANDLE nodeHandle, PEAK_BOOL8 * isFeature)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetIsFeature)
    {
        return inst.m_PEAK_Node_GetIsFeature(nodeHandle, isFeature);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetCachingMode(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CACHING_MODE * cachingMode)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetCachingMode)
    {
        return inst.m_PEAK_Node_GetCachingMode(nodeHandle, cachingMode);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetPollingTime(PEAK_NODE_HANDLE nodeHandle, int64_t * pollingTime_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetPollingTime)
    {
        return inst.m_PEAK_Node_GetPollingTime(nodeHandle, pollingTime_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetToolTip(PEAK_NODE_HANDLE nodeHandle, char * toolTip, size_t * toolTipSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetToolTip)
    {
        return inst.m_PEAK_Node_GetToolTip(nodeHandle, toolTip, toolTipSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetDescription(PEAK_NODE_HANDLE nodeHandle, char * description, size_t * descriptionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetDescription)
    {
        return inst.m_PEAK_Node_GetDescription(nodeHandle, description, descriptionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetType(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_TYPE * type)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetType)
    {
        return inst.m_PEAK_Node_GetType(nodeHandle, type);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetParentNodeMap(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_MAP_HANDLE * nodeMapHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetParentNodeMap)
    {
        return inst.m_PEAK_Node_GetParentNodeMap(nodeHandle, nodeMapHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_FindInvalidatedNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * invalidatedNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_FindInvalidatedNode)
    {
        return inst.m_PEAK_Node_FindInvalidatedNode(nodeHandle, name, nameSize, invalidatedNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetNumInvalidatedNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numInvalidatedNodes)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetNumInvalidatedNodes)
    {
        return inst.m_PEAK_Node_GetNumInvalidatedNodes(nodeHandle, numInvalidatedNodes);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetInvalidatedNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * invalidatedNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetInvalidatedNode)
    {
        return inst.m_PEAK_Node_GetInvalidatedNode(nodeHandle, index, invalidatedNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_FindInvalidatingNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * invalidatingNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_FindInvalidatingNode)
    {
        return inst.m_PEAK_Node_FindInvalidatingNode(nodeHandle, name, nameSize, invalidatingNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetNumInvalidatingNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numInvalidatingNodes)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetNumInvalidatingNodes)
    {
        return inst.m_PEAK_Node_GetNumInvalidatingNodes(nodeHandle, numInvalidatingNodes);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetInvalidatingNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * invalidatingNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetInvalidatingNode)
    {
        return inst.m_PEAK_Node_GetInvalidatingNode(nodeHandle, index, invalidatingNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_FindSelectedNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * selectedNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_FindSelectedNode)
    {
        return inst.m_PEAK_Node_FindSelectedNode(nodeHandle, name, nameSize, selectedNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetNumSelectedNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numSelectedNodes)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetNumSelectedNodes)
    {
        return inst.m_PEAK_Node_GetNumSelectedNodes(nodeHandle, numSelectedNodes);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetSelectedNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * selectedNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetSelectedNode)
    {
        return inst.m_PEAK_Node_GetSelectedNode(nodeHandle, index, selectedNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_FindSelectingNode(PEAK_NODE_HANDLE nodeHandle, const char * name, size_t nameSize, PEAK_NODE_HANDLE * selectingNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_FindSelectingNode)
    {
        return inst.m_PEAK_Node_FindSelectingNode(nodeHandle, name, nameSize, selectingNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetNumSelectingNodes(PEAK_NODE_HANDLE nodeHandle, size_t * numSelectingNodes)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetNumSelectingNodes)
    {
        return inst.m_PEAK_Node_GetNumSelectingNodes(nodeHandle, numSelectingNodes);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_GetSelectingNode(PEAK_NODE_HANDLE nodeHandle, size_t index, PEAK_NODE_HANDLE * selectingNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_GetSelectingNode)
    {
        return inst.m_PEAK_Node_GetSelectingNode(nodeHandle, index, selectingNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_RegisterChangedCallback(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK callback, void * callbackContext, PEAK_NODE_CHANGED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_RegisterChangedCallback)
    {
        return inst.m_PEAK_Node_RegisterChangedCallback(nodeHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Node_UnregisterChangedCallback(PEAK_NODE_HANDLE nodeHandle, PEAK_NODE_CHANGED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Node_UnregisterChangedCallback)
    {
        return inst.m_PEAK_Node_UnregisterChangedCallback(nodeHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_ToNode(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_ToNode)
    {
        return inst.m_PEAK_IntegerNode_ToNode(integerNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetMinimum(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * minimum)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetMinimum)
    {
        return inst.m_PEAK_IntegerNode_GetMinimum(integerNodeHandle, minimum);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetMaximum(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * maximum)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetMaximum)
    {
        return inst.m_PEAK_IntegerNode_GetMaximum(integerNodeHandle, maximum);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetIncrement(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * increment)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetIncrement)
    {
        return inst.m_PEAK_IntegerNode_GetIncrement(integerNodeHandle, increment);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetIncrementType(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_INCREMENT_TYPE * incrementType)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetIncrementType)
    {
        return inst.m_PEAK_IntegerNode_GetIncrementType(integerNodeHandle, incrementType);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetValidValues(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t * validValues, size_t * validValuesSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetValidValues)
    {
        return inst.m_PEAK_IntegerNode_GetValidValues(integerNodeHandle, validValues, validValuesSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetRepresentation(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_REPRESENTATION * representation)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetRepresentation)
    {
        return inst.m_PEAK_IntegerNode_GetRepresentation(integerNodeHandle, representation);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetUnit(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, char * unit, size_t * unitSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetUnit)
    {
        return inst.m_PEAK_IntegerNode_GetUnit(integerNodeHandle, unit, unitSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_GetValue(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, int64_t * value)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_GetValue)
    {
        return inst.m_PEAK_IntegerNode_GetValue(integerNodeHandle, cacheUsePolicy, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_IntegerNode_SetValue(PEAK_INTEGER_NODE_HANDLE integerNodeHandle, int64_t value)
{
    auto& inst = instance();
    if(inst.m_PEAK_IntegerNode_SetValue)
    {
        return inst.m_PEAK_IntegerNode_SetValue(integerNodeHandle, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BooleanNode_ToNode(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_BooleanNode_ToNode)
    {
        return inst.m_PEAK_BooleanNode_ToNode(booleanNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BooleanNode_GetValue(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_BOOL8 * value)
{
    auto& inst = instance();
    if(inst.m_PEAK_BooleanNode_GetValue)
    {
        return inst.m_PEAK_BooleanNode_GetValue(booleanNodeHandle, cacheUsePolicy, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_BooleanNode_SetValue(PEAK_BOOLEAN_NODE_HANDLE booleanNodeHandle, PEAK_BOOL8 value)
{
    auto& inst = instance();
    if(inst.m_PEAK_BooleanNode_SetValue)
    {
        return inst.m_PEAK_BooleanNode_SetValue(booleanNodeHandle, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CommandNode_ToNode(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_CommandNode_ToNode)
    {
        return inst.m_PEAK_CommandNode_ToNode(commandNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CommandNode_GetIsDone(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, PEAK_BOOL8 * isDone)
{
    auto& inst = instance();
    if(inst.m_PEAK_CommandNode_GetIsDone)
    {
        return inst.m_PEAK_CommandNode_GetIsDone(commandNodeHandle, isDone);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CommandNode_Execute(PEAK_COMMAND_NODE_HANDLE commandNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_CommandNode_Execute)
    {
        return inst.m_PEAK_CommandNode_Execute(commandNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CommandNode_WaitUntilDoneInfinite(PEAK_COMMAND_NODE_HANDLE commandNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_CommandNode_WaitUntilDoneInfinite)
    {
        return inst.m_PEAK_CommandNode_WaitUntilDoneInfinite(commandNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CommandNode_WaitUntilDone(PEAK_COMMAND_NODE_HANDLE commandNodeHandle, uint64_t waitTimeout_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_CommandNode_WaitUntilDone)
    {
        return inst.m_PEAK_CommandNode_WaitUntilDone(commandNodeHandle, waitTimeout_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_ToNode(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_ToNode)
    {
        return inst.m_PEAK_FloatNode_ToNode(floatNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetMinimum(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * minimum)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetMinimum)
    {
        return inst.m_PEAK_FloatNode_GetMinimum(floatNodeHandle, minimum);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetMaximum(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * maximum)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetMaximum)
    {
        return inst.m_PEAK_FloatNode_GetMaximum(floatNodeHandle, maximum);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetIncrement(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * increment)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetIncrement)
    {
        return inst.m_PEAK_FloatNode_GetIncrement(floatNodeHandle, increment);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetIncrementType(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_INCREMENT_TYPE * incrementType)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetIncrementType)
    {
        return inst.m_PEAK_FloatNode_GetIncrementType(floatNodeHandle, incrementType);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetValidValues(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double * validValues, size_t * validValuesSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetValidValues)
    {
        return inst.m_PEAK_FloatNode_GetValidValues(floatNodeHandle, validValues, validValuesSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetRepresentation(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_REPRESENTATION * representation)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetRepresentation)
    {
        return inst.m_PEAK_FloatNode_GetRepresentation(floatNodeHandle, representation);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetUnit(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, char * unit, size_t * unitSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetUnit)
    {
        return inst.m_PEAK_FloatNode_GetUnit(floatNodeHandle, unit, unitSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetDisplayNotation(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_DISPLAY_NOTATION * displayNotation)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetDisplayNotation)
    {
        return inst.m_PEAK_FloatNode_GetDisplayNotation(floatNodeHandle, displayNotation);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetDisplayPrecision(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, int64_t * displayPrecision)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetDisplayPrecision)
    {
        return inst.m_PEAK_FloatNode_GetDisplayPrecision(floatNodeHandle, displayPrecision);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetHasConstantIncrement(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_BOOL8 * hasConstantIncrement)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetHasConstantIncrement)
    {
        return inst.m_PEAK_FloatNode_GetHasConstantIncrement(floatNodeHandle, hasConstantIncrement);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_GetValue(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, double * value)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_GetValue)
    {
        return inst.m_PEAK_FloatNode_GetValue(floatNodeHandle, cacheUsePolicy, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FloatNode_SetValue(PEAK_FLOAT_NODE_HANDLE floatNodeHandle, double value)
{
    auto& inst = instance();
    if(inst.m_PEAK_FloatNode_SetValue)
    {
        return inst.m_PEAK_FloatNode_SetValue(floatNodeHandle, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_StringNode_ToNode(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_StringNode_ToNode)
    {
        return inst.m_PEAK_StringNode_ToNode(stringNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_StringNode_GetMaximumLength(PEAK_STRING_NODE_HANDLE stringNodeHandle, int64_t * maximumLength)
{
    auto& inst = instance();
    if(inst.m_PEAK_StringNode_GetMaximumLength)
    {
        return inst.m_PEAK_StringNode_GetMaximumLength(stringNodeHandle, maximumLength);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_StringNode_GetValue(PEAK_STRING_NODE_HANDLE stringNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, char * value, size_t * valueSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_StringNode_GetValue)
    {
        return inst.m_PEAK_StringNode_GetValue(stringNodeHandle, cacheUsePolicy, value, valueSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_StringNode_SetValue(PEAK_STRING_NODE_HANDLE stringNodeHandle, const char * value, size_t valueSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_StringNode_SetValue)
    {
        return inst.m_PEAK_StringNode_SetValue(stringNodeHandle, value, valueSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RegisterNode_ToNode(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_RegisterNode_ToNode)
    {
        return inst.m_PEAK_RegisterNode_ToNode(registerNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RegisterNode_GetAddress(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, uint64_t * address)
{
    auto& inst = instance();
    if(inst.m_PEAK_RegisterNode_GetAddress)
    {
        return inst.m_PEAK_RegisterNode_GetAddress(registerNodeHandle, address);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RegisterNode_GetLength(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, size_t * length)
{
    auto& inst = instance();
    if(inst.m_PEAK_RegisterNode_GetLength)
    {
        return inst.m_PEAK_RegisterNode_GetLength(registerNodeHandle, length);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RegisterNode_Read(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, uint8_t * bytesToRead, size_t bytesToReadSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_RegisterNode_Read)
    {
        return inst.m_PEAK_RegisterNode_Read(registerNodeHandle, cacheUsePolicy, bytesToRead, bytesToReadSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_RegisterNode_Write(PEAK_REGISTER_NODE_HANDLE registerNodeHandle, const uint8_t * bytesToWrite, size_t bytesToWriteSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_RegisterNode_Write)
    {
        return inst.m_PEAK_RegisterNode_Write(registerNodeHandle, bytesToWrite, bytesToWriteSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CategoryNode_ToNode(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_CategoryNode_ToNode)
    {
        return inst.m_PEAK_CategoryNode_ToNode(categoryNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CategoryNode_GetNumSubNodes(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t * numSubNodes)
{
    auto& inst = instance();
    if(inst.m_PEAK_CategoryNode_GetNumSubNodes)
    {
        return inst.m_PEAK_CategoryNode_GetNumSubNodes(categoryNodeHandle, numSubNodes);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_CategoryNode_GetSubNode(PEAK_CATEGORY_NODE_HANDLE categoryNodeHandle, size_t index, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_CategoryNode_GetSubNode)
    {
        return inst.m_PEAK_CategoryNode_GetSubNode(categoryNodeHandle, index, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_ToNode(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_ToNode)
    {
        return inst.m_PEAK_EnumerationNode_ToNode(enumerationNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_GetCurrentEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_NODE_CACHE_USE_POLICY cacheUsePolicy, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_GetCurrentEntry)
    {
        return inst.m_PEAK_EnumerationNode_GetCurrentEntry(enumerationNodeHandle, cacheUsePolicy, enumerationEntryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_SetCurrentEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_SetCurrentEntry)
    {
        return inst.m_PEAK_EnumerationNode_SetCurrentEntry(enumerationNodeHandle, enumerationEntryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char * symbolicValue, size_t symbolicValueSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue)
    {
        return inst.m_PEAK_EnumerationNode_SetCurrentEntryBySymbolicValue(enumerationNodeHandle, symbolicValue, symbolicValueSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_SetCurrentEntryByValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_SetCurrentEntryByValue)
    {
        return inst.m_PEAK_EnumerationNode_SetCurrentEntryByValue(enumerationNodeHandle, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_FindEntryBySymbolicValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, const char * symbolicValue, size_t symbolicValueSize, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_FindEntryBySymbolicValue)
    {
        return inst.m_PEAK_EnumerationNode_FindEntryBySymbolicValue(enumerationNodeHandle, symbolicValue, symbolicValueSize, enumerationEntryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_FindEntryByValue(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, int64_t value, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_FindEntryByValue)
    {
        return inst.m_PEAK_EnumerationNode_FindEntryByValue(enumerationNodeHandle, value, enumerationEntryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_GetNumEntries(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t * numEntries)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_GetNumEntries)
    {
        return inst.m_PEAK_EnumerationNode_GetNumEntries(enumerationNodeHandle, numEntries);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationNode_GetEntry(PEAK_ENUMERATION_NODE_HANDLE enumerationNodeHandle, size_t index, PEAK_ENUMERATION_ENTRY_NODE_HANDLE * enumerationEntryNodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationNode_GetEntry)
    {
        return inst.m_PEAK_EnumerationNode_GetEntry(enumerationNodeHandle, index, enumerationEntryNodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationEntryNode_ToNode(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_NODE_HANDLE * nodeHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationEntryNode_ToNode)
    {
        return inst.m_PEAK_EnumerationEntryNode_ToNode(enumerationEntryNodeHandle, nodeHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationEntryNode_GetIsSelfClearing(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, PEAK_BOOL8 * isSelfClearing)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationEntryNode_GetIsSelfClearing)
    {
        return inst.m_PEAK_EnumerationEntryNode_GetIsSelfClearing(enumerationEntryNodeHandle, isSelfClearing);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationEntryNode_GetSymbolicValue(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, char * symbolicValue, size_t * symbolicValueSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationEntryNode_GetSymbolicValue)
    {
        return inst.m_PEAK_EnumerationEntryNode_GetSymbolicValue(enumerationEntryNodeHandle, symbolicValue, symbolicValueSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EnumerationEntryNode_GetValue(PEAK_ENUMERATION_ENTRY_NODE_HANDLE enumerationEntryNodeHandle, int64_t * value)
{
    auto& inst = instance();
    if(inst.m_PEAK_EnumerationEntryNode_GetValue)
    {
        return inst.m_PEAK_EnumerationEntryNode_GetValue(enumerationEntryNodeHandle, value);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetInfo(PEAK_PORT_HANDLE portHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetInfo)
    {
        return inst.m_PEAK_Port_GetInfo(portHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetID(PEAK_PORT_HANDLE portHandle, char * id, size_t * idSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetID)
    {
        return inst.m_PEAK_Port_GetID(portHandle, id, idSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetName(PEAK_PORT_HANDLE portHandle, char * name, size_t * nameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetName)
    {
        return inst.m_PEAK_Port_GetName(portHandle, name, nameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetVendorName(PEAK_PORT_HANDLE portHandle, char * vendorName, size_t * vendorNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetVendorName)
    {
        return inst.m_PEAK_Port_GetVendorName(portHandle, vendorName, vendorNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetModelName(PEAK_PORT_HANDLE portHandle, char * modelName, size_t * modelNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetModelName)
    {
        return inst.m_PEAK_Port_GetModelName(portHandle, modelName, modelNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetVersion(PEAK_PORT_HANDLE portHandle, char * version, size_t * versionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetVersion)
    {
        return inst.m_PEAK_Port_GetVersion(portHandle, version, versionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetTLType(PEAK_PORT_HANDLE portHandle, char * tlType, size_t * tlTypeSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetTLType)
    {
        return inst.m_PEAK_Port_GetTLType(portHandle, tlType, tlTypeSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetModuleName(PEAK_PORT_HANDLE portHandle, char * moduleName, size_t * moduleNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetModuleName)
    {
        return inst.m_PEAK_Port_GetModuleName(portHandle, moduleName, moduleNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetDataEndianness(PEAK_PORT_HANDLE portHandle, PEAK_ENDIANNESS * dataEndianness)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetDataEndianness)
    {
        return inst.m_PEAK_Port_GetDataEndianness(portHandle, dataEndianness);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetIsReadable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isReadable)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetIsReadable)
    {
        return inst.m_PEAK_Port_GetIsReadable(portHandle, isReadable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetIsWritable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isWritable)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetIsWritable)
    {
        return inst.m_PEAK_Port_GetIsWritable(portHandle, isWritable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetIsAvailable(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isAvailable)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetIsAvailable)
    {
        return inst.m_PEAK_Port_GetIsAvailable(portHandle, isAvailable);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetIsImplemented(PEAK_PORT_HANDLE portHandle, PEAK_BOOL8 * isImplemented)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetIsImplemented)
    {
        return inst.m_PEAK_Port_GetIsImplemented(portHandle, isImplemented);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_Read(PEAK_PORT_HANDLE portHandle, uint64_t address, uint8_t * bytesToRead, size_t bytesToReadSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_Read)
    {
        return inst.m_PEAK_Port_Read(portHandle, address, bytesToRead, bytesToReadSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_Write(PEAK_PORT_HANDLE portHandle, uint64_t address, const uint8_t * bytesToWrite, size_t bytesToWriteSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_Write)
    {
        return inst.m_PEAK_Port_Write(portHandle, address, bytesToWrite, bytesToWriteSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetNumURLs(PEAK_PORT_HANDLE portHandle, size_t * numUrls)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetNumURLs)
    {
        return inst.m_PEAK_Port_GetNumURLs(portHandle, numUrls);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Port_GetURL(PEAK_PORT_HANDLE portHandle, size_t index, PEAK_PORT_URL_HANDLE * portUrlHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Port_GetURL)
    {
        return inst.m_PEAK_Port_GetURL(portHandle, index, portUrlHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetInfo(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetInfo)
    {
        return inst.m_PEAK_PortURL_GetInfo(portUrlHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetURL(PEAK_PORT_URL_HANDLE portUrlHandle, char * url, size_t * urlSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetURL)
    {
        return inst.m_PEAK_PortURL_GetURL(portUrlHandle, url, urlSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetScheme(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_URL_SCHEME * scheme)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetScheme)
    {
        return inst.m_PEAK_PortURL_GetScheme(portUrlHandle, scheme);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileName(PEAK_PORT_URL_HANDLE portUrlHandle, char * fileName, size_t * fileNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileName)
    {
        return inst.m_PEAK_PortURL_GetFileName(portUrlHandle, fileName, fileNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileRegisterAddress(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t * fileRegisterAddress)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileRegisterAddress)
    {
        return inst.m_PEAK_PortURL_GetFileRegisterAddress(portUrlHandle, fileRegisterAddress);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileSize(PEAK_PORT_URL_HANDLE portUrlHandle, uint64_t * fileSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileSize)
    {
        return inst.m_PEAK_PortURL_GetFileSize(portUrlHandle, fileSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileSHA1Hash(PEAK_PORT_URL_HANDLE portUrlHandle, uint8_t * fileSha1Hash, size_t * fileSha1HashSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileSHA1Hash)
    {
        return inst.m_PEAK_PortURL_GetFileSHA1Hash(portUrlHandle, fileSha1Hash, fileSha1HashSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileVersionMajor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionMajor)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileVersionMajor)
    {
        return inst.m_PEAK_PortURL_GetFileVersionMajor(portUrlHandle, fileVersionMajor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileVersionMinor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionMinor)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileVersionMinor)
    {
        return inst.m_PEAK_PortURL_GetFileVersionMinor(portUrlHandle, fileVersionMinor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileVersionSubminor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileVersionSubminor)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileVersionSubminor)
    {
        return inst.m_PEAK_PortURL_GetFileVersionSubminor(portUrlHandle, fileVersionSubminor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileSchemaVersionMajor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileSchemaVersionMajor)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileSchemaVersionMajor)
    {
        return inst.m_PEAK_PortURL_GetFileSchemaVersionMajor(portUrlHandle, fileSchemaVersionMajor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetFileSchemaVersionMinor(PEAK_PORT_URL_HANDLE portUrlHandle, int32_t * fileSchemaVersionMinor)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetFileSchemaVersionMinor)
    {
        return inst.m_PEAK_PortURL_GetFileSchemaVersionMinor(portUrlHandle, fileSchemaVersionMinor);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_PortURL_GetParentPort(PEAK_PORT_URL_HANDLE portUrlHandle, PEAK_PORT_HANDLE * portHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_PortURL_GetParentPort)
    {
        return inst.m_PEAK_PortURL_GetParentPort(portUrlHandle, portHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventSupportingModule_EnableEvents(PEAK_EVENT_SUPPORTING_MODULE_HANDLE eventSupportingModuleHandle, PEAK_EVENT_TYPE eventType, PEAK_EVENT_CONTROLLER_HANDLE * eventControllerHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventSupportingModule_EnableEvents)
    {
        return inst.m_PEAK_EventSupportingModule_EnableEvents(eventSupportingModuleHandle, eventType, eventControllerHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_GetInfo(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_GetInfo)
    {
        return inst.m_PEAK_EventController_GetInfo(eventControllerHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_GetNumEventsInQueue(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * numEventsInQueue)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_GetNumEventsInQueue)
    {
        return inst.m_PEAK_EventController_GetNumEventsInQueue(eventControllerHandle, numEventsInQueue);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_GetNumEventsFired(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t * numEventsFired)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_GetNumEventsFired)
    {
        return inst.m_PEAK_EventController_GetNumEventsFired(eventControllerHandle, numEventsFired);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_GetEventMaxSize(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * eventMaxSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_GetEventMaxSize)
    {
        return inst.m_PEAK_EventController_GetEventMaxSize(eventControllerHandle, eventMaxSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_GetEventDataMaxSize(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, size_t * eventDataMaxSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_GetEventDataMaxSize)
    {
        return inst.m_PEAK_EventController_GetEventDataMaxSize(eventControllerHandle, eventDataMaxSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_GetControlledEventType(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, PEAK_EVENT_TYPE * controlledEventType)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_GetControlledEventType)
    {
        return inst.m_PEAK_EventController_GetControlledEventType(eventControllerHandle, controlledEventType);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_WaitForEvent(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle, uint64_t timeout_ms, PEAK_EVENT_HANDLE * eventHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_WaitForEvent)
    {
        return inst.m_PEAK_EventController_WaitForEvent(eventControllerHandle, timeout_ms, eventHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_KillWait(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_KillWait)
    {
        return inst.m_PEAK_EventController_KillWait(eventControllerHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_FlushEvents(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_FlushEvents)
    {
        return inst.m_PEAK_EventController_FlushEvents(eventControllerHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_EventController_Destruct(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_EventController_Destruct)
    {
        return inst.m_PEAK_EventController_Destruct(eventControllerHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Event_GetInfo(PEAK_EVENT_HANDLE eventHandle, int32_t infoCommand, int32_t * infoDataType, uint8_t * info, size_t * infoSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Event_GetInfo)
    {
        return inst.m_PEAK_Event_GetInfo(eventHandle, infoCommand, infoDataType, info, infoSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Event_GetID(PEAK_EVENT_HANDLE eventHandle, uint64_t * id)
{
    auto& inst = instance();
    if(inst.m_PEAK_Event_GetID)
    {
        return inst.m_PEAK_Event_GetID(eventHandle, id);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Event_GetData(PEAK_EVENT_HANDLE eventHandle, uint8_t * data, size_t * dataSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Event_GetData)
    {
        return inst.m_PEAK_Event_GetData(eventHandle, data, dataSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Event_GetType(PEAK_EVENT_HANDLE eventHandle, PEAK_EVENT_TYPE * type)
{
    auto& inst = instance();
    if(inst.m_PEAK_Event_GetType)
    {
        return inst.m_PEAK_Event_GetType(eventHandle, type);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Event_GetRawData(PEAK_EVENT_HANDLE eventHandle, uint8_t * rawData, size_t * rawDataSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_Event_GetRawData)
    {
        return inst.m_PEAK_Event_GetRawData(eventHandle, rawData, rawDataSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_Event_Destruct(PEAK_EVENT_HANDLE eventHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_Event_Destruct)
    {
        return inst.m_PEAK_Event_Destruct(eventHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_Construct(PEAK_FIRMWARE_UPDATER_HANDLE * firmwareUpdaterHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_Construct)
    {
        return inst.m_PEAK_FirmwareUpdater_Construct(firmwareUpdaterHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char * gufPath, size_t gufPathSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation)
    {
        return inst.m_PEAK_FirmwareUpdater_CollectAllFirmwareUpdateInformation(firmwareUpdaterHandle, gufPath, gufPathSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, const char * gufPath, size_t gufPathSize, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation)
    {
        return inst.m_PEAK_FirmwareUpdater_CollectFirmwareUpdateInformation(firmwareUpdaterHandle, gufPath, gufPathSize, deviceDescriptorHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t * numFirmwareUpdateInformation)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation)
    {
        return inst.m_PEAK_FirmwareUpdater_GetNumFirmwareUpdateInformation(firmwareUpdaterHandle, numFirmwareUpdateInformation);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_GetFirmwareUpdateInformation(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, size_t index, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE * firmwareUpdateInformationHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation)
    {
        return inst.m_PEAK_FirmwareUpdater_GetFirmwareUpdateInformation(firmwareUpdaterHandle, index, firmwareUpdateInformationHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_UpdateDevice(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_UpdateDevice)
    {
        return inst.m_PEAK_FirmwareUpdater_UpdateDevice(firmwareUpdaterHandle, deviceDescriptorHandle, firmwareUpdateInformationHandle, firmwareUpdateProgressObserverHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle, PEAK_DEVICE_DESCRIPTOR_HANDLE deviceDescriptorHandle, PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, uint64_t deviceResetDiscoveryTimeout_ms)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout)
    {
        return inst.m_PEAK_FirmwareUpdater_UpdateDeviceWithResetTimeout(firmwareUpdaterHandle, deviceDescriptorHandle, firmwareUpdateInformationHandle, firmwareUpdateProgressObserverHandle, deviceResetDiscoveryTimeout_ms);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdater_Destruct(PEAK_FIRMWARE_UPDATER_HANDLE firmwareUpdaterHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdater_Destruct)
    {
        return inst.m_PEAK_FirmwareUpdater_Destruct(firmwareUpdaterHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetIsValid(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_BOOL8 * isValid)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetIsValid)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetIsValid(firmwareUpdateInformationHandle, isValid);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetFileName(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * fileName, size_t * fileNameSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetFileName)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetFileName(firmwareUpdateInformationHandle, fileName, fileNameSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetDescription(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * description, size_t * descriptionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetDescription)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetDescription(firmwareUpdateInformationHandle, description, descriptionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetVersion(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * version, size_t * versionSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetVersion)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetVersion(firmwareUpdateInformationHandle, version, versionSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * versionExtractionPattern, size_t * versionExtractionPatternSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetVersionExtractionPattern(firmwareUpdateInformationHandle, versionExtractionPattern, versionExtractionPatternSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetVersionStyle(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_VERSION_STYLE * versionStyle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetVersionStyle)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetVersionStyle(firmwareUpdateInformationHandle, versionStyle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetReleaseNotes(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * releaseNotes, size_t * releaseNotesSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetReleaseNotes)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetReleaseNotes(firmwareUpdateInformationHandle, releaseNotes, releaseNotesSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetReleaseNotesURL(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, char * releaseNotesUrl, size_t * releaseNotesUrlSize)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetReleaseNotesURL(firmwareUpdateInformationHandle, releaseNotesUrl, releaseNotesUrlSize);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetUserSetPersistence(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PERSISTENCE * userSetPersistence)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetUserSetPersistence)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetUserSetPersistence(firmwareUpdateInformationHandle, userSetPersistence);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence(PEAK_FIRMWARE_UPDATE_INFORMATION_HANDLE firmwareUpdateInformationHandle, PEAK_FIRMWARE_UPDATE_PERSISTENCE * sequencerSetPersistence)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence)
    {
        return inst.m_PEAK_FirmwareUpdateInformation_GetSequencerSetPersistence(firmwareUpdateInformationHandle, sequencerSetPersistence);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_Construct(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE * firmwareUpdateProgressObserverHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_Construct)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_Construct(firmwareUpdateProgressObserverHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStartedCallback(firmwareUpdateProgressObserverHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STARTED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStartedCallback(firmwareUpdateProgressObserverHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepStartedCallback(firmwareUpdateProgressObserverHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_STARTED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepStartedCallback(firmwareUpdateProgressObserverHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepProgressChangedCallback(firmwareUpdateProgressObserverHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_PROGRESS_CHANGED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepProgressChangedCallback(firmwareUpdateProgressObserverHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateStepFinishedCallback(firmwareUpdateProgressObserverHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_STEP_FINISHED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateStepFinishedCallback(firmwareUpdateProgressObserverHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFinishedCallback(firmwareUpdateProgressObserverHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FINISHED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFinishedCallback(firmwareUpdateProgressObserverHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK callback, void * callbackContext, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE * callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_RegisterUpdateFailedCallback(firmwareUpdateProgressObserverHandle, callback, callbackContext, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle, PEAK_FIRMWARE_UPDATE_FAILED_CALLBACK_HANDLE callbackHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_UnregisterUpdateFailedCallback(firmwareUpdateProgressObserverHandle, callbackHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

inline PEAK_RETURN_CODE DynamicLoader::PEAK_FirmwareUpdateProgressObserver_Destruct(PEAK_FIRMWARE_UPDATE_PROGRESS_OBSERVER_HANDLE firmwareUpdateProgressObserverHandle)
{
    auto& inst = instance();
    if(inst.m_PEAK_FirmwareUpdateProgressObserver_Destruct)
    {
        return inst.m_PEAK_FirmwareUpdateProgressObserver_Destruct(firmwareUpdateProgressObserverHandle);
    }
    else
    {
        throw std::runtime_error("Library not loaded!");
    }
}

} /* namespace dynamic */
} /* namespace peak */

