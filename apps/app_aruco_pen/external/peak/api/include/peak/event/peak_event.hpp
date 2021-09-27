/*!
 * \file    peak_event.hpp
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
#include <peak/dll_interface/peak_dll_interface_util.hpp>

#include <cstdint>
#include <vector>


namespace peak
{
namespace core
{

/*!
 * \brief Different types of events.
 *
 * See GenTL EVENT_TYPE.
 *
 * GenTL EVENT_NEW_BUFFER is missing, since new buffer events are handled via DataStream::WaitForFinishedBuffer().
 */
enum class EventType
{
    Error,
    FeatureInvalidate = 2,
    FeatureChange,
    RemoteDevice,
    Module,

    Custom = 1000
};

class NodeMap;

/*!
 * \brief Encapsulates the GenTL event data functions.
 *
 */
class Event
{
public:
    Event() = delete;
    ~Event();
    Event(const Event& other) = delete;
    Event& operator=(const Event& other) = delete;
    Event(Event&& other) = delete;
    Event& operator=(Event&& other) = delete;

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
    uint64_t ID() const;
    /*!
     * \brief Returns the event payload data.
     *
     * The delivered data depend on the event type.
     *
     * \return Payload data
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::vector<uint8_t> Data() const;

    /*!
     * \brief Returns the type.
     *
     * \return Type
     *
     * \since 1.0
     */
    EventType Type() const;

    /*!
     * \brief Returns the event raw data.
     *
     * The delivered data depend on the underlying transport layer
     * (GEV, USB3, ...) and the event type.
     * (e.g. If the underlying CTI implements GEV and the event is a remote device event,
     * the delivered data will be the event raw data of a GEV event)
     *
     * \return Raw data
     *
     * \since 1.2
     */
    std::vector<uint8_t> RawData() const;

private:
    friend NodeMap;
    friend ClassCreator<Event>;
    explicit Event(PEAK_EVENT_HANDLE eventHandle);
    PEAK_EVENT_HANDLE m_backendHandle;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline Event::Event(PEAK_EVENT_HANDLE eventHandle)
    : m_backendHandle(eventHandle)
{}

inline RawInformation Event::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_Event_GetInfo(m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline uint64_t Event::ID() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>(
        [&](uint64_t* id) { return PEAK_C_ABI_PREFIX PEAK_Event_GetID(m_backendHandle, id); });
}

inline std::vector<uint8_t> Event::Data() const
{
    return QueryNumericArrayFromCInterfaceFunction<uint8_t>([&](uint8_t* data, size_t* dataSize) {
        return PEAK_C_ABI_PREFIX PEAK_Event_GetData(m_backendHandle, data, dataSize);
    });
}

inline EventType Event::Type() const
{
    return static_cast<EventType>(QueryNumericFromCInterfaceFunction<PEAK_EVENT_TYPE>(
        [&](PEAK_EVENT_TYPE* type) { return PEAK_C_ABI_PREFIX PEAK_Event_GetType(m_backendHandle, type); }));
}

inline std::vector<uint8_t> Event::RawData() const
{
    return QueryNumericArrayFromCInterfaceFunction<uint8_t>([&](uint8_t* rawData, size_t* rawDataSize) {
        return PEAK_C_ABI_PREFIX PEAK_Event_GetRawData(m_backendHandle, rawData, rawDataSize);
    });
}

inline Event::~Event()
{
    (void)PEAK_C_ABI_PREFIX PEAK_Event_Destruct(m_backendHandle);
}

inline std::string ToString(EventType entry)
{
    std::string entryString;

    if (entry == EventType::Error)
    {
        entryString = "Error";
    }
    else if (entry == EventType::FeatureInvalidate)
    {
        entryString = "FeatureInvalidate";
    }
    else if (entry == EventType::FeatureChange)
    {
        entryString = "FeatureChange";
    }
    else if (entry == EventType::RemoteDevice)
    {
        entryString = "RemoteDevice";
    }
    else if (entry == EventType::Module)
    {
        entryString = "Module";
    }
    else if (entry >= EventType::Custom)
    {
        entryString = "Custom";
    }

    return entryString;
}

} /* namespace core */
} /* namespace peak */
