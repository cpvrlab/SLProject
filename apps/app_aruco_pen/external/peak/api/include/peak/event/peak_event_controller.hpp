/*!
 * \file    peak_event_controller.hpp
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
#include <peak/common/peak_timeout.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/event/peak_event.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>


namespace peak
{
namespace core
{

/*!
 * \brief Encapsulates the GenTL event functionality associated with one GenTL Event handle.
 *
 * This class is returned by EventSupportingModule::EnableEvents() and acts as a controller for events of the type given
 * to EventSupportingModule::EnableEvents().
 *
 */
class EventController
{
public:
    EventController() = delete;
    ~EventController();
    EventController(const EventController& other) = delete;
    EventController& operator=(const EventController& other) = delete;
    EventController(EventController&& other) = delete;
    EventController& operator=(EventController&& other) = delete;

    /*! @copydoc SystemDescriptor::Info() */
    RawInformation Info(int32_t infoCommand) const;
    /*!
     * \brief Returns the number of events in queue.
     *
     * \return Number of events in queue
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t NumEventsInQueue() const;
    /*!
     * \brief Returns the number of fired events.
     *
     * \return Number of fired events
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    uint64_t NumEventsFired() const;
    /*!
     * \brief Returns the maximum size of an event in bytes.
     *
     * \return Maximum size of an event in bytes.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t EventMaxSize() const;
    /*!
     * \brief Returns the maximum size of the data of an event in bytes.
     *
     * \return Maximum size of the data of an event in bytes.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    size_t EventDataMaxSize() const;

    /*!
     * \brief Blocking wait for an event.
     *
     * \param[in] timeout_ms The maximum waiting time in milliseconds.
     *                       When called with Timeout::INFINITE_TIMEOUT, the function will only return
     *                       if the event is triggered or KillWait() is called.
     *
     * \return Event
     *
     * \since 1.0
     *
     * \throws AbortedException The wait was aborted
     * \throws TimeoutException The function call timed out
     * \throws InternalErrorException An internal error has occurred.
     */
    std::unique_ptr<Event> WaitForEvent(Timeout timeout_ms);
    /*!
     * \brief Terminates a wait for an event.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void KillWait();
    /*!
     * \brief Discards all events in queue.
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    void FlushEvents();

    /*!
     * \brief Returns the controlled event type.
     *
     * \return Controlled event type
     *
     * \since 1.0
     */
    EventType ControlledEventType() const;

private:
    friend ClassCreator<EventController>;
    explicit EventController(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle);
    PEAK_EVENT_CONTROLLER_HANDLE m_backendHandle;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline EventController::EventController(PEAK_EVENT_CONTROLLER_HANDLE eventControllerHandle)
    : m_backendHandle(eventControllerHandle)
{}

inline EventController::~EventController()
{
    (void)PEAK_C_ABI_PREFIX PEAK_EventController_Destruct(m_backendHandle);
}

inline RawInformation EventController::Info(int32_t infoCommand) const
{
    return QueryRawInformationFromCInterfaceFunction([&](int32_t* dataType, uint8_t* buffer, size_t* bufferSize) {
        return PEAK_C_ABI_PREFIX PEAK_EventController_GetInfo(
            m_backendHandle, infoCommand, dataType, buffer, bufferSize);
    });
}

inline size_t EventController::NumEventsInQueue() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* numEventsInQueue) {
        return PEAK_C_ABI_PREFIX PEAK_EventController_GetNumEventsInQueue(m_backendHandle, numEventsInQueue);
    });
}

inline uint64_t EventController::NumEventsFired() const
{
    return QueryNumericFromCInterfaceFunction<uint64_t>([&](uint64_t* numEventsFired) {
        return PEAK_C_ABI_PREFIX PEAK_EventController_GetNumEventsFired(m_backendHandle, numEventsFired);
    });
}

inline size_t EventController::EventMaxSize() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* eventMaxSize) {
        return PEAK_C_ABI_PREFIX PEAK_EventController_GetEventMaxSize(m_backendHandle, eventMaxSize);
    });
}

inline size_t EventController::EventDataMaxSize() const
{
    return QueryNumericFromCInterfaceFunction<size_t>([&](size_t* eventDataMaxSize) {
        return PEAK_C_ABI_PREFIX PEAK_EventController_GetEventDataMaxSize(m_backendHandle, eventDataMaxSize);
    });
}

inline std::unique_ptr<Event> EventController::WaitForEvent(Timeout timeout_ms)
{
    auto eventHandle = QueryNumericFromCInterfaceFunction<PEAK_EVENT_HANDLE>(
        [&](PEAK_EVENT_HANDLE* _eventHandle) {
            return PEAK_C_ABI_PREFIX PEAK_EventController_WaitForEvent(m_backendHandle, timeout_ms, _eventHandle);
        });

    return std::make_unique<ClassCreator<Event>>(eventHandle);
}

inline void EventController::KillWait()
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_EventController_KillWait(m_backendHandle); });
}

inline void EventController::FlushEvents()
{
    CallAndCheckCInterfaceFunction(
        [&] { return PEAK_C_ABI_PREFIX PEAK_EventController_FlushEvents(m_backendHandle); });
}

inline EventType EventController::ControlledEventType() const
{
    return static_cast<EventType>(
        QueryNumericFromCInterfaceFunction<PEAK_EVENT_TYPE>([&](PEAK_EVENT_TYPE* controlledEventType) {
            return PEAK_C_ABI_PREFIX PEAK_EventController_GetControlledEventType(
                m_backendHandle, controlledEventType);
        }));
}

} /* namespace core */
} /* namespace peak */
