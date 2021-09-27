/*!
 * \file    peak_event_supporting_module.hpp
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once


#include <peak/backend/peak_backend.h>
#include <peak/common/peak_module.hpp>
#include <peak/dll_interface/peak_dll_interface_util.hpp>
#include <peak/event/peak_event_controller.hpp>

#include <memory>


namespace peak
{
namespace core
{

/*!
 * \brief The base class for all modules being able to raise events.
 *
 * This class generalizes all modules supporting events (System, Interface, Device, DataStream, Buffer).
 *
 */
class EventSupportingModule : public Module
{
public:
    EventSupportingModule() = default;
    ~EventSupportingModule() override = default;
    EventSupportingModule(const EventSupportingModule& other) = delete;
    EventSupportingModule& operator=(const EventSupportingModule& other) = delete;
    EventSupportingModule(EventSupportingModule&& other) = delete;
    EventSupportingModule& operator=(EventSupportingModule&& other) = delete;

    /*!
     * \brief Enables events of the given event type.
     *
     * \param[in] type The event type to enable.
     *
     * \return Event controller for the given event type
     *
     * \since 1.0
     *
     * \throws InternalErrorException An internal error has occurred.
     */
    std::unique_ptr<EventController> EnableEvents(EventType type);

protected:
    virtual PEAK_EVENT_SUPPORTING_MODULE_HANDLE EventSupportingModuleHandle() const = 0;
};

} /* namespace core */
} /* namespace peak */

/* Implementation */
namespace peak
{
namespace core
{

inline std::unique_ptr<EventController> EventSupportingModule::EnableEvents(EventType type)
{
    auto eventSupportingModuleHandle = EventSupportingModuleHandle();

    auto eventControllerHandle = QueryNumericFromCInterfaceFunction<PEAK_EVENT_CONTROLLER_HANDLE>(
        [&](PEAK_EVENT_CONTROLLER_HANDLE* _eventControllerHandle) {
            return PEAK_C_ABI_PREFIX PEAK_EventSupportingModule_EnableEvents(
                eventSupportingModuleHandle, static_cast<PEAK_EVENT_TYPE>(type), _eventControllerHandle);
        });

    return std::make_unique<ClassCreator<EventController>>(eventControllerHandle);
}

} /* namespace core */
} /* namespace peak */
