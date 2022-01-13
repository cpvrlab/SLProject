#ifndef SM_EVENT_SENDER_H
#define SM_EVENT_SENDER_H

#include <sm/EventHandler.h>

namespace sm
{

// state is event sender
class EventSender
{
public:
    EventSender(EventHandler& handler)
      : _handler(handler)
    {
    }
    EventSender() = delete;
    void sendEvent(Event* event)
    {
        _handler.addEvent(event);
    }

private:
    EventHandler& _handler;
};

}

#endif
