#ifndef SM_EVENT_HANDLER_H
#define SM_EVENT_HANDLER_H

#include <queue>
#include <sm/Event.h>

namespace sm
{
class EventHandler
{
public:
    void addEvent(Event* e)
    {
        _events.push(e);
    }

protected:
    std::queue<Event*> _events;
};

}
#endif // !
