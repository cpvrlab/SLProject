#ifndef SM_EVENT_H
#define SM_EVENT_H

#include <map>
#include <sm/EventData.h>

namespace sm
{

class Event
{
public:
    enum
    {
        EVENT_IGNORED = 0xFE,
    };

    virtual ~Event(){};

    void enableTransition(unsigned int from, unsigned int to)
    {
        _transitions[from] = to;
    }

    unsigned int getNewState(unsigned int currentState)
    {
        auto it = _transitions.find(currentState);
        if (it != _transitions.end())
        {
            return it->second;
        }
        else
        {
            return EVENT_IGNORED;
        }
    }

    EventData* getEventData()
    {
        return _eventData;
    }

protected:
    EventData* _eventData = nullptr;

private:
    std::map<unsigned int, unsigned int> _transitions;
};
}

#endif
