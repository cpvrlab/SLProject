#ifndef SM_EVENT_H
#define SM_EVENT_H

#include <map>
#include <string>
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

    Event(std::string name, std::string senderInfo)
      : _name(name),
        _senderInfo(senderInfo)
    {
    }

    virtual ~Event(){};

    // enables a transition from one state to the other
    void enableTransition(unsigned int from, unsigned int to)
    {
        _transitions[from] = to;
    }

    // Check if there is a transition to a new state. The current state is used to lookup the new state.
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
    // get event data that was possibly send with this event. If the function returns nullptr, it contains no data.
    EventData* getEventData()
    {
        return _eventData;
    }

    const char* name() const { return _name.c_str(); }
    const char* senderInfo() const { return _senderInfo.c_str(); }

protected:
    EventData* _eventData = nullptr;

private:
    std::map<unsigned int, unsigned int> _transitions;

    std::string _name;
    std::string _senderInfo;
};
}

#endif
