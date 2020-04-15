#ifndef SM_EVENT_DATA_H
#define SM_EVENT_DATA_H

namespace sm
{

class EventData
{
public:
    virtual ~EventData() {}
};

class NoEventData : public EventData
{
public:
    NoEventData()
    {
    }
};

}
#endif
