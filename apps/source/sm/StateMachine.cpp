#include "StateMachine.h"

namespace sm
{
StateMachine::StateMachine(unsigned int initialStateId)
  : _currentStateId(initialStateId)
{
}

StateMachine::~StateMachine()
{
    for (auto it : _stateActions)
    {
        delete it.second;
    }
};

bool StateMachine::update()
{
    sm::EventData* data = nullptr;
    //we only handle one event per update call!
    if (_events.size())
    {
        Event* e = _events.front();
        _events.pop();

        unsigned int newState = e->getNewState(_currentStateId);
        data                  = e->getEventData();
        if (newState != Event::EVENT_IGNORED)
        {
            _currentStateId = newState;
        }
        else
        {
            Utils::log("StateMachine", "Event ignored");
        }

        delete e;
    }

    _stateActions[_currentStateId]->invokeStateAction(this, data);

    if (data)
        delete data;

    return true;
}
}
