#include "StateMachine.h"

#define LOG_STATEMACHINE_WARN(...) Utils::log("StateMachine", __VA_ARGS__);
#define LOG_STATEMACHINE_INFO(...) Utils::log("StateMachine", __VA_ARGS__);
#define LOG_STATEMACHINE_DEBUG(...) Utils::log("StateMachine", __VA_ARGS__);

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
    sm::EventData* data            = nullptr;
    bool           stateEntry      = false;
    bool           stateWasUpdated = false;
    //invoke state action for every valid event, but at least once
    while (!_events.empty())
    {
        Event* e = _events.front();
        _events.pop();

        unsigned int newState = e->getNewState(_currentStateId);
        data                  = e->getEventData();

        if (newState != Event::EVENT_IGNORED)
        {
            if (_currentStateId != newState)
            {
                stateEntry      = true;
                _currentStateId = newState;
            }
            _stateActions[_currentStateId]->invokeStateAction(this, data, stateEntry);
            stateWasUpdated = true;
        }
        else
        {
            LOG_STATEMACHINE_DEBUG("Event ignored");
        }

        delete e;
    }

    if (!stateWasUpdated)
        _stateActions[_currentStateId]->invokeStateAction(this, data, stateEntry);

    //ATTENTION: data ownership is not transferred to state
    delete data;

    return true;
}
}
