#include "StateMachine.h"
#include <sstream>

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
    // invoke state action for every valid event, but at least once
    while (!_events.empty())
    {
        Event* e = _events.front();
        _events.pop();

        unsigned int newState = e->getNewState(_currentStateId);
        data                  = e->getEventData();

        LOG_STATEMACHINE_DEBUG("Event %s received sent by %s", e->name(), e->senderInfo());

        if (newState != Event::EVENT_IGNORED)
        {
            if (_currentStateId != newState)
            {
                stateEntry = true;
                LOG_STATEMACHINE_DEBUG("State change: %s -> %s", getPrintableState(_currentStateId).c_str(), getPrintableState(newState).c_str());

                // inform old state that we will leave it soon
                auto itStateAction = _stateActions.find(_currentStateId);
                if (itStateAction != _stateActions.end())
                {
                    itStateAction->second->invokeStateAction(this, data, false, true);
                }
                else
                {
                    std::stringstream ss;
                    ss << "You forgot to register state " << getPrintableState(_currentStateId) << "!";
                    Utils::exitMsg("StateMachine", ss.str().c_str(), __LINE__, __FILE__);
                }

                // update state
                _currentStateId = newState;
            }
            auto itStateAction = _stateActions.find(_currentStateId);
            if (itStateAction != _stateActions.end())
            {
                itStateAction->second->invokeStateAction(this, data, stateEntry, false);
            }
            else
            {
                std::stringstream ss;
                ss << "You forgot to register state " << getPrintableState(_currentStateId) << "!";
                Utils::exitMsg("StateMachine", ss.str().c_str(), __LINE__, __FILE__);
            }
            stateWasUpdated = true;
        }
        else
        {
            LOG_STATEMACHINE_DEBUG("Event %s ignored in state %s", e->name(), getPrintableState(_currentStateId).c_str());
        }

        delete e;
    }

    if (!stateWasUpdated)
        _stateActions[_currentStateId]->invokeStateAction(this, data, stateEntry, false);

    // ATTENTION: data ownership is not transferred to state
    delete data;

    return true;
}
}
