#ifndef SM_STATE_MACHINE_H
#define SM_STATE_MACHINE_H

#include <sm/EventData.h>
#include <sm/EventHandler.h>
#include <Utils.h>
#include <cassert>

namespace sm
{

class StateMachine;

/// @brief Abstract state base class that all states inherit from.
class StateBase
{
public:
    virtual ~StateBase() { ; }

    /*!
     * Called by the state machine to execute a state action.
     * @param sm A state machine instance
     * @param data The event data
     * @param stateEntry
     * @param stateExit
     */
    virtual void invokeStateAction(StateMachine* sm, const EventData* data, const bool stateEntry, const bool stateExit) const {};
};

/*!
 * StateAction takes three template arguments: A state machine class,
 * a state function event data type (derived from EventData) and a state machine
 * member function pointer.
 * @tparam SM
 * @tparam Data
 * @tparam Func
 */
template<class SM, class Data, void (SM::*Func)(const Data*, const bool, const bool)>
class StateAction : public StateBase
{
public:
    virtual void invokeStateAction(StateMachine* sm, const EventData* data, const bool stateEntry, const bool stateExit) const
    {
        // Downcast the state machine and event data to the correct derived type
        SM* derivedSM = static_cast<SM*>(sm);

        const Data* derivedData = dynamic_cast<const Data*>(data);

        // Call the state function
        (derivedSM->*Func)(derivedData, stateEntry, stateExit);
    }
};

/*!
- Transfer id of initial state in constructor of StateMachine
- Define state functions like: void <name>(const sm::EventData* data);
- call registerState in constructor which maps a state function to an state id
*/
class StateMachine : public EventHandler
{
public:
    explicit StateMachine(unsigned int initialStateId);
    virtual ~StateMachine();

    //! process events and update current state
    bool update();

    virtual std::string getPrintableState(unsigned int state) = 0;

protected:
    //! register state processing functions from deriving class
    template<class SM, class Data, void (SM::*Func)(const Data*, const bool, const bool)>
    void registerState(unsigned int stateId)
    {
        assert(_stateActions.find(stateId) == _stateActions.end());
        _stateActions[stateId] = new StateAction<SM, Data, Func>();
    }

private:
    unsigned int _currentStateId = 0;

    std::map<unsigned int, sm::StateBase*> _stateActions;
};

} // namespace SM

#endif
