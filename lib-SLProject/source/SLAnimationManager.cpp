
#include <stdafx.h>
#include <SLScene.h>
#include <SLAnimation.h>
#include <SLAnimationState.h>
#include <SLAnimationManager.h>



SLAnimationManager::SLAnimationManager()
{

}
SLAnimationManager::~SLAnimationManager()
{
    clear();
}

void SLAnimationManager::clear()
{
    map<SLstring, SLAnimation*>::iterator it;
    for (it = _nodeAnimations.begin(); it != _nodeAnimations.end(); it++)
        delete it->second;
    _nodeAnimations.clear();
    
    map<SLstring, SLAnimationState*>::iterator it2;
    for (it2 = _nodeAnimationStates.begin(); it2 != _nodeAnimationStates.end(); it2++)
        delete it2->second;
    _nodeAnimationStates.clear();

    for (SLint i = 0; i < _skeletons.size(); ++i)
        delete _skeletons[i];
    _skeletons.clear();
}

void SLAnimationManager::addNodeAnimation(SLAnimation* anim)
{
    assert(_nodeAnimations.find(anim->name()) == _nodeAnimations.end() && "node animation with same name already exists!");
    _nodeAnimations[anim->name()] = anim;
}

SLAnimationState* SLAnimationManager::getNodeAnimationState(const SLstring& name)
{
    if (_nodeAnimationStates.find(name) != _nodeAnimationStates.end())
        return _nodeAnimationStates[name];

    else if (_nodeAnimations.find(name) != _nodeAnimations.end())
    {
        _nodeAnimationStates[name] = new SLAnimationState(_nodeAnimations[name]);
        return _nodeAnimationStates[name];
    }

    return NULL;
}

void SLAnimationManager::update()
{
    SLScene* s = SLScene::current;

    // advance time for node animations and apply them
    // @todo currently we can't blend between normal node animations because we reset them
    // per state. so the last state that affects a node will have its animation applied.
    // we need to save the states differently if we want them.


    map<SLstring, SLAnimationState*>::iterator it;
    for (it = _nodeAnimationStates.begin(); it != _nodeAnimationStates.end(); it++)
    {
        SLAnimationState* state = it->second;
        if (state->enabled())
        {
            state->parentAnimation()->resetNodes(); 
            state->advanceTime(s->elapsedTimeSec());
            state->parentAnimation()->apply(state->localTime(), state->weight());
        }
    }
    
    // update the skeletons seperately 
    for (SLint i = 0; i < _skeletons.size(); ++i)
    {
        _skeletons[i]->updateAnimations();
    }
}