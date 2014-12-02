
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

    for (SLint i = 0; i < _nodeAnimationStates.size(); ++i)
        delete _nodeAnimationStates[i];
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

SLAnimationState* SLAnimationManager::createNodeAnimationState(SLAnimation* parent, SLfloat weight)
{
    SLAnimationState* state = new SLAnimationState(parent, weight);
    _nodeAnimationStates.push_back(state);
    return state;
}


void SLAnimationManager::update()
{
    SLScene* s = SLScene::current;

    // advance time for node animations and apply them
    // @todo currently we can't blend between normal node animations because we reset them
    // per state. so the last state that affects a node will have its animation applied.
    // we need to save the states differently if we want them.
    for (SLint i = 0; i < _nodeAnimationStates.size(); ++i)
    {
        SLAnimationState* state = _nodeAnimationStates[i];
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