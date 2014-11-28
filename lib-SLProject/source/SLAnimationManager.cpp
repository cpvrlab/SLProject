
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
            //state->advanceTime(s->elapsedTime());
            state->advanceTime(0.016f);
            state->parentAnimation()->apply(state->localTime(), state->weight());
        }
    }
    
    // update the skeletons seperately 
    for (SLint i = 0; i < _skeletons.size(); ++i)
    {
        _skeletons[i]->updateAnimations();
    }
}