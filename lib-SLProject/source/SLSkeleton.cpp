//#############################################################################
//  File:      SLSkeleton.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLSkeleton.h>
#include <SLScene.h>
#include <SLAnimationPlay.h>


//-----------------------------------------------------------------------------
/*! Constructor
*/
SLSkeleton::SLSkeleton()
: _minOS(-1, -1, -1), _maxOS(1, 1, 1), _minMaxOutOfDate(true)
{
    SLScene::current->animManager().addSkeleton(this);
}

//-----------------------------------------------------------------------------
/*! Destructor
*/
SLSkeleton::~SLSkeleton()
{
    delete _root;

    SLMAnimation::iterator it1;
    for (it1 = _animations.begin(); it1 != _animations.end(); it1++)
        delete it1->second;
    
    SLMAnimationPlay::iterator it2;
    for (it2 = _animationPlays.begin(); it2 != _animationPlays.end(); it2++)
        delete it2->second;
}

//-----------------------------------------------------------------------------
/*! Creates a new joint owned by this skeleton with a default name.
*/
SLJoint* SLSkeleton::createJoint(SLuint id)
{
    ostringstream oss;
    oss << "Joint " << id;
    return createJoint(oss.str(), id);
}

//-----------------------------------------------------------------------------
/*! Creates a new joint owned by this skeleton.
*/
SLJoint* SLSkeleton::createJoint(const SLstring& name, SLuint id)
{
    SLJoint* result = new SLJoint(name, id, this);
    
    assert((id >= _joints.size() ||
           (id < _joints.size() && _joints[id] == NULL)) &&
          "Trying to create a joint with an already existing id.");

    if (_joints.size() <= id)
        _joints.resize(id+1);
    
    _joints[id] = result;
    return result;
}


//-----------------------------------------------------------------------------
/*! Returns an animation state by name.
*/
SLAnimationPlay* SLSkeleton::getAnimationPlay(const SLstring& name)
{
    if (_animationPlays.find(name) != _animationPlays.end())
        return _animationPlays[name];
    else if (_animations.find(name) != _animations.end())
    {
        _animationPlays[name] = new SLAnimationPlay(_animations[name]);
        return _animationPlays[name];
    }

    return NULL;
}

//-----------------------------------------------------------------------------
/*! Returns an SLJoint by it's internal id.
*/
SLJoint* SLSkeleton::getJoint(SLuint id)
{
    assert(id < _joints.size() && "Index out of bounds");
    return _joints[id];
}

//-----------------------------------------------------------------------------
/*! returns an SLJoint by name.
*/
SLJoint* SLSkeleton::getJoint(const SLstring& name)
{
    if (!_root) return NULL;
    SLJoint* result = _root->find<SLJoint>(name);
    return result;
}

//-----------------------------------------------------------------------------
/*! Fills a SLMat4f array with the final joint matrices for this skeleton.
*/
void SLSkeleton::getJointWorldMatrices(SLMat4f* jointWM)
{
    // @todo this is asking for a crash...
    for (SLint i = 0; i < _joints.size(); i++)
    {
        jointWM[i] = _joints[i]->updateAndGetWM() * _joints[i]->offsetMat();
    }
}

//-----------------------------------------------------------------------------
/*! Setter for the root joint of this skeleton.
*/
void SLSkeleton::root(SLJoint* joint)
{
    if (_root)
    _root = joint;
}

//-----------------------------------------------------------------------------
/*! Create a nw animation owned by this skeleton.
*/
SLAnimation* SLSkeleton::createAnimation(const SLstring& name, SLfloat duration)
{
    assert(_animations.find(name) == _animations.end() &&
           "animation with same name already exists!");
    SLAnimation* anim = new SLAnimation(name, duration);
    _animations[name] = anim;
    return anim;
}

//-----------------------------------------------------------------------------
/*! Resets all joints.
*/
void SLSkeleton::reset()
{
    // update all joints
    for (SLint i = 0; i < _joints.size(); i++)
        _joints[i]->resetToInitialState();
}

//-----------------------------------------------------------------------------
/*! Updates the skeleton based on its active animation states
*/
SLbool SLSkeleton::updateAnimations(SLfloat elapsedTimeSec)
{
    /// @todo IMPORTANT: don't do this if we don't have any enabled animations,
    /// current workaround won't allow for blending!
    //reset();

    // reset the changed flag
    _changed = false;

    SLbool activeAnims = false;
    SLMAnimationPlay::iterator it;
    for (it = _animationPlays.begin(); it != _animationPlays.end(); it++)
    {
        SLAnimationPlay* state = it->second;
        if (state->enabled())
        {
            state->advanceTime(elapsedTimeSec);

            // mark skeleton as changed if a state is different
            activeAnims = state->changed();
        }
    }

    // return if nothing changed
    if (!activeAnims)
        return false;

    // reset the skeleton and apply all enabled animations
    reset();

    for (it = _animationPlays.begin(); it != _animationPlays.end(); it++)
    {
        SLAnimationPlay* state = it->second;
        if (state->enabled())
        {
            state->parentAnimation()->apply(this, state->localTime(), state->weight());
            state->changed(false); // remove changed dirty flag from the state again
        }
    }

    _minMaxOutOfDate = true;
    return true;
}


//-----------------------------------------------------------------------------
/*! getter for current the current min object space vertex.
*/
const SLVec3f& SLSkeleton::minOS()
{
    if (_minMaxOutOfDate)
        updateMinMax();
    
    return _minOS;
}

//-----------------------------------------------------------------------------
/*! getter for current the current max object space vertex.
*/
const SLVec3f& SLSkeleton::maxOS()
{
    if (_minMaxOutOfDate)
        updateMinMax();

    return _maxOS;
}

//-----------------------------------------------------------------------------
/*! Calculate the current min and max values in local space based on joint
radii.
*/
void SLSkeleton::updateMinMax()
{    
    // recalculate the new min and max os based on bone radius
    SLbool firstSet = false;
    for (SLint i = 0; i < _joints.size(); i++)
    {
        SLfloat r = _joints[i]->radius();
        // ignore joints with a zero radius
        if (r == 0.0f)
            continue;

        SLVec3f jointPos = _joints[i]->updateAndGetWM().translation();
        SLVec3f curMin = jointPos - SLVec3f(r, r, r);
        SLVec3f curMax = jointPos + SLVec3f(r, r, r);
        if (!firstSet)
        {
            _minOS = curMin;
            _maxOS = curMax;
            firstSet = true;
        }
        else
        {
            _minOS.x = min(_minOS.x, curMin.x);
            _minOS.y = min(_minOS.y, curMin.y);
            _minOS.z = min(_minOS.z, curMin.z);

            _maxOS.x = max(_maxOS.x, curMax.x);
            _maxOS.y = max(_maxOS.y, curMax.y);
            _maxOS.z = max(_maxOS.z, curMax.z);
        }
    }
    _minMaxOutOfDate = false;
}
//-----------------------------------------------------------------------------
