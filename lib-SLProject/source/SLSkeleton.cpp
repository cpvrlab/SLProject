//#############################################################################
//  File:      SLSkeleton.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLSkeleton.h>
#include <SLScene.h>
#include <SLAnimPlayback.h>

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
    for (auto it : _animations) delete it.second;
    for (auto it : _animPlaybacks) delete it.second;
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
           (id < _joints.size() && _joints[id] == nullptr)) &&
          "Trying to create a joint with an already existing id.");

    if (_joints.size() <= id)
        _joints.resize(id+1);
    
    _joints[id] = result;
    return result;
}


//-----------------------------------------------------------------------------
/*! Returns an animation state by name.
*/
SLAnimPlayback* SLSkeleton::getAnimPlayback(const SLstring& name)
{
    if (_animPlaybacks.find(name) != _animPlaybacks.end())
        return _animPlaybacks[name];
    else if (_animations.find(name) != _animations.end())
    {
        _animPlaybacks[name] = new SLAnimPlayback(_animations[name]);
        return _animPlaybacks[name];
    }

    return nullptr;
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
    if (!_root) return nullptr;
    SLJoint* result = _root->find<SLJoint>(name);
    return result;
}

//-----------------------------------------------------------------------------
/*! Fills a SLMat4f array with the final joint matrices for this skeleton.
*/
void SLSkeleton::getJointWorldMatrices(SLMat4f* jointWM)
{
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
    for (auto j : _joints)
        j->resetToInitialState();
}

//-----------------------------------------------------------------------------
/*! Updates the skeleton based on its active animation states
*/
SLbool SLSkeleton::updateAnimations(SLfloat elapsedTimeSec)
{
    SLbool animated = false;

    for (auto it : _animPlaybacks)
    {   SLAnimPlayback* pb = it.second;
        if (pb->enabled())
        {   pb->advanceTime(elapsedTimeSec);
            animated |= pb->changed();
        }
    }

    // return if nothing changed
    if (!animated)
        return false;

    // reset the skeleton and apply all enabled animations
    reset();

    for (auto it : _animPlaybacks)
    {
        SLAnimPlayback* pb = it.second;
        if (pb->enabled())
        {   pb->parentAnimation()->apply(this, pb->localTime(), pb->weight());
            pb->changed(false); // remove changed dirty flag from the pb again
        }
    }
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
    for (auto joint : _joints)
    {
        SLfloat r = joint->radius();

        // ignore joints with a zero radius
        if (r == 0.0f)
            continue;

        SLVec3f jointPos = joint->updateAndGetWM().translation();
        SLVec3f curMin = jointPos - SLVec3f(r, r, r);
        SLVec3f curMax = jointPos + SLVec3f(r, r, r);
        
        if (!firstSet)
        {   _minOS = curMin;
            _maxOS = curMax;
            firstSet = true;
        } else
        {   _minOS.x = min(_minOS.x, curMin.x);
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
