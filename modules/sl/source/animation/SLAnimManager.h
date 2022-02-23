//#############################################################################
//  File:      SLAnimManager.h
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMMANAGER_H
#define SLANIMMANAGER_H

#include <SLAnimManager.h>
#include <SLAnimPlayback.h>
#include <SLAnimation.h>
#include <SLAnimSkeleton.h>

//-----------------------------------------------------------------------------
//! SLAnimManager is the central class for all animation handling.
/*!
A single instance of this class is hold by the SLScene instance and is
responsible for updating the enabled animations and to manage their life time.
If keeps a list of all skeletons and node animations and also holds a list of
all animation playback controllers.
The update of all animations is done before the rendering of all SLSceneView in
SLScene::updateIfAllViewsGotPainted by calling the SLAnimManager::update.
*/
class SLAnimManager
{
public:
    ~SLAnimManager();

    void            addSkeleton(SLAnimSkeleton* skel);
    void            addNodeAnimation(SLAnimation* anim);
    SLbool          hasNodeAnimations() { return (_nodeAnimations.size() > 0); }
    SLAnimPlayback* nodeAnimPlayback(const SLstring& name);
    SLAnimPlayback* allAnimPlayback(SLuint ix) { return _allAnimPlaybacks[ix]; }
    SLAnimPlayback* lastAnimPlayback() { return _allAnimPlaybacks.back(); }

    SLAnimation* createNodeAnimation(SLfloat duration);
    SLAnimation* createNodeAnimation(const SLstring& name,
                                     SLfloat         duration);
    SLAnimation* createNodeAnimation(const SLstring& name,
                                     SLfloat         duration,
                                     SLbool          enabled,
                                     SLEasingCurve   easing,
                                     SLAnimLooping   looping);

    SLMAnimation& animations()
    {
        return _nodeAnimations;
    }
    SLVSkeleton&     skeletons() { return _skeletons; }
    SLVstring&       allAnimNames() { return _allAnimNames; }
    SLVAnimPlayback& allAnimPlaybacks() { return _allAnimPlaybacks; }

    SLbool update(SLfloat elapsedTimeSec);
    void   drawVisuals(SLSceneView* sv);
    void   clear();

private:
    SLVSkeleton     _skeletons;         //!< all skeleton instances
    SLMAnimation    _nodeAnimations;    //!< node animations
    SLMAnimPlayback _nodeAnimPlaybacks; //!< node animation playbacks
    SLVstring       _allAnimNames;      //!< vector with all animation names
    SLVAnimPlayback _allAnimPlaybacks;  //!< vector with all animation playbacks
};
//-----------------------------------------------------------------------------
#endif
