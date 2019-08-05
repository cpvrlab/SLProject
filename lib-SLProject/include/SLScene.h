//#############################################################################
//  File:      SLScene.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSCENE_H
#define SLSCENE_H

#include <SL.h>
#include <SLAnimManager.h>
#include <SLAverage.h>
#include <SLEventHandler.h>
#include <SLGLOculus.h>
#include <SLLight.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLRect.h>
#include <SLTimer.h>
#include <SLVec3.h>
#include <SLVec4.h>
#include <vector>

class SLSceneView;
class SLCVTracked;
class SLCamera;

//-----------------------------------------------------------------------------
typedef std::vector<SLSceneView*> SLVSceneView; //!< Vector of SceneView pointers
typedef std::vector<SLCVTracked*> SLVCVTracker; //!< Vector of CV tracker pointers
//-----------------------------------------------------------------------------
//! C-Callback function typedef for scene load function
typedef void(SL_STDCALL* cbOnSceneLoad)(SLScene* s, SLSceneView* sv, SLint sceneID);
//-----------------------------------------------------------------------------
//! The SLScene class represents the top level instance holding the scene structure
/*!      
 The SLScene class holds everything that is common for all scene views such as
 the root pointer (_root3D) to the scene, an array of lights as well as the
 global resources (_meshes (SLMesh), _materials (SLMaterial), _textures
 (SLGLTexture) and _shaderProgs (SLGLProgram)).
 All these resources and the scene with all nodes to which _root3D pointer points
 get deleted in the method unInit.\n
 A scene could have multiple scene views. A pointer of each is stored in the
 vector _sceneViews.\n
 A single instance of this SLScene class is holded by the SLApplication.
 The scene assembly takes place outside of the library in function of the application.
 A pointer for this function must be passed to the SLScene constructor. For the
 demo project this function is in AppDemoSceneLoad.cpp.
*/
class SLScene : public SLObject
{
    friend class SLNode;

    public:
    SLScene(SLstring      name,
            cbOnSceneLoad onSceneLoadCallback);
    ~SLScene();
    // Setters
    void root3D(SLNode* root3D) { _root3D = root3D; }
    void root2D(SLNode* root2D) { _root2D = root2D; }
    void globalAmbiLight(SLCol4f gloAmbi) { _globalAmbiLight = gloAmbi; }
    void stopAnimations(SLbool stop) { _stopAnimations = stop; }
    void videoType(SLVideoType vt);
    void showDetection(SLbool st) { _showDetection = st; }
    void info(SLstring i) { _info = i; }

    void calculateFrameTime(SLSceneView* sv);
    void updateAnimations(SLbool sceneHasChanged,
                          SLbool renderTypeIsRT,
                          SLbool voxelsAreShown);
    void updateAABBs();

    // Getters
    SLAnimManager& animManager() { return _animManager; }
    //SLSceneView*     sv(SLuint index) { return _sceneViews[index]; }
    //SLVSceneView&    sceneViews() { return _sceneViews; }
    SLNode*          root3D() { return _root3D; }
    SLNode*          root2D() { return _root2D; }
    SLstring&        info() { return _info; }
    void             timerStart() { _timer.start(); }
    SLfloat          timeSec() { return (SLfloat)_timer.elapsedTimeInSec(); }
    SLfloat          timeMilliSec() { return (SLfloat)_timer.elapsedTimeInMilliSec(); }
    SLfloat          elapsedTimeMS() { return _frameTimeMS; }
    SLfloat          elapsedTimeSec() { return _frameTimeMS * 0.001f; }
    SLVEventHandler& eventHandlers() { return _eventHandlers; }

    SLCol4f       globalAmbiLight() const { return _globalAmbiLight; }
    SLVLight&     lights() { return _lights; }
    SLfloat       fps() { return _fps; }
    SLAvgFloat&   frameTimesMS() { return _frameTimesMS; }
    SLAvgFloat&   updateTimesMS() { return _updateTimesMS; }
    SLAvgFloat&   trackingTimesMS() { return _trackingTimesMS; }
    SLAvgFloat&   detectTimesMS() { return _detectTimesMS; }
    SLAvgFloat&   detect1TimesMS() { return _detect1TimesMS; }
    SLAvgFloat&   detect2TimesMS() { return _detect2TimesMS; }
    SLAvgFloat&   matchTimesMS() { return _matchTimesMS; }
    SLAvgFloat&   optFlowTimesMS() { return _optFlowTimesMS; }
    SLAvgFloat&   poseTimesMS() { return _poseTimesMS; }
    SLAvgFloat&   updateAnimTimesMS() { return _updateAnimTimesMS; }
    SLAvgFloat&   updateAABBTimesMS() { return _updateAABBTimesMS; }
    SLAvgFloat&   cullTimesMS() { return _cullTimesMS; }
    SLAvgFloat&   draw2DTimesMS() { return _draw2DTimesMS; }
    SLAvgFloat&   draw3DTimesMS() { return _draw3DTimesMS; }
    SLAvgFloat&   captureTimesMS() { return _captureTimesMS; }
    SLAvgFloat&   vsyncTimesMS() { return _vsyncTimesMS; }
    SLVMaterial&  materials() { return _materials; }
    SLVMesh&      meshes() { return _meshes; }
    SLVGLTexture& textures() { return _textures; }
    SLVGLProgram& programs() { return _programs; }
    SLGLProgram*  programs(SLShaderProg i) { return _programs[i]; }
    SLNode*       selectedNode() { return _selectedNode; }
    SLMesh*       selectedMesh() { return _selectedMesh; }
    SLRectf&      selectedRect() { return _selectedRect; }
    SLbool        stopAnimations() const { return _stopAnimations; }
    SLGLOculus*   oculus() { return &_oculus; }
    SLint         numSceneCameras();
    SLCamera*     nextCameraInScene(SLSceneView* activeSV);

    // Video stuff
    SLVideoType   videoType() { return _videoType; }
    SLGLTexture*  videoTexture() { return &_videoTexture; }
    SLGLTexture*  videoTextureErr() { return &_videoTextureErr; }
    SLVCVTracker& trackers() { return _trackers; }
    SLbool        showDetection() { return _showDetection; }

    cbOnSceneLoad onLoad; //!< C-Callback for scene load

    // Misc.
    //virtual  void            onLoad              (SLSceneView* sv, SLCommand _currentID);
    virtual void onLoadAsset(SLstring assetFile,
                             SLuint   processFlags);
    virtual void onAfterLoad(SLSceneView* sv);
    bool         onUpdate();
    void         init();
    void         unInit();
    void         selectNode(SLNode* nodeToSelect);
    void         selectNodeMesh(SLNode* nodeToSelect, SLMesh* meshToSelect);
    bool         removeMesh(SLMesh* mesh);
    bool         deleteTexture(SLGLTexture* texture);

    protected:
    //SLVSceneView    _sceneViews;    //!< Vector of all sceneview pointers
    SLVMesh         _meshes;        //!< Vector of all meshes
    SLVMaterial     _materials;     //!< Vector of all materials pointers
    SLVGLTexture    _textures;      //!< Vector of all texture pointers
    SLVGLProgram    _programs;      //!< Vector of all shader program pointers
    SLVLight        _lights;        //!< Vector of all lights
    SLVEventHandler _eventHandlers; //!< Vector of all event handler
    SLAnimManager   _animManager;   //!< Animation manager instance

    SLNode*  _root3D;       //!< Root node for 3D scene
    SLNode*  _root2D;       //!< Root node for 2D scene displayed in ortho projection
    SLstring _info;         //!< scene info string
    SLNode*  _selectedNode; //!< Pointer to the selected node
    SLMesh*  _selectedMesh; //!< Pointer to the selected mesh
    SLRectf  _selectedRect; //!< Mouse selection rectangle

    SLTimer _timer;           //!< high precision timer
    SLCol4f _globalAmbiLight; //!< global ambient light intensity
    SLbool  _rootInitialized; //!< Flag if scene is initialized
    SLint   _numProgsPreload; //!< No. of preloaded shaderProgs

    SLfloat    _frameTimeMS;      //!< Last frame time in ms
    SLfloat    _lastUpdateTimeMS; //!< Last time after update in ms
    SLfloat    _fps;              //!< Averaged no. of frames per second
    SLAvgFloat _frameTimesMS;     //!< Averaged total time per frame in ms

    SLfloat _startUpdateMS;
    SLfloat _sumCullTimeMS   = 0.0f;
    SLfloat _sumDraw3DTimeMS = 0.0f;
    SLfloat _sumDraw2DTimeMS = 0.0f;

    // major part times
    SLAvgFloat _vsyncTimesMS;   //!< Averaged time for vertical sync in ms
    SLAvgFloat _captureTimesMS; //!< Averaged time for video capturing in ms
    SLAvgFloat _updateTimesMS;  //!< Averaged time for update in ms
    SLAvgFloat _cullTimesMS;    //!< Averaged time for culling in ms
    SLAvgFloat _draw3DTimesMS;  //!< Averaged time for 3D drawing in ms
    SLAvgFloat _draw2DTimesMS;  //!< Averaged time for 2D drawing in ms

    // minor part times of update traversal
    SLAvgFloat _trackingTimesMS;   //!< Averaged time for video tracking in ms
    SLAvgFloat _detectTimesMS;     //!< Averaged time for video feature detection & description in ms
    SLAvgFloat _detect1TimesMS;    //!< Averaged time for video feature detection subpart 1 in ms
    SLAvgFloat _detect2TimesMS;    //!< Averaged time for video feature detection subpart 2 in ms
    SLAvgFloat _matchTimesMS;      //!< Averaged time for video feature matching in ms
    SLAvgFloat _optFlowTimesMS;    //!< Averaged time for video feature optical flow tracking in ms
    SLAvgFloat _poseTimesMS;       //!< Averaged time for video feature pose estimation in ms
    SLAvgFloat _updateAABBTimesMS; //!< Averaged time for update the nodes AABB in ms
    SLAvgFloat _updateAnimTimesMS; //!< Averaged time for update the animations in ms

    SLbool _stopAnimations; //!< Global flag for stopping all animations

    SLGLOculus _oculus; //!< Oculus Rift interface

    // Video stuff
    SLVideoType  _videoType;       //!< Flag for using the live video image
    SLGLTexture  _videoTexture;    //!< Texture for live video image
    SLGLTexture  _videoTextureErr; //!< Texture for live video error
    SLVCVTracker _trackers;        //!< Vector of all AR trackers
    SLbool       _showDetection;   //!< Flag if detection should be visualized
};
//-----------------------------------------------------------------------------
#endif
