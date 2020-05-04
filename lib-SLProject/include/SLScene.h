//#############################################################################
//  File:      SLScene.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSCENE_H
#define SLSCENE_H

#include <SL.h>
#include <utility>
#include <vector>
#include <map>
#include <SLAnimManager.h>
#include <Averaged.h>
#include <SLGLOculus.h>
#include <SLLight.h>
#include <SLMesh.h>

class SLCamera;

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
 The scene assembly takes place outside of the library in function of the application.
 A pointer for this function must be passed to the SLScene constructor. For the
 demo project this function is in AppDemoSceneLoad.cpp.
*/
class SLScene : public SLObject
{
    friend class SLNode;

public:
    SLScene(const SLstring&      name,
            cbOnSceneLoad onSceneLoadCallback);
    ~SLScene() override;

    // Setters
    void root3D(SLNode* root3D) { _root3D = root3D; }
    void root2D(SLNode* root2D) { _root2D = root2D; }
    void globalAmbiLight(const SLCol4f& gloAmbi) { _globalAmbiLight = gloAmbi; }
    void stopAnimations(SLbool stop) { _stopAnimations = stop; }
    void info(SLstring i) { _info = std::move(i); }

    // Getters
    SLAnimManager&   animManager() { return _animManager; }
    SLNode*          root3D() { return _root3D; }
    SLNode*          root2D() { return _root2D; }
    SLstring&        info() { return _info; }
    SLfloat          elapsedTimeMS() const { return _frameTimeMS; }
    SLfloat          elapsedTimeSec() const { return _frameTimeMS * 0.001f; }
    SLVEventHandler& eventHandlers() { return _eventHandlers; }

    SLCol4f   globalAmbiLight() const { return _globalAmbiLight; }
    SLVLight& lights() { return _lights; }
    SLfloat   fps() const { return _fps; }
    AvgFloat& frameTimesMS() { return _frameTimesMS; }
    AvgFloat& updateTimesMS() { return _updateTimesMS; }
    AvgFloat& updateAnimTimesMS() { return _updateAnimTimesMS; }
    AvgFloat& updateAABBTimesMS() { return _updateAABBTimesMS; }

    SLNode*   selectedNode() { return _selectedNode; }
    SLMesh*   selectedMesh() { return _selectedMesh; }
    SLbool    stopAnimations() const { return _stopAnimations; }
    SLint     numSceneCameras();
    SLCamera* nextCameraInScene(SLCamera* activeSVCam);

    cbOnSceneLoad onLoad; //!< C-Callback for scene load

    // Misc.
    bool         onUpdate(bool renderTypeIsRT,
                          bool voxelsAreShown);
    void         init();
    virtual void unInit();
    void         selectNode(SLNode* nodeToSelect);
    void         selectNodeMesh(SLNode* nodeToSelect, SLMesh* meshToSelect);
    SLGLOculus* oculus() { return &_oculus; }

protected:
    SLVLight        _lights;        //!< Vector of all lights
    SLVEventHandler _eventHandlers; //!< Vector of all event handler
    SLAnimManager   _animManager;   //!< Animation manager instance

    SLNode*  _root3D;       //!< Root node for 3D scene
    SLNode*  _root2D;       //!< Root node for 2D scene displayed in ortho projection
    SLstring _info;         //!< scene info string
    SLNode*  _selectedNode; //!< Pointer to the selected node
    SLMesh*  _selectedMesh; //!< Pointer to the selected mesh

    SLCol4f _globalAmbiLight; //!< global ambient light intensity
    SLbool  _rootInitialized; //!< Flag if scene is initialized

    SLfloat _frameTimeMS;      //!< Last frame time in ms
    SLfloat _lastUpdateTimeMS; //!< Last time after update in ms
    SLfloat _fps;              //!< Averaged no. of frames per second

    // major part times
    AvgFloat _frameTimesMS;      //!< Averaged total time per frame in ms
    AvgFloat _updateTimesMS;     //!< Averaged time for update in ms
    AvgFloat _updateAABBTimesMS; //!< Averaged time for update the nodes AABB in ms
    AvgFloat _updateAnimTimesMS; //!< Averaged time for update the animations in ms

    SLbool _stopAnimations; //!< Global flag for stopping all animations

    SLGLOculus _oculus; //!< Oculus Rift interface
};

#endif
