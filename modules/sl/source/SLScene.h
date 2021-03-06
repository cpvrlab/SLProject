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

#include <utility>
#include <vector>
#include <map>
#include <SL.h>
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
    SLScene(const SLstring& name,
            cbOnSceneLoad   onSceneLoadCallback);
    ~SLScene() override;

    void initOculus(SLstring shaderDir);

    // Setters
    void root3D(SLNode* root3D) { _root3D = root3D; }
    void root2D(SLNode* root2D) { _root2D = root2D; }
    void stopAnimations(SLbool stop) { _stopAnimations = stop; }
    void info(SLstring i) { _info = std::move(i); }
    void loadTimeMS(SLfloat loadTimeMS) { _loadTimeMS = loadTimeMS; }
    void assetManager(SLAssetManager* am) { _assetManager = am; }

    // Getters
    SLAnimManager&   animManager() { return _animManager; }
    SLAssetManager*  assetManager() { return _assetManager; }
    SLNode*          root3D() { return _root3D; }
    SLNode*          root2D() { return _root2D; }
    SLstring&        info() { return _info; }
    SLfloat          elapsedTimeMS() const { return _frameTimeMS; }
    SLfloat          elapsedTimeSec() const { return _frameTimeMS * 0.001f; }
    SLVEventHandler& eventHandlers() { return _eventHandlers; }
    SLfloat          loadTimeMS() const { return _loadTimeMS; }
    SLVLight&        lights() { return _lights; }
    SLfloat          fps() const { return _fps; }
    AvgFloat&        frameTimesMS() { return _frameTimesMS; }
    AvgFloat&        updateTimesMS() { return _updateTimesMS; }
    AvgFloat&        updateAnimTimesMS() { return _updateAnimTimesMS; }
    AvgFloat&        updateAABBTimesMS() { return _updateAABBTimesMS; }

    //! Returns the node if only one is selected. See also SLMesh::selectNodeMesh
    SLNode* singleNodeSelected() { return _selectedNodes.size() == 1 ? _selectedNodes[0] : nullptr; }

    //! Returns the node if only one is selected. See also SLMesh::selectNodeMesh
    SLMesh*  singleMeshFullSelected() { return (_selectedNodes.size() == 1 &&
                                               _selectedMeshes.size() == 1 &&
                                               _selectedMeshes[0]->IS32.empty())
                                                ? _selectedMeshes[0]
                                                : nullptr; }
    SLVNode& selectedNodes() { return _selectedNodes; }
    SLVMesh& selectedMeshes() { return _selectedMeshes; }

    SLbool    stopAnimations() const { return _stopAnimations; }
    SLint     numSceneCameras();
    SLCamera* nextCameraInScene(SLCamera* activeSVCam);

    cbOnSceneLoad onLoad; //!< C-Callback for scene load

    // Misc.
    bool         onUpdate(bool renderTypeIsRT,
                          bool voxelsAreShown);
    void         init();
    virtual void unInit();
    void         selectNodeMesh(SLNode* nodeToSelect, SLMesh* meshToSelect);
    void         deselectAllNodesAndMeshes();

    SLGLOculus* oculus() { return _oculus.get(); }

protected:
    SLVLight        _lights;        //!< Vector of all lights
    SLVEventHandler _eventHandlers; //!< Vector of all event handler
    SLAnimManager   _animManager;   //!< Animation manager instance
    SLAssetManager* _assetManager;  //!< Pointer to the external assetManager

    SLNode*  _root3D;         //!< Root node for 3D scene
    SLNode*  _root2D;         //!< Root node for 2D scene displayed in ortho projection
    SLstring _info;           //!< scene info string
    SLVNode  _selectedNodes;  //!< Vector of selected nodes. See SLMesh::selectNodeMesh.
    SLVMesh  _selectedMeshes; //!< Vector of selected meshes. See SLMesh::selectNodeMesh.

    SLfloat _loadTimeMS;       //!< time to load scene in ms
    SLfloat _frameTimeMS;      //!< Last frame time in ms
    SLfloat _lastUpdateTimeMS; //!< Last time after update in ms
    SLfloat _fps;              //!< Averaged no. of frames per second

    // major part times
    AvgFloat _frameTimesMS;      //!< Averaged total time per frame in ms
    AvgFloat _updateTimesMS;     //!< Averaged time for update in ms
    AvgFloat _updateAABBTimesMS; //!< Averaged time for update the nodes AABB in ms
    AvgFloat _updateAnimTimesMS; //!< Averaged time for update the animations in ms

    SLbool _stopAnimations; //!< Global flag for stopping all animations

    std::unique_ptr<SLGLOculus> _oculus; //!< Oculus Rift interface
};

#endif
