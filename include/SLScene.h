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

#include <stdafx.h>
#include <SLMaterial.h>
#include <SLEventHandler.h>
#include <SLLight.h>
#include <SLNode.h>
#include <SLSkeleton.h>
#include <SLGLOculus.h>
#include <SLAnimManager.h>
#include <SLAverage.h>

class SLSceneView;
class SLButton;
class SLText;

//-----------------------------------------------------------------------------
typedef std::vector<SLSceneView*> SLVSceneView; //!< Vector of SceneView pointers
//-----------------------------------------------------------------------------
//! The SLScene class represents the top level instance holding the scene structure
/*!      
The SLScene class holds everything that is common for all scene views such as 
the root pointer (_root3D) to the scene, the background color, an array of
lights as well as the global resources (_meshes (SLMesh), _materials (SLMaterial), 
_textures (SLGLTexture) and _shaderProgs (SLGLProgram)).
All these resources and the scene with all nodes to whitch _root3D pointer points
get deleted in the method unInit. A scene could have multiple scene views. 
A pointer of each is stored in the vector _sceneViews. 
The onLoad method can build a of several built in test and demo scenes.
You can access the current scene from everywhere with the static pointer _current.
*/
class SLScene: public SLObject    
{  
    friend class SLNode;
    friend class SLSceneView;
   
    public:           
                            SLScene         (SLstring name="");
                           ~SLScene         ();
            // Setters
            void            root3D          (SLNode* root3D){_root3D = root3D;}
            void            menu2D          (SLButton* menu2D){_menu2D = menu2D;}
            void            backColor       (SLCol4f backColor){_backColor=backColor;}
            void            globalAmbiLight (SLCol4f gloAmbi){_globalAmbiLight=gloAmbi;}
            void            info            (SLSceneView* sv, SLstring infoText, 
                                             SLCol4f color=SLCol4f::WHITE);
            void            stopAnimations  (SLbool stop){_stopAnimations = stop;}
                           
            // Getters
     inline SLAnimManager&  animManager     () {return _animManager;}
     inline SLSceneView*    sv              (SLuint index) {return _sceneViews[index];}
     inline SLNode*         root3D          () {return _root3D;}
            SLint           currentID       () const {return _currentID;}
            SLfloat         timeSec         () {return (SLfloat)_timer.getElapsedTimeInSec();}
            SLfloat         timeMilliSec    () {return (SLfloat)_timer.getElapsedTimeInMilliSec();}
            SLfloat         elapsedTimeSec  () {return _elapsedTimeMS * 0.001f;}
            SLButton*       menu2D          () {return _menu2D;}
            SLButton*       menuGL          () {return _menuGL;}
            SLGLTexture*    texCursor       () {return _texCursor;}
            SLCol4f         globalAmbiLight () const {return _globalAmbiLight;}
            SLCol4f         backColor       () const {return _backColor;}
            SLCol4f*        backColorV      () {return &_backColor;}
            SLVLight&       lights          () {return _lights;}
            SLVEventHandler& eventHandlers  () {return _eventHandlers;}
            SLVMaterial&    materials       () {return _materials;}
            SLVMesh&        meshes          () {return _meshes;}
            SLVGLTexture&   textures        () {return _textures;}
            SLVGLProgram&   programs        () {return _programs;}
            SLGLProgram*    programs        (SLStdShaderProg i) {return _programs[i];}
            SLText*         info            (SLSceneView* sv);
            SLstring        infoAbout_en    () const {return _infoAbout_en;}
            SLstring        infoCredits_en  () const {return _infoCredits_en;}
            SLstring        infoHelp_en     () const {return _infoHelp_en;}
            SLNode*         selectedNode    () {return _selectedNode;}
            SLMesh*         selectedMesh    () {return _selectedMesh;}
            SLbool          stopAnimations  () const {return _stopAnimations;}
            SLGLOculus*     oculus          () {return &_oculus;}   
            
            // Misc.
   virtual  void            onLoad          (SLSceneView* sv, SLCmd sceneName);
            void            init            ();
            void            unInit          ();
            bool            onUpdate        ();
            SLbool          onCommandAllSV  (const SLCmd cmd);
            void            selectNode      (SLNode* nodeToSelect);
            void            selectNodeMesh  (SLNode* nodeToSelect,
                                             SLMesh* meshToSelect);

     static SLScene*        current;            //!< global static scene pointer

   protected:
            SLVSceneView    _sceneViews;        //!< Vector of all sceneview pointers
            SLVMesh         _meshes;            //!< Vector of all meshes
            SLVMaterial     _materials;         //!< Vector of all materials pointers
            SLVGLTexture    _textures;          //!< Vector of all texture pointers
            SLVGLProgram    _programs;          //!< Vector of all shader program pointers
            SLVLight        _lights;            //!< Vector of all lights
            SLVEventHandler _eventHandlers;     //!< Vector of all event handler
            SLAnimManager   _animManager;       //!< Animation manager instance
            
            SLNode*         _root3D;            //!< Root node for 3D scene
            SLNode*         _selectedNode;      //!< Pointer to the selected node
            SLMesh*         _selectedMesh;      //!< Pointer to the selected mesh

            SLTimer         _timer;             //!< high precision timer
            SLCol4f         _backColor;         //!< Background color
            SLCol4f         _globalAmbiLight;   //!< global ambient light intensity
            SLint           _currentID;         //!< Identifier of current scene
            SLbool          _rootInitialized;   //!< Flag if scene is intialized
            SLint           _numProgsPreload;   //!< No. of preloaded shaderProgs
            
            SLText*         _info;              //!< Text node for scene info
            SLText*         _infoGL;            //!< Root text node for 2D GL stats infos
            SLText*         _infoRT;            //!< Root text node for 2D RT stats infos
            SLText*         _infoLoading;       //!< Root text node for 2D loading text
            SLstring        _infoAbout_en;      //!< About info text
            SLstring        _infoCredits_en;    //!< Credits info text
            SLstring        _infoHelp_en;       //!< Help info text

            SLButton*       _menu2D;            //!< Root button node for 2D GUI
            SLButton*       _menuGL;            //!< Root button node for OpenGL menu
            SLButton*       _menuRT;            //!< Root button node for RT menu
            SLButton*       _menuPT;            //!< Root button node for PT menu
            SLButton*       _btnAbout;          //!< About button
            SLButton*       _btnHelp;           //!< Help button
            SLButton*       _btnCredits;        //!< Credits button
            SLGLTexture*    _texCursor;         //!< Texture for the virtual cursor
            
            SLfloat         _elapsedTimeMS;     //!< Last frame time in ms
            SLfloat         _lastUpdateTimeMS;  //!< Last time after update in ms
            SLfloat         _fps;               //!< Averaged no. of frames per second
            SLAvgFloat      _updateTimesMS;     //!< Averaged time for update in ms
            SLAvgFloat      _frameTimesMS;      //!< Averaged time per frame in ms
            SLAvgFloat      _cullTimesMS;       //!< Averaged time for culling in ms
            SLAvgFloat      _draw3DTimesMS;     //!< Averaged time for 3D drawing in ms
            SLAvgFloat      _draw2DTimesMS;     //!< Averaged time for 2D drawing in ms
            
            SLbool          _stopAnimations;    //!< Global flag for stopping all animations
            
            SLGLOculus      _oculus;            //!< Oculus Rift interface
};
//-----------------------------------------------------------------------------
#endif
