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
#include <SLBackground.h>
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
All these resources and the scene with all nodes to witch _root3D pointer points
get deleted in the method unInit. A scene could have multiple scene views. 
A pointer of each is stored in the vector _sceneViews. 
The onLoad method can build a of several built in test and demo scenes.
You can access the current scene from everywhere with the static pointer _current.
*/
class SLScene: public SLObject    
{  
    friend class SLNode;
   
    public:

                            SLScene         (SLstring name="");
                           ~SLScene         ();
            // Setters
            void            root3D          (SLNode* root3D){_root3D = root3D;}
            void            globalAmbiLight (SLCol4f gloAmbi){_globalAmbiLight=gloAmbi;}
            void            info            (SLSceneView* sv, SLstring infoText, 
                                             SLCol4f color=SLCol4f::WHITE);
            void            stopAnimations  (SLbool stop) {_stopAnimations = stop;}
            void            infoGL          (SLText* t) {_infoGL = t;}
            void            infoRT          (SLText* t) {_infoRT = t;}
            void            infoLoading     (SLText* t) {_infoLoading = t;}
            void            menu2D          (SLButton* b) {_menu2D = b;}
            void            menuGL          (SLButton* b) {_menuGL = b;}
            void            menuRT          (SLButton* b) {_menuRT = b;}
            void            menuPT          (SLButton* b) {_menuPT = b;}
            void            btnAbout        (SLButton* b) {_btnAbout = b;}
            void            btnCredits      (SLButton* b) {_btnCredits = b;}
            void            btnHelp         (SLButton* b) {_btnHelp = b;}
                           
            // Getters
            SLAnimManager&  animManager     () {return _animManager;}
            SLSceneView*    sv              (SLuint index) {return _sceneViews[index];}
            SLVSceneView&   sceneViews      () {return _sceneViews;}
            SLNode*         root3D          () {return _root3D;}
            SLCommand       currentSceneID  () const {return _currentSceneID;}
            SLBackground&   background      () {return _background;}
            void            timerStart      () {_timer.start();}
            SLfloat         timeSec         () {return (SLfloat)_timer.getElapsedTimeInSec();}
            SLfloat         timeMilliSec    () {return (SLfloat)_timer.getElapsedTimeInMilliSec();}
            SLfloat         elapsedTimeMS   () {return _elapsedTimeMS;}
            SLfloat         elapsedTimeSec  () {return _elapsedTimeMS * 0.001f;}
            SLVEventHandler& eventHandlers  () {return _eventHandlers;}
            SLButton*       menu2D          () {return _menu2D;}
            SLButton*       menuGL          () {return _menuGL;}
            SLButton*       menuRT          () {return _menuRT;}
            SLButton*       menuPT          () {return _menuPT;}
            SLButton*       btnAbout        () {return _btnAbout;}
            SLButton*       btnHelp         () {return _btnHelp;}
            SLButton*       btnCredits      () {return _btnCredits;}
            SLstring        infoAbout_en    () const {return _infoAbout_en;}
            SLstring        infoCredits_en  () const {return _infoCredits_en;}
            SLstring        infoHelp_en     () const {return _infoHelp_en;}
            SLText*         info            (SLSceneView* sv);
            SLText*         info            () {return _info;}
            SLText*         infoGL          () {return _infoGL;}
            SLText*         infoRT          () {return _infoRT;}
            SLText*         infoLoading     () {return _infoLoading;}
            SLGLTexture*    texCursor       () {return _texCursor;}
            SLCol4f         globalAmbiLight () const {return _globalAmbiLight;}
            SLVLight&       lights          () {return _lights;}
            SLfloat         fps             () {return _fps;}
            SLAvgFloat&     frameTimesMS    () {return _frameTimesMS;}
            SLAvgFloat&     updateTimesMS   () {return _updateTimesMS;}
            SLAvgFloat&     cullTimesMS     () {return _cullTimesMS;}
            SLAvgFloat&     draw2DTimesMS   () {return _draw2DTimesMS;}
            SLAvgFloat&     draw3DTimesMS   () {return _draw3DTimesMS;}
            SLVMaterial&    materials       () {return _materials;}
            SLVMesh&        meshes          () {return _meshes;}
            SLVGLTexture&   textures        () {return _textures;}
            SLVGLProgram&   programs        () {return _programs;}
            SLGLProgram*    programs        (SLShaderProg i) {return _programs[i];}
            SLNode*         selectedNode    () {return _selectedNode;}
            SLMesh*         selectedMesh    () {return _selectedMesh;}
            SLbool          stopAnimations  () const {return _stopAnimations;}
            SLGLOculus*     oculus          () {return &_oculus;}   
            SLbool          usesVideoImage  () {return _usesVideoImage;}
            SLGLTexture*    videoTexture    () {return &_videoTexture;}
            
            // Misc.
   virtual  void            onLoad          (SLSceneView* sv, SLCommand _currentID);
   virtual  void            onAfterLoad     ();
            bool            onUpdate();
            void            init            ();
            void            unInit          ();
            void            deleteAllMenus  ();
            SLbool          onCommandAllSV  (const SLCommand cmd);
            void            selectNode      (SLNode* nodeToSelect);
            void            selectNodeMesh  (SLNode* nodeToSelect, SLMesh* meshToSelect);
            void            copyVideoImage  (SLint width, 
                                             SLint height, 
                                             SLPixelFormat srcPixelFormat,
                                             SLuchar* data, 
                                             bool isTopLeft=false);

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
            SLBackground    _background;        //!< Background colors or texture
            SLCol4f         _globalAmbiLight;   //!< global ambient light intensity
            SLCommand       _currentSceneID;    //!< Identifier of current scene
            SLbool          _rootInitialized;   //!< Flag if scene is initialized
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

            SLbool          _usesVideoImage;    //!< Flag for updating the video image
            SLGLTexture     _videoTexture;      //!< Texture for live video image
};
//-----------------------------------------------------------------------------
#endif
