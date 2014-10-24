//#############################################################################
//  File:      SLScene.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#ifdef SL_MEMLEAKDETECT     // set in SL.h for debug config only
#include <debug_new.h>      // memory leak detector
#endif

#include <SLScene.h>
#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLText.h>
#include <SLLight.h>
#include <SLTexFont.h>
#include <SLButton.h>

//-----------------------------------------------------------------------------
/*! Global static scene pointer that can be used throughout the entire library
to access the current scene and its sceneviews. 
*/
SLScene* SLScene::current = 0;
//-----------------------------------------------------------------------------
/*! The constructor of the scene does all one time initialization such as 
loading the standard shader programs from which the pointers are stored in
the dynamic array _shaderProgs. Custom shader programs that are loaded in a
scene must be deleted when the scene changes.
The following standard shaders are preloaded:
  - ColorAttribute.vert, Color.frag
  - ColorUniform.vert, Color.frag
  - PerVrtBlinn.vert, PerVrtBlinn.frag
  - PerVrtBlinnTex.vert, PerVrtBlinnTex.frag
  - TextureOnly.vert, TextureOnly.frag
  - PerPixBlinn.vert, PerPixBlinn.frag
  - PerPixBlinnTex.vert, PerPixBlinnTex.frag
  - BumpNormal.vert, BumpNormal.frag
  - BumpNormal.vert, BumpNormalParallax.frag
  - FontTex.vert, FontTex.frag
  - StereoOculus.vert, StereoOculus.frag
  - StereoOculusDistortionMesh.vert, StereoOculusDistortionMesh.frag

There will be only one scene for an application and it gets constructed in
the C-interface function SLInterface::slCreateScene that is called by the platform
and GUI-toolkit dependent window initialization. As examples you can see it in:
  - app-Demo-GLFW: glfwMain.cpp in function main()
  - app-Demo-Qt: qtGLWidget::initializeGL()
  - app-Viewer-Qt: qtGLWidget::initializeGL()
  - app-Demo-Android: Java_ch_fhnw_comgr_GLES2Lib_onInit()
  - app-Demo-iOS: ViewController.m in method viewDidLoad()
*/
SLScene::SLScene(SLstring name) : SLObject(name)
{  
    current = this;

    _root3D       = 0;
    _menu2D       = 0;
    _menuGL       = 0;
    _menuRT       = 0;
    _menuPT       = 0;
    _info         = 0;
    _infoGL       = 0;
    _infoRT       = 0;
    _infoLoading  = 0;
    _btnHelp      = 0;
    _btnAbout     = 0;
    _btnCredits   = 0;
    _selectedMesh = 0;
    _selectedNode = 0;
     
    // Load std. shader programs in order as defined in SLStdShaderProgs enum
    // In the constructor they are added the _shaderProgs vector
    SLGLShaderProg* p;
    p = new SLGLShaderProgGeneric("ColorAttribute.vert","Color.frag");
    p = new SLGLShaderProgGeneric("ColorUniform.vert","Color.frag");
    p = new SLGLShaderProgGeneric("PerVrtBlinn.vert","PerVrtBlinn.frag");
    p = new SLGLShaderProgGeneric("PerVrtBlinnTex.vert","PerVrtBlinnTex.frag");
    p = new SLGLShaderProgGeneric("TextureOnly.vert","TextureOnly.frag");
    p = new SLGLShaderProgGeneric("PerPixBlinn.vert","PerPixBlinn.frag");
    p = new SLGLShaderProgGeneric("PerPixBlinnTex.vert","PerPixBlinnTex.frag");
    p = new SLGLShaderProgGeneric("BumpNormal.vert","BumpNormal.frag");
    p = new SLGLShaderProgGeneric("BumpNormal.vert","BumpNormalParallax.frag");
    p = new SLGLShaderProgGeneric("FontTex.vert","FontTex.frag");
    p = new SLGLShaderProgGeneric("StereoOculus.vert","StereoOculus.frag");
    p = new SLGLShaderProgGeneric("StereoOculusDistortionMesh.vert","StereoOculusDistortionMesh.frag");
    _numProgsPreload = (SLint)_shaderProgs.size();
   
    // Generate std. fonts   
    SLTexFont::generateFonts();

    _infoAbout_en =
"Welcome to the SLProject demo app (v1.0.000). It is developed at the \
Computer Science Department of the Berne University of Applied Sciences. \
The app shows what you can learn in one semester about 3D computer graphics \
in real time rendering and ray tracing. The framework is developed \
in C++ with OpenGL ES2 so that it can run also on mobile devices. \
Ray tracing provides in addition highquality transparencies, reflections and soft shadows. \
Click to close and use the menu to choose different scenes and view settings. \
For more information please visit: http://code.google.com/p/slproject/";

    _infoCredits_en =
"Credits for external libraries: \\n\
- assimp: assimp.sourceforge.net \\n\
- glew: glew.sourceforge.net \\n\
- glfw: www.glfw.org \\n\
- jpeg: www.jpeg.org \\n\
- nvwa: sourceforge.net/projects/nvwa \\n\
- png: www.libpng.org \\n\
- Qt: www.qt-project.org \\n\
- randomc: www.agner.org/random \\n\
- zlib: zlib.net";

    _infoHelp_en =
"Help for mouse or finger control: \\n\
- Use mouse or your finger to rotate the scene \\n\
- Use mouse-wheel or pinch 2 fingers to go forward/backward \\n\
- Use CTRL-mouse or 2 fingers to move sidewards/up-down \\n\
- Double click or double tap to select object \\n\
- Screenshot: Use a screenshot tool,\\n\
on iOS: Quick hold down home & power button, \\n\
on Android: Quick hold down back & home button \\n\
on desktop: Use a screenshot tool";
}
//-----------------------------------------------------------------------------
/*! The destructor does the final total deallocation of all global resources.
The destructor is called in slTerminate.
*/
SLScene::~SLScene()
{
    // Delete all remaining sceneviews
    for (SLint i = 0; i < _sceneViews.size(); ++i)
        if (_sceneViews[i])
            delete _sceneViews[i];

    unInit();
   
    // delete global SLGLState instance
    SLGLState::deleteInstance();

    // clear light pointers
    _lights.clear();
   
    // delete materials 
    for (SLuint i=0; i<_materials.size(); ++i) delete _materials[i];
        _materials.clear();
   
    // delete materials 
    for (SLuint i=0; i<_meshes.size(); ++i) delete _meshes[i];
        _meshes.clear();
   
    // delete textures
    for (SLuint i=0; i<_textures.size(); ++i) delete _textures[i];
        _textures.clear();
   
    // delete shader programs
    for (SLuint i=0; i<_shaderProgs.size(); ++i) delete _shaderProgs[i];
        _shaderProgs.clear();
   
    // delete fonts   
    SLTexFont::deleteFonts();
   
    // delete menus & statistic texts
    delete _menuGL;     _menuGL     = 0;
    delete _menuRT;     _menuRT     = 0;
    delete _menuPT;     _menuPT     = 0;
    delete _info;       _info       = 0;
    delete _infoGL;     _infoGL     = 0;
    delete _infoRT;     _infoRT     = 0;
    delete _btnAbout;   _btnAbout   = 0;
    delete _btnHelp;    _btnHelp    = 0;
    delete _btnCredits; _btnCredits = 0;
   
    current = 0;

    SL_LOG("~SLScene\n");
    SL_LOG("------------------------------------------------------------------\n");
}
//-----------------------------------------------------------------------------
/*! The scene init is called whenever the scene is new loaded.
*/
void SLScene::init()
{     
    unInit();
   
    _backColor.set(0.1f,0.4f,0.8f,1.0f);
    _globalAmbiLight.set(0.2f,0.2f,0.2f,0.0f);
    _selectedNode = 0;

    _timer.start();

    // load virtual cursor texture
    _texCursor = new SLGLTexture("cursor.tga");
}
//-----------------------------------------------------------------------------
/*! The scene uninitializing clears the scenegraph (_root3D) and all global
global resources such as materials, textures & custom shaders loaded with the 
scene. The standard shaders, the fonts and the 2D-GUI elements remain. They are
destructed at process end.
*/
void SLScene::unInit()
{  
    _selectedMesh = 0;
    _selectedNode = 0;

    // reset existing sceneviews
    for (SLint i = 0; i < _sceneViews.size(); ++i)
    {   if(_sceneViews[i] != NULL) 
        {   
            // Resets sceneview camera to their own
            _sceneViews[i]->camera(_sceneViews[i]->sceneViewCamera());
        }
    }

    // delete entire scene graph
    delete _root3D;
    _root3D = 0;

    // clear light pointers
    _lights.clear();

    // delete textures
    for (SLuint i=0; i<_textures.size(); ++i) 
        delete _textures[i];
    _textures.clear();
   
    // manually clear the default material (it will get deleted below)
    SLMaterial::defaultMaterial(NULL);
    
    // delete materials 
    for (SLuint i=0; i<_materials.size(); ++i) 
        delete _materials[i];
    _materials.clear();

    // delete meshs 
    for (SLuint i=0; i<_meshes.size(); ++i) 
        delete _meshes[i];
    _meshes.clear();
   
    SLMaterial::current = 0;
   
    // delete custom shader programs but not default shaders
    while (_shaderProgs.size() > _numProgsPreload) 
    {   SLGLShaderProg* sp = _shaderProgs.back();
        delete sp;
        _shaderProgs.pop_back();
    }
   
    // clear eventHandlers
    _eventHandlers.clear();

    // reset all states
    SLGLState::getInstance()->initAll();
}
//-----------------------------------------------------------------------------
/*!
SLScene::info deletes previous info text and sets new one with a max. width 
*/
void SLScene::info(SLSceneView* sv, SLstring infoText, SLCol4f color)
{  
    delete _info;
   
    // Set font size depending on DPI
    SLTexFont* f = SLTexFont::getFont(1.5f, sv->dpi());

    SLfloat minX = 11 * sv->dpmm();
    _info = new SLText(infoText, f, color, 
                       sv->scrW()-minX-5.0f,
                       1.2f);

    _info->translate(minX, SLButton::minMenuPos.y, 0, TS_Local);
}
//-----------------------------------------------------------------------------
/*! 
SLScene::info returns the info text. If null it creates an empty one
*/
SLText* SLScene::info(SLSceneView* sv)
{
    if (_info == 0) info(sv, "", SLCol4f::WHITE);
    return _info;
}
//-----------------------------------------------------------------------------
//! Sets the _selectedNode to the passed Node and flags it as selected
void SLScene::selectNode(SLNode* nodeToSelect)
{
    if (_selectedNode)
        _selectedNode->drawBits()->off(SL_DB_SELECTED);

    if (nodeToSelect)
    {  if (_selectedNode == nodeToSelect)
        {   _selectedNode = 0;
        } else
        {   _selectedNode = nodeToSelect;
            _selectedNode->drawBits()->on(SL_DB_SELECTED);
        }
    } else _selectedNode = 0;
}
//-----------------------------------------------------------------------------
//! Sets the _selectedNode and _selectedMesh and flags it as selected
void SLScene::selectNodeMesh(SLNode* nodeToSelect, SLMesh* meshToSelect)
{
    if (_selectedNode)
        _selectedNode->drawBits()->off(SL_DB_SELECTED);

    if (nodeToSelect)
    {  if (_selectedNode == nodeToSelect && _selectedMesh == meshToSelect)
        {   _selectedNode = 0;
            _selectedMesh = 0;
        } else
        {   _selectedNode = nodeToSelect;
            _selectedMesh = meshToSelect;
            _selectedNode->drawBits()->on(SL_DB_SELECTED);
        }
    } else 
    {   _selectedNode = 0;
        _selectedMesh = 0;
    }
}
//-----------------------------------------------------------------------------
//! Executes a command on all sceneview
SLbool SLScene::onCommandAllSV(const SLCmd cmd)
{
    SLbool result = false;
    for(SLint i=0; i<_sceneViews.size(); ++i)
    {
        result = _sceneViews[i]->onCommand(cmd) ? true : result;
    }
    return true;
}
//-----------------------------------------------------------------------------
