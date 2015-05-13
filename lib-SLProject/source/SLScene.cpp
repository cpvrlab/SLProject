//#############################################################################
//  File:      SLScene.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
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
#include <SLAnimation.h>
#include <SLAnimManager.h>
#include <SLInputManager.h>

//-----------------------------------------------------------------------------
/*! Global static scene pointer that can be used throughout the entire library
to access the current scene and its sceneviews. 
*/
SLScene* SLScene::current = nullptr;
//-----------------------------------------------------------------------------
/*! The constructor of the scene does all one time initialization such as 
loading the standard shader programs from which the pointers are stored in
the vector _shaderProgs. Custom shader programs that are loaded in a
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
the C-interface function slCreateScene in SLInterface.cpp that is called by the
platform and GUI-toolkit dependent window initialization.
As examples you can see it in:
  - app-Demo-GLFW: glfwMain.cpp in function main()
  - app-Demo-Qt: qtGLWidget::initializeGL()
  - app-Viewer-Qt: qtGLWidget::initializeGL()
  - app-Demo-Android: Java_ch_fhnw_comgr_GLES2Lib_onInit()
  - app-Demo-iOS: ViewController.m in method viewDidLoad()
*/
SLScene::SLScene(SLstring name) : SLObject(name)
{  
    current = this;

    _root3D       = nullptr;
    _menu2D       = nullptr;
    _menuGL       = nullptr;
    _menuRT       = nullptr;
    _menuPT       = nullptr;
    _info         = nullptr;
    _infoGL       = nullptr;
    _infoRT       = nullptr;
    _infoLoading  = nullptr;
    _btnHelp      = nullptr;
    _btnAbout     = nullptr;
    _btnCredits   = nullptr;
    _selectedMesh = nullptr;
    _selectedNode = nullptr;
    _stopAnimations = false;

    _fps = 0;
    _elapsedTimeMS = 0;
    _lastUpdateTimeMS = 0;
    _frameTimesMS.init();
    _updateTimesMS.init();
    _cullTimesMS.init();
    _draw3DTimesMS.init();
    _draw2DTimesMS.init();
     
    // Load std. shader programs in order as defined in SLStdShaderProgs enum
    // In the constructor they are added the _shaderProgs vector
    SLGLProgram* p;
    p = new SLGLGenericProgram("ColorAttribute.vert","Color.frag");
    p = new SLGLGenericProgram("ColorUniform.vert","Color.frag");
    p = new SLGLGenericProgram("PerVrtBlinn.vert","PerVrtBlinn.frag");
    p = new SLGLGenericProgram("PerVrtBlinnTex.vert","PerVrtBlinnTex.frag");
    p = new SLGLGenericProgram("TextureOnly.vert","TextureOnly.frag");
    p = new SLGLGenericProgram("PerPixBlinn.vert","PerPixBlinn.frag");
    p = new SLGLGenericProgram("PerPixBlinnTex.vert","PerPixBlinnTex.frag");
    p = new SLGLGenericProgram("BumpNormal.vert","BumpNormal.frag");
    p = new SLGLGenericProgram("BumpNormal.vert","BumpNormalParallax.frag");
    p = new SLGLGenericProgram("FontTex.vert","FontTex.frag");
    p = new SLGLGenericProgram("StereoOculus.vert","StereoOculus.frag");
    p = new SLGLGenericProgram("StereoOculusDistortionMesh.vert","StereoOculusDistortionMesh.frag");
    _numProgsPreload = (SLint)_programs.size();
   
    // Generate std. fonts   
    SLTexFont::generateFonts();

    _infoAbout_en =
"Welcome to the SLProject demo app (v1.1.100). It is developed at the \
Computer Science Department of the Berne University of Applied Sciences. \
The app shows what you can learn in one semester about 3D computer graphics \
in real time rendering and ray tracing. The framework is developed \
in C++ with OpenGL ES2 so that it can run also on mobile devices. \
Ray tracing provides in addition highquality transparencies, reflections and soft shadows. \
Click to close and use the menu to choose different scenes and view settings. \
For more information please visit: https://github.com/cpvrlab/SLProject";

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
    for (auto sv : _sceneViews)
        if (sv != nullptr)
            delete sv;

    unInit();
   
    // delete global SLGLState instance
    SLGLState::deleteInstance();

    // clear light pointers
    _lights.clear();
   
    // delete materials 
    for (auto m : _materials) delete m;
        _materials.clear();
   
    // delete materials 
    for (auto m : _meshes) delete m;
        _meshes.clear();
   
    // delete textures
    for (auto t : _textures) delete t;
        _textures.clear();
   
    // delete shader programs
    for (auto p : _programs) delete p;
        _programs.clear();
   
    // delete fonts   
    SLTexFont::deleteFonts();
   
    // delete menus & statistic texts
    delete _menuGL;     _menuGL     = nullptr;
    delete _menuRT;     _menuRT     = nullptr;
    delete _menuPT;     _menuPT     = nullptr;
    delete _info;       _info       = nullptr;
    delete _infoGL;     _infoGL     = nullptr;
    delete _infoRT;     _infoRT     = nullptr;
    delete _btnAbout;   _btnAbout   = nullptr;
    delete _btnHelp;    _btnHelp    = nullptr;
    delete _btnCredits; _btnCredits = nullptr;
   
    current = nullptr;

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
    _selectedMesh = nullptr;
    _selectedNode = nullptr;

    // reset existing sceneviews
    for (auto sv : _sceneViews)
        if (sv != nullptr)
            sv->camera(sv->sceneViewCamera());

    // delete entire scene graph
    delete _root3D;
    _root3D = nullptr;

    // clear light pointers
    _lights.clear();

    // delete textures
    for (auto t : _textures) delete t;
    _textures.clear();
   
    // manually clear the default material (it will get deleted below)
    SLMaterial::defaultMaterial(nullptr);
    
    // delete materials 
    for (auto m : _materials) delete m;
    _materials.clear();

    // delete meshs 
    for (auto m : _meshes) delete m;
    _meshes.clear();
   
    SLMaterial::current = nullptr;
   
    // delete custom shader programs but not default shaders
    while (_programs.size() > _numProgsPreload)
    {   SLGLProgram* sp = _programs.back();
        delete sp;
        _programs.pop_back();
    }
   
    // clear eventHandlers
    _eventHandlers.clear();

    _animManager.clear();

    // reset all states
    SLGLState::getInstance()->initAll();
}
//-----------------------------------------------------------------------------
//! Updates all animations in the scene after all views got painted.
/*! Updates all animations in the scene after all views got painted and
calculates the elapsed time for one frame in all views. A scene can be displayed
in multiple views as demonstrated in the app-Viewer-Qt example.
\return true if realy something got updated
*/
bool SLScene::onUpdate()
{
    // Return if not all sceneview got repainted
    for (auto sv : _sceneViews)
        if (sv != nullptr && !sv->gotPainted())
            return false;

    // Reset all _gotPainted flags
    for (auto sv : _sceneViews)
        if (sv != nullptr)
            sv->gotPainted(false);

    // Calculate the elapsed time for the animation
    _elapsedTimeMS = timeMilliSec() - _lastUpdateTimeMS;
    _lastUpdateTimeMS = timeMilliSec();

    // Sum up times of all scene views
    SLfloat sumCullTimeMS   = 0.0f;
    SLfloat sumDraw3DTimeMS = 0.0f;
    SLfloat sumDraw2DTimeMS = 0.0f;
    SLbool renderTypeIsRT = false;
    SLbool voxelsAreShown = false;

    for (auto sv : _sceneViews)
    {   if (sv != nullptr)
        {   sumCullTimeMS   += sv->cullTimeMS();
            sumDraw3DTimeMS += sv->draw3DTimeMS();
            sumDraw2DTimeMS += sv->draw2DTimeMS();
            if (!renderTypeIsRT && sv->renderType()==renderRT)
                renderTypeIsRT = true;
            if (!voxelsAreShown && sv->drawBit(SL_DB_VOXELS))
                voxelsAreShown = true;
        }
    }
    _cullTimesMS.set(sumCullTimeMS);
    _draw3DTimesMS.set(sumDraw3DTimeMS);
    _draw2DTimesMS.set(sumDraw2DTimeMS);

    // Calculate the frames per second metric
    _frameTimesMS.set(_elapsedTimeMS);
    _fps = 1 / _frameTimesMS.average() * 1000.0f;
    if (_fps < 0.0f) _fps = 0.0f;

    // Do animations
    SLfloat startUpdateMS = timeMilliSec();

    // reset the dirty flag on all skeletons
    // @todo    put this functionality in the anim manager
    // @note    This would not be necessary if we had a 1 to 1 relationship of meshes to skeletons
    //          then  the mesh could just mark the skeleton as clean after retrieving the new data.
    //          Currently however we could have multiple meshes that reference the same skeleton.
    //          This could be solved by taking the mesh/submesh architecture approach. All logically
    //          grouped meshes are submeshes to one mesh. For example a character with glasses and clothes
    //          would consist of a submesh for the glasses, the clothing and the character body. The 
    //          skeleton data update would then be done on mesh level which in turn updates all of its submeshes.
    //
    //          For now we need to reset the dirty flag manually at the start of each frame because of the above note.
    for(auto skeleton : _animManager.skeletons())
        skeleton->changed(false);

    // Process queued up system events and poll custom input devices
    SLbool animatedOrChanged = SLInputManager::instance().pollEvents();

    ////////////////////////////////////////////////////////////////////////////
    animatedOrChanged |= !_stopAnimations && _animManager.update(elapsedTimeSec());
    ////////////////////////////////////////////////////////////////////////////
    
    // Do software skinning on all changed skeletons
    for (auto mesh : _meshes) 
    {   if (mesh->skeleton() && 
            mesh->skeleton()->changed() && 
            mesh->skinMethod() == SM_SoftwareSkinning)
        {   mesh->transformSkin();
            animatedOrChanged = true;
        }

        // update any out of date acceleration structure for RT or if they're being rendered.
        if (renderTypeIsRT || voxelsAreShown)
            mesh->updateAccelStruct();
    }
    
    // Update AABBs efficiently. The updateAABBRec call won't generate any overhead if nothing changed
    SLGLState::getInstance()->modelViewMatrix.identity();
    _root3D->updateAABBRec();


    _updateTimesMS.set(timeMilliSec()-startUpdateMS);
    return animatedOrChanged;
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

    _info->translate(minX, SLButton::minMenuPos.y, 0, TS_Object);
}
//-----------------------------------------------------------------------------
/*! 
SLScene::info returns the info text. If null it creates an empty one
*/
SLText* SLScene::info(SLSceneView* sv)
{
    if (_info == nullptr) info(sv, "", SLCol4f::WHITE);
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
    for(auto sv : _sceneViews)
        if (sv != nullptr)
            result = sv->onCommand(cmd) ? true : result;

    return true;
}
//-----------------------------------------------------------------------------
