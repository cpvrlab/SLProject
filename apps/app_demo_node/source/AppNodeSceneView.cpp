//#############################################################################
//  File:      AppNodeSceneView.cpp
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <AppDemo.h>
#include <SLAssimpImporter.h>
#include <SLAssetManager.h>
#include <SLBox.h>
#include <SLGLVertexArrayExt.h>
#include <SLLightSpot.h>
#include <SLTexFont.h>
#include <GlobalTimer.h>

#include "AppNodeGui.h"
#include "AppNodeSceneView.h"

//-----------------------------------------------------------------------------
/*! AppNodeSceneView inherits the base class SLSceneView and overrides some
 eventhandler.
 Most events such as all mouse and keyboard events from the OS is forwarded to
 SLSceneview. SLSceneview implements a default behaviour. If you want a
 different or additional behaviour for a certain eventhandler you have to sub-
 class SLSceneView and override the eventhandler.
 */
AppNodeSceneView::AppNodeSceneView(SLScene*        s,
                                   int             dpi,
                                   SLInputManager& inputManager)
  : SLSceneView(s, dpi, inputManager),
    _modifiers(K_none),
    _continuousInput(true),
    _curMode(TranslationMode),
    _curObject(nullptr),
    _curSpace(TS_parent)
{
    for (bool& ks : _keyStates)
        ks = false;

    _pivotPos.set(0, 0, 0);
}
//-----------------------------------------------------------------------------
AppNodeSceneView::~AppNodeSceneView()
{
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::postSceneLoad()
{
    assert(_s->assetManager() && "No asset manager assigned to scene");

    SLAssetManager* am   = _s->assetManager();
    SLMaterial*     rMat = new SLMaterial(am, "rMat", SLCol4f(1.0f, 0.7f, 0.7f));
    SLMaterial*     gMat = new SLMaterial(am, "gMat", SLCol4f(0.7f, 1.0f, 0.7f));

    // build parent box
    _moveBox = new SLNode("Parent");
    _moveBox->translation(0, 0, 2);
    _moveBox->rotation(22.5f, SLVec3f(0, -1, 0));
    _moveBox->addMesh(new SLBox(am, -0.3f, -0.3f, -0.3f, 0.3f, 0.3f, 0.3f, "Box", rMat));
    _moveBox->setInitialState();

    // build child box
    _moveBoxChild = new SLNode("Child");
    _moveBoxChild->translation(0, 1, 0);
    _moveBoxChild->rotation(22.5f, SLVec3f(0, -1, 0));
    _moveBoxChild->setInitialState();
    _moveBoxChild->addMesh(new SLBox(am, -0.2f, -0.2f, -0.2f, 0.2f, 0.2f, 0.2f, "Box", gMat));
    _moveBox->addChild(_moveBoxChild);

    // load coordinate axis arrows
    SLAssimpImporter importer;
    _axesNode = importer.load(_s->animManager(),
                              am,
                              AppDemo::modelPath + "FBX/Axes/axes_blender.fbx",
                              AppDemo::texturePath);

    _s->root3D()->addChild(_moveBox);
    _s->root3D()->addChild(_axesNode);

    if (!_curObject)
    {
        _curObject = _moveBoxChild;
        _s->selectNodeMesh(_curObject, _curObject->mesh());
    }
    updateInfoText();
    updateCurOrigin();
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::preDraw()
{
    static SLfloat lastTime    = GlobalTimer::timeS();
    SLfloat        currentTime = GlobalTimer::timeS();
    _deltaTime                 = currentTime - lastTime;
    lastTime                   = currentTime;

    if (_keyStates['W'])
        if (_deltaTime > 0.1f)
            cout << _deltaTime << endl;

    bool updated = false;
    for (int i = 0; i < 65536; ++i)
    {
        if (_keyStates[i])
            updated = onContinuousKeyPress((SLKey)i);
    }

    if (updated)
    {
        updateInfoText();
        updateCurOrigin();
    }
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::reset()
{
    _pivotPos.set(0, 0, 0);
    _moveBox->resetToInitialState();
    _moveBoxChild->resetToInitialState();
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::translateObject(SLVec3f val) const
{
    if (_continuousInput)
        val *= _deltaTime;
    else
        val *= 0.1f;

    _curObject->translate(val, _curSpace);
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::rotateObject(const SLVec3f& val) const
{
    SLfloat angle = 22.5;
    if (_continuousInput)
        angle *= _deltaTime;

    _curObject->rotate(angle, val, _curSpace);
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::rotateObjectAroundPivot(SLVec3f val) const
{
    SLfloat angle = 22.5;
    if (_continuousInput)
        angle *= _deltaTime;

    _curObject->rotateAround(_pivotPos, val, angle, _curSpace);
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::translatePivot(SLVec3f val)
{
    if (_continuousInput)
        val *= _deltaTime;
    else
        val *= 0.1f;

    _pivotPos += val;
}
//-----------------------------------------------------------------------------
SLbool AppNodeSceneView::onContinuousKeyPress(SLKey key)
{
    if (!_continuousInput)
        _keyStates[key] = false;

    if (_curMode == TranslationMode)
    {
        switch (key)
        {
            case 'W': translateObject(SLVec3f(0, 0, -1)); return true;
            case 'S': translateObject(SLVec3f(0, 0, 1)); return true;
            case 'A': translateObject(SLVec3f(-1, 0, 0)); return true;
            case 'D': translateObject(SLVec3f(1, 0, 0)); return true;
            case 'Q': translateObject(SLVec3f(0, 1, 0)); return true;
            case 'E': translateObject(SLVec3f(0, -1, 0)); return true;
        }
    }
    else if (_curMode == RotationMode)
    {
        switch (key)
        {
            case 'W': rotateObject(SLVec3f(-1, 0, 0)); return true;
            case 'S': rotateObject(SLVec3f(1, 0, 0)); return true;
            case 'A': rotateObject(SLVec3f(0, 1, 0)); return true;
            case 'D': rotateObject(SLVec3f(0, -1, 0)); return true;
            case 'Q': rotateObject(SLVec3f(0, 0, 1)); return true;
            case 'E': rotateObject(SLVec3f(0, 0, -1)); return true;
        }
    }
    else if (_curMode == RotationAroundMode)
    {
        if (_modifiers & K_shift)
        {
            switch (key)
            {
                case 'W': translatePivot(SLVec3f(0, 0, -1)); return true;
                case 'S': translatePivot(SLVec3f(0, 0, 1)); return true;
                case 'A': translatePivot(SLVec3f(-1, 0, 0)); return true;
                case 'D': translatePivot(SLVec3f(1, 0, 0)); return true;
                case 'Q': translatePivot(SLVec3f(0, 1, 0)); return true;
                case 'E': translatePivot(SLVec3f(0, -1, 0)); return true;
            }
        }
        else
        {
            switch (key)
            {
                case 'W': rotateObjectAroundPivot(SLVec3f(-1, 0, 0)); return true;
                case 'S': rotateObjectAroundPivot(SLVec3f(1, 0, 0)); return true;
                case 'A': rotateObjectAroundPivot(SLVec3f(0, -1, 0)); return true;
                case 'D': rotateObjectAroundPivot(SLVec3f(0, 1, 0)); return true;
                case 'Q': rotateObjectAroundPivot(SLVec3f(0, 0, 1)); return true;
                case 'E': rotateObjectAroundPivot(SLVec3f(0, 0, -1)); return true;
            }
        }
    }
    else if (_curMode == LookAtMode)
    {
        switch (key)
        {
            case 'W': translatePivot(SLVec3f(0, 0, -1)); break;
            case 'S': translatePivot(SLVec3f(0, 0, 1)); break;
            case 'A': translatePivot(SLVec3f(-1, 0, 0)); break;
            case 'D': translatePivot(SLVec3f(1, 0, 0)); break;
            case 'Q': translatePivot(SLVec3f(0, 1, 0)); break;
            case 'E': translatePivot(SLVec3f(0, -1, 0)); break;
        }

        // if we look at a point in local space then the local space will change.
        // we want to keep the old look at position in world space though so that
        // the user can confirm that his object is in fact looking at the point it should.
        if (_curSpace == TS_object)
        {
            SLVec3f pivotWorldPos = _curObject->updateAndGetWM() * _pivotPos;
            _curObject->lookAt(_pivotPos, SLVec3f::AXISY, _curSpace);
            _pivotPos = _curObject->updateAndGetWMI() * pivotWorldPos;
        }
        else // else just look at the point
            _curObject->lookAt(_pivotPos, SLVec3f::AXISY, _curSpace);

        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
SLbool AppNodeSceneView::onKeyPress(const SLKey key, const SLKey mod)
{
    _keyStates[key] = true;
    _modifiers      = mod;

    switch (key)
    {
        // general input
        case '1': _curMode = TranslationMode; break;
        case '2': _curMode = RotationMode; break;
        case '3': _curMode = RotationAroundMode; break;
        case '4': _curMode = LookAtMode; break;

        // select parent object
        case K_F1:
            _curObject = (_curObject == _moveBox) ? _moveBoxChild : _moveBox;
            AppDemo::scene->selectNodeMesh(_curObject,
                                           _curObject->mesh());
            break;
        case K_F2: _continuousInput = !_continuousInput; break;
        case 'R': reset(); break;
        case 'Y':
        case 'Z': _curSpace = TS_object; break;
        case 'X': _curSpace = TS_parent; break;
        case 'C': _curSpace = TS_world; break;
    }

    updateInfoText();
    updateCurOrigin();

    return false;
}
//-----------------------------------------------------------------------------
SLbool AppNodeSceneView::onKeyRelease(const SLKey key, const SLKey mod)
{
    _keyStates[key] = false;
    _modifiers      = mod;

    return false;
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::updateCurOrigin()
{
    switch (_curSpace)
    {
        case TS_world:
            _curOrigin.identity();
            _axesNode->resetToInitialState();
            break;

        case TS_parent:
            _curOrigin.setMatrix(_curObject->parent()->updateAndGetWM());
            _axesNode->om(_curObject->parent()->updateAndGetWM());
            break;

        case TS_object:
            _curOrigin.setMatrix(_curObject->updateAndGetWM());
            _axesNode->om(_curObject->updateAndGetWM());
            break;
    }

    // if in rotation mode, move the axis to the objects origin, but keep the orientation
    if (_curMode == TranslationMode || _curMode == RotationMode)
    {
        _axesNode->translation(_curObject->updateAndGetWM().translation(), TS_world);
    }
    // look at nd rotate around mode both move the pivot relative to the current system
    if (_curMode == RotationAroundMode || _curMode == LookAtMode)
    {
        _axesNode->translate(_pivotPos.x, _pivotPos.y, _pivotPos.z, TS_object);
    }

    _curOrigin;
}
//-----------------------------------------------------------------------------
void AppNodeSceneView::updateInfoText()
{
    SLchar m[2550]; // message character array
    m[0] = 0;       // set zero length

    SLstring keyBinds;
    keyBinds = "Key bindings: \n";
    keyBinds += "F1: toggle current object \n";
    keyBinds += "F2: toggle continuous input \n\n";
    keyBinds += "1: translation mode \n";
    keyBinds += "2: rotation mode \n";
    keyBinds += "3: rotate around point mode \n";
    keyBinds += "4: look at mode \n\n";

    keyBinds += "Y: Set relative space to Object\n";
    keyBinds += "X: Set relative space to Parent\n";
    keyBinds += "C: Set relative space to World\n\n";

    SLstring space;
    switch (_curSpace)
    {
        case TS_object: space = "TS_Object"; break;
        case TS_parent: space = "TS_Parent"; break;
        case TS_world: space = "TS_World"; break;
    }

    SLstring mode;
    switch (_curMode)
    {
        case TranslationMode:
            mode = "Translate";
            keyBinds += "W: forward in " + space + " space \n";
            keyBinds += "S: backward in " + space + " space \n";
            keyBinds += "A: left in " + space + " space \n";
            keyBinds += "D: right in " + space + " space \n";
            keyBinds += "Q: up in " + space + " space \n";
            keyBinds += "E: down in " + space + " space \n";
            break;
        case RotationMode:
            mode = "Rotate";
            keyBinds += "W: rotate around -X in " + space + "\n";
            keyBinds += "S: rotate around  X in " + space + "\n";
            keyBinds += "A: rotate around  Y in " + space + "\n";
            keyBinds += "D: rotate around -Y in " + space + "\n";
            keyBinds += "Q: rotate around  Z in " + space + "\n";
            keyBinds += "E: rotate around -Z in " + space + "\n";
            break;
        case RotationAroundMode:
            mode = "RotateAround";
            keyBinds += "W: rotate around -X in " + space + "\n";
            keyBinds += "S: rotate around  X in " + space + "\n";
            keyBinds += "A: rotate around -Y in " + space + "\n";
            keyBinds += "D: rotate around  Y in " + space + "\n";
            keyBinds += "Q: rotate around  Z in " + space + "\n";
            keyBinds += "E: rotate around -Z in " + space + "\n\n";

            keyBinds += "Shift-W: pivot forward in " + space + "\n";
            keyBinds += "Shift-S: pivot left in " + space + "\n";
            keyBinds += "Shift-A: pivot backward in " + space + "\n";
            keyBinds += "Shift-D: pivot right in " + space + "\n";
            keyBinds += "Shift-Q: pivot up in " + space + "\n";
            keyBinds += "Shift-E: pivot down in " + space + "\n";
            break;
        case LookAtMode:
            mode = "LookAt";
            keyBinds += "W: move lookAt point forward in " + space + "\n";
            keyBinds += "S: move lookAt point left in " + space + "\n";
            keyBinds += "A: move lookAt point backward in " + space + "\n";
            keyBinds += "D: move lookAt point right in " + space + "\n";
            keyBinds += "Q: move lookAt point up in " + space + "\n";
            keyBinds += "E: move lookAt point down in " + space + "\n";
            break;
    }

    keyBinds += "\nR: Reset \n";
    snprintf(m + strlen(m), sizeof(m), "%s", keyBinds.c_str());

    SLTexFont* f         = SLAssetManager::getFont(1.2f, dpi());
    AppNodeGui::infoText = m;
}
//-----------------------------------------------------------------------------
