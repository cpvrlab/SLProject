//#############################################################################
//  File:      AppNodeSceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
enum TransformMode
{
    TranslationMode,
    RotationMode,
    RotationAroundMode,
    LookAtMode
};

//-----------------------------------------------------------------------------
/*!
 SLSceneView derived class for a node transform test application that
 demonstrates all transform possibilities in SLNode. The SLSceneView class is
 inherited because we override here the default event-handling for the keyboard
 (onKeyPress and onKeyRelease)
*/
class AppNodeSceneView : public SLSceneView
{
public:
    AppNodeSceneView(SLScene* s, int dpi, SLInputManager& inputManager);
    ~AppNodeSceneView();

    // From SLSceneView overwritten
    void   preDraw();
    void   postSceneLoad();
    SLbool onKeyPress(const SLKey key, const SLKey mod);
    SLbool onKeyRelease(const SLKey key, const SLKey mod);

    void reset();
    void translateObject(SLVec3f vec) const;
    void rotateObject(const SLVec3f& val) const;
    void rotateObjectAroundPivot(SLVec3f val) const;
    void translatePivot(SLVec3f vec);

    SLbool onContinuousKeyPress(SLKey key);

    void updateCurOrigin();
    void updateInfoText();

    SLMat4f _curOrigin;          //!< current origin of relative space (orientation and position of axes)

    SLNode* _moveBox;            //!< big parent cube
    SLNode* _moveBoxChild;       //!< little child cube
    SLVec3f _pivotPos;           //!< position of the pivot point
    SLNode* _axesNode;           //!< node for axis mesh

    bool  _keyStates[65536];     //!< key press states of all keys
    SLKey _modifiers;            //!< pressed modifier keys
    bool  _continuousInput;      //!< flag for continuous input processing

    SLfloat          _deltaTime; //!< delta time of a frame
    TransformMode    _curMode;   //!< current transform mode
    SLNode*          _curObject; //!< current object to transform
    SLTransformSpace _curSpace;  //!< current transform space
};
//-----------------------------------------------------------------------------
