//#############################################################################
//  File:      NewNodeSceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>

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
demonstrates all transform possibilities in SLNode
*/
class NewNodeSceneView : public SLSceneView
{
    public:
                            NewNodeSceneView();
                           ~NewNodeSceneView();

        // From SLSceneView overwritten
        void                preDraw();
        void                postDraw();
        void                postSceneLoad();
        SLbool              onKeyPress(const SLKey key, const SLKey mod);
        SLbool              onKeyRelease(const SLKey key, const SLKey mod);

        void                reset();
        void                translateObject(SLVec3f vec);
        void                rotateObject(SLVec3f val);
        void                rotateObjectAroundPivot(SLVec3f val);
        void                translatePivot(SLVec3f vec);

        SLbool              onContinuousKeyPress(SLKey key);

        SLbool              update();
        void                updateCurOrigin();
        void                updateInfoText();
        void                renderText();

        SLMat4f             _curOrigin;     //!< current origin of relative space (orientation and position of axes)

        SLNode*             _moveBox;       //!< big parent cube
        SLNode*             _moveBoxChild;  //!< little child cube
        SLVec3f             _pivotPos;      //!< position of the pivot point
        SLNode*             _axesNode;      //!< node for axis mesh
        SLText*             _infoText;      //!< node for all text display

        bool                _keyStates[65536];  //!< key press states of all keys
        SLKey               _modifiers;         //!< pressed modifier keys
        bool                _continuousInput;   //!< flag for continuous input processing

        SLfloat             _deltaTime;     //!< delta time of a frame
        TransformMode       _curMode;       //!< current transform mode
        SLNode*             _curObject;     //!< current object to transform
        SLTransformSpace    _curSpace;      //!< current transform space
};
//-----------------------------------------------------------------------------
