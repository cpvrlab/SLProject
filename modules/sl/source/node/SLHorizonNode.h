//#############################################################################
//  File:      SLHorizonNode.h
//  Authors:   Michael Göttlicher
//  Date:      November 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Göttlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_HORIZON_NODE_H
#define SL_HORIZON_NODE_H

#include <SLNode.h>
#include <SLTexFont.h>
#include <SLDeviceRotation.h>
#include <SLPolyline.h>
#include <SLGLProgramGeneric.h>
#include <SLMaterial.h>

//-----------------------------------------------------------------------------
class SLHorizonNode : public SLNode
{
public:
    SLHorizonNode(SLstring          name,
                  SLDeviceRotation* devRot,
                  SLTexFont*        font,
                  SLstring          shaderDir,
                  int               scrW,
                  int               scrH);
    ~SLHorizonNode();

    void doUpdate() override;

private:
    SLDeviceRotation* _devRot = nullptr;
    SLTexFont*        _font   = nullptr;
    SLstring          _shaderDir;

    SLGLProgram* _prog        = nullptr;
    SLMaterial*  _mat         = nullptr;
    SLPolyline*  _line        = nullptr;
    SLNode*      _horizonNode = nullptr;
    SLNode*      _textNode    = nullptr;

    SLMat3f _sRc;
};
//-----------------------------------------------------------------------------
#endif
