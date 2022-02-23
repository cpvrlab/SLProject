//#############################################################################
//  File:      SLHorizonNode.cpp
//  Date:      November 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Göttlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLHorizonNode.h>
#include <SLText.h>
#include <SLAlgo.h>
#include <cmath>

//-----------------------------------------------------------------------------
SLHorizonNode::SLHorizonNode(SLstring          name,
                             SLDeviceRotation* devRot,
                             SLTexFont*        font,
                             SLstring          shaderDir,
                             int               scrW,
                             int               scrH)
  : SLNode(name),
    _devRot(devRot),
    _font(font),
    _shaderDir(shaderDir)
{
    // make sure device rotation is enabled
    if (!_devRot->isUsed())
        _devRot->isUsed(true);

    // rotation of camera w.r.t sensor
    _sRc.rotation(-90, 0, 0, 1);

    // init visualization node and meshes
    //(this node is owner of instantiated programs, meshes and materials)
    _prog = new SLGLProgramGeneric(nullptr,
                                   shaderDir + "ColorUniformPoint.vert",
                                   shaderDir + "Color.frag");
    _prog->addUniform1f(new SLGLUniform1f(UT_const,
                                          "u_pointSize",
                                          1.0f));
    _mat = new SLMaterial(nullptr, _prog, SLCol4f::WHITE, "White");
    // define mesh points
    int     refLen = std::min(scrW, scrH);
    SLfloat cs; // center size
    if (_font)
    {
        SLfloat  scale = 1.f;
        SLstring txt   = "-359.9";
        SLVec2f  size  = _font->calcTextSize(txt);
        cs             = size.x;
    }
    else
        cs = (float)refLen * 0.01f; // center size

    float l = (float)refLen * 0.35f;

    SLVVec3f points = {{-l, 0, 0},
                       {-cs, 0, 0},
                       {0, -cs, 0},
                       {cs, 0, 0},
                       {l, 0, 0},
                       {cs, 0, 0},
                       {0, cs, 0},
                       {-cs, 0, 0}};

    _line        = new SLPolyline(nullptr,
                           points,
                           true,
                           "Horizon line",
                           _mat);
    _horizonNode = new SLNode(_line, "Horizon node");
    this->addChild(_horizonNode);
}
//-----------------------------------------------------------------------------
SLHorizonNode::~SLHorizonNode()
{
    delete _prog;
    delete _mat;
    delete _line;
}
//-----------------------------------------------------------------------------
void SLHorizonNode::doUpdate()
{
    // get latest orientation and update horizon
    SLVec3f horizon;
    SLAlgo::estimateHorizon(_devRot->rotationAveraged(), _sRc, horizon);

    // rotate node to align it to horizon
    float horizonAngle = std::atan2(horizon.y, horizon.x) * RAD2DEG;
    _horizonNode->rotation(horizonAngle,
                           SLVec3f(0, 0, 1),
                           SLTransformSpace::TS_object);

    // update text
    if (_font)
    {
        if (_textNode)
            this->deleteChild(_textNode);

        stringstream ss;
        // we invert the sign to express the rotation of the device w.r.t the horizon
        ss << std::fixed << std::setprecision(1) << -horizonAngle;
        SLstring txt = ss.str();

        SLVec2f size = _font->calcTextSize(txt);
        _textNode    = new SLText(txt, _font, SLCol4f::WHITE);
        _textNode->translate(-size.x * 0.5f, -size.y * 0.5f, 0);
        this->addChild(_textNode);
    }
}
//-----------------------------------------------------------------------------
