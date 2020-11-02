//#############################################################################
//  File:      SLHorizonNode.cpp
//  Author:    Michael Göttlicher
//  Date:      November 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLHorizonNode.h>

SLHorizonNode::SLHorizonNode(SLstring name, SLDeviceRotation* devRot, SLTexFont* font, SLstring shaderDir, int scrW, int scrH)
  : SLNode(name),
    _devRot(devRot),
    _font(font),
    _shaderDir(shaderDir)
{
    //make sure device rotation is enabled
    if (!_devRot->isUsed())
        _devRot->isUsed(true);

    //rotation of camera w.r.t sensor
    _sRc.rotation(-90, 0, 0, 1);

    //init visualization node and meshes
    //(this node is owner of instantiated programs, meshes and materials)
    _prog = new SLGLGenericProgram(nullptr, shaderDir + "ColorUniformPoint.vert", shaderDir + "Color.frag");
    _prog->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 1.0f));
    //_matR = new SLMaterial(nullptr, "Red Opaque", SLCol4f::RED, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _matR = new SLMaterial(nullptr, _prog, SLCol4f::RED, "Red");
    //define mesh points
    int      refLen = std::min(scrW, scrH);
    SLfloat  cs     = refLen * 0.01f; // center size
    float    l      = refLen * 0.35;

    SLVVec3f points = {{-l, 0, 0},
                       {-cs, 0, 0},
                       {0, -cs, 0},
                       {cs, 0, 0},
                       {l, 0, 0},
                       {cs, 0, 0},
                       {0, cs, 0},
                       {-cs, 0, 0}};

    _line = new SLPolyline(nullptr, points, true, "Horizon line", _matR);
    this->addMesh(_line);
}

SLHorizonNode::~SLHorizonNode()
{
    delete _prog;
    delete _matR;
    delete _line;
}

void SLHorizonNode::doUpdate()
{
    //get latest orientation and update horizon
    SLVec3f horizon;
    estimateHorizon(_devRot->rotationAveraged(), _sRc, horizon);
    //rotate node to align it to horizon
    float horizonAngle = std::atan2f(horizon.y, horizon.x);
    this->rotation(horizonAngle * RAD2DEG, SLVec3f(0, 0, 1), SLTransformSpace::TS_object);
}
