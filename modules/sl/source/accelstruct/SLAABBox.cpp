//#############################################################################
//  File:      SLAABBox.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAABBox.h>
#include <SLRay.h>
#include <SLScene.h>
#include <SLGLState.h>

//-----------------------------------------------------------------------------
//! Default constructor with default zero vector initialization
SLAABBox::SLAABBox()
{
    reset();
}
//-----------------------------------------------------------------------------
//! Resets initial state without contents
void SLAABBox::reset()
{
    _minWS    = SLVec3f::ZERO;
    _maxWS    = SLVec3f::ZERO;
    _centerWS = SLVec3f::ZERO;
    _radiusWS = 0;

    _minOS    = SLVec3f::ZERO;
    _maxOS    = SLVec3f::ZERO;
    _centerOS = SLVec3f::ZERO;
    _radiusOS = 0;

    _sqrViewDist = 0;
    _axis0WS     = SLVec3f::ZERO;
    _axisXWS     = SLVec3f::ZERO;
    _axisYWS     = SLVec3f::ZERO;
    _axisZWS     = SLVec3f::ZERO;
    _isVisible   = true;

    _rectSS.setZero();
}
//-----------------------------------------------------------------------------
//! Recalculate min and max after transformation in world coords
void SLAABBox::fromOStoWS(const SLVec3f& minOS,
                          const SLVec3f& maxOS,
                          const SLMat4f& wm)
{
    // Do not transform empty AABB (such as from the camera)
    if (minOS == SLVec3f::ZERO && maxOS == SLVec3f::ZERO)
        return;

    _minOS.set(minOS);
    _maxOS.set(maxOS);
    _minWS.set(minOS);
    _maxWS.set(maxOS);

    // we need to transform all 8 corners for a non-optimal bounding box
    SLVec3f vCorner[8];
    SLint   i;

    vCorner[0].set(_minWS);
    vCorner[1].set(_maxWS.x, _minWS.y, _minWS.z);
    vCorner[2].set(_maxWS.x, _minWS.y, _maxWS.z);
    vCorner[3].set(_minWS.x, _minWS.y, _maxWS.z);
    vCorner[4].set(_maxWS.x, _maxWS.y, _minWS.z);
    vCorner[5].set(_minWS.x, _maxWS.y, _minWS.z);
    vCorner[6].set(_minWS.x, _maxWS.y, _maxWS.z);
    vCorner[7].set(_maxWS);

    // apply world transform
    for (i = 0; i < 8; ++i)
        vCorner[i] = wm.multVec(vCorner[i]);

    // sets the minimum and maximum of the vertex components of the 8 corners
    _minWS.set(vCorner[0]);
    _maxWS.set(vCorner[0]);
    for (i = 1; i < 8; ++i)
    {
        _minWS.setMin(vCorner[i]);
        _maxWS.setMax(vCorner[i]);
    }

    // set coordinate axis in world space
    _axis0WS = wm.multVec(SLVec3f::ZERO);
    _axisXWS = wm.multVec(SLVec3f::AXISX);
    _axisYWS = wm.multVec(SLVec3f::AXISY);
    _axisZWS = wm.multVec(SLVec3f::AXISZ);

    // Delete OpenGL vertex array
    if (_vao.vaoID()) _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Recalculate min and max before transformation in object coords
void SLAABBox::fromWStoOS(const SLVec3f& minWS,
                          const SLVec3f& maxWS,
                          const SLMat4f& wmI)
{
    _minOS.set(minWS);
    _maxOS.set(maxWS);
    _minWS.set(minWS);
    _maxWS.set(maxWS);

    // we need to transform all 8 corners for a non-optimal bounding box
    SLVec3f vCorner[8];
    SLint   i;

    vCorner[0].set(_minOS);
    vCorner[1].set(_maxOS.x, _minOS.y, _minOS.z);
    vCorner[2].set(_maxOS.x, _minOS.y, _maxOS.z);
    vCorner[3].set(_minOS.x, _minOS.y, _maxOS.z);
    vCorner[4].set(_maxOS.x, _maxOS.y, _minOS.z);
    vCorner[5].set(_minOS.x, _maxOS.y, _minOS.z);
    vCorner[6].set(_minOS.x, _maxOS.y, _maxOS.z);
    vCorner[7].set(_maxOS);

    // apply world transform
    for (i = 0; i < 8; ++i)
        vCorner[i] = wmI.multVec(vCorner[i]);

    // sets the minimum and maximum of the vertex components of the 8 corners
    _minOS.set(vCorner[0]);
    _maxOS.set(vCorner[0]);
    for (i = 1; i < 8; ++i)
    {
        _minOS.setMin(vCorner[i]);
        _maxOS.setMax(vCorner[i]);
    }

    // Delete OpenGL vertex array
    if (_vao.vaoID()) _vao.clearAttribs();

    // Set center & radius of the bounding sphere around the AABB
    _centerOS.set((_minOS + _maxOS) * 0.5f);
    SLVec3f extent(_maxOS - _centerOS);
    _radiusOS = extent.length();
}
//-----------------------------------------------------------------------------
//! Updates the axis of the owning node
void SLAABBox::updateAxisWS(const SLMat4f& wm)
{
    // set coordinate axis in world space
    _axis0WS = wm.multVec(SLVec3f::ZERO);
    _axisXWS = wm.multVec(SLVec3f::AXISX);
    _axisYWS = wm.multVec(SLVec3f::AXISY);
    _axisZWS = wm.multVec(SLVec3f::AXISZ);

    // Delete OpenGL vertex array
    if (_vao.vaoID()) _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Updates joints axis and the bone line from the parent to us
/*! If the node has a skeleton assigned the method updates the axis and bone
visualization lines of the joint. Note that joints bone line is drawn by its
children. So the bone line in here is the bone from the parent to us.
If this parent bone direction is not along the parents Y axis we interpret the
connection not as a bone but as an offset displacement. Bones will be drawn in
SLAABBox::drawBoneWS in yellow and displacements in magenta.
If the joint has no parent (the root) no line is drawn.
*/
void SLAABBox::updateBoneWS(const SLMat4f& parentWM,
                            const SLbool   isRoot,
                            const SLMat4f& nodeWM)
{
    // set coordinate axis centre point
    _axis0WS = nodeWM.multVec(SLVec3f::ZERO);

    // set scale factor for coordinate axis
    SLfloat axisScaleFactor = 0.03f;

    if (!isRoot)
    {
        // build the parent pos in WM
        _parent0WS = parentWM.multVec(SLVec3f::ZERO);

        // set the axis scale factor depending on the length of the parent bone
        SLVec3f parentToMe = _axis0WS - _parent0WS;
        axisScaleFactor    = std::max(parentToMe.length() / 10.0f, axisScaleFactor);

        // check if the parent to me direction is parallel to the parents actual y-axis
        parentToMe.normalize();
        SLVec3f parentY = parentWM.axisY();
        parentY.normalize();
        _boneIsOffset = parentToMe.dot(parentY) < (1.0f - FLT_EPSILON);
    }
    else
    {
        // for the root node don't draw a parent bone
        _parent0WS    = _axis0WS;
        _boneIsOffset = false;
    }

    // set coordinate axis end points
    _axisXWS = nodeWM.multVec(SLVec3f::AXISX * axisScaleFactor);
    _axisYWS = nodeWM.multVec(SLVec3f::AXISY * axisScaleFactor);
    _axisZWS = nodeWM.multVec(SLVec3f::AXISZ * axisScaleFactor);

    // Delete OpenGL vertex array
    if (_vao.vaoID()) _vao.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Calculates center & radius of the bounding sphere around the AABB
void SLAABBox::setCenterAndRadiusWS()
{
    _centerWS.set((_minWS + _maxWS) * 0.5f);
    SLVec3f ext(_maxWS - _centerWS);
    _radiusWS = ext.length();
}
//-----------------------------------------------------------------------------
//! Generates the vertex buffer for the line visualization
void SLAABBox::generateVAO()
{
    SLVVec3f P; // vertex positions

    // Bounding box lines in world space
    P.push_back(SLVec3f(_minWS.x, _minWS.y, _minWS.z)); // lower rect
    P.push_back(SLVec3f(_maxWS.x, _minWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _minWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _minWS.y, _maxWS.z));
    P.push_back(SLVec3f(_maxWS.x, _minWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _minWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _minWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _minWS.y, _minWS.z));

    P.push_back(SLVec3f(_minWS.x, _maxWS.y, _minWS.z)); // upper rect
    P.push_back(SLVec3f(_maxWS.x, _maxWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _maxWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _maxWS.y, _maxWS.z));
    P.push_back(SLVec3f(_maxWS.x, _maxWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _maxWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _maxWS.y, _maxWS.z));

    P.push_back(SLVec3f(_minWS.x, _maxWS.y, _minWS.z)); // vertical lines
    P.push_back(SLVec3f(_minWS.x, _minWS.y, _minWS.z));
    P.push_back(SLVec3f(_minWS.x, _maxWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _minWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _maxWS.y, _minWS.z));
    P.push_back(SLVec3f(_maxWS.x, _minWS.y, _maxWS.z));
    P.push_back(SLVec3f(_maxWS.x, _maxWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _minWS.y, _maxWS.z));
    P.push_back(SLVec3f(_minWS.x, _maxWS.y, _maxWS.z)); // 24

    // Axis lines in world space
    P.push_back(SLVec3f(_axis0WS.x, _axis0WS.y, _axis0WS.z)); // x-axis
    P.push_back(SLVec3f(_axisXWS.x, _axisXWS.y, _axisXWS.z));
    P.push_back(SLVec3f(_axis0WS.x, _axis0WS.y, _axis0WS.z)); // y-axis
    P.push_back(SLVec3f(_axisYWS.x, _axisYWS.y, _axisYWS.z));
    P.push_back(SLVec3f(_axis0WS.x, _axis0WS.y, _axis0WS.z)); // z-axis
    P.push_back(SLVec3f(_axisZWS.x, _axisZWS.y, _axisZWS.z)); // 30

    // Bone points in world space
    P.push_back(SLVec3f(_parent0WS.x, _parent0WS.y, _parent0WS.z));
    P.push_back(SLVec3f(_axis0WS.x, _axis0WS.y, _axis0WS.z));

    _vao.generateVertexPos(&P);
}
//-----------------------------------------------------------------------------
//! Draws the AABB in world space with lines in a color
void SLAABBox::drawWS(const SLCol4f& color)
{
    if (!_vao.vaoID()) generateVAO();
    _vao.drawArrayAsColored(PT_lines,
                            color,
                            1.0f,
                            0,
                            24);
}
//-----------------------------------------------------------------------------
//! Draws the axis in world space with lines in a color
void SLAABBox::drawAxisWS()
{
    if (!_vao.vaoID()) generateVAO();
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::RED,
                            2.0f,
                            24,
                            2);
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::GREEN,
                            2.0f,
                            26,
                            2);
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::BLUE,
                            2.0f,
                            28,
                            2);
}
//-----------------------------------------------------------------------------
//! Draws the joint axis and the parent bone in world space
/*! The joints x-axis is drawn in red, the y-axis in green and the z-axis in
blue. If the parent displacement is a bone it is drawn in yellow, if it is a
an offset displacement in magenta. See also SLAABBox::updateBoneWS.
*/
void SLAABBox::drawBoneWS()
{
    if (!_vao.vaoID()) generateVAO();
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::RED,
                            2.0f,
                            24,
                            2);
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::GREEN,
                            2.0f,
                            26,
                            2);
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::BLUE,
                            2.0f,
                            28,
                            2);

    // draw either an offset line or a bone line as the parent
    if (!_boneIsOffset)
        _vao.drawArrayAsColored(PT_lines,
                                SLCol4f::YELLOW,
                                1.0f,
                                30,
                                2);
    else
        _vao.drawArrayAsColored(PT_lines,
                                SLCol4f::MAGENTA,
                                1.0f,
                                30,
                                2);
}
//-----------------------------------------------------------------------------
//! SLAABBox::isHitInWS: Ray - AABB Intersection Test in object space
SLbool SLAABBox::isHitInOS(SLRay* ray)
{
    // See: "An Efficient and Robust Ray Box Intersection Algorithm"
    // by Amy L. Williams, Steve Barrus, R. Keith Morley, Peter Shirley
    // This test is about 10% faster than the test from Woo
    // It need the pre computed values invDir and sign in SLRay

    SLVec3f params[2] = {_minOS, _maxOS};
    SLfloat tymin, tymax, tzmin, tzmax;

    ray->tmin = (params[ray->signOS[0]].x - ray->originOS.x) * ray->invDirOS.x;
    ray->tmax = (params[1 - ray->signOS[0]].x - ray->originOS.x) * ray->invDirOS.x;
    tymin     = (params[ray->signOS[1]].y - ray->originOS.y) * ray->invDirOS.y;
    tymax     = (params[1 - ray->signOS[1]].y - ray->originOS.y) * ray->invDirOS.y;

    if ((ray->tmin > tymax) || (tymin > ray->tmax)) return false;
    if (tymin > ray->tmin) ray->tmin = tymin;
    if (tymax < ray->tmax) ray->tmax = tymax;

    tzmin = (params[ray->signOS[2]].z - ray->originOS.z) * ray->invDirOS.z;
    tzmax = (params[1 - ray->signOS[2]].z - ray->originOS.z) * ray->invDirOS.z;

    if ((ray->tmin > tzmax) || (tzmin > ray->tmax)) return false;
    if (tzmin > ray->tmin) ray->tmin = tzmin;
    if (tzmax < ray->tmax) ray->tmax = tzmax;

    return ((ray->tmin < ray->length) && (ray->tmax > 0));
}
//-----------------------------------------------------------------------------
//! SLAABBox::isHitInWS: Ray - AABB Intersection Test in world space
SLbool SLAABBox::isHitInWS(SLRay* ray)
{
    // See: "An Efficient and Robust Ray Box Intersection Algorithm"
    // by Amy L. Williams, Steve Barrus, R. Keith Morley, Peter Shirley
    // This test is about 10% faster than the test from Woo
    // It needs the pre-computed values invDir and sign in SLRay
    SLVec3f params[2] = {_minWS, _maxWS};
    SLfloat tymin, tymax, tzmin, tzmax;

    ray->tmin = (params[ray->sign[0]].x - ray->origin.x) * ray->invDir.x;
    ray->tmax = (params[1 - ray->sign[0]].x - ray->origin.x) * ray->invDir.x;
    tymin     = (params[ray->sign[1]].y - ray->origin.y) * ray->invDir.y;
    tymax     = (params[1 - ray->sign[1]].y - ray->origin.y) * ray->invDir.y;

    if ((ray->tmin > tymax) || (tymin > ray->tmax)) return false;
    if (tymin > ray->tmin) ray->tmin = tymin;
    if (tymax < ray->tmax) ray->tmax = tymax;

    tzmin = (params[ray->sign[2]].z - ray->origin.z) * ray->invDir.z;
    tzmax = (params[1 - ray->sign[2]].z - ray->origin.z) * ray->invDir.z;

    if ((ray->tmin > tzmax) || (tzmin > ray->tmax)) return false;
    if (tzmin > ray->tmin) ray->tmin = tzmin;
    if (tzmax < ray->tmax) ray->tmax = tzmax;

    return ((ray->tmin < ray->length) && (ray->tmax > 0));
}
//-----------------------------------------------------------------------------
//! Merges the bounding box bb to this one by extending this one axis aligned
void SLAABBox::mergeWS(SLAABBox& bb)
{
    if (bb.minWS() != SLVec3f::ZERO && bb.maxWS() != SLVec3f::ZERO)
    {
        _minWS.setMin(bb.minWS());
        _maxWS.setMax(bb.maxWS());
    }
}
//-----------------------------------------------------------------------------
//! Calculates the AABBs min. and max. corners in screen space
void SLAABBox::calculateRectSS()
{
    SLVec3f corners[8];

    // Back corners in world space
    corners[0] = _minWS;
    corners[1] = SLVec3f(_maxWS.x, _minWS.y, _minWS.z);
    corners[2] = SLVec3f(_minWS.x, _maxWS.y, _minWS.z);
    corners[3] = SLVec3f(_maxWS.x, _maxWS.y, _minWS.z);

    // Front corners in world space
    corners[4] = SLVec3f(_minWS.x, _minWS.y, _maxWS.z);
    corners[5] = SLVec3f(_maxWS.x, _minWS.y, _maxWS.z);
    corners[6] = SLVec3f(_minWS.x, _maxWS.y, _maxWS.z);
    corners[7] = _maxWS;

    // build view-projection-viewport matrix
    SLGLState* stateGL = SLGLState::instance();
    SLMat4f    vpvpMat = stateGL->viewportMatrix() *
                      stateGL->projectionMatrix *
                      stateGL->viewMatrix;

    // transform corners from world to screen space
    for (SLint i = 0; i < 8; ++i)
        corners[i] = vpvpMat.multVec(corners[i]);

    // Build min. and max. in screen space
    SLVec2f minSS(FLT_MAX, FLT_MAX);
    SLVec2f maxSS(FLT_MIN, FLT_MIN);

    for (SLint i = 0; i < 8; ++i)
    {
        minSS.x = std::min(minSS.x, corners[i].x);
        minSS.y = std::min(minSS.y, corners[i].y);
        maxSS.x = std::max(maxSS.x, corners[i].x);
        maxSS.y = std::max(maxSS.y, corners[i].y);
    }

    _rectSS.set(minSS.x,
                minSS.y,
                maxSS.x - minSS.x,
                maxSS.y - minSS.y);
    //_rectSS.print("_rectSS: ");
}
//-----------------------------------------------------------------------------
//! Calculates the bounding rectangle in screen space and returns coverage in SS
SLfloat SLAABBox::rectCoverageInSS()
{
    calculateRectSS();

    SLGLState* stateGL        = SLGLState::instance();
    SLfloat    areaSS         = _rectSS.width * _rectSS.height;
    SLVec4i    vp             = stateGL->viewport();
    SLfloat    areaFullScreen = (float)vp.z * (float)vp.w;
    SLfloat    coverage       = areaSS / areaFullScreen;
    return coverage;
}
//-----------------------------------------------------------------------------
