//#############################################################################
//  File:      SLAABBox.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLAABBox_H
#define SLAABBox_H

#include <SLGLVertexArrayExt.h>
#include <SLMat4.h>
#include <SLRect.h>

class SLRay;

//-----------------------------------------------------------------------------
//! Defines an axis aligned bounding box
/*!
The SLAABBox class defines an axis aligned bounding box with a minimal and
maximal point. Each node (SLNode) will have an AABB the will be calculated
in buildAABB. A mesh (SLMesh) will implement buildAABB and calculate the
minimal and maximal coordinates in object space (stored in _minOS and _maxOS).
For a fast ray-AABB intersection in world space we transform _minOS and _maxOS
into world space (with the shapes world matrix) and store it in _minWS and
_maxWS.
For an even faster intersection test with the plane of the view frustum we
calculate in addition the bounding sphere around the AABB. The radius and the
center point are stored in _radiusOS/_centerOS and _radiusWS/_centerWS.
*/
class SLAABBox
{
public:
    SLAABBox();

    // Setters
    void minWS(const SLVec3f& minC) { _minWS = minC; }
    void maxWS(const SLVec3f& maxC) { _maxWS = maxC; }
    void minOS(const SLVec3f& minC) { _minOS = minC; }
    void maxOS(const SLVec3f& maxC) { _maxOS = maxC; }

    void isVisible(SLbool visible) { _isVisible = visible; }
    void sqrViewDist(SLfloat sqrVD) { _sqrViewDist = sqrVD; }

    // Getters
    SLVec3f  minWS() { return _minWS; }
    SLVec3f  maxWS() { return _maxWS; }
    SLVec3f  centerWS() { return _centerWS; }
    SLfloat  radiusWS() { return _radiusWS; }
    SLVec3f  minOS() { return _minOS; }
    SLVec3f  maxOS() { return _maxOS; }
    SLVec3f  centerOS() { return _centerOS; }
    SLfloat  radiusOS() { return _radiusOS; }
    SLbool   isVisible() { return _isVisible; }
    SLfloat  sqrViewDist() { return _sqrViewDist; }
    SLRectf& rectSS() { return _rectSS; }

    // Misc.
    void    reset();
    void    fromOStoWS(const SLVec3f& minOS,
                       const SLVec3f& maxOS,
                       const SLMat4f& wm);
    void    fromWStoOS(const SLVec3f& minWS,
                       const SLVec3f& maxWS,
                       const SLMat4f& wmI);
    void    updateAxisWS(const SLMat4f& wm);
    void    updateBoneWS(const SLMat4f& parentWM,
                         SLbool         isRoot,
                         const SLMat4f& nodeWM);
    void    mergeWS(SLAABBox& bb);
    void    drawWS(const SLCol4f& color);
    void    drawAxisWS();
    void    drawBoneWS();
    void    setCenterAndRadiusWS();
    void    generateVAO();
    SLbool  isHitInOS(SLRay* ray);
    SLbool  isHitInWS(SLRay* ray);
    void    calculateRectSS();
    SLfloat rectCoverageInSS();

private:
    SLVec3f            _minWS;        //!< Min. corner in world space
    SLVec3f            _minOS;        //!< Min. corner in object space
    SLVec3f            _maxWS;        //!< Max. corner in world space
    SLVec3f            _maxOS;        //!< Max. corner in object space
    SLVec3f            _centerWS;     //!< Center of AABB in world space
    SLVec3f            _centerOS;     //!< Center of AABB in object space
    SLfloat            _radiusWS;     //!< Radius of sphere around AABB in WS
    SLfloat            _radiusOS;     //!< Radius of sphere around AABB in OS
    SLfloat            _sqrViewDist;  //!< Squared dist. from center to viewpoint
    SLVec3f            _axis0WS;      //!< World space axis center point
    SLVec3f            _axisXWS;      //!< World space x-axis vector
    SLVec3f            _axisYWS;      //!< World space y-axis vector
    SLVec3f            _axisZWS;      //!< World space z-axis vector
    SLbool             _boneIsOffset; //!< Flag if the connection parent to us is a bone or an offset
    SLVec3f            _parent0WS;    //!< World space vector to the parent position
    SLbool             _isVisible;    //!< Flag if AABB is in the view frustum
    SLRectf            _rectSS;       //!< Bounding rectangle in screen space
    SLGLVertexArrayExt _vao;          //!< Vertex array object for rendering
};
//-----------------------------------------------------------------------------

#endif
