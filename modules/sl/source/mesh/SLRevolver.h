//#############################################################################
//  File:      SLRevolver.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLREVOLVER_H
#define SLREVOLVER_H

#include <SLMesh.h>

#include <utility>

class SLAssetManager;

//-----------------------------------------------------------------------------
//! SLRevolver is an SLMesh object built out of revolving points.
/*!
SLRevolver is an SLMesh object that is built out of points that are revolved
in slices around and axis. The surface will be outwards if the points in the
array _revPoints increase towards the axis direction.
If all points in the array _revPoints are different the normals will be
smoothed. If two consecutive points are identical the normals will define a
hard edge. Texture coords. are cylindrically mapped.
*/
class SLRevolver : public SLMesh
{
public:
    //! ctor for generic revolver mesh
    SLRevolver(SLAssetManager* assetMgr,
               SLVVec3f        revolvePoints,
               SLVec3f         revolveAxis,
               SLuint          slices      = 36,
               SLbool          smoothFirst = false,
               SLbool          smoothLast  = false,
               SLstring        name        = "revolver mesh",
               SLMaterial*     mat         = nullptr);

    //! ctor for derived revolver shapes
    SLRevolver(SLAssetManager* assetMgr, SLstring name) : SLMesh(assetMgr, std::move(name)) { ; }

    void   buildMesh(SLMaterial* mat = nullptr);
    SLuint stacks() { return _stacks; }
    SLuint slices() { return _slices; }

protected:
    SLVVec3f _revPoints; //!< Array revolving points
    SLVec3f  _revAxis;   //!< axis of revolution
    SLuint   _stacks;    //!< No. of stacks (mostly used)
    SLuint   _slices;    //!< NO. of slices

    //! flag if the normal of the first point is eqaual to -revAxis
    SLbool _smoothFirst;

    //! flag if the normal of the last point is eqaual to revAxis
    SLbool _smoothLast;
};
//-----------------------------------------------------------------------------
#endif // SLREVOLVER_H
