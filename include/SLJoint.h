//#############################################################################
//  File:      SLJoint.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLBONE_H
#define SLBONE_H

#include <stdafx.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class SLSkeleton;
//-----------------------------------------------------------------------------
//! Specialized SLNode that represents a single joint (or bone) in a skeleton
/*!
The main addition of SLJoint to the base SLNode is the offset matrix.
The offset matrix is the inverse transformation of the joint's binding pose in
mesh space. It is used to transform the vertices of a rigged mesh to the origin
of the joint to be able to manipulate them in the join's space.
The ID of the joint must be unique among all joints in the parent skeleton.
*/
class SLJoint : public SLNode
{
public:
                    SLJoint     (SLuint handle,
                                 SLSkeleton* creator);
                    SLJoint     (const SLstring& name,
                                 SLuint handle,
                                 SLSkeleton* creator);
    
    SLJoint*        createChild (SLuint id);
    SLJoint*        createChild (const SLstring& name, SLuint id);

    void            calcMaxRadius(const SLVec3f& vec);
    SLMat4f         calcFinalMat();

    void            needUpdate();

    // Setters
    void            offsetMat   (const SLMat4f& mat) { _offsetMat = mat; }

    // Getters
    SLuint          id          () const { return _id; }
    const SLMat4f&  offsetMat   () const { return _offsetMat; }
    SLfloat         radius      () const { return _radius; }

protected:
    SLuint          _id;        //!< unique id inside its parent skeleton
    SLSkeleton*     _skeleton;  //!< the skeleton this joint belongs to
    SLMat4f         _offsetMat; //!< matrix transforming this joint from bind pose to world pose
    SLfloat         _radius;    //!< info for the mesh this skeleton is bound to (should be moved to a skeleton instance class later, or removed entierely)
};
//-----------------------------------------------------------------------------
typedef std::vector<SLJoint*> SLVJoint;
//-----------------------------------------------------------------------------
#endif
