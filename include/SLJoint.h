//#############################################################################
//  File:      SLJoint.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
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
/*! Specialized SLNode that represents a single joint (or bone) in a skeleton */
class SLJoint : public SLNode
{
public:
    SLJoint(SLuint handle, SLSkeleton* creator);
    SLJoint(const SLstring& name, SLuint handle, SLSkeleton* creator);
    
    SLJoint*        createChild(SLuint handle);
    SLJoint*        createChild(const SLstring& name, SLuint handle);
    
    SLuint          handle() const { return _handle; }

    void            offsetMat           (const SLMat4f& mat);
    const SLMat4f&  offsetMat() const { return _offsetMat; }

    void            calcMaxRadius(const SLVec3f& vec);
    SLfloat         radius() const { return _radius; }

    SLMat4f         calculateFinalMat();

protected:
    SLuint          _handle;        //!< unique handle inside its parent skeleton
    SLSkeleton*     _creator;       //!< the skeleton this joint belongs to
    SLMat4f         _offsetMat;     //!< matrix transforming this joint from bind pose to world pose

    // specific information for the mesh this skeleton is bound to (should be moved to a skeleton instance class later, or removed entierely)
    SLfloat         _radius;
};
//-----------------------------------------------------------------------------
typedef std::vector<SLJoint*> SLVJoint;
//-----------------------------------------------------------------------------
#endif
