//#############################################################################
//  File:      SLNodeMesh.h
//  Author:    Marcus Hudritsch
//  Date:      August 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLNODEMESH_H
#define SLNODEMESH_H

#include <SLNode.h>
#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLNodeMesh
/*!
 *
 */
class SLNodeMesh
{
public:
    SLNodeMesh(SLNode* node, SLMesh* mesh) : _node(node), _mesh(mesh)
    {
        assert(node && mesh);
    }

    void draw(SLSceneView* sv, SLMaterial* _mat);

private:
    SLNode*            _node;      //!< Pointer to the associated node
    SLMesh*            _mesh;      //!< Pointer to the associated mesh
    SLGLVertexArray    _vao;       //!< OpenGL Vertex Array Object for drawing
    SLGLVertexArrayExt _vaoN;      //!< OpenGL VAO for optional normal drawing
    SLGLVertexArrayExt _vaoT;      //!< OpenGL VAO for optional tangent drawing
    SLGLVertexArrayExt _vaoS;      //!< OpenGL VAO for optional selection drawing
    SLGLPrimitiveType  _primitive; //!< Primitive type (default triangles)
    bool               _isVisible; //!< flag if the nodes AABB is in frustum
};
//-----------------------------------------------------------------------------
typedef vector<SLNodeMesh> SLVNodeMesh;
//-----------------------------------------------------------------------------
#endif