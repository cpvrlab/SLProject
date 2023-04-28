#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <Object.h>
#include <Material.h>
#include <Mesh.h>
#include <SLMat4.h>

struct NodeStats
{
    SLuint numNodes      = 0; //!< No. of nodes
    SLuint NodesBytes    = 0; //!< Size of nodes in bytes
    SLuint numEmptyNodes = 0; //!< No. of empty nodes
    SLuint numMeshes     = 0; //!< No. of meshs
    SLuint MeshBytes     = 0; //!< Size of meshs in bytes

    void print()
    {
        // SL_LOG("Nodes          : %d", numNodes);
        // SL_LOG("MB Nodes       : %f", (SLfloat)NodesBytes / 1000000.0f);
        // SL_LOG("Empty Nodes    : %d", numEmptyNodes);
        // SL_LOG("Meshes         : %d", numMeshes);
        // SL_LOG("MB Meshes      : %f", (SLfloat)numMeshBytes / 1000000.0f);

        std::cout << "Nodes          :" << numNodes << std::endl;
        std::cout << "MB Nodes       :" << (SLfloat)NodesBytes / 1000000.0f << std::endl;
        std::cout << "Empty Nodes    :" << numEmptyNodes << std::endl;
        std::cout << "Meshes         :" << numMeshes << std::endl;
        std::cout << "MB Meshes      :" << (SLfloat)MeshBytes / 1000000.0f << std::endl;
    }
};

using namespace std;

//-----------------------------------------------------------------------------
class Node : public Object
{
public:
    Node(string name) : Object(name) { ; }
    ~Node() { ; }

    // Getter
    SLMat4f&      om() { return _om; }
    const Mesh*   mesh() { return _mesh; }
    const Node*   parent() { return _parent; }
    vector<Node*> children() { return _children; }

    // Setter
    void SetMesh(Mesh* mesh);
    void om(const SLMat4f& mat);
    void AddChild(Node* child);

    void           updateWM() const;
    const SLMat4f& updateAndGetWM() const;
    const SLMat4f& updateAndGetWMI() const;
    const SLMat4f& updateAndGetWMN() const;

protected:
    Node*           _parent = nullptr; //!< pointer to the parent node
    vector<Node*>   _children;         //!< vector of children nodes
    Mesh*           _mesh = nullptr;   //!< vector of meshes of the node
    SLMat4f         _om;               //!< object matrix for local transforms
    mutable SLMat4f _wm;               //!< world matrix for world transform
    mutable SLMat4f _wmI;              //!< inverse world matrix;
    mutable SLMat3f _wmN;              //!< normal world matrix;
    // SLAABBox      _aabb;             //!< axis aligned bounding box
    mutable bool _isAABBUpToDate = false;
    mutable bool _isWMUpToDate   = false;
};
//-----------------------------------------------------------------------------
typedef vector<Node> VNode;
//-----------------------------------------------------------------------------
#endif // NODE_H
