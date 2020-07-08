#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <Object.h>
#include <Material.h>
#include <Mesh.h>
#include <SLMat4.h>

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

    // update()
    // cull()
    // draw()

protected:
    Node*         _parent = nullptr; //!< pointer to the parent node
    vector<Node*> _children;         //!< vector of children nodes
    Mesh*         _mesh = nullptr;   //!< vector of meshes of the node
    SLMat4f       _om;               //!< object matrix for local transforms
};
//-----------------------------------------------------------------------------
typedef vector<Node> VNode;
//-----------------------------------------------------------------------------
#endif // NODE_H
