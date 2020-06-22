#include "Node.h"

void Node::AddMesh(Mesh* mesh)
{
    _meshes.push_back(mesh);
}

void Node::AddChild(Node* child)
{
    _children.push_back(child);
}
