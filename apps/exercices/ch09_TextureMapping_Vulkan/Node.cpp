#include "Node.h"

void Node::SetMesh(Mesh* mesh)
{
    _mesh = mesh;
}

void Node::om(const SLMat4f& mat)
{
    _om = mat;
}

void Node::AddChild(Node* child)
{
    _children.push_back(child);
}
