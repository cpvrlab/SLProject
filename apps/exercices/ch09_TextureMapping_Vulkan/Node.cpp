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

void Node::updateWM() const
{
    if (_parent)
        _wm.setMatrix(_parent->updateAndGetWM() * _om);
    else
        _wm.setMatrix(_om);

    _wmI.setMatrix(_wm);
    _wmI.invert();
    _wmN.setMatrix(_wm.mat3());

    _isWMUpToDate = true;
}

const SLMat4f& Node::updateAndGetWM() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wm;
}

const SLMat4f& Node::updateAndGetWMI() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wmI;
}

const SLMat4f& Node::updateAndGetWMN() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wmN;
}
