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

/*
SLAABBox& Node::updateAABBRec()
{
    if (_isAABBUpToDate)
        return _aabb;

    // empty the AABB (= max negative AABB)
    if (_mesh != nullptr || !_children.empty())
    {
        _aabb.minWS(SLVec3f(FLT_MAX, FLT_MAX, FLT_MAX));
        _aabb.maxWS(SLVec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    }

    // TODO: Find solution for camera exception
    // if (typeid(*this) == typeid(Camera))
    //     ((Camera*)this)->buildAABB(_aabb, updateAndGetWM());

    // Build or updateRec AABB of meshes & merge them to the nodes aabb in WS
    SLAABBox aabbMesh;
    _mesh->buildAABB(aabbMesh, updateAndGetWM());
    _aabb.mergeWS(aabbMesh);

    // Merge children in WS except for cameras except if cameras have children
    for (Node* child : _children)
    {
        bool childIsCamera = typeid(*child)==typeid(SLCamera);
        bool cameraHasChildren = false;
        if (childIsCamera)
            cameraHasChildren = !child->children().empty();

        if (!childIsCamera || cameraHasChildren)

child->updateAABBRec();
_aabb.mergeWS(child->updateAABBRec());
}

// We need min & max also in OS for the uniform grid intersection in OS
_aabb.fromWStoOS(_aabb.minWS(), _aabb.maxWS(), updateAndGetWMI());

// For visualizing the nodes orientation we finally updateRec the axis in WS
_aabb.updateAxisWS(updateAndGetWM());

_isAABBUpToDate = true;

return _aabb;
}
*/
