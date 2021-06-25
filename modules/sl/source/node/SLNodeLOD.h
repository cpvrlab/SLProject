#ifndef SLNODELOD_H
#define SLNODELOD_H

#include <SLNode.h>

class SLNodeLOD : public SLNode
{
public:
    SLNodeLOD();
    void         addLODChild(SLNode* child, SLfloat minValue, SLfloat maxValue);
    virtual void cullChildren3D(SLSceneView* sv);

private:
    SLint _childIndices[101];
};

#endif