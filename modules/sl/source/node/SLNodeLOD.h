#ifndef SLNODELOD_H
#define SLNODELOD_H

#include <SLNode.h>

class SLNodeLOD : public SLNode
{
    virtual void cullChildren3D(SLSceneView* sv);
};

#endif