#ifndef SL_CIRCLE_H
#define SL_CIRCLE_H

#include <SLNode.h>

class SLCircle : public SLNode
{
public:
    SLCircle(float r) : _r(r){};
    virtual void drawMeshes(SLSceneView* sv);

private:
    float              _r;
    SLGLVertexArrayExt _vao; //!< Vertex array for rendering
};

#endif