#ifndef SL_CIRCLE_H
#define SL_CIRCLE_H

#include <SLNode.h>

class SLCircle : public SLNode
{
public:
    SLCircle(float r) : _screenOffset(0.0f, 0.0f), _r(r){};
    virtual void drawMeshes(SLSceneView* sv);

    void    screenOffset(SLVec2f screenOffset) { _screenOffset = screenOffset; }
    SLVec2f screenOffset() { return _screenOffset; }

    void scaleRadius(float s);

private:
    SLVec2f            _screenOffset;
    float              _r;
    SLGLVertexArrayExt _vao; //!< Vertex array for rendering
};

#endif