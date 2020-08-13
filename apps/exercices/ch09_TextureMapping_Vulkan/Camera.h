#ifndef CAMERA_H
#define CAMERA_H

#include <Node.h>
#include <SLPlane.h>

class Camera : public Node
{
public:
    Camera() : Node("Camera") { ; }

    // void setFrustumPlanes();
    // bool isInFrustum(SLAABBox* aabb);

    // Getter
    float fov() { return _fov; }
    float viewportWidth() { return _viewportWidth; }
    float viewportHeight() { return _viewportHeight; }
    float viewportRatio() { return _viewportWidth / _viewportHeight; }
    float clipNear() { return _clipNear; }
    float clipFar() { return _clipFar; }
    // Setter
    void fov(float fov) { _fov = fov; }
    void clipNear(float clipNear) { _clipNear = clipNear; }
    void clipFar(float clipFar) { _clipFar = clipFar; }
    void setViewport(float width, float height)
    {
        assert(height > 0 && width > 0);
        _viewportWidth  = width;
        _viewportHeight = height;
        _viewportRatio  = width / height;
    }

protected:
    float   _fov = 40.0f;
    float   _viewportWidth;
    float   _viewportHeight;
    float   _viewportRatio;
    float   _clipNear = 0.1f;
    float   _clipFar  = 200.0f;
    SLPlane _plane[6]; //!< 6 frustum planes (t, b, l, r, n, f)
};

#endif
