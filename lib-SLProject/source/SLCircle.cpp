#include <SLCircle.h>
#include <SLGLState.h>

void SLCircle::drawMeshes(SLSceneView* sv)
{
    SLGLState* stateGL = SLGLState::instance();

    stateGL->pushModelViewMatrix();
    stateGL->modelViewMatrix.translate(0, 0, 1.0f);

    SLVVec3f rombusAndCirclePoints;

    // Add points for circle over window
    SLint   circlePoints = 60;
    SLfloat deltaPhi     = Utils::TWOPI / (SLfloat)circlePoints;

    SLVec2f scrCoords = SLVec2f(_wm.translation().x,
                                _wm.translation().y);

    for (SLint i = 0; i < circlePoints; ++i)
    {
        SLVec2f c;
        c.fromPolar(_r, i * deltaPhi);
        c.add(c, scrCoords);
        rombusAndCirclePoints.push_back(SLVec3f(c.x, c.y, 0));
    }

    _vao.clearAttribs();
    _vao.generateVertexPos(&rombusAndCirclePoints);
    SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);

    _vao.drawArrayAsColored(PT_lineLoop, yelloAlpha);

    stateGL->popModelViewMatrix();
}
