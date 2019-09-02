#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <CVCapture.h>
#include <AppDemoGuiStatsTiming.h>
#include <Utils.h>
//-----------------------------------------------------------------------------
AppDemoGuiStatsTiming::AppDemoGuiStatsTiming(string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiStatsTiming::buildInfos(SLScene* s, SLSceneView* sv)
{
    SLRenderType rType = sv->renderType();
    SLfloat      ft    = s->frameTimesMS().average();
    SLchar       m[2550]; // message character array
    m[0] = 0;             // set zero length

    // Get averages from average variables (see SLAverage)
    SLfloat captureTime  = CVCapture::instance()->captureTimesMS().average();
    SLfloat updateTime   = s->updateTimesMS().average();
    SLfloat draw3DTime   = s->draw3DTimesMS().average();
    SLfloat draw2DTime   = s->draw2DTimesMS().average();
    SLfloat cullTime     = s->cullTimesMS().average();

    // Calculate percentage from frame time
    SLfloat captureTimePC  = Utils::clamp(captureTime / ft * 100.0f, 0.0f, 100.0f);
    SLfloat updateTimePC   = Utils::clamp(updateTime / ft * 100.0f, 0.0f, 100.0f);

    SLfloat draw3DTimePC   = Utils::clamp(draw3DTime / ft * 100.0f, 0.0f, 100.0f);
    SLfloat draw2DTimePC   = Utils::clamp(draw2DTime / ft * 100.0f, 0.0f, 100.0f);
    SLfloat cullTimePC     = Utils::clamp(cullTime / ft * 100.0f, 0.0f, 100.0f);

    sprintf(m + strlen(m), "Renderer      : OpenGL\n");
    sprintf(m + strlen(m), "Frame size    : %d x %d\n", sv->scrW(), sv->scrH());
    sprintf(m + strlen(m), "NO. drawcalls : %d\n", SLGLVertexArray::totalDrawCalls);
    sprintf(m + strlen(m), "Frames per s. : %4.1f\n", s->fps());
    sprintf(m + strlen(m), "Frame time    : %4.1f ms (100%%)\n", ft);
    sprintf(m + strlen(m), "  Capture     : %4.1f ms (%3d%%)\n", captureTime, (SLint)captureTimePC);
    sprintf(m + strlen(m), "  Update      : %4.1f ms (%3d%%)\n", updateTime, (SLint)updateTimePC);
    sprintf(m + strlen(m), "  Culling     : %4.1f ms (%3d%%)\n", cullTime, (SLint)cullTimePC);
    sprintf(m + strlen(m), "  Drawing 3D  : %4.1f ms (%3d%%)\n", draw3DTime, (SLint)draw3DTimePC);
    sprintf(m + strlen(m), "  Drawing 2D  : %4.1f ms (%3d%%)\n", draw2DTime, (SLint)draw2DTimePC);
}

