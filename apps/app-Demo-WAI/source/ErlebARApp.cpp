#include "ErlebARApp.h"
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLAssimpImporter.h>

void ErlebARApp::init(int            scrWidth,
                      int            scrHeight,
                      float          scr2fbX,
                      float          scr2fbY,
                      int            dpi,
                      AppDirectories dirs)
{
    addEvent(new InitEvent(scrWidth, scrHeight, scr2fbX, scr2fbY, dpi, dirs));
}

void ErlebARApp::goBack()
{
    addEvent(new GoBackEvent());
}

void ErlebARApp::IDLE(const sm::NoEventData* data)
{
}

void ErlebARApp::INIT(const InitData* data)
{
    assert(data != nullptr);

    const std::string& slDataRoot = data->deviceData.dirs().slDataRoot;
    // setup magic paths
    SLGLProgram::defaultPath      = slDataRoot + "/shaders/";
    SLGLTexture::defaultPath      = slDataRoot + "/images/textures/";
    SLGLTexture::defaultPathFonts = slDataRoot + "/images/fonts/";
    SLAssimpImporter::defaultPath = slDataRoot + "/models/";

    //instantiation of views
}

void ErlebARApp::TERMINATE(const sm::NoEventData* data)
{
    addEvent(new StateDoneEvent());
}

void ErlebARApp::SELECTION(const sm::NoEventData* data)
{
}

void ErlebARApp::TEST(const sm::NoEventData* data)
{
}
