//#############################################################################
//  File:      ImGuiWrapper.cpp
//  Purpose:   Wrapper Class around the external ImGui GUI-framework
//             See also: https://github.com/ocornut/imgui
//  Author:    Marcus Hudritsch
//  Date:      October 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef IMGUIWRAPPER_H
#define IMGUIWRAPPER_H

#include <imgui.h>
#include <SL.h>
#include <SLEnums.h>
#include <SLVec2.h>
#include <SLRect.h>
#include <SLUiInterface.h>
#include <HighResTimer.h>

class SLScene;
class SLSceneView;
class ImGuiContext;

//e.g. scrolling of child window by touch down and move.
//We need the possibility to turn it off because it conflicts with drag and drop of windows
//if a window is not fixed.
class PanScrolling
{
public:
    void enable()
    {
        _enabled  = true;
        _lastPosY = 0.f;
        _diff     = 0.f;
        _vMW      = 0.f;
        _tOld     = 0.f;
    }

    void disable()
    {
        _enabled = false;
    }

    //call on mouse move
    void moveTo(const float yPos)
    {
        _diff -= (_lastPosY - yPos);
        _lastPosY = yPos;
    }

    //call on mouse down
    void start(const float yPos)
    {
        _lastPosY = yPos;
        _diff     = 0.f;
        _vMW      = 0.f;
        _tOld     = 0.f;
    }

    //call to update mouse wheel in render function
    //As we are using the io.mouseWheel from imgui to set the window position,
    //we have to convert to mouseWheel coordinates.
    float getScrollInMouseWheelCoords(const bool mouseDown, const float fontSize, const float t)
    {
        float dt = t - _tOld;
        _tOld    = t;

        if (mouseDown)
        {
            //Convertion to mouse wheel coords: One mouse wheel unit scrolls about 5 lines of text
            //(see io.MouseWheel comment)
            float diffMW = _diff / (fontSize * 5.f);
            _diff        = 0; //diff consumed, reset it

            //calculate v (of mouse wheel), we need it when left mouse button goes up
            if (dt > 0.000001f)
            {
                //v = s / t
                _vMW = diffMW / dt;
            }

            return diffMW;
        }
        else if (std::abs(_vMW) > 0.000001f)
        {
            //velocity damping
            //v = v - a * t
            if (_vMW > 0)
            {
                _vMW = _vMW - _aMW * dt;
                if (_vMW < 0.f)
                    _vMW = 0.f;
            }
            else
            {
                _vMW = _vMW + _aMW * dt;
                if (_vMW > 0.f)
                    _vMW = 0.f;
            }

            //s = v * t
            return _vMW * dt;
        }
        else
            return 0.f;
    }

    bool enabled() { return _enabled; }

private:
    bool        _enabled  = false;
    float       _lastPosY = 0.f;    //!< mouse down start position
    float       _diff     = 0.f;    //!< Summed up y difference between mouse move events
    float       _vMW      = 0.f;    //!< Mouse wheel velocity: used after pan scrolling for additional scrolling
    float       _tOld     = 0.f;    //!< Time at last call to getScrollInMouseWheelCoords
    const float _aMW      = 100.0f; //!< Mouse wheel acceleration (subtended velocity direction)
};

//-----------------------------------------------------------------------------
//! ImGui Interface class for forwarding all events to the ImGui Handlers
/*! ImGui is a super easy GUI library for the rendering of a UI with OpenGL.
For more information see: https://github.com/ocornut/imgui\n
\n
This class provides only the interface into ImGui. In the event handlers of
SLSceneView the according callback in ImGui is called.\n
There is no UI drawn with this class. It must be defined in another class
that provides the build function. For the Demo apps this is done in the class
SLDemoGui and the build function is passed e.g. in glfwMain function of the
app-Demo-SLProject project.\n
\n
The full call stack for rendering one frame is:\n
- The top-level onPaint of the app (Win, Linux, MacOS, Android or iOS)
  - slUpdateAndPaint: C-Interface function of SLProject
    - SLSceneView::onPaint: Main onPaint function of a sceneview
      - SLGLImGui::onInitNewFrame: Initializes a new GUI frame
        - ImGui::NewFrame()
        - SLGLImGui::build: The UI build function
      - ... normal scene rendering of SLProject
      - SLSceneView::draw2DGL:
        - ImGui::Render
          - SLGLImGui::onPaint(ImGui::GetDrawData())
          - SLDemoGui::buildDemoGui: Builds the full UI
*/

class ImGuiWrapper : public SLUiInterface
{
public:
    ImGuiWrapper();
    ~ImGuiWrapper();
    void init(std::string configPath) override;

    void onInitNewFrame(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;
    void onPaint(const SLRecti& viewport) override;
    void onMouseDown(SLMouseButton button, SLint x, SLint y) override;
    void onMouseUp(SLMouseButton button, SLint x, SLint y) override;
    //returns true if it wants to capture mouse
    void onMouseMove(SLint xPos, SLint yPos) override;
    void onMouseWheel(SLfloat yoffset) override;
    void onKeyPress(SLKey key, SLKey mod) override;
    void onKeyRelease(SLKey key, SLKey mod) override;
    void onCharInput(SLuint c) override;
    void onClose() override;
    void renderExtraFrame(SLScene* s, SLSceneView* sv, SLint mouseX, SLint mouseY) override;
    bool doNotDispatchKeyboard() override;
    bool doNotDispatchMouse() override;
    // gui build function pattern
    virtual void build(SLScene* s, SLSceneView* sv) = 0;

protected:
    ImGuiContext* _context = nullptr;

    PanScrolling _panScroll;

private:
    void deleteOpenGLObjects();
    void createOpenGLObjects();
    void printCompileErrors(SLint         shaderHandle,
                            const SLchar* src);

    SLfloat _timeSec;           //!< Time in seconds
    SLVec2f _mousePosPX;        //!< Mouse cursor position
    SLfloat _mouseWheel;        //!< Mouse wheel position
    SLbool  _mousePressed[3];   //!< Mouse button press state
    SLuint  _fontTexture;       //!< OpenGL texture id for font
    SLint   _progHandle;        //!< OpenGL handle for shader program
    SLint   _vertHandle;        //!< OpenGL handle for vertex shader
    SLint   _fragHandle;        //!< OpenGL handle for fragment shader
    SLint   _attribLocTex;      //!< OpenGL attribute location for texture
    SLint   _attribLocProjMtx;  //!< OpenGL attribute location for ???
    SLint   _attribLocPosition; //!< OpenGL attribute location for vertex pos.
    SLint   _attribLocUV;       //!< OpenGL attribute location for texture coords
    SLint   _attribLocColor;    //!< OpenGL attribute location for color
    SLuint  _vboHandle;         //!< OpenGL handle for vertex buffer object
    SLuint  _vaoHandle;         //!< OpenGL vertex array object handle
    SLuint  _elementsHandle;    //!< OpenGL handle for vertex indexes

    std::string _inifile;
};
//-----------------------------------------------------------------------------
#endif
