//#############################################################################
//  File:      ImGuiWrapper.cpp
//  Purpose:   Wrapper Class around the external ImGui GUI-framework
//             See also: https://github.com/ocornut/imgui
//  Date:      October 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <ImGuiWrapper.h>
#include <SLSceneView.h>
#include <SLScene.h>
#include <GlobalTimer.h>

#include <imgui_internal.h>

//-----------------------------------------------------------------------------
//! Prints the compile errors in case of a GLSL compile failure
void ImGuiRendererOpenGL::printCompileErrors(SLint shaderHandle, const SLchar* src)
{
    // Check compiler log
    SLint compileSuccess = 0;
    glGetShaderiv((SLuint)shaderHandle, GL_COMPILE_STATUS, &compileSuccess);
    if (compileSuccess == GL_FALSE)
    {
        GLchar log[512];
        glGetShaderInfoLog((SLuint)shaderHandle,
                           sizeof(log),
                           nullptr,
                           &log[0]);
        SL_LOG("*** COMPILER ERROR ***");
        SL_LOG("%s\n---", log);
        SL_LOG("%s", src);
    }
}
//-----------------------------------------------------------------------------
//! Creates all OpenGL objects for drawing the imGui
void ImGuiRendererOpenGL::createOpenGLObjects()
{
    // Backup GL state
    GLint last_texture, last_array_buffer, last_vertex_array;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &last_vertex_array);

    // Build version string as the first statement
    SLGLState* state         = SLGLState::instance();
    SLstring   verGLSL       = state->glSLVersionNO();
    SLstring   vertex_shader = "#version " + verGLSL;
    if (state->glIsES3()) vertex_shader += " es";
    vertex_shader +=
      "\n"
      "#ifdef GL_ES\n"
      "precision mediump float;\n"
      "#endif\n"
      "\n"
      "uniform mat4 ProjMtx;\n"
      "in vec2 Position;\n"
      "in vec2 UV;\n"
      "in vec4 Color;\n"
      "out vec2 Frag_UV;\n"
      "out vec4 Frag_Color;\n"
      "void main()\n"
      "{\n"
      "	Frag_UV = UV;\n"
      "	Frag_Color = Color;\n"
      "	gl_Position = ProjMtx * vec4(Position.xy,0,1);\n"
      "}\n";

    SLstring fragment_shader = "#version " + verGLSL;
    if (state->glIsES3()) fragment_shader += " es";
    fragment_shader +=
      "\n"
      "#ifdef GL_ES\n"
      "precision mediump float;\n"
      "#endif\n"
      "\n"
      "uniform sampler2D Texture;\n"
      "in vec2 Frag_UV;\n"
      "in vec4 Frag_Color;\n"
      "out vec4 Out_Color;\n"
      "void main()\n"
      "{\n"
      "	Out_Color = Frag_Color * texture( Texture, Frag_UV.st);\n"
      "}\n";

    _vertHandle         = (SLint)glCreateShader(GL_VERTEX_SHADER);
    _fragHandle         = (SLint)glCreateShader(GL_FRAGMENT_SHADER);
    const char* srcVert = vertex_shader.c_str();
    const char* srcFrag = fragment_shader.c_str();
    glShaderSource((SLuint)_vertHandle, 1, &srcVert, nullptr);
    glShaderSource((SLuint)_fragHandle, 1, &srcFrag, nullptr);
    glCompileShader((SLuint)_vertHandle);
    printCompileErrors(_vertHandle, srcVert);
    glCompileShader((SLuint)_fragHandle);
    printCompileErrors(_fragHandle, srcFrag);

    _progHandle = (SLint)glCreateProgram();
    glAttachShader((SLuint)_progHandle, (SLuint)_vertHandle);
    glAttachShader((SLuint)_progHandle, (SLuint)_fragHandle);
    glLinkProgram((SLuint)_progHandle);

    GET_GL_ERROR;

    _attribLocTex      = glGetUniformLocation((SLuint)_progHandle, "Texture");
    _attribLocProjMtx  = glGetUniformLocation((SLuint)_progHandle, "ProjMtx");
    _attribLocPosition = glGetAttribLocation((SLuint)_progHandle, "Position");
    _attribLocUV       = glGetAttribLocation((SLuint)_progHandle, "UV");
    _attribLocColor    = glGetAttribLocation((SLuint)_progHandle, "Color");

    GET_GL_ERROR;

    glGenBuffers(1, &_vboHandle);
    glGenBuffers(1, &_elementsHandle);

    glGenVertexArrays(1, &_vaoHandle);
    glBindVertexArray(_vaoHandle);
    glBindBuffer(GL_ARRAY_BUFFER, _vboHandle);
    glEnableVertexAttribArray((SLuint)_attribLocPosition);
    glEnableVertexAttribArray((SLuint)_attribLocUV);
    glEnableVertexAttribArray((SLuint)_attribLocColor);

    GET_GL_ERROR;

#define OFFSETOF(TYPE, ELEMENT) ((size_t) & (((TYPE*)nullptr)->ELEMENT))
    glVertexAttribPointer((SLuint)_attribLocPosition,
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(ImDrawVert),
                          (GLvoid*)OFFSETOF(ImDrawVert, pos));
    glVertexAttribPointer((SLuint)_attribLocUV,
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(ImDrawVert),
                          (GLvoid*)OFFSETOF(ImDrawVert, uv));
    glVertexAttribPointer((SLuint)_attribLocColor,
                          4,
                          GL_UNSIGNED_BYTE,
                          GL_TRUE,
                          sizeof(ImDrawVert),
                          (GLvoid*)OFFSETOF(ImDrawVert, col));
#undef OFFSETOF

    GET_GL_ERROR;

    // Build texture atlas
    ImGuiIO& io = _context->IO;
    SLuchar* pixels;
    int      width, height;

    // Load as RGBA 32-bits (75% of the memory is wasted, but default font is
    // so small) because it is more likely to be compatible with user's
    // existing shaders. If your ImTextureId represent a higher-level concept
    // than just a GL texture id, consider calling GetTexDataAsAlpha8()
    // instead to save on GPU memory.
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    // Upload texture to graphics system
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    glGenTextures(1, &_fontTexture);
    glBindTexture(GL_TEXTURE_2D, _fontTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    GET_GL_ERROR;

    // Store our identifier
    io.Fonts->TexID = (void*)(intptr_t)_fontTexture;

    // Restore state
    glBindTexture(GL_TEXTURE_2D, (SLuint)last_texture);

    // Restore modified GL state
    glBindTexture(GL_TEXTURE_2D, (SLuint)last_texture);
    glBindBuffer(GL_ARRAY_BUFFER, (SLuint)last_array_buffer);
    glBindVertexArray((SLuint)last_vertex_array);

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Deletes all OpenGL objects for drawing the imGui
void ImGuiRendererOpenGL::deleteOpenGLObjects()
{
    if (_vaoHandle) glDeleteVertexArrays(1, &_vaoHandle);
    if (_vboHandle) glDeleteBuffers(1, &_vboHandle);
    if (_elementsHandle) glDeleteBuffers(1, &_elementsHandle);
    _vaoHandle = _vboHandle = _elementsHandle = 0;

    if (_progHandle && _vertHandle) glDetachShader((SLuint)_progHandle,
                                                   (SLuint)_vertHandle);
    if (_vertHandle) glDeleteShader((SLuint)_vertHandle);
    _vertHandle = 0;

    if (_progHandle && _fragHandle) glDetachShader((SLuint)_progHandle,
                                                   (SLuint)_fragHandle);
    if (_fragHandle) glDeleteShader((SLuint)_fragHandle);
    _fragHandle = 0;

    if (_progHandle) glDeleteProgram((SLuint)_progHandle);
    _progHandle = 0;

    if (_fontTexture)
    {
        glDeleteTextures(1, &_fontTexture);
        _context->IO.Fonts->TexID = nullptr;
        _fontTexture              = 0;
    }
}
//-----------------------------------------------------------------------------
void ImGuiRendererOpenGL::render(const SLRecti& viewportRect)
{
    ImGui::SetCurrentContext(_context);
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    ImGuiIO& io = _context->IO;

    // Avoid rendering when minimized, scale coordinates for retina displays
    // (screen coordinates != framebuffer coordinates)
    int fbWidth  = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
    int fbHeight = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
    if (fbWidth == 0 || fbHeight == 0)
        return;
    draw_data->ScaleClipRects(io.DisplayFramebufferScale);

    // Backup GL state
    GLint last_active_texture;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &last_active_texture);
    glActiveTexture(GL_TEXTURE0);

    GLint last_program;
    glGetIntegerv(GL_CURRENT_PROGRAM, &last_program);
    GLint last_texture;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    GLint last_array_buffer;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
    GLint last_element_array_buffer;
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &last_element_array_buffer);
    GLint last_vertex_array;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &last_vertex_array);
    GLint last_blend_src_rgb;
    glGetIntegerv(GL_BLEND_SRC_RGB, &last_blend_src_rgb);
    GLint last_blend_dst_rgb;
    glGetIntegerv(GL_BLEND_DST_RGB, &last_blend_dst_rgb);
    GLint last_blend_src_alpha;
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &last_blend_src_alpha);
    GLint last_blend_dst_alpha;
    glGetIntegerv(GL_BLEND_DST_ALPHA, &last_blend_dst_alpha);
    GLint last_blend_equation_rgb;
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &last_blend_equation_rgb);
    GLint last_blend_equation_alpha;
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &last_blend_equation_alpha);
    GLint last_viewport[4];
    glGetIntegerv(GL_VIEWPORT, last_viewport);
    GLint last_scissor_box[4];
    glGetIntegerv(GL_SCISSOR_BOX, last_scissor_box);

    GLboolean last_enable_blend        = glIsEnabled(GL_BLEND);
    GLboolean last_enable_cull_face    = glIsEnabled(GL_CULL_FACE);
    GLboolean last_enable_depth_test   = glIsEnabled(GL_DEPTH_TEST);
    GLboolean last_enable_scissor_test = glIsEnabled(GL_SCISSOR_TEST);

    // Setup render state: alpha-blending enabled, no face culling, no depth testing, scissor enabled
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);

    // Setup viewport
    if (viewportRect.isEmpty())
        glViewport(0,
                   0,
                   (GLsizei)fbWidth,
                   (GLsizei)fbHeight);
    else
        glViewport((GLsizei)(viewportRect.x * io.DisplayFramebufferScale.x),
                   (GLsizei)(viewportRect.y * io.DisplayFramebufferScale.y),
                   (GLsizei)(viewportRect.width * io.DisplayFramebufferScale.x),
                   (GLsizei)(viewportRect.height * io.DisplayFramebufferScale.y));

    // Setup orthographic projection matrix
    // clang-format off
    const float ortho_projection[4][4] =
    {
        {2.0f / io.DisplaySize.x, 0.0f,                     0.0f, 0.0f},
        {0.0f,                    2.0f / -io.DisplaySize.y, 0.0f, 0.0f},
        {0.0f,                    0.0f,                    -1.0f, 0.0f},
        {-1.0f,                   1.0f,                     0.0f, 1.0f},
    };
    // clang-format on

    glUseProgram((SLuint)_progHandle);
    glUniform1i(_attribLocTex, 0);
    glUniformMatrix4fv(_attribLocProjMtx, 1, GL_FALSE, &ortho_projection[0][0]);
    glBindVertexArray((SLuint)_vaoHandle);

    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list          = draw_data->CmdLists[n];
        const ImDrawIdx*  idx_buffer_offset = nullptr;

        glBindBuffer(GL_ARRAY_BUFFER, _vboHandle);
        glBufferData(GL_ARRAY_BUFFER,
                     (GLsizeiptr)cmd_list->VtxBuffer.Size * (GLsizeiptr)sizeof(ImDrawVert),
                     (const GLvoid*)cmd_list->VtxBuffer.Data,
                     GL_STREAM_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _elementsHandle);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     (GLsizeiptr)cmd_list->IdxBuffer.Size * (GLsizeiptr)sizeof(ImDrawIdx),
                     (const GLvoid*)cmd_list->IdxBuffer.Data,
                     GL_STREAM_DRAW);

        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback)
            {
                pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D, (GLuint)(intptr_t)pcmd->TextureId);

                if (viewportRect.isEmpty())
                    glScissor((int)pcmd->ClipRect.x,
                              (int)(fbHeight - pcmd->ClipRect.w),
                              (int)(pcmd->ClipRect.z - pcmd->ClipRect.x),
                              (int)(pcmd->ClipRect.w - pcmd->ClipRect.y));
                else
                    glScissor((GLsizei)(viewportRect.x * io.DisplayFramebufferScale.x),
                              (GLsizei)(viewportRect.y * io.DisplayFramebufferScale.y),
                              (GLsizei)(viewportRect.width * io.DisplayFramebufferScale.x),
                              (GLsizei)(viewportRect.height * io.DisplayFramebufferScale.y));

                glDrawElements(GL_TRIANGLES,
                               (GLsizei)pcmd->ElemCount,
                               sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                               idx_buffer_offset);
            }
            idx_buffer_offset += pcmd->ElemCount;
        }
    }

    // Restore modified GL state
    glUseProgram((SLuint)last_program);
    glBindTexture(GL_TEXTURE_2D, (SLuint)last_texture);
    glActiveTexture((SLuint)last_active_texture);
    glBindVertexArray((SLuint)last_vertex_array);
    glBindBuffer(GL_ARRAY_BUFFER, (SLuint)last_array_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, (SLuint)last_element_array_buffer);
    glBlendEquationSeparate((SLuint)last_blend_equation_rgb,
                            (SLuint)last_blend_equation_alpha);
    glBlendFuncSeparate((SLuint)last_blend_src_rgb,
                        (SLuint)last_blend_dst_rgb,
                        (SLuint)last_blend_src_alpha,
                        (SLuint)last_blend_dst_alpha);
    if (last_enable_blend)
        glEnable(GL_BLEND);
    else
        glDisable(GL_BLEND);

    if (last_enable_cull_face)
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);

    if (last_enable_depth_test)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    if (last_enable_scissor_test)
        glEnable(GL_SCISSOR_TEST);
    else
        glDisable(GL_SCISSOR_TEST);

    glViewport(last_viewport[0],
               last_viewport[1],
               (GLsizei)last_viewport[2],
               (GLsizei)last_viewport[3]);

    glScissor(last_scissor_box[0],
              last_scissor_box[1],
              (GLsizei)last_scissor_box[2],
              (GLsizei)last_scissor_box[3]);

    ImGui::SetCurrentContext(nullptr);
}
//-----------------------------------------------------------------------------
void ImGuiEngine::init(const std::string& configPath)
{
    ImGuiIO& io    = _context->IO;
    _iniFilename   = configPath + "imgui.ini";
    io.IniFilename = _iniFilename.c_str();

    io.KeyMap[ImGuiKey_Tab]        = K_tab;
    io.KeyMap[ImGuiKey_LeftArrow]  = K_left;
    io.KeyMap[ImGuiKey_RightArrow] = K_right;
    io.KeyMap[ImGuiKey_UpArrow]    = K_up;
    io.KeyMap[ImGuiKey_DownArrow]  = K_down;
    io.KeyMap[ImGuiKey_PageUp]     = K_pageUp;
    io.KeyMap[ImGuiKey_PageDown]   = K_pageUp;
    io.KeyMap[ImGuiKey_Home]       = K_home;
    io.KeyMap[ImGuiKey_End]        = K_end;
    io.KeyMap[ImGuiKey_Delete]     = K_delete;
    io.KeyMap[ImGuiKey_Backspace]  = K_backspace;
    io.KeyMap[ImGuiKey_Enter]      = K_enter;
    io.KeyMap[ImGuiKey_Escape]     = K_esc;
    io.KeyMap[ImGuiKey_A]          = 'A';
    io.KeyMap[ImGuiKey_C]          = 'C';
    io.KeyMap[ImGuiKey_V]          = 'V';
    io.KeyMap[ImGuiKey_X]          = 'X';
    io.KeyMap[ImGuiKey_Y]          = 'Y';
    io.KeyMap[ImGuiKey_Z]          = 'Z';

    // The screen size is set again in onResize
    io.DisplaySize             = ImVec2(0, 0);
    io.DisplayFramebufferScale = ImVec2(1, 1);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
ImGuiWrapper::ImGuiWrapper(ImGuiContext* context, ImGuiRenderer* renderer)
  : _context(context),
    _renderer(renderer)
{
    assert(_context);
    assert(_renderer);
}
//-----------------------------------------------------------------------------
ImGuiWrapper::~ImGuiWrapper()
{
}
//-----------------------------------------------------------------------------
//! Initializes OpenGL handles to zero and sets the ImGui key map
void ImGuiWrapper::init(const std::string& configPath)
{
    _mouseWheel = 0.0f;
}
//-----------------------------------------------------------------------------
//! Inits a new frame for the ImGui system
void ImGuiWrapper::onInitNewFrame(SLScene* s, SLSceneView* sv)
{
    // If no build function is provided there is no ImGui
    // if (!build) return;

    // if ((SLint)ImGuiWrapper::fontPropDots != (SLint)_fontPropDots ||
    //     (SLint)ImGuiWrapper::fontFixedDots != (SLint)_fontFixedDots)
    //     loadFonts(ImGuiWrapper::fontPropDots, ImGuiWrapper::fontFixedDots);

    // if (!_fontTexture)
    //     createOpenGLObjects();

    ImGuiIO& io = _context->IO;

    // Setup time step
    SLfloat nowSec = GlobalTimer::timeS();
    io.DeltaTime   = _timeSec > 0.0 ? nowSec - _timeSec : 1.0f / 60.0f;
    if (io.DeltaTime < 0) io.DeltaTime = 1.0f / 60.0f;
    _timeSec = nowSec;

    if (_panScroll.enabled())
    {
        io.MouseWheel = _panScroll.getScrollInMouseWheelCoords(io.MouseDown[0], _context->FontSize, nowSec);
    }
    else
    {
        io.MouseWheel = _mouseWheel;
        _mouseWheel   = 0.0f;
    }

    // Start the frame
    ImGui::SetCurrentContext(_context);
    ImGui::NewFrame();

    // Call the build function to construct the ui
    build(s, sv);
    ImGui::SetCurrentContext(nullptr);
    // SL_LOG(".");
}
//-----------------------------------------------------------------------------
//! Callback if window got resized
void ImGuiWrapper::onResize(SLint scrW,
                            SLint scrH)
{
    ImGuiIO& io                = _context->IO;
    io.DisplaySize             = ImVec2((SLfloat)scrW, (SLfloat)scrH);
    io.DisplayFramebufferScale = ImVec2(1, 1);
}
//-----------------------------------------------------------------------------
//! Callback for main rendering for the ImGui GUI system
void ImGuiWrapper::onPaint(const SLRecti& viewportRect)
{
    // do the opengl rendering
    _renderer->render(viewportRect);
}
//-----------------------------------------------------------------------------
//! Callback on mouse button down event
void ImGuiWrapper::onMouseDown(SLMouseButton button, SLint x, SLint y)
{
    ImGuiIO& io = _context->IO;
    io.MousePos = ImVec2((SLfloat)x, (SLfloat)y);
    if (button == MB_left)
    {
        io.MouseDown[0] = true;
        if (_panScroll.enabled())
            _panScroll.start((SLfloat)y);
    }
    if (button == MB_right) io.MouseDown[1] = true;
    if (button == MB_middle) io.MouseDown[2] = true;
    // SL_LOG("D");
}
//-----------------------------------------------------------------------------
//! Callback on mouse button up event
void ImGuiWrapper::onMouseUp(SLMouseButton button, SLint x, SLint y)
{
    ImGuiIO& io = _context->IO;
    io.MousePos = ImVec2((SLfloat)x, (SLfloat)y);
    if (button == MB_left) io.MouseDown[0] = false;
    if (button == MB_right) io.MouseDown[1] = false;
    if (button == MB_middle) io.MouseDown[2] = false;
    // SL_LOG("U");
}
//-----------------------------------------------------------------------------
//! Updates the mouse cursor position
void ImGuiWrapper::onMouseMove(SLint xPos, SLint yPos)
{
    _context->IO.MousePos = ImVec2((SLfloat)xPos, (SLfloat)yPos);
    ImGuiIO& io           = _context->IO;
    if (io.MouseDown[0] && _panScroll.enabled())
        _panScroll.moveTo((float)yPos);
    // SL_LOG("M");
}
//-----------------------------------------------------------------------------
//! Callback for the mouse scroll movement
void ImGuiWrapper::onMouseWheel(SLfloat yoffset)
{
    // Use fractional mouse wheel, 1.0 unit 5 lines.
    _mouseWheel += yoffset;
}
//-----------------------------------------------------------------------------
//! Callback on key press event
void ImGuiWrapper::onKeyPress(SLKey key, SLKey mod)
{
    ImGuiIO& io      = _context->IO;
    io.KeysDown[key] = true;
    io.KeyCtrl       = mod & K_ctrl ? true : false;
    io.KeyShift      = mod & K_shift ? true : false;
    io.KeyAlt        = mod & K_alt ? true : false;
}
//-----------------------------------------------------------------------------
//! Callback on key release event
void ImGuiWrapper::onKeyRelease(SLKey key, SLKey mod)
{
    ImGuiIO& io      = _context->IO;
    io.KeysDown[key] = false;
    io.KeyCtrl       = mod & K_ctrl ? true : false;
    io.KeyShift      = mod & K_shift ? true : false;
    io.KeyAlt        = mod & K_alt ? true : false;
}
//-----------------------------------------------------------------------------
//! Callback on character input
void ImGuiWrapper::onCharInput(SLuint c)
{
    ImGuiIO& io = _context->IO;
    if (c > 0 && c < 0x10000)
        io.AddInputCharacter((unsigned short)c);
}
//-----------------------------------------------------------------------------
//! Callback on closing the application
void ImGuiWrapper::onClose()
{
}
//-----------------------------------------------------------------------------
//! Renders an extra frame with the current mouse position
void ImGuiWrapper::renderExtraFrame(SLScene* s, SLSceneView* sv, SLint mouseX, SLint mouseY)
{
    // If ImGui build function exists render the ImGui
    _context->IO.MousePos = ImVec2((SLfloat)mouseX, (SLfloat)mouseY);
    onInitNewFrame(s, sv);
    onPaint(sv->viewportRect());
}
//-----------------------------------------------------------------------------
bool ImGuiWrapper::doNotDispatchKeyboard()
{
    return _context->IO.WantCaptureKeyboard;
}
//-----------------------------------------------------------------------------
bool ImGuiWrapper::doNotDispatchMouse()
{
    return _context->IO.WantCaptureMouse;
}
//-----------------------------------------------------------------------------
