//#############################################################################
//  File:      GL/SLGLImGui.cpp
//  Purpose:   Wrapper Class around the external ImGui GUI-framework
//             See also: https://github.com/ocornut/imgui
//  Author:    Marcus Hudritsch
//  Date:      October 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLGLImGui.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! Static global instance for render callback
SLGLImGui* SLGLImGui::globalInstance = 0; 
//-----------------------------------------------------------------------------
//! Function called whan no GUI build function is provided
void noGuiBuilt(SLScene* s, SLSceneView* sv)
{
    static SLbool showOnce = true;
    if (showOnce)
    {   ImGui::SetNextWindowSize(ImVec2(200,80), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Error", &showOnce);
        ImGui::Text("There is no GUI build function\nprovided for SLGLImGui::build.");
        ImGui::End();
    }
}
//-----------------------------------------------------------------------------
//! Initializes OpenGL handles to zero and sets the ImGui key map
void SLGLImGui::init(SLint scrW, SLint scrH, SLint fbW, SLint fbH)
{
    SLGLImGui::globalInstance = this;

    _fontTexture = 0;
    _progHandle = 0;
    _vertHandle = 0;
    _fragHandle = 0;
    _attribLocTex = 0;
    _attribLocProjMtx = 0;
    _attribLocPosition = 0;
    _attribLocUV = 0;
    _attribLocColor = 0;
    _vboHandle = 0;
    _vaoHandle = 0;
    _elementsHandle = 0;

    _mouseWheel = 0.0f;
    _mousePressed[0] = false;
    _mousePressed[1] = false;
    _mousePressed[2] = false;

    ImGuiIO& io = ImGui::GetIO();
    io.KeyMap[ImGuiKey_Tab]         = K_tab;
    io.KeyMap[ImGuiKey_LeftArrow]   = K_left;
    io.KeyMap[ImGuiKey_RightArrow]  = K_right;
    io.KeyMap[ImGuiKey_UpArrow]     = K_up;
    io.KeyMap[ImGuiKey_DownArrow]   = K_down;
    io.KeyMap[ImGuiKey_PageUp]      = K_pageUp;
    io.KeyMap[ImGuiKey_PageDown]    = K_pageUp;
    io.KeyMap[ImGuiKey_Home]        = K_home;
    io.KeyMap[ImGuiKey_End]         = K_end;
    io.KeyMap[ImGuiKey_Delete]      = K_delete;
    io.KeyMap[ImGuiKey_Backspace]   = K_backspace;
    io.KeyMap[ImGuiKey_Enter]       = K_enter;
    io.KeyMap[ImGuiKey_Escape]      = K_esc;
    io.KeyMap[ImGuiKey_A]           = 'A';
    io.KeyMap[ImGuiKey_C]           = 'C';
    io.KeyMap[ImGuiKey_V]           = 'V';
    io.KeyMap[ImGuiKey_X]           = 'X';
    io.KeyMap[ImGuiKey_Y]           = 'Y';
    io.KeyMap[ImGuiKey_Z]           = 'Z';

    // The screen size is set again in onResize
    io.DisplaySize = ImVec2((SLfloat)scrW, (SLfloat)scrH);
    io.DisplayFramebufferScale = ImVec2(1,1);

    // Load different default font
    SLstring fontFilename = SLGLTexture::defaultPathFonts + "DroidSans.ttf";
    if (SLFileSystem::fileExists(fontFilename))
        io.Fonts->AddFontFromFileTTF(fontFilename.c_str(), 16.0f);
    else SL_LOG("\n*** Error ***: \nFont doesn't exist: %s\n\n",
                fontFilename.c_str());

    // Pass C render function to ImGui
    io.RenderDrawListsFn = SLGLImGui::imgui_renderFunction;

    // Provide default build function
    // The build function must be afterwards reassigned
    build = noGuiBuilt;
}
//-----------------------------------------------------------------------------
//! Static C function the that is provided to ImGui::GetIO().RenderDrawListsFn
void SLGLImGui::imgui_renderFunction(ImDrawData* draw_data)
{
    globalInstance->onPaint(draw_data);
}
//-----------------------------------------------------------------------------
//! Creates all OpenGL objects for drawing the imGui
void SLGLImGui::createOpenGLObjects()
{
    // Backup GL state
    GLint last_texture, last_array_buffer, last_vertex_array;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &last_vertex_array);

    /*
    const GLchar *vertex_shader =
        "uniform mat4 ProjMtx;\n"
        "attribute vec2 Position;\n"
        "attribute vec2 UV;\n"
        "attribute vec4 Color;\n"
        "varying vec2 Frag_UV;\n"
        "varying vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "	Frag_UV = UV;\n"
        "	Frag_Color = Color;\n"
        "	gl_Position = ProjMtx * vec4(Position.xy,0,1);\n"
        "}\n";

    const GLchar* fragment_shader =
        "uniform sampler2D Texture;\n"
        "varying vec2 Frag_UV;\n"
        "varying vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "	gl_FragColor = Frag_Color * texture(Texture, Frag_UV.st);\n"
        "}\n";
    */

    const GLchar *vertex_shader =
        "#version 330\n"
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

    const GLchar* fragment_shader =
        "#version 330\n"
        "uniform sampler2D Texture;\n"
        "in vec2 Frag_UV;\n"
        "in vec4 Frag_Color;\n"
        "out vec4 Out_Color;\n"
        "void main()\n"
        "{\n"
        "	Out_Color = Frag_Color * texture( Texture, Frag_UV.st);\n"
        "}\n";
    

    _progHandle = glCreateProgram();
    _vertHandle = glCreateShader(GL_VERTEX_SHADER);
    _fragHandle = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(_vertHandle, 1, &vertex_shader, 0);
    glShaderSource(_fragHandle, 1, &fragment_shader, 0);
    glCompileShader(_vertHandle);
    glCompileShader(_fragHandle);
    glAttachShader(_progHandle, _vertHandle);
    glAttachShader(_progHandle, _fragHandle);
    glLinkProgram(_progHandle);

    GET_GL_ERROR;

    _attribLocTex      = glGetUniformLocation(_progHandle, "Texture");
    _attribLocProjMtx  = glGetUniformLocation(_progHandle, "ProjMtx");
    _attribLocPosition = glGetAttribLocation (_progHandle, "Position");
    _attribLocUV       = glGetAttribLocation (_progHandle, "UV");
    _attribLocColor    = glGetAttribLocation (_progHandle, "Color");

    GET_GL_ERROR;

    glGenBuffers(1, &_vboHandle);
    glGenBuffers(1, &_elementsHandle);

    glGenVertexArrays(1, &_vaoHandle);
    glBindVertexArray(_vaoHandle);
    glBindBuffer(GL_ARRAY_BUFFER, _vboHandle);
    glEnableVertexAttribArray(_attribLocPosition);
    glEnableVertexAttribArray(_attribLocUV);
    glEnableVertexAttribArray(_attribLocColor);

    GET_GL_ERROR;

    #define OFFSETOF(TYPE, ELEMENT) ((size_t)&(((TYPE *)0)->ELEMENT))
    glVertexAttribPointer(_attribLocPosition, 2, GL_FLOAT,         GL_FALSE, sizeof(ImDrawVert), (GLvoid*)OFFSETOF(ImDrawVert, pos));
    glVertexAttribPointer(_attribLocUV,       2, GL_FLOAT,         GL_FALSE, sizeof(ImDrawVert), (GLvoid*)OFFSETOF(ImDrawVert, uv));
    glVertexAttribPointer(_attribLocColor,    4, GL_UNSIGNED_BYTE, GL_TRUE,  sizeof(ImDrawVert), (GLvoid*)OFFSETOF(ImDrawVert, col)); 
    #undef OFFSETOF

    GET_GL_ERROR;

    // Build texture atlas
    ImGuiIO& io = ImGui::GetIO();
    SLuchar* pixels;
    int width, height;

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
    io.Fonts->TexID = (void *)(intptr_t)_fontTexture;

    // Restore state
    glBindTexture(GL_TEXTURE_2D, last_texture);

    // Restore modified GL state
    glBindTexture(GL_TEXTURE_2D, last_texture);
    glBindBuffer(GL_ARRAY_BUFFER, last_array_buffer);
    glBindVertexArray(last_vertex_array);

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Deletes all OpenGL objects for drawing the imGui
void SLGLImGui::deleteOpenGLObjects()
{
    if (_vaoHandle) glDeleteVertexArrays(1, &_vaoHandle);
    if (_vboHandle) glDeleteBuffers(1, &_vboHandle);
    if (_elementsHandle) glDeleteBuffers(1, &_elementsHandle);
    _vaoHandle = _vboHandle = _elementsHandle = 0;

    if (_progHandle && _vertHandle) glDetachShader(_progHandle, _vertHandle);
    if (_vertHandle) glDeleteShader(_vertHandle);
    _vertHandle = 0;

    if (_progHandle && _fragHandle) glDetachShader(_progHandle, _fragHandle);
    if (_fragHandle) glDeleteShader(_fragHandle);
    _fragHandle = 0;

    if (_progHandle) glDeleteProgram(_progHandle);
    _progHandle = 0;

    if (_fontTexture)
    {
        glDeleteTextures(1, &_fontTexture);
        ImGui::GetIO().Fonts->TexID = 0;
        _fontTexture = 0;
    }
}
//-----------------------------------------------------------------------------
//! Inits a new frame for the ImGui system
void SLGLImGui::onInitNewFrame()
{
    if (!_fontTexture)
        createOpenGLObjects();

    ImGuiIO& io = ImGui::GetIO();
    
    // Setup time step
    SLfloat nowSec =  SLScene::current->timeSec();
    io.DeltaTime = _timeSec > 0.0 ? nowSec-_timeSec : 1.0f/60.0f;
    if (io.DeltaTime < 0) io.DeltaTime = 1.0f/60.0f;
    _timeSec = nowSec;

    // Start the frame
    ImGui::NewFrame();
}
//-----------------------------------------------------------------------------
//! Callback if window got resized
void SLGLImGui::onResize(SLint scrW, SLint scrH)
{
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((SLfloat)scrW, (SLfloat)scrH);
}
//-----------------------------------------------------------------------------
//! Callback for main rendering for the ImGui GUI system
void SLGLImGui::onPaint(ImDrawData* draw_data)
{
    ImGuiIO& io = ImGui::GetIO();

    // Avoid rendering when minimized, scale coordinates for retina displays 
    // (screen coordinates != framebuffer coordinates)
    int fb_width  = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
    int fb_height = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
    if (fb_width == 0 || fb_height == 0)
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

    GLboolean last_enable_blend         = glIsEnabled(GL_BLEND);
    GLboolean last_enable_cull_face     = glIsEnabled(GL_CULL_FACE);
    GLboolean last_enable_depth_test    = glIsEnabled(GL_DEPTH_TEST);
    GLboolean last_enable_scissor_test  = glIsEnabled(GL_SCISSOR_TEST);

    // Setup render state: alpha-blending enabled, no face culling, no depth testing, scissor enabled
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);

    // Setup viewport, orthographic projection matrix
    glViewport(0, 0, (GLsizei)fb_width, (GLsizei)fb_height);
    const float ortho_projection[4][4] =
    {   { 2.0f/io.DisplaySize.x, 0.0f,                   0.0f, 0.0f },
        { 0.0f,                  2.0f/-io.DisplaySize.y, 0.0f, 0.0f },
        { 0.0f,                  0.0f,                  -1.0f, 0.0f },
        {-1.0f,                  1.0f,                   0.0f, 1.0f },
    };

    glUseProgram(_progHandle);
    glUniform1i(_attribLocTex, 0);
    glUniformMatrix4fv(_attribLocProjMtx, 1, GL_FALSE, &ortho_projection[0][0]);
    glBindVertexArray(_vaoHandle);

    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        const ImDrawIdx* idx_buffer_offset = 0;

        glBindBuffer(GL_ARRAY_BUFFER, _vboHandle);
        glBufferData(GL_ARRAY_BUFFER, 
                     (GLsizeiptr)cmd_list->VtxBuffer.Size * sizeof(ImDrawVert), 
                     (const GLvoid*)cmd_list->VtxBuffer.Data, GL_STREAM_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _elementsHandle);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                     (GLsizeiptr)cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx), 
                     (const GLvoid*)cmd_list->IdxBuffer.Data, GL_STREAM_DRAW);

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
                glScissor((int)pcmd->ClipRect.x, 
                          (int)(fb_height - pcmd->ClipRect.w), 
                          (int)(pcmd->ClipRect.z - pcmd->ClipRect.x), 
                          (int)(pcmd->ClipRect.w - pcmd->ClipRect.y));
                glDrawElements(GL_TRIANGLES, 
                               (GLsizei)pcmd->ElemCount, 
                               sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, 
                               idx_buffer_offset);
            }
            idx_buffer_offset += pcmd->ElemCount;
        }
    }

    // Restore modified GL state
    glUseProgram(last_program);
    glBindTexture(GL_TEXTURE_2D, last_texture);
    glActiveTexture(last_active_texture);
    glBindVertexArray(last_vertex_array);
    glBindBuffer(GL_ARRAY_BUFFER, last_array_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, last_element_array_buffer);
    glBlendEquationSeparate(last_blend_equation_rgb, last_blend_equation_alpha);
    glBlendFuncSeparate(last_blend_src_rgb, 
                        last_blend_dst_rgb, 
                        last_blend_src_alpha, 
                        last_blend_dst_alpha);
    if (last_enable_blend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (last_enable_cull_face) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (last_enable_depth_test) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (last_enable_scissor_test) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    glViewport(last_viewport[0], 
               last_viewport[1], 
               (GLsizei)last_viewport[2], 
               (GLsizei)last_viewport[3]);
    glScissor(last_scissor_box[0], 
              last_scissor_box[1], 
              (GLsizei)last_scissor_box[2], 
              (GLsizei)last_scissor_box[3]);
}
//-----------------------------------------------------------------------------
//! Updates the mouse cursor position
void SLGLImGui::onMouseMove(SLint xPos, SLint yPos)
{
    ImGui::GetIO().MousePos = ImVec2((SLfloat)xPos, (SLfloat)yPos);
}
//-----------------------------------------------------------------------------
//! Callback for the mouse scroll movement
void SLGLImGui::onMouseWheel(SLfloat yoffset)
{
    // Use fractional mouse wheel, 1.0 unit 5 lines.
    _mouseWheel += yoffset; 
}
//-----------------------------------------------------------------------------
//! Callback on mouse button down event
void SLGLImGui::onMouseDown(SLMouseButton button)
{
    ImGuiIO& io = ImGui::GetIO();
    if (button == MB_left)   io.MouseDown[0] = true;
    if (button == MB_middle) io.MouseDown[1] = true;
    if (button == MB_right)  io.MouseDown[2] = true;
}
//-----------------------------------------------------------------------------
//! Callback on mouse button up event
void SLGLImGui::onMouseUp(SLMouseButton button)
{
    ImGuiIO& io = ImGui::GetIO();
    if (button == MB_left)   io.MouseDown[0] = false;
    if (button == MB_middle) io.MouseDown[1] = false;
    if (button == MB_right)  io.MouseDown[2] = false;
}
//-----------------------------------------------------------------------------
//! Callback on key press event
void SLGLImGui::onKeyPress(SLKey key, SLKey mod)
{
    ImGuiIO& io = ImGui::GetIO();
    io.KeysDown[key] = true;
    io.KeyCtrl  = mod & K_ctrl ? true : false;
    io.KeyShift = mod & K_shift ? true : false;
    io.KeyAlt   = mod & K_alt ? true : false;
}
//-----------------------------------------------------------------------------
//! Callback on key release event
void SLGLImGui::onKeyRelease(SLKey key, SLKey mod)
{
    ImGuiIO& io = ImGui::GetIO();
    io.KeysDown[key] = false;
    io.KeyCtrl  = mod & K_ctrl ? true : false;
    io.KeyShift = mod & K_shift ? true : false;
    io.KeyAlt   = mod & K_alt ? true : false;
}
//-----------------------------------------------------------------------------
//! Callback on closing the application
void SLGLImGui::onClose()
{
    deleteOpenGLObjects();
    ImGui::Shutdown();
}
//-----------------------------------------------------------------------------
