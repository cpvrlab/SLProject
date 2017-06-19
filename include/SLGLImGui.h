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

#include <stdafx.h>
#include <imgui.h>

#ifndef SLGLIMGUI_H
#define SLGLIMGUI_H

class SLScene;
class SLSceneView;

//-----------------------------------------------------------------------------
class SLGLImGui
{
    public:
                    SLGLImGui               (){;}

        void        init                    (SLint scrW=0, 
                                             SLint scrH=0, 
                                             SLint fbW=0, 
                                             SLint fbH=0);
        void        createOpenGLObjects     ();
        void        deleteOpenGLObjects     ();

        void        onInitNewFrame          ();
        void        onResize                (SLint scrW, SLint scrH);
        void        onPaint                 (ImDrawData* draw_data);
        void        onMouseDown             (SLMouseButton button);  
        void        onMouseUp               (SLMouseButton button); 
        void        onMouseMove             (SLint xPos, SLint yPos);
        void        onMouseWheel            (SLfloat yoffset);
        void        onKeyPress              (SLKey key, SLKey mod);
        void        onKeyRelease            (SLKey key, SLKey mod);
        void        onClose                 ();

        // Static global instance for render callback
        static SLGLImGui* globalInstance; 

        // Static C-function for render callback
        static void imgui_renderFunction    (ImDrawData* draw_data);
        
        // gui build function
        void        (*build)                (SLScene* s, SLSceneView* sv);   

    private:
        SLfloat     _timeSec;               //!< Time in seconds
        SLVec2f     _mousePosPX;            //!< Mouse cursor position
        SLfloat     _mouseWheel;            //!< Mouse wheel position
        SLbool      _mousePressed[3];       //!< Mouse button press state
        SLuint      _fontTexture;           //!< OpenGL texture id for font
        SLint       _progHandle;            //!< OpenGL handle for shader program
        SLint       _vertHandle;            //!< OpenGL handle for vertex shader
        SLint       _fragHandle;            //!< OpenGL handle for fragment shader
        SLint       _attribLocTex;          //!< OpenGL attribute location for texture
        SLint       _attribLocProjMtx;      //!< OpenGL attribute location for ???
        SLint       _attribLocPosition;     //!< OpenGL attribute location for vertex pos.
        SLint       _attribLocUV;           //!< OpenGL attribute location for texture coords
        SLint       _attribLocColor;        //!< OpenGL attribute location for color
        SLuint      _vboHandle;             //!< OpenGL handle for vertex buffer object
        SLuint      _vaoHandle;             //!< OpenGL vertex array object handle
        SLuint      _elementsHandle;        //!< OpenGL handle for vertex indexes
};
//-----------------------------------------------------------------------------
#endif
