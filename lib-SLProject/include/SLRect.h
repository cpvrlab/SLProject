//#############################################################################
//  File:      SLRect.h
//  Author:    Marcus Hudritsch
//  Date:      August 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLRECT_H
#define SLRECT_H

#include <SL.h>
#include <SLGLVertexArrayExt.h>
#include <SLVec2.h>
#include <SLVec3.h>

//-----------------------------------------------------------------------------
//! A rectangle template class
/*! Defines a rectangle with a top-left corner at x,y measured from top-left
of the window and with its width and height. It is used e.g. to draw a
selection rectangle in SLSceneView::draw2DGL.
*/
// clang-format off
template<class T, class V>
class SLRect
{
    public:
            T x, y, width, height;

            SLRect      ()                      {setZero();}
            SLRect      (const T WIDTH,
                         const T HEIGHT)        {x=0;y=0;width=WIDTH;height=HEIGHT;}
            SLRect      (const T X,
                         const T Y,
                         const T WIDTH,
                         const T HEIGHT)        {x=X;y=Y;width=WIDTH;height=HEIGHT;}
            SLRect      (const V& tl,
                         const V& br)           {x=tl.x; y=tl.y;
                                                 width=br.x-tl.x;
                                                 height=br.y-tl.y;}
    void     set        (const T X,
                         const T Y,
                         const T WIDTH,
                         const T HEIGHT)        {x=X;y=Y;width=WIDTH;height=HEIGHT;}
    void     set        (const T v[2])          {x=v[0]; y=v[1];}
    void     set        (const V tl,
                         const V br)            {tl(tl); br(br);}
    void     setZero    ()                      {x=0; y=0; width=0; height=0;}

    // Component wise compare
    SLbool  operator == (const SLRect& r) const {return (x==r.x && y==r.y && width==r.width && height==r.height);}
    SLbool  operator != (const SLRect& r) const {return (x!=r.x || y!=r.y || width!=r.width || height!=r.height);}

    // Assign operators
    SLRect&  operator =  (const SLRect& r)       {x=r.x; y=r.y; width=r.width; height=r.height; return *this;}
   
    // Stream output operator
    friend ostream& operator << (ostream& output, 
                                const SLRect& r){output<<"["<<r.x<<","<<r.y<<","<<r.width<<","<<r.height<<"]"; return output;}

    // Misc. setters
    void    tl          (V v)                   {x = v.x; y = v.y;}              //!< top-left corner
    void    br          (V v)                   {if (v.x < x) x = v.x;
                                                 if (v.y < y) y = v.y;
                                                 width = v.x-x; height = v.y-y;} //!< bottom-right corner
    void    setScnd     (V v)                   {if (v.x>x) {width =v.x-x;} else {width+=x-v.x; x=v.x;}
                                                 if (v.y>y) {height=v.y-y;} else {height+=y-v.y; y=v.y;}}

    // Misc. getters
    V       tl          ()                      {return V(x,y);}              //!< top-left corner
    V       br          ()                      {return V(x+width,y+height);} //!< bottom-right corner
    T       area        ()                      {return width * height;}
    SLbool  isEmpty     () const                     {return (width==0 || height==0);} ;
    SLbool  isZero      ()                      {return (x==0 && y==0 && width==0 && height==0);}
    SLbool  contains    (T X, T Y)              {return (X>=x && X<=x+width && Y>=y && Y<=y+height);}
    SLbool  contains    (V v)                   {return (v.x>=x && v.x<=x+width && v.y>=y && v.y<=y+height);}
    SLbool  contains    (const SLRect& r)       {return (r.x>=x && r.x<=x+width && r.y>=y && r.y<=y+height && r.width <= width && r.height <= height);}

    void    print       (const char* str=nullptr) 
    {
        if (str) 
            SL_LOG("%s",str);
        SL_LOG("% 3.3f, % 3.3f, % 3.3f, % 3.3f",x, y, width, height);
    }

    // Draw rectangle with OpenGL VAO
    void drawGL(const SLCol4f& color)
    {
        SLVVec3f P;
        P.push_back(SLVec3f(x,      -y,        0));
        P.push_back(SLVec3f(x+width,-y,        0));
        P.push_back(SLVec3f(x+width,-y-height, 0));
        P.push_back(SLVec3f(x      ,-y-height, 0));
        _vao.generateVertexPos(&P);
        _vao.drawArrayAsColored(PT_lineLoop, color);
    }

    private:
        SLGLVertexArrayExt _vao;

};
//-----------------------------------------------------------------------------
typedef SLRect<SLint, SLVec2i>      SLRecti;
typedef SLRect<SLfloat, SLVec2f>    SLRectf;
typedef std::vector<SLRecti>        SLVRecti;
typedef std::vector<SLRectf>        SLVRectf;
//-----------------------------------------------------------------------------
#endif
