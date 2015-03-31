//#############################################################################
//  File:      math/SLVec2.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLVEC2_H
#define SLVEC2_H

#include <SL.h>

//-----------------------------------------------------------------------------
//! 2D vector template class for standard 2D vector algebra.
/*!
2D vector template class and type definitions for 2D vectors;
\n Use SLVec2f for a specific float type vector
\n Use SLVec2d for a specific double type vector
\n Use SLVec2r for a precision independent real type.
*/
template<class T> 
class SLVec2
{
    public:
            union
            {  struct {T x, y;};
               struct {T comp[2];};
            };

            SLVec2      ()                      {}
            SLVec2      (const T X, 
                         const T Y)             {x=X;y=Y;}
            SLVec2      (const T v[2])          {x=v[0]; y=v[1];}
            SLVec2      (const SLVec2& v)       {x=v.x; y=v.y;}

    void     set        (const T X,
                         const T Y)             {x=X; y=Y;}
    void     set        (const T v[2])          {x=v[0]; y=v[1];}
    void     set        (const SLVec2& v)       {x=v.x; y=v.y;}

    // Component wise compare
    inline  SLint    operator == (const SLVec2& v) const {return (x==v.x && y==v.y);}
    inline  SLint    operator != (const SLVec2& v) const {return (x!=v.x || y!=v.y);}
    inline  SLint    operator <= (const SLVec2& v) const {return (x<=v.x && y<=v.y);}
    inline  SLint    operator >= (const SLVec2& v) const {return (x>=v.x && y>=v.y);}

    // Operators with temp. allocation
    inline  SLVec2   operator +  (const SLVec2& v) const {return SLVec2(x+v.x, y+v.y);}
    inline  SLVec2   operator -  (const SLVec2& v) const {return SLVec2(x-v.x, y-v.y);}
    inline  SLVec2   operator -  () const                {return SLVec2(-x, -y);}
    inline  T        operator *  (const SLVec2& v) const {return x*v.x+y*v.y;};     //dot
    inline  SLVec2   operator *  (const T s) const       {return SLVec2(x*s, y*s);}
    inline  SLVec2   operator /  (const T s) const       {return SLVec2(x/s, y/s);}
    inline  SLVec2   operator &  (const SLVec2& v) const {return SLVec2(x*v.x, y*v.y);}
    friend inline 
            SLVec2   operator *  (T s, const SLVec2& v)  {return SLVec2(v.x*s, v.y*s);}

    // Assign operators
    SLVec2&  operator =  (const SLVec2& v)       {x=v.x; y=v.y;   return *this;}
    SLVec2&  operator += (const SLVec2& v)       {x+=v.x; y+=v.y; return *this;}
    SLVec2&  operator -= (const SLVec2& v)       {x-=v.x; y-=v.y; return *this;}
    SLVec2&  operator *= (const T s)             {x*=s; y*=s;     return *this;}
    SLVec2&  operator /= (const T s)             {x/=s; y/=s;     return *this;}
    SLVec2&  operator &= (const SLVec2& v)       {x*=v.x; y*=v.y; return *this;}
   
    // Stream output operator
    friend ostream& operator << (ostream& output, 
                                const SLVec2& v){output<<"["<<v.x<<","<<v.y<<"]"; return output;}

    // Operations without temp. allocation
    void     add         (const SLVec2& a,
                            const SLVec2& b)     {x=a.x+b.x; y=a.y+b.y;}
    void     sub         (const SLVec2& a,
                            const SLVec2& b)     {x=a.x-b.x; y=a.y-b.y;}
    void     scale       (const T s)             {x*=s; y*=s;}
    T        dot         (const SLVec2& v)       {return x*v.x+y*v.y;};
    T        length      () const                {return ((T)sqrt(x*x+y*y));}
    T        lengthSqr   () const                {return (x*x+y*y); }
    SLVec2&  normalize   ()                      {T L=length(); 
                                                  if (L>0) {x/=L; y/=L;} 
                                                  return *this;}  
    void     clampMinMax (const T min, 
                            const T max)         {x = (x>max)?max : (x<min)?min : x;
                                                  y = (y>max)?max : (y<min)?min : y;}
   
    //! Calculates polar coords with radius & angle phi in radians (-pi < phi < pi)
    void     toPolar     (T& r, T& phiRAD)       {r = length();
                                                  phiRAD = atan2(y,x);}                                              
   
    //! Calculates the vector from polar coords r & phi in radians (-pi < phi < pi)
    void     fromPolar   (T r, T phiRAD)         {x = r * cos(phiRAD);
                                                  y = r * sin(phiRAD);}
   
    //! Calculate the absolute to the vector v
    T        diff        (const SLVec2& v)       {return SL_abs(x-v.x) + 
                                                        SL_abs(y-v.y);}  
    void     setMin      (const SLVec2& v)       {if (v.x < x) x=v.x;
                                                  if (v.y < y) y=v.y;}
    void     setMax      (const SLVec2& v)       {if (v.x > x) x=v.x;
                                                  if (v.y > y) y=v.y;}
    void     print       (const char* str=0)     {if (str) SL_LOG("%s",str); 
                                                  SL_LOG("% 3.3f, % 3.3f\n",x, y);}
   
    static 
    SLVec2   ZERO;
};
//-----------------------------------------------------------------------------
template<class T> SLVec2<T> SLVec2<T>::ZERO = SLVec2<T>(0,0);
//-----------------------------------------------------------------------------
typedef SLVec2<SLint>         SLVec2i;
typedef SLVec2<SLfloat>       SLVec2f;

typedef std::vector<SLVec2f>  SLVVec2f;

#ifdef SL_HAS_DOUBLE
typedef SLVec2<SLdouble>      SLVec2d;
typedef std::vector<SLVec2d>  SLVVec2d;
#endif
//-----------------------------------------------------------------------------
#endif
