//#############################################################################
//  File:      math/Math/SLVec4.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLVEC4_H
#define SLVEC4_H

#include <SL.h>
#include <SLVec3.h>
#include <SLUtils.h>

//-----------------------------------------------------------------------------
//! 4D vector template class for standard 4D vector algebra.
/*!
4D vector template class with type definitions for 4D vectors and colors:
\n Use SLVec4f for a specific float type vector
\n Use SLVec4d for a specific double type vector
\n Use SLVec4r for a precision independent real type.
*/
template <class T>
class SLVec4
{
    public:
        union
        {   struct {T x, y, z, w;};
            struct {T r, g, b, a;};
            struct {T comp[4];};
        };

            SLVec4      (void)                  {}
            SLVec4      (const T X, 
                            const T Y, 
                            const T Z=0, 
                            const T W=1)           {x=X; y=Y; z=Z; 
                                                    w=W;}
            SLVec4      (const T v[4])          {x=v[0]; y=v[1]; z=v[2]; w=v[3];}
            SLVec4      (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;    w=1;}
            SLVec4      (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=1;}
            SLVec4      (const SLVec4<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=v.w;}
            SLVec4      (const SLstring fourFloatsWithDelimiter) {fromString(fourFloatsWithDelimiter);}

    void     set         (const T X, 
                            const T Y, 
                            const T Z, 
                            const T W=1)           {x=X;    y=Y;    z=Z;    w=W;}
    void     set         (const T xyz)           {x=xyz;  y=xyz;  z=xyz;  w=1;}
    void     set         (const T v[4])          {x=v[0]; y=v[1]; z=v[2]; w=v[3];}
    void     set         (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;    w=1;}
    void     set         (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=1;}
    void     set         (const SLVec4<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=v.w;}

    SL_INLINE SLint    operator == (const SLVec4& v) const {return (x==v.x && y==v.y && z==v.z && w==v.w);}
    SL_INLINE SLint    operator != (const SLVec4& v) const {return (x!=v.x || y!=v.y || z!=v.z || w!=v.w);}
    SL_INLINE SLint    operator <= (const SLVec4& v) const {return (x<=v.x && y<=v.y && z<=v.z && w<=v.w);}
    SL_INLINE SLint    operator >= (const SLVec4& v) const {return (x>=v.x && y>=v.y && z>=v.z && w>=v.w);}
   
   
    // Operators with temp. allocation
    SL_INLINE SLVec4   operator +  (const SLVec4& v) const {return SLVec4(x+v.x, y+v.y, z+v.z, w+v.w);}
    SL_INLINE SLVec4   operator -  (const SLVec4& v) const {return SLVec4(x-v.x, y-v.y, z-v.z, w-v.w);}
    SL_INLINE SLVec4   operator -  (void) const            {return SLVec4(-x, -y, -z, -w);}
    SL_INLINE T        operator *  (const SLVec4& v) const {return x*v.x+y*v.y+z*v.z+w*v.w;};
    SL_INLINE SLVec4   operator *  (const T s) const       {return SLVec4(x*s, y*s, z*s);}
    SL_INLINE SLVec4   operator /  (const T s) const       {return SLVec4(x/s, y/s, z/s, w/s);}
    SL_INLINE SLVec4   operator &  (const SLVec4& v) const {return SLVec4(x*v.x, y*v.y, z*v.z, w*v.w);}
    friend inline 
    SLVec4   operator *  (T s, const SLVec4& v)  {return SLVec4(v.x*s, v.y*s, v.z*s);}
   
    // Stream output operator
    friend ostream& operator << (ostream& output, 
                                const SLVec4& v){output<<"["<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"]"; return output;}

    // Assign operators
    SLVec4&  operator =  (const SLVec2<T>& v)    {x=v.x; y=v.y; z=0; w=1;         return *this;}
    SLVec4&  operator =  (const SLVec3<T>& v)    {x=v.x; y=v.y; z=v.z; w=1;       return *this;}
    SLVec4&  operator =  (const SLVec4& v)       {x=v.x; y=v.y; z=v.z; w=v.w;     return *this;}
    SLVec4&  operator += (const SLVec4& v)       {x+=v.x; y+=v.y; z+=v.z; w+=v.w; return *this;}
    SLVec4&  operator += (const SLVec3<T>& v)    {x+=v.x; y+=v.y; z+=v.z; w+=0;   return *this;}
    SLVec4&  operator -= (const SLVec4& v)       {x-=v.x; y-=v.y; z-=v.z; w-=v.w; return *this;}
    SLVec4&  operator -= (const SLVec3<T>& v)    {x-=v.x; y-=v.y; z-=v.z; w-=0;   return *this;}
    SLVec4&  operator *= (const T s)             {x*=s; y*=s; z*=s; w*=s;         return *this;}
    SLVec4&  operator /= (const T s)             {x/=s; y/=s; z/=s; w/=s;         return *this;}
    SLVec4&  operator &= (const SLVec4& v)       {x*=v.x; y*=v.y; z*=v.z; w*=v.w; return *this;}
    SLVec4&  operator &= (const SLVec3<T>& v)    {x*=v.x; y*=v.y; z*=v.z; w*=1;   return *this;}
   
    // Operations without temp. allocation
    SL_INLINE void     add         (const SLVec4& a,
                                    const SLVec4& b)       {x=a.x+b.x; y=a.y+b.y, z=a.z+b.z; w=a.w+b.w;}
    SL_INLINE void     sub         (const SLVec4& a,
                                    const SLVec4& b)       {x=a.x-b.x; y=a.y-b.y, z=a.z-b.z; w=a.w-b.w;}
    SL_INLINE void     scale       (const T s)             {x*=s; y*=s; z*=s; w*=s;}
    SL_INLINE T        dot         (const SLVec4& v)       {return x*v.x+y*v.y+z*v.z+w*v.w;};
    SL_INLINE void     cross       (const SLVec4& a,
                                    const SLVec4& b)       {x = a.y*b.z - a.z*b.y; 
                                                            y = a.z*b.x - a.x*b.z; 
                                                            z = a.x*b.y - a.y*b.x;
                                                            w = 1;}
    SL_INLINE T        length      () const                {return (T)sqrt(x*x+y*y+z*z+w*w);}
    SL_INLINE T        lengthSqr   () const                {return (x*x+y*y+z*z+w*w);}
    SL_INLINE SLVec4&  normalize   ()                      {T L = length(); 
                                                            if (L>0){x/=L; y/=L; z/=L; w/=L;} 
                                                            return *this;}
    SL_INLINE void     wdiv        ()                      {x/=w; y/=w; z/=w; w=1;} 
    SL_INLINE void     clampMinMax (const T min, 
                                    const T max)           {x = (x>max)?max : (x<min)?min : x;
                                                            y = (y>max)?max : (y<min)?min : y;
                                                            z = (z>max)?max : (z<min)?min : z;
                                                            w = 1;}
    SL_INLINE T        diff        (const SLVec4& v)       {return SL_abs(x-v.x) + 
                                                                    SL_abs(y-v.y) + 
                                                                    SL_abs(z-v.z) + 
                                                                    SL_abs(w-v.w);}
    SL_INLINE T        diffRGB     (const SLVec4& v)       {return SL_abs(x-v.x) + 
                                                                    SL_abs(y-v.y) + 
                                                                    SL_abs(z-v.z);}
    SL_INLINE void     mix         (const SLVec4& a,
                                    const SLVec4& b,
                                    const T factor_b)      {T factor_a = 1-factor_b;
                                                            x = a.x*factor_a + b.x*factor_b;
                                                            y = a.y*factor_a + b.y*factor_b;
                                                            z = a.z*factor_a + b.z*factor_b; 
                                                            w = a.w*factor_a + b.w*factor_b;}  
    SL_INLINE T        maxXYZ      ()                      {if (x>=y && x>=z)   return x;
                                                            else if (y>=z)      return y;
                                                            else                return z;}
            //! Gamma correction
            void      gamma       (T gammaVal)            {x= pow(x,1.0/gammaVal);
                                                            y= pow(y,1.0/gammaVal);
                                                            z= pow(z,1.0/gammaVal);}

            void      print       (const SLchar* str=0)   {if (str) SL_LOG("%s\n",str); 
                                    SL_LOG("% 3.3f, % 3.3f, % 3.3f, % 3.3f\n",x, y, z, w);}
            
            //! Conversion to string
            SLstring  toString(SLstring delimiter=", ")
                        {  return SLUtils::toString(x) + delimiter +
                                SLUtils::toString(y) + delimiter +
                                SLUtils::toString(z) + delimiter +
                                SLUtils::toString(w);
                        }
            
            //! Conversion from string
            void      fromString(SLstring fourFloatsWithDelimiter, SLchar delimiter=',')
                        {  SLVstring comp;
                            SLUtils::split(fourFloatsWithDelimiter, delimiter, comp);
                            float f[4] = {0.0, 0.0f, 0.0f, 1.0f};
                            for (SLint i=0; i<comp.size(); ++i)
                            f[i] = (SLfloat)atof(comp[i].c_str());
                            x = f[0]; y = f[1]; z = f[2]; w = f[3];
                        }

    static SLVec4 ZERO;
    static SLVec4 BLACK;
    static SLVec4 GRAY;
    static SLVec4 WHITE;
    static SLVec4 RED;
    static SLVec4 GREEN;
    static SLVec4 BLUE;
    static SLVec4 YELLOW;
    static SLVec4 CYAN;
    static SLVec4 MAGENTA;
};
//-----------------------------------------------------------------------------
template<class T> SLVec4<T> SLVec4<T>::ZERO   = SLVec4<T>(0.0f, 0.0f, 0.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::BLACK  = SLVec4<T>(0.0f, 0.0f, 0.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::GRAY   = SLVec4<T>(0.5f, 0.5f, 0.5f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::WHITE  = SLVec4<T>(1.0f, 1.0f, 1.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::RED    = SLVec4<T>(1.0f, 0.0f, 0.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::GREEN  = SLVec4<T>(0.0f, 1.0f, 0.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::BLUE   = SLVec4<T>(0.0f, 0.0f, 1.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::YELLOW = SLVec4<T>(1.0f, 1.0f, 0.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::CYAN   = SLVec4<T>(0.0f, 1.0f, 1.0f, 1.0f);
template<class T> SLVec4<T> SLVec4<T>::MAGENTA= SLVec4<T>(1.0f, 0.0f, 1.0f, 1.0f);
//-----------------------------------------------------------------------------
typedef SLVec4<SLfloat>       SLVec4f;
typedef SLVec4<SLfloat>       SLCol4f;

typedef std::vector<SLVec4f>  SLVVec4f;
typedef std::vector<SLCol4f>  SLVCol4f;

#ifdef SL_HAS_DOUBLE
typedef SLVec4<SLdouble>      SLVec4d;
typedef std::vector<SLVec4d>  SLVVec4d;
#endif
//-----------------------------------------------------------------------------
#endif


