//#############################################################################
//  File:      math/Math/SLVec4.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
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
                                 const T W=1)           {x=X;    y=Y;    z=Z;    w=W;}
                    SLVec4      (const T v[4])          {x=v[0]; y=v[1]; z=v[2]; w=v[3];}
                    SLVec4      (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;    w=1;}
                    SLVec4      (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=1;}
                    SLVec4      (const SLVec4<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=v.w;}
                    SLVec4      (const SLstring fourFloatsWithDelimiter) {fromString(fourFloatsWithDelimiter);}

            void    set         (const T X,
                                 const T Y,
                                 const T Z,
                                 const T W=1)           {x=X;    y=Y;    z=Z;    w=W;}
            void    set         (const T xyz)           {x=xyz;  y=xyz;  z=xyz;  w=1;}
            void    set         (const T v[4])          {x=v[0]; y=v[1]; z=v[2]; w=v[3];}
            void    set         (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;    w=1;}
            void    set         (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=1;}
            void    set         (const SLVec4<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=v.w;}

    inline SLint    operator == (const SLVec4& v) const {return (x==v.x && y==v.y && z==v.z && w==v.w);}
    inline SLint    operator != (const SLVec4& v) const {return (x!=v.x || y!=v.y || z!=v.z || w!=v.w);}
    inline SLint    operator <= (const SLVec4& v) const {return (x<=v.x && y<=v.y && z<=v.z && w<=v.w);}
    inline SLint    operator >= (const SLVec4& v) const {return (x>=v.x && y>=v.y && z>=v.z && w>=v.w);}
   
   
    // Operators with temp. allocation
    inline SLVec4   operator +  (const SLVec4& v) const {return SLVec4(x+v.x, y+v.y, z+v.z, w+v.w);}
    inline SLVec4   operator -  (const SLVec4& v) const {return SLVec4(x-v.x, y-v.y, z-v.z, w-v.w);}
    inline SLVec4   operator -  (void) const            {return SLVec4(-x, -y, -z, -w);}
    inline T        operator *  (const SLVec4& v) const {return x*v.x+y*v.y+z*v.z+w*v.w;};
    inline SLVec4   operator *  (const T s) const       {return SLVec4(x*s, y*s, z*s);}
    inline SLVec4   operator /  (const T s) const       {return SLVec4(x/s, y/s, z/s, w/s);}
    inline SLVec4   operator &  (const SLVec4& v) const {return SLVec4(x*v.x, y*v.y, z*v.z, w*v.w);}
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
    inline void     add         (const SLVec4& a,
                                 const SLVec4& b)       {x=a.x+b.x; y=a.y+b.y, z=a.z+b.z; w=a.w+b.w;}
    inline void     sub         (const SLVec4& a,
                                 const SLVec4& b)       {x=a.x-b.x; y=a.y-b.y, z=a.z-b.z; w=a.w-b.w;}
    inline void     scale       (const T s)             {x*=s; y*=s; z*=s; w*=s;}
    inline T        dot         (const SLVec4& v)       {return x*v.x+y*v.y+z*v.z+w*v.w;};
    inline void     cross       (const SLVec4& a,
                                 const SLVec4& b)       {x = a.y*b.z - a.z*b.y;
                                                         y = a.z*b.x - a.x*b.z;
                                                         z = a.x*b.y - a.y*b.x;
                                                         w = 1;}
    inline T        length      () const                {return (T)sqrt(x*x+y*y+z*z+w*w);}
    inline T        lengthSqr   () const                {return (x*x+y*y+z*z+w*w);}
    inline SLVec4&  normalize   ()                      {T L = length(); 
                                                            if (L>0){x/=L; y/=L; z/=L; w/=L;} 
                                                            return *this;}
    inline void     wdiv        ()                      {x/=w; y/=w; z/=w; w=1;} 
    inline void     clampMinMax (const T min, 
                                 const T max)           {x = (x>max)?max : (x<min)?min : x;
                                                         y = (y>max)?max : (y<min)?min : y;
                                                         z = (z>max)?max : (z<min)?min : z;
                                                         w = 1;}
    inline T        diff        (const SLVec4& v)       {return SL_abs(x-v.x) + 
                                                                    SL_abs(y-v.y) + 
                                                                    SL_abs(z-v.z) + 
                                                                    SL_abs(w-v.w);}
    inline T        diffRGB     (const SLVec4& v)       {return SL_abs(x-v.x) + 
                                                                    SL_abs(y-v.y) + 
                                                                    SL_abs(z-v.z);}
    inline void     mix         (const SLVec4& a,
                                 const SLVec4& b,
                                 const T factor_b)      {T factor_a = 1-factor_b;
                                                         x = a.x*factor_a + b.x*factor_b;
                                                         y = a.y*factor_a + b.y*factor_b;
                                                         z = a.z*factor_a + b.z*factor_b;
                                                         w = a.w*factor_a + b.w*factor_b;}
    inline T        maxXYZ      ()                      {if (x>=y && x>=z)   return x;
                                                         else if (y>=z)      return y;
                                                         else                return z;}
            //! Gamma correction
            void    gamma       (T gammaVal)            {x= pow(x,1.0/gammaVal);
                                                         y= pow(y,1.0/gammaVal);
                                                         z= pow(z,1.0/gammaVal);}

            void    print       (const SLchar* str=0)
            {   if (str) SL_LOG("%s\n",str);
                SL_LOG("% 3.3f, % 3.3f, % 3.3f, % 3.3f\n",x, y, z, w);
            }

            //! Conversion to string
            SLstring toString   (SLstring delimiter=", ")
             {   return SLUtils::toString(x) + delimiter +
                        SLUtils::toString(y) + delimiter +
                        SLUtils::toString(z) + delimiter +
                        SLUtils::toString(w);
             }

            //! Conversion from string
            void    fromString  (SLstring fourFloatsWithDelimiter, SLchar delimiter=',')
            {   SLVstring components;
                SLUtils::split(fourFloatsWithDelimiter, delimiter, components);
                float f[4] = {0.0, 0.0f, 0.0f, 1.0f};
                for (SLint i=0; i<components.size(); ++i)
                    f[i] = (SLfloat)atof(components[i].c_str());
                x = f[0]; y = f[1]; z = f[2]; w = f[3];
            }

            //! HSVA to RGBA color conversion (http://www.rapidtables.com/convert/color/hsv-to-rgb.htm)
            void    hsva2rgba   (const SLVec4 &hsva)
            {
                T h = fmod(fmod(hsva.x, SL_2PI) + SL_2PI, SL_2PI); // 0° <= H <= 360°
                T s = SL_clamp(hsva.y, 0.0f, 1.0f);
                T v = SL_clamp(hsva.z, 0.0f, 1.0f);
                T a = SL_clamp(hsva.w, 0.0f, 1.0f);

                T c = v * s;
                T x = c * (1.0f - fabs(fmod(h*3.0f / M_PI, 2.0f) - 1.0f));
                T m = v - c;

                switch (SLint(floor(h*3.0f / SL_PI)))
                {   case 0: return set(m + c, m + x, m    , a); // [  0°, 60°]
                    case 1: return set(m + x, m + c, m    , a); // [ 60°,120°]
                    case 2: return set(m    , m + c, m + x, a); // [120°,180°]
                    case 3: return set(m    , m + x, m + c, a); // [180°,240°]
                    case 4: return set(m + x, m    , m + c, a); // [240°,300°]
                    case 5: return set(m + c, m    , m + x, a); // [300°,360°]
                }
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
typedef SLVec4<SLint>         SLVec4i;
typedef SLVec4<SLfloat>       SLCol4f;

typedef std::vector<SLVec4f>  SLVVec4f;
typedef std::vector<SLCol4f>  SLVCol4f;

#ifdef SL_HAS_DOUBLE
typedef SLVec4<SLdouble>      SLVec4d;
typedef std::vector<SLVec4d>  SLVVec4d;
#endif
//-----------------------------------------------------------------------------
#endif


