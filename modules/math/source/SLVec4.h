//#############################################################################
//  File:      math/Math/SLVec4.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLVEC4_H
#define SLVEC4_H

#include <SL.h>
#include <SLVec2.h>
#include <SLVec3.h>
#include <Utils.h>

//-----------------------------------------------------------------------------
//! 4D vector template class for standard 4D vector algebra.
/*!
4D vector template class with type definitions for 4D vectors and colors:
\n Use SLVec4f for a specific float type vector
\n Use SLVec4d for a specific double type vector
\n Use SLVec4r for a precision independent real type.
*/
// clang-format off
template <class T>
class SLVec4
{
    public:
        union
        {   struct {T x, y, z, w;};
            struct {T r, g, b, a;};
            struct {T comp[4];};
        };

                    SLVec4      ()                      {x=0;y=0;z=0;w=0;}
           explicit SLVec4      (const T V)             {x=V;    y=V;    z=V;    w=1;}
                    SLVec4      (const T X,
                                 const T Y,
                                 const T Z=0,
                                 const T W=1)           {x=X;    y=Y;    z=Z;    w=W;}
           explicit SLVec4      (const T v[4])          {x=v[0]; y=v[1]; z=v[2]; w=v[3];}
           explicit SLVec4      (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;    w=1;}
           explicit SLVec4      (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=1;}
                    SLVec4      (const SLVec4<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=v.w;}
           explicit SLVec4      (const SLstring& fourFloatsWithDelimiter) {fromString(fourFloatsWithDelimiter);}

            void    set         (const T X,
                                 const T Y,
                                 const T Z,
                                 const T W=1)           {x=X;    y=Y;    z=Z;    w=W;}
            void    set         (const T xyz)           {x=xyz;  y=xyz;  z=xyz;  w=1;}
            void    set         (const T v[4])          {x=v[0]; y=v[1]; z=v[2]; w=v[3];}
            void    set         (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;    w=1;}
            void    set         (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=1;}
            void    set         (const SLVec4<T>& v)    {x=v.x;  y=v.y;  z=v.z;  w=v.w;}

    inline SLbool   operator == (const SLVec4& v) const {return (x==v.x && y==v.y && z==v.z && w==v.w);}
    inline SLbool   operator != (const SLVec4& v) const {return (x!=v.x || y!=v.y || z!=v.z || w!=v.w);}
    inline SLbool   operator <= (const SLVec4& v) const {return (x<=v.x && y<=v.y && z<=v.z && w<=v.w);}
    inline SLbool   operator >= (const SLVec4& v) const {return (x>=v.x && y>=v.y && z>=v.z && w>=v.w);}
    inline SLbool   operator <  (const SLVec4& v) const {return (x<v.x && y<v.y && z<v.z && w<v.w);}
    inline SLbool   operator >  (const SLVec4& v) const {return (x>v.x && y>v.y && z>v.z && w>v.w);}
    inline SLbool   operator <= (const T v)       const {return (x<=v && y<=v && z<=v && w<=v);}
    inline SLbool   operator >= (const T v)       const {return (x>=v && y>=v && z>=v && w>=v);}
    inline SLbool   operator <  (const T v)       const {return (x<v  && y<v  && z<v  && w<v);}
    inline SLbool   operator >  (const T v)       const {return (x>v  && y>v  && z>v  && w>v);}
   
   
    // Operators with temp. allocation
    inline SLVec4   operator +  (const SLVec4& v) const {return SLVec4(x+v.x, y+v.y, z+v.z, w+v.w);}
    inline SLVec4   operator -  (const SLVec4& v) const {return SLVec4(x-v.x, y-v.y, z-v.z, w-v.w);}
    inline SLVec4   operator -  ()                const {return SLVec4(-x, -y, -z, -w);}
    inline T        operator *  (const SLVec4& v) const {return x*v.x+y*v.y+z*v.z+w*v.w;}
    inline SLVec4   operator *  (const T s)       const {return SLVec4(x*s, y*s, z*s);}
    inline SLVec4   operator /  (const T s)       const {return SLVec4(x/s, y/s, z/s, w/s);}
    inline SLVec4   operator &  (const SLVec4& v) const {return SLVec4(x*v.x, y*v.y, z*v.z, w*v.w);}
    friend inline 
    SLVec4   operator *  (T s, const SLVec4& v)  {return SLVec4(v.x*s, v.y*s, v.z*s);}
   
    // Stream output operator
    friend std::ostream& operator << (std::ostream& output,
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
    inline T        dot         (const SLVec4& v)       {return x*v.x+y*v.y+z*v.z+w*v.w;}
    inline void     cross       (const SLVec4& a,
                                 const SLVec4& b)       {x = a.y*b.z - a.z*b.y;
                                                         y = a.z*b.x - a.x*b.z;
                                                         z = a.x*b.y - a.y*b.x;
                                                         w = 1;}
    inline SLVec3<T> vec3       () const                {return SLVec3<T>(x,y,z);}
    inline SLVec2<T> vec2       () const                {return SLVec2<T>(x,y);}
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
    inline T        diff        (const SLVec4& v)       {return Utils::abs(x-v.x) +
                                                                Utils::abs(y-v.y) +
                                                                Utils::abs(z-v.z) +
                                                                Utils::abs(w-v.w);}
    inline T        diffRGB     (const SLVec4& v)       {return Utils::abs(x-v.x) +
                                                                Utils::abs(y-v.y) +
                                                                Utils::abs(z-v.z);}
    inline void     mix         (const SLVec4& a,
                                 const SLVec4& b,
                                 const T factor_b)      {T factor_a = 1-factor_b;
                                                         x = a.x*factor_a + b.x*factor_b;
                                                         y = a.y*factor_a + b.y*factor_b;
                                                         z = a.z*factor_a + b.z*factor_b;
                                                         w = a.w*factor_a + b.w*factor_b;}
    inline T        minXYZ      ()                      {if (x<=y && x<=z)   return x;
                                                         else if (y<=z)      return y;
                                                         else                return z;}
    inline T        maxXYZ      ()                      {if (x>=y && x>=z)   return x;
                                                         else if (y>=z)      return y;
                                                         else                return z;}
    inline T        minXYZW     ()                      {if (x<=y && x<=z && x<=w) return x;
                                                         else if (y<=z && y<=w)    return y;
                                                         else if (z<=w)            return z;
                                                         else                      return w;}
    inline T        maxXYZW     ()                      {if (x>=y && x>=z && x>=w) return x;
                                                         else if (y>=z && y>=w)    return y;
                                                         else if (z>=w)            return z;
                                                         else                      return w;}
    inline SLint    maxComp     ()                      {if (x>=y && x>=z && x>=w) return 0;
                                                         else if (y>=z && y>=w)    return 1;
                                                         else if (z>=w)            return 2;
                                                         else                      return 3;}
    inline SLint    minComp     ()                      {if (x<=y && x<=z && x<=w) return 0;
                                                         else if (y<=z && y<=w)    return 1;
                                                         else if (z<=w)            return 2;
                                                         else                      return 3;}
    inline  SLbool  isZero      ()                      {return (x==0 && y==0 && z==0 && w==0);}

            //! Gamma correction
            void    gammaCorrect(T oneOverGamma)        {x= pow(x,oneOverGamma);
                                                         y= pow(y,oneOverGamma);
                                                         z= pow(z,oneOverGamma);}

            void    print       (const SLchar* str=nullptr)
            {   if (str) SL_LOG("%s",str);
                SL_LOG("% 3.3f, % 3.3f, % 3.3f, % 3.3f",x, y, z, w);
            }

            //! Conversion to string
            SLstring toString   (SLstring delimiter=", ")
             {   return Utils::toString(x) + delimiter +
                        Utils::toString(y) + delimiter +
                        Utils::toString(z) + delimiter +
                        Utils::toString(w);
             }

            //! Conversion from string
            void    fromString  (const SLstring& fourFloatsWithDelimiter, SLchar delimiter=',')
            {   SLVstring components;
                Utils::splitString(fourFloatsWithDelimiter, delimiter, components);
                float f[4] = {0.0, 0.0f, 0.0f, 1.0f};
                for (SLuint i=0; i<components.size(); ++i)
                    f[i] = (SLfloat)atof(components[i].c_str());
                x = f[0]; y = f[1]; z = f[2]; w = f[3];
            }

            //! HSVA to RGBA color conversion (http://www.rapidtables.com/convert/color/hsv-to-rgb.htm)
            void    hsva2rgba   (const SLVec4 &hsva)
            {
                T h = fmod(fmod(hsva.x, Utils::TWOPI) + Utils::TWOPI, Utils::TWOPI); // 0 deg <= H <= 360 deg
                T s = Utils::clamp(hsva.y, 0.0f, 1.0f);
                T v = Utils::clamp(hsva.z, 0.0f, 1.0f);
                T a = Utils::clamp(hsva.w, 0.0f, 1.0f);

                T c = v * s;
                T x = c * (1.0f - (T)fabs((T)fmod(h*3.0f / Utils::PI, 2.0f) - 1.0f));
                T m = v - c;

                switch (SLint(floor(h*3.0f * Utils::ONEOVERPI)))
                {   case 0: return set(m + c, m + x, m    , a); // [  0 deg, 60 deg]
                    case 1: return set(m + x, m + c, m    , a); // [ 60 deg,120 deg]
                    case 2: return set(m    , m + c, m + x, a); // [120 deg,180 deg]
                    case 3: return set(m    , m + x, m + c, a); // [180 deg,240 deg]
                    case 4: return set(m + x, m    , m + c, a); // [240 deg,300 deg]
                    case 5: return set(m + c, m    , m + x, a); // [300 deg,360 deg]
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

typedef vector<SLVec4f>  SLVVec4f;
typedef vector<SLCol4f>  SLVCol4f;

#ifdef SL_HAS_DOUBLE
typedef SLVec4<SLdouble>      SLVec4d;
typedef vector<SLVec4d>  SLVVec4d;
#endif
//-----------------------------------------------------------------------------
#endif


