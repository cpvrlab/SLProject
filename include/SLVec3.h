//#############################################################################
//  File:      math/Math/SLVec3.h
//  Purpose:   3 Component vector class
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLVEC3_H
#define SLVEC3_H

#include <SL.h>
#include <SLVec2.h>
#include <SLUtils.h>

//-----------------------------------------------------------------------------
//! 3D vector template class for standard 3D vector algebra.
/*!
3D vector template class with type definitions for 3D vectors and colors:
\n Use SLVec3f for a specific float type vector
\n Use SLVec3d for a specific double type vector
\n Use SLVec3r for a precision independent real type.
*/

template<class T> 
class SLVec3
{
    public:     
            union
            {   struct {T x, y, z;};
                struct {T r, g, b;};
                struct {T comp[3];};
            };
            
                    SLVec3      (void)                  {}
                    SLVec3      (const T X,
                                 const T Y,
                                 const T Z=0)           {x=X;    y=Y;    z=Z;}
                    SLVec3      (const T v[3])          {x=v[0]; y=v[1]; z=v[2];}
                    SLVec3      (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;}
                    SLVec3      (const SLVec3<T>& v)    { x = v.x;  y = v.y;  z = v.z; }
                    SLVec3      (const SLstring threeFloatsWithDelimiter) {fromString(threeFloatsWithDelimiter);}

            void    set         (const T X,
                                 const T Y,
                                 const T Z)             {x=X;    y=Y;    z=Z;}
            void    set         (const T X,
                                 const T Y)             {x=X;    y=Y;    z=0;}
            void    set         (const T v[3])          {x=v[0]; y=v[1]; z=v[2];}
            void    set         (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;}
            void    set         (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;}

    // Component wise compare
    inline  SLint   operator == (const SLVec3& v) const {return (x==v.x && y==v.y && z==v.z);}
    inline  SLint   operator != (const SLVec3& v) const {return (x!=v.x || y!=v.y || z!=v.z);}
    inline  SLint   operator <= (const SLVec3& v) const {return (x<=v.x && y<=v.y && z<=v.z);}
    inline  SLint   operator >= (const SLVec3& v) const {return (x>=v.x && y>=v.y && z>=v.z);}
   
    // Operators with temp. allocation
    inline  SLVec3  operator -  (void) const            {return SLVec3(-x, -y, -z);}
    inline  SLVec3  operator +  (const SLVec3& v) const {return SLVec3(x+v.x, y+v.y, z+v.z);}
    inline  SLVec3  operator -  (const SLVec3& v) const {return SLVec3(x-v.x, y-v.y, z-v.z);}
    inline  T       operator *  (const SLVec3& v) const {return x*v.x+y*v.y+z*v.z;};  // dot
    inline  SLVec3  operator ^  (const SLVec3& v) const {return SLVec3(y*v.z-z*v.y,   // cross
                                                                           z*v.x-x*v.z,
                                                                           x*v.y-y*v.x);}
    inline  SLVec3  operator *  (const T s) const       {return SLVec3(x*s, y*s, z*s);}
    inline  SLVec3  operator /  (const T s) const       {return SLVec3(x/s, y/s, z/s);}
    inline  SLVec3  operator &  (const SLVec3& v) const {return SLVec3(x*v.x, y*v.y, z*v.z);}
    friend inline
            SLVec3  operator *  (T s, const SLVec3& v)  {return SLVec3(v.x*s, v.y*s, v.z*s);}

    // Assign operators
    inline  SLVec3& operator =  (const SLVec2<T>& v)    {x=v.x; y=v.y; z=0;      return *this;}
    inline  SLVec3& operator =  (const SLVec3& v)       {x=v.x; y=v.y; z=v.z;    return *this;}
    inline  SLVec3& operator += (const SLVec3& v)       {x+=v.x; y+=v.y; z+=v.z; return *this;}
    inline  SLVec3& operator -= (const SLVec3& v)       {x-=v.x; y-=v.y; z-=v.z; return *this;}
    inline  SLVec3& operator += (const T s)             {x+=s; y+=s; z+=s;       return *this;}
    inline  SLVec3& operator -= (const T s)             {x-=s; y-=s; z-=s;       return *this;}
    inline  SLVec3& operator *= (const T s)             {x*=s; y*=s; z*=s;       return *this;}
    inline  SLVec3& operator /= (const T s)             {x/=s; y/=s; z/=s;       return *this;}
    inline  SLVec3& operator &= (const SLVec3& v)       {x*=v.x; y*=v.y; z*=v.z; return *this;}
   
    // Stream output operator
    friend  ostream& operator << (ostream& output,
                                  const SLVec3& v)      {output<<"["<<v.x<<","<<v.y<<","<<v.z<<"]"; return output;}

    // Operations without temp. allocation (use these for speed!)
    inline  void    add         (const SLVec3& a,
                                 const SLVec3& b)       {x=a.x+b.x; y=a.y+b.y, z=a.z+b.z;}
    inline  void    add         (const SLVec3& a)       {x+=a.x; y+=a.y, z+=a.z;}
    inline  void    sub         (const SLVec3& a,
                                 const SLVec3& b)       {x=a.x-b.x; y=a.y-b.y, z=a.z-b.z;}
    inline  void    sub         (const SLVec3& a)       {x-=a.x; y-=a.y, z-=a.z;}
    inline  void    scale       (const T s)             {x*=s; y*=s; z*=s;}
    inline  T       dot         (const SLVec3& v) const {return x*v.x+y*v.y+z*v.z;}
    inline  void    cross       (const SLVec3& a,
                                 const SLVec3& b)       {x = a.y*b.z - a.z*b.y;
                                                         y = a.z*b.x - a.x*b.z;
                                                         z = a.x*b.y - a.y*b.x;}
    inline  T       length      () const                {return (T)sqrt(x*x+y*y+z*z);}
    inline  T       lengthSqr   () const                {return (x*x+y*y+z*z);}
    inline  SLVec3& normalize   ()                      {T L=length();
                                                         if (L>0) {x/=L; y/=L; z/=L;}
                                                         return *this;}
    inline  SLVec3  normalized  () const                {T L=length();
                                                         SLVec3 ret(*this);
                                                         if(L>0) { ret.x /= L; ret.y /= L; ret.z /= L; }
                                                         return ret;}
    inline  void    clampMinMax (const T min,
                                 const T max)           {x = (x>max)?max : (x<min)?min : x;
                                                         y = (y>max)?max : (y<min)?min : y;
                                                         z = (z>max)?max : (z<min)?min : z;}
    inline  T       diff        (const SLVec3& v)       {return SL_abs(x-v.x) +
                                                                SL_abs(y-v.y) +
                                                                SL_abs(z-v.z);}
    inline  void    mix         (const SLVec3& a,
                                 const SLVec3& b,
                                 const T factor_b)      {T factor_a = 1-factor_b;
                                                         x = a.x*factor_a + b.x*factor_b;
                                                         y = a.y*factor_a + b.y*factor_b;
                                                         z = a.z*factor_a + b.z*factor_b;}
    inline  void    setMin      (const SLVec3& v)       {if (v.x < x) x=v.x;
                                                         if (v.y < y) y=v.y;
                                                         if (v.z < z) z=v.z;}
    inline  void    setMax      (const SLVec3& v)       {if (v.x > x) x=v.x;
                                                         if (v.y > y) y=v.y;
                                                         if (v.z > z) z=v.z;}
    inline  T       maxXYZ      ()                      {if (x>=y && x>=z) return x;
                                                         else if (y>=z)    return y;
                                                         else              return z;}
    inline  T       maxXYZ      (SLint &axis)           {if (x>=y && x>=z){axis=0; return x;}
                                                         else if (y>=z)   {axis=1; return y;}
                                                         else             {axis=2; return z;}}
    inline  SLint   maxAxis     ()                      {if (x>=y && x>=z) return 0;
                                                         else if (y>=z)    return 1;
                                                         else              return 2;}
            //! Calculate the distance to point p
            T       distance (const SLVec3& p) const    {SLVec3 d(x-p.x, y-p.y, z-p.z);
                                                         return d.length();}

            //! Calculate the squard distance from the vector to point q
            T       distSquared (const SLVec3& q)       {SLVec3 dir(x,y,z); dir.normalize();
                                                         T t = dir.dot(q);
                                                         SLVec3 qPrime = t*dir;
                                                         SLVec3 v = q-qPrime;
                                                         T distanceSquared = v.dot(v);
                                                         return distanceSquared;}

            //! Calculates the spherical coords into r, theta & phi in radians
            void    toSpherical (T& r, T& theta, T& phi){r = length();
                                                         theta = acos(z/r); // 0 < theta < pi
                                                         phi = atan2(y,x);} // -pi < phi < pi

            //! Calculates the vector from spherical coords r, theta & phi in radians
            void    fromSpherical(T r, T theta, T phi)  {T sin_theta = sin(theta);
                                                         x = r*sin_theta*cos(phi);
                                                         y = r*sin_theta*sin(phi);
                                                         z = r*cos(theta);}
            //! Gamma correction
            void    gamma(T gammaVal)                   {x= pow(x,1.0/gammaVal);
                                                         y= pow(y,1.0/gammaVal);
                                                         z= pow(z,1.0/gammaVal);}
            //! Prints the vector to std out
            void    print       (const SLchar* str=0)   {if (str) SL_LOG("%s",str);
                                                         SL_LOG("% 3.6f, % 3.6f, % 3.6f\n",x, y, z);}

            //! Conversion to string
            SLstring toString   (SLstring delimiter = ", ")
            {   return SLUtils::toString(x) + delimiter +
                       SLUtils::toString(y) + delimiter +
                       SLUtils::toString(z);
            }

            //! Conversion from string
            void    fromString  (SLstring threeFloatsWithDelimiter, SLchar delimiter = ',')
            {   SLVstring components;
                SLUtils::split(threeFloatsWithDelimiter, delimiter, components);
                float f[3] = {0.0, 0.0f, 0.0f};
                for (int i=0; i<components.size(); ++i)
                    f[i] = (SLfloat)atof(components[i].c_str());
                x = f[0]; y = f[1]; z = f[2];
            }

            //! HSV to RGB color conversion (http://www.rapidtables.com/convert/color/hsv-to-rgb.htm)
            void    hsv2rgb     (const SLVec3 &hsv)
            {
                T h = fmod(fmod(hsv.x, SL_2PI) + SL_2PI, SL_2PI); // 0° <= H <= 360°
                T s = SL_clamp(hsv.y, 0.0f, 1.0f);
                T v = SL_clamp(hsv.z, 0.0f, 1.0f);
                T a = SL_clamp(hsv.w, 0.0f, 1.0f);

                T c = v * s;
                T x = c * (1.0f - fabs(fmod(h*3.0f / M_PI, 2.0f) - 1.0f));
                T m = v - c;

                switch (SLint(floor(h*3.0f / SL_PI)))
                {   case 0: return set(m + c, m + x, m    ); // [  0°, 60°]
                    case 1: return set(m + x, m + c, m    ); // [ 60°,120°]
                    case 2: return set(m    , m + c, m + x); // [120°,180°]
                    case 3: return set(m    , m + x, m + c); // [180°,240°]
                    case 4: return set(m + x, m    , m + c); // [240°,300°]
                    case 5: return set(m + c, m    , m + x); // [300°,360°]
                }
            }

    static SLVec3 ZERO;
    static SLVec3 BLACK;
    static SLVec3 GRAY;
    static SLVec3 WHITE;
    static SLVec3 RED;
    static SLVec3 GREEN;
    static SLVec3 BLUE;
    static SLVec3 CYAN;
    static SLVec3 MAGENTA;
    static SLVec3 YELLOW;
    static SLVec3 ORANGE;
    static SLVec3 COLBFH;
    static SLVec3 AXISX;
    static SLVec3 AXISY;
    static SLVec3 AXISZ;
};
//-----------------------------------------------------------------------------
template<class T> SLVec3<T> SLVec3<T>::ZERO   = SLVec3<T>(0.0f, 0.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::BLACK  = SLVec3<T>(0.0f, 0.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::GRAY   = SLVec3<T>(0.5f, 0.5f, 0.5f);
template<class T> SLVec3<T> SLVec3<T>::WHITE  = SLVec3<T>(1.0f, 1.0f, 1.0f);
template<class T> SLVec3<T> SLVec3<T>::RED    = SLVec3<T>(1.0f, 0.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::GREEN  = SLVec3<T>(0.0f, 1.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::BLUE   = SLVec3<T>(0.0f, 0.0f, 1.0f);
template<class T> SLVec3<T> SLVec3<T>::YELLOW = SLVec3<T>(1.0f, 1.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::ORANGE = SLVec3<T>(0.5f, 0.5f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::CYAN   = SLVec3<T>(0.0f, 1.0f, 1.0f);
template<class T> SLVec3<T> SLVec3<T>::MAGENTA= SLVec3<T>(1.0f, 0.0f, 1.0f);
template<class T> SLVec3<T> SLVec3<T>::COLBFH = SLVec3<T>(0.8f, 0.5f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::AXISX  = SLVec3<T>(1.0f, 0.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::AXISY  = SLVec3<T>(0.0f, 1.0f, 0.0f);
template<class T> SLVec3<T> SLVec3<T>::AXISZ  = SLVec3<T>(0.0f, 0.0f, 1.0f);
//-----------------------------------------------------------------------------
typedef SLVec3<SLfloat>       SLVec3f;
typedef SLVec3<SLfloat>       SLCol3f;
typedef SLVec3<SLint>         SLVec3i; 
typedef SLVec3<SLuint>        SLVec3ui; 
typedef SLVec3<SLshort>       SLVec3s; 

typedef std::vector<SLVec3f>  SLVVec3f;
typedef std::vector<SLCol3f>  SLVCol3f;

#ifdef SL_HAS_DOUBLE
typedef SLVec3<SLdouble>      SLVec3d;
typedef std::vector<SLVec3d>  SLVVec3d;
#endif
//-----------------------------------------------------------------------------
#endif


