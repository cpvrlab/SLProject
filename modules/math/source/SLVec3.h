//#############################################################################
//  File:      math/Math/SLVec3.h
//  Purpose:   3 Component vector class
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLVEC3_H
#define SLVEC3_H

#include <math.h>
#include <SLMath.h>
#include <SLVec2.h>
#include <Utils.h>

// Some constants for ecef to lla conversions
static const double EARTH_RADIUS_A          = 6378137;
static const double EARTH_ECCENTRICTIY      = 8.1819190842622e-2;
static const double EARTH_RADIUS_A_SQR      = EARTH_RADIUS_A * EARTH_RADIUS_A;
static const double EARTH_ECCENTRICTIY_SQR  = EARTH_ECCENTRICTIY * EARTH_ECCENTRICTIY;
static const double EARTH_RADIUS_B          = sqrt(EARTH_RADIUS_A_SQR * (1 - EARTH_ECCENTRICTIY_SQR));
static const double EARTH_RADIUS_B_SQR      = EARTH_RADIUS_B * EARTH_RADIUS_B;
static const double EARTH_ECCENTRICTIY2     = sqrt((EARTH_RADIUS_A_SQR - EARTH_RADIUS_B_SQR) / EARTH_RADIUS_B_SQR);
static const double EARTH_ECCENTRICTIY2_SQR = EARTH_ECCENTRICTIY2 * EARTH_ECCENTRICTIY2;

//-----------------------------------------------------------------------------
//! 3D vector template class for standard 3D vector algebra.
/*!
3D vector template class with type definitions for 3D vectors and colors:
\n Use SLVec3f for a specific float type vector
\n Use SLVec3d for a specific double type vector
\n Use SLVec3r for a precision independent real type.
*/
// clang-format off
template<class T> 
class SLVec3
{
    public:
            union
            {   struct {T x, y, z;};        // 3D cartesian coordinates
                struct {T r, g, b;};        // Red, green & blue color components
                struct {T lat, lon, alt;};  // WGS84 latitude (deg), longitude (deg) & altitude (m)
                struct {T comp[3];};
            };
            
                    SLVec3      ()                      {x=0;y=0;z=0;}
           explicit SLVec3      (const T V)             {x=V;    y=V;    z=V;}
                    SLVec3      (const T X,
                                 const T Y,
                                 const T Z=0)           {x=X;    y=Y;    z=Z;}
           explicit SLVec3      (const T v[3])          {x=v[0]; y=v[1]; z=v[2];}
           explicit SLVec3      (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;}
                    SLVec3      (const SLVec3<T>& v)    {x = v.x; y = v.y; z = v.z; }
           explicit SLVec3      (const SLstring& threeFloatsWithDelimiter) {fromString(threeFloatsWithDelimiter);}

            void    set         (const T X,
                                 const T Y,
                                 const T Z)             {x=X;    y=Y;    z=Z;}
            void    set         (const T X,
                                 const T Y)             {x=X;    y=Y;    z=0;}
            void    set         (const T v[3])          {x=v[0]; y=v[1]; z=v[2];}
            void    set         (const SLVec2<T>& v)    {x=v.x;  y=v.y;  z=0;}
            void    set         (const SLVec3<T>& v)    {x=v.x;  y=v.y;  z=v.z;}

    // Component wise compare
    inline  SLbool  operator == (const SLVec3& v) const {return (x==v.x && y==v.y && z==v.z);}
    inline  SLbool  operator != (const SLVec3& v) const {return (x!=v.x || y!=v.y || z!=v.z);}
    inline  SLbool  operator <= (const SLVec3& v) const {return (x<=v.x && y<=v.y && z<=v.z);}
    inline  SLbool  operator >= (const SLVec3& v) const {return (x>=v.x && y>=v.y && z>=v.z);}
    inline  SLbool  operator <  (const SLVec3& v) const {return (x< v.x && y< v.y && z< v.z);}
    inline  SLbool  operator >  (const SLVec3& v) const {return (x> v.x && y> v.y && z> v.z);}
    inline  SLbool  operator <= (const T v)       const {return (x<=v && y<=v && z<=v);}
    inline  SLbool  operator >= (const T v)       const {return (x>=v && y>=v && z>=v);}
    inline  SLbool  operator <  (const T v)       const {return (x< v && y< v && z< v);}
    inline  SLbool  operator >  (const T v)       const {return (x> v && y> v && z> v);}
   
    // Operators with temp. allocation
    inline  SLVec3  operator -  () const                {return SLVec3(-x, -y, -z);}
    inline  SLVec3  operator +  (const SLVec3& v) const {return SLVec3(x+v.x, y+v.y, z+v.z);}
    inline  SLVec3  operator -  (const SLVec3& v) const {return SLVec3(x-v.x, y-v.y, z-v.z);}
    inline  T       operator *  (const SLVec3& v) const {return x*v.x+y*v.y+z*v.z;} // dot
    inline  SLVec3  operator ^  (const SLVec3& v) const {return SLVec3(y*v.z-z*v.y, // cross
                                                                           z*v.x-x*v.z,
                                                                           x*v.y-y*v.x);}
    inline  SLVec3  operator *  (const T s) const       {return SLVec3(x*s, y*s, z*s);}
    inline  SLVec3  operator /  (const T s) const       {return SLVec3(x/s, y/s, z/s);}
    inline  SLVec3  operator &  (const SLVec3& v) const {return SLVec3(x*v.x, y*v.y, z*v.z);}
    friend
    inline  SLVec3  operator *  (T s, const SLVec3& v)  {return SLVec3(v.x*s, v.y*s, v.z*s);}

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
    friend  std::ostream& operator << (std::ostream& output,
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
    inline  T       diff        (const SLVec3& v)       {return Utils::abs(x-v.x) +
                                                                Utils::abs(y-v.y) +
                                                                Utils::abs(z-v.z);}
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
    inline  T       minXYZ      ()                      {if (x<=y && x<=z) return x;
                                                         else if (y<=z)    return y;
                                                         else              return z;}
    inline  T       maxXYZ      (SLint &comp)           {if (x>=y && x>=z){comp=0; return x;}
                                                         else if (y>=z)   {comp=1; return y;}
                                                         else             {comp=2; return z;}}
    inline  T       minXYZ      (SLint &comp)           {if (x<=y && x<=z){comp=0; return x;}
                                                         else if (y<=z)   {comp=1; return y;}
                                                         else             {comp=2; return z;}}
    inline  SLint   maxComp     ()                      {if (x>=y && x>=z) return 0;
                                                         else if (y>=z)    return 1;
                                                         else              return 2;}
    inline  SLbool  isZero      ()                      {return (x==0 && y==0 && z==0);}

            //! Calculate the distance to point p
            T       distance (const SLVec3& p) const    {SLVec3 d(x-p.x, y-p.y, z-p.z);
                                                         return d.length();}

            //! Calculate the squared distance from the vector to point q
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
            void    gammaCorrect(T oneOverGamma)        {x= pow(x,oneOverGamma);
                                                         y= pow(y,oneOverGamma);
                                                         z= pow(z,oneOverGamma);}

            //! Prints the vector to std out
            void    print       (const SLchar* str=nullptr){if (str) SLMATH_LOG("%s",str);
                SLMATH_LOG("% 3.2f, % 3.2f, % 3.2f",x, y, z);}

            //! Conversion to string
            SLstring toString   (SLstring delimiter = ", ", int decimals = 2)
            {   return Utils::toString(x,decimals) + delimiter +
                       Utils::toString(y,decimals) + delimiter +
                       Utils::toString(z,decimals);
            }

            //! Conversion from string
            void fromString (const SLstring& threeFloatsWithDelimiter, SLchar delimiter = ',')
            {   SLVstring components;
                Utils::splitString(threeFloatsWithDelimiter, delimiter, components);
                float f[3] = {0.0, 0.0f, 0.0f};
                for (SLulong i=0; i<components.size(); ++i)
                    f[i] = (SLfloat)atof(components[i].c_str());
                x = f[0]; y = f[1]; z = f[2];
            }

            //! HSV (0-1) to RGB (0-1) color conversion (http://www.rapidtables.com/convert/color/hsv-to-rgb.htm)
            void hsv2rgb (const SLVec3 &hsv)
            {
                T h = fmod(fmod(hsv.x, Utils::TWOPI) + Utils::TWOPI, Utils::TWOPI); // 0 deg <= H <= 360 deg
                T s = clamp(hsv.y, 0.0f, 1.0f);
                T v = clamp(hsv.z, 0.0f, 1.0f);
                T a = clamp(hsv.w, 0.0f, 1.0f);

                T c = v * s;
                T x = c * (1.0f - fabs(fmod(h*3.0f / M_PI, 2.0f) - 1.0f));
                T m = v - c;

                switch (SLint(floor(h*3.0f * Utils::ONEOVERPI)))
                {   case 0: return set(m + c, m + x, m    ); // [  0 deg, 60 deg]
                    case 1: return set(m + x, m + c, m    ); // [ 60 deg,120 deg]
                    case 2: return set(m    , m + c, m + x); // [120 deg,180 deg]
                    case 3: return set(m    , m + x, m + c); // [180 deg,240 deg]
                    case 4: return set(m + x, m    , m + c); // [240 deg,300 deg]
                    case 5: return set(m + c, m    , m + x); // [300 deg,360 deg]
                }
            }

            //! Earth Centered Earth Fixed (ecef) to Latitude Longitude Altitude (LatLonAlt) using the WGS84 model
            /*!
            Longitude and latitude are in decimal degrees and altitude in meters.
            The Cartesian ecef coordinates are in meters.
            See for more details: https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
            */
            void ecef2LatLonAlt(const SLVec3 &ecef)
            {
                double a    = EARTH_RADIUS_A;
                double b    = EARTH_RADIUS_B;
                double esq  = EARTH_ECCENTRICTIY_SQR;
                double epsq = EARTH_ECCENTRICTIY2_SQR;

                double p   = sqrt(ecef.x * ecef.x + ecef.y*ecef.y);
                double th  = atan2(a*ecef.z, b*p);

                double longitude = atan2(ecef.y, ecef.x);
                double latitude = atan2((ecef.z + epsq*b*pow(sin(th),3.0)), (p - esq*a*pow(cos(th),3.0)) );
                double N   = a/(sqrt(1-esq*pow(sin(latitude),2)));
                double altitude = p/cos(latitude) - N;

                lat = latitude * Utils::RAD2DEG;
                lon = fmod(longitude,Utils::TWOPI) * Utils::RAD2DEG;
                alt = altitude * Utils::RAD2DEG;
            }

            //! Latitude Longitude Altitude (LatLonAlt) to Earth Centered Earth Fixed (ecef) using the WGS84 model
            /*!
            Latitude and longitude are in decimal degrees and altitude in meters.
            The Cartesian ecef coordinates are in meters.
            See for more details: https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
            */
            void latlonAlt2ecef(const SLVec3 &latDegLonDegAltM)
            {
                double latitude = latDegLonDegAltM.lat * Utils::DEG2RAD;
                double longitude = latDegLonDegAltM.lon * Utils::DEG2RAD;
                double altitude = latDegLonDegAltM.alt;
                double a   = EARTH_RADIUS_A;
                double esq = EARTH_ECCENTRICTIY_SQR;
                double cosLat = cos(latitude);

                double N = a / sqrt(1 - esq * pow(sin(latitude),2));

                x = (N+altitude) * cosLat * cos(longitude);
                y = (N+altitude) * cosLat * sin(longitude);
                z = ((1-esq) * N + altitude) * sin(latitude);
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
typedef SLVec3<SLfloat>  SLVec3f;
typedef SLVec3<SLfloat>  SLCol3f;
typedef SLVec3<SLint>    SLVec3i;
typedef SLVec3<SLuint>   SLVec3ui;
typedef SLVec3<SLshort>  SLVec3s;
typedef SLVec3<double>   SLVec3d;

typedef vector<SLVec3f>  SLVVec3f;
typedef vector<SLCol3f>  SLVCol3f;
typedef vector<SLVec3d>  SLVVec3d;

typedef vector<SLfloat>  SLVfloat;
//-----------------------------------------------------------------------------
#endif


