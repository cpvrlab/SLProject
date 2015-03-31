//#############################################################################
//  File:      math/SLMat4.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLQUAT_H
#define SLQUAT_H

#include <SL.h>
#include <SLMath.h>
#include <SLMat4.h>
#include <SLVec4.h>

//-----------------------------------------------------------------------------
//!Quaternion class for angle-axis rotation representation. 
/*!Quaternion class for angle-axis rotation representation. For rotations
quaternions must have unit length. See http://en.wikipedia.org/wiki/Quaternion
Quaternions can be interpolated at low cost with the method lerp or slerp.
@todo Add more doxygen documentation.
*/
template <class T>
class SLQuat4 
{  
    public:         SLQuat4        ();
                    SLQuat4        (T x, T y, T z, T w);
                    SLQuat4        (const SLMat3<T>& m);
                    SLQuat4        (const T angleDEG, const SLVec3<T>& axis);
                    SLQuat4        (const SLVec3<T>& v0, const SLVec3<T>& v1);
                    SLQuat4        (const T pitchRAD, const T yawRAD, const T rollRAD);
      
        T           x              () const { return _x; }
        T           y              () const { return _y; }
        T           z              () const { return _z; }
        T           w              () const { return _w; }

        void        set            (T x, T y, T z, T w);
        void        fromMat3       (const SLMat3<T>& m);
        void        fromAngleAxis  (const T angleRAD, 
                                    const T axisX, const T axisY, const T axisZ);
        void        fromEulerAngles(const T pitchRAD, const T yawRAD, const T rollRAD);
        void        fromVec3       (const SLVec3<T>& v0, const SLVec3<T>& v1);

 static SLQuat4<T>  fromLookRotation(const SLVec3<T>& forward, const SLVec3<T>& up);

        SLMat3<T>   toMat3         () const;
        SLMat4<T>   toMat4         () const;
        SLVec4<T>   toVec4         () const;
        void        toAngleAxis    (T& angleDEG, SLVec3<T>& axis) const;
      
        T           dot         (const SLQuat4<T>& q) const;
        T           length      () const;
        SLQuat4<T>  normalized  () const;
        T           normalize   ();
        SLQuat4<T>  inverted    () const;
        void        invert      ();
        SLQuat4<T>  conjugated  () const;
        void        conjugate   ();
        SLQuat4<T>  rotated     (const SLQuat4<T>& b) const;
        void        rotate      (const SLQuat4<T>& q);
        SLVec3<T>   rotate      (const SLVec3<T>& vec) const;
        SLQuat4<T>  scaled      (T scale) const;
        void        scale       (T scale);
        SLQuat4<T>  lerp        (const SLQuat4<T>& q2, const T t) const;
        void        lerp        (const SLQuat4<T>& q1, 
                                const SLQuat4<T>& q2, const T t);
        SLQuat4<T>  slerp       (const SLQuat4<T>& q2, const T t) const;
        void        slerp       (const SLQuat4<T>& q1, 
                                const SLQuat4<T>& q2, const T t);
           
        SLQuat4<T>& operator=   (const SLQuat4<T> q);
        SLQuat4<T>  operator-   (const SLQuat4<T>& q) const;
        SLQuat4<T>  operator+   (const SLQuat4<T>& q) const;
        SLQuat4<T>  operator*   (const SLQuat4<T>& q) const;
        SLQuat4<T>  operator*   (const T s) const;
        SLVec3<T>   operator*   (const SLVec3<T>& v) const;
        SLbool      operator==  (const SLQuat4<T>& q) const;
        SLbool      operator!=  (const SLQuat4<T>& q) const;
        SLQuat4<T>& operator*=  (const SLQuat4<T>& q2);
        SLQuat4<T>& operator*=  (const T s);
        
    static SLQuat4 IDENTITY;

    private:
        T  _x, _y, _z, _w;
};

//-----------------------------------------------------------------------------
template<class T> SLQuat4<T> SLQuat4<T>::IDENTITY   = SLQuat4<T>(0.0f, 0.0f, 0.0f, 1.0f);

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T>::SLQuat4() : _x(0), _y(0), _z(0), _w(1)
{
}

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T>::SLQuat4(T x, T y, T z, T w) : _x(x), _y(y), _z(z), _w(w)
{
}

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T>::SLQuat4(const SLMat3<T>& m)
{
    fromMat3(m);
}

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T>::SLQuat4(const T angleDEG, const SLVec3<T>& axis)
{
    fromAngleAxis(angleDEG*SL_DEG2RAD, axis.x, axis.y, axis.z);
}

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T>::SLQuat4(const SLVec3<T>& v0, const SLVec3<T>& v1)
{
    fromVec3(v0, v1);
}
                    
//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T>::SLQuat4(const T pitchRAD, const T yawRAD, const T rollRAD)
{
    fromEulerAngles(pitchRAD, yawRAD, rollRAD);
}

//-----------------------------------------------------------------------------
template <class T>
void SLQuat4<T>::set(T x, T y, T z, T w)
{
    _x = x;
    _y = y;
    _z = z;
    _w = w;
}
//-----------------------------------------------------------------------------
template <class T>
void SLQuat4<T>::fromMat3 (const SLMat3<T>& m)
{
    // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
    // article "Quaternion Calculus and Fast Animation".

    const int next[3] = {1, 2, 0};

    T trace = m.trace();
    T root;

    if (trace > (T)0)
    {
        // |_w| > 1/2, may as well choose _w > 1/2
        root = sqrt(trace + (T)1);  // 2w
        _w = ((T)0.5)*root;
        root = ((T)0.5)/root;       // 1/(4w)
        _x = (m(2,1) - m(1,2)) * root;
        _y = (m(0,2) - m(2,0)) * root;
        _z = (m(1,0) - m(0,1)) * root;
    }
    else
    {
        // |_w| <= 1/2
        int i = 0;
        if (m(1,1) > m(0,0)) i = 1;
        if (m(2,2) > m(i,i)) i = 2;
        int j = next[i];
        int k = next[j];

        root = sqrt(m(i,i) - m(j,j) - m(k,k) + (T)1);
        T* quat[3] = { &_x, &_y, &_z };
        *quat[i] = ((T)0.5)*root;
        root = ((T)0.5)/root;
        _w = (m(k,j) - m(j,k))*root;
        *quat[j] = (m(j,i) + m(i,j))*root;
        *quat[k] = (m(k,i) + m(i,k))*root;
    }
}

//-----------------------------------------------------------------------------
template <class T>
void SLQuat4<T>::fromVec3(const SLVec3<T>& v0, const SLVec3<T>& v1)
{  
    // Code from "The Shortest Arc Quaternion" 
    // by Stan Melax in "Game Programming Gems".
    if (v0 == -v1)
    {
        fromAngleAxis(SL_PI, 1, 0, 0);
        return;
    }
      
    SLVec3<T> c;
    c.cross(v0, v1);
    T d = v0.dot(v1);
    T s = sqrt((1 + d) * (T)2);

    _x = c.x / s;
    _y = c.y / s;
    _z = c.z / s;
    _w = s * (T)0.5;
}

//-----------------------------------------------------------------------------
template <class T>
SLQuat4<T>  SLQuat4<T>::fromLookRotation(const SLVec3<T>& forward, const SLVec3<T>& up)
{
    SLVec3f vector = forward;
    vector.normalize();
    SLVec3f vector2;
    vector2.cross(up, vector);
    vector2.normalize();
    SLVec3f vector3;
    vector3.cross(vector, vector2);
    vector3.normalize();
    float m00 = vector2.x;
    float m01 = vector2.y;
    float m02 = vector2.z;
    float m10 = vector3.x;
    float m11 = vector3.y;
    float m12 = vector3.z;
    float m20 = vector.x;
    float m21 = vector.y;
    float m22 = vector.z;
 
    float num8 = (m00 + m11) + m22;
    SLQuat4<T> quaternion;
    if (num8 > 0.0f)
    {
        float num = (float)sqrt(num8 + 1.0f);
        quaternion._w = num * 0.5f;
        num = 0.5f / num;
        quaternion._x = (m12 - m21) * num;
        quaternion._y = (m20 - m02) * num;
        quaternion._z = (m01 - m10) * num;
    }
    else if ((m00 >= m11) && (m00 >= m22))
    {
        float num7 = (float)sqrt(((1.0f + m00) - m11) - m22);
        float num4 = 0.5f / num7;
        quaternion._x = 0.5f * num7;
        quaternion._y = (m01 + m10) * num4;
        quaternion._z = (m02 + m20) * num4;
        quaternion._w = (m12 - m21) * num4;
    }
    else if (m11 > m22)
    {
        float num6 = (float)sqrt(((1.0f + m11) - m00) - m22);
        float num3 = 0.5f / num6;
        quaternion._x = (m10+ m01) * num3;
        quaternion._y = 0.5f * num6;
        quaternion._z = (m21 + m12) * num3;
        quaternion._w = (m20 - m02) * num3;
    }
    else
    {
        float num5 = (float)sqrt(((1.0f + m22) - m00) - m11);
        float num2 = 0.5f / num5;
        quaternion._x = (m20 + m02) * num2;
        quaternion._y = (m21 + m12) * num2;
        quaternion._z = 0.5f * num5;
        quaternion._w = (m01 - m10) * num2;
    }
    return quaternion;
}
//-----------------------------------------------------------------------------
template <class T>
void SLQuat4<T>::fromAngleAxis(const T angleRAD, 
                               const T axisX, const T axisY, const T axisZ)
{
    _w = (T)cos(angleRAD * (T)0.5);
    _x = _y = _z = (T)sin(angleRAD * (T)0.5);
    _x *= axisX;
    _y *= axisY;
    _z *= axisZ;
}

//-----------------------------------------------------------------------------
/*! Sets the quaternion from 3 Euler angles in radians
Source: Essential Mathematics for Games and Interactive Applications
A Programmer’s Guide 2nd edition by James M. Van Verth and Lars M. Bishop
*/
template <class T>
void SLQuat4<T>::fromEulerAngles(const T pitchRAD, const T yawRAD, const T rollRAD)
{
    // Basically we create 3 Quaternions, one for pitch (x), one for yaw (y), 
    // one for roll (z) and multiply those together.
    // The calculation below does the same, just shorter
 
    T p = pitchRAD * (T)0.5;
    T y = yawRAD   * (T)0.5;
    T r = rollRAD  * (T)0.5;
 
    T Sx = (T)sin(p);
    T Cx = (T)cos(p);
    T Sy = (T)sin(y);
    T Cy = (T)cos(y);
    T Sz = (T)sin(r);
    T Cz = (T)cos(r);
 
    _x = Sx*Cy*Cz + Cx*Sy*Sz;
    _y = Cx*Sy*Cz - Sx*Cy*Sz;
    _z = Cx*Cy*Sz + Sx*Sy*Cx;
    _w = Cx*Cy*Cz - Sx*Sy*Sz;
}

//-----------------------------------------------------------------------------
template <class T>
SLMat3<T> SLQuat4<T>::toMat3() const
{
    T  x2 = _x *(T)2;  
    T  y2 = _y *(T)2;  
    T  z2 = _z *(T)2;

    T wx2 = _w * x2;  T wy2 = _w * y2;  T wz2 = _w * z2;
    T xx2 = _x * x2;  T xy2 = _x * y2;  T xz2 = _x * z2;
    T yy2 = _y * y2;  T yz2 = _y * z2;  T zz2 = _z * z2;

    SLMat3<T> m(1 -(yy2 + zz2),    xy2 - wz2,     xz2 + wy2,
                    xy2 + wz2, 1 -(xx2 + zz2),    yz2 - wx2,
                    xz2 - wy2,     yz2 + wx2, 1 -(xx2 + yy2));
    return m;
}

//-----------------------------------------------------------------------------
template <class T>
SLMat4<T> SLQuat4<T>::toMat4() const
{
    T  x2 = _x *(T)2;  
    T  y2 = _y *(T)2;  
    T  z2 = _z *(T)2;

    T wx2 = _w * x2;  T wy2 = _w * y2;  T wz2 = _w * z2;
    T xx2 = _x * x2;  T xy2 = _x * y2;  T xz2 = _x * z2;
    T yy2 = _y * y2;  T yz2 = _y * z2;  T zz2 = _z * z2;

    SLMat4<T> m(1 -(yy2 + zz2),    xy2 - wz2,     xz2 + wy2,  0,
                    xy2 + wz2, 1 -(xx2 + zz2),    yz2 - wx2,  0,
                    xz2 - wy2,     yz2 + wx2, 1 -(xx2 + yy2), 0,
                            0,             0,              0, 1);
    return m;
}

//-----------------------------------------------------------------------------
template <class T>
inline SLVec4<T> SLQuat4<T>::toVec4() const
{
    return SLVec4<T>(_x, _y, _z, _w);
}

//-----------------------------------------------------------------------------
template <typename T>
void SLQuat4<T>::toAngleAxis (T& angleDEG, SLVec3<T>& axis) const
{
    // The quaternion representing the rotation is
    // q = cos(A/2) + sin(A/2)*(_x*i+_y*j+_z*k)

    T sqrLen = _x*_x + _y*_y + _z*_z;

    if (sqrLen > FLT_EPSILON)
    {
        angleDEG = ((T)2) * acos(_w) * SL_RAD2DEG;
        T len = sqrt(sqrLen);
        axis.x = _x / len;
        axis.y = _y / len;
        axis.z = _z / len;
    }
    else
    {
        // Angle is 0 (mod 2*pi), so any axis will do.
        angleDEG = (T)0;
        axis.x   = (T)1;
        axis.y   = (T)0;
        axis.z   = (T)0;
    }
}

//-----------------------------------------------------------------------------
template<class T>
SLQuat4<T>& SLQuat4<T>::operator= (const SLQuat4<T> q)
{
    _x = q._x;
    _y = q._y;
    _z = q._z;
    _w = q._w;
    return(*this);
}
//-----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::operator- (const SLQuat4<T>& q) const
{
    return SLQuat4<T>(_x - q._x, _y - q._y, _z - q._z, _w - q._w);
}

//-----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::operator+ (const SLQuat4<T>& q) const
{
    return SLQuat4<T>(_x + q._x, _y + q._y, _z + q._z, _w + q._w);
}

//-----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::operator* (const SLQuat4<T>& q) const
{
    return rotated(q);
}

//-----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::operator* (const T s) const
{
    return scaled(s);
}

//-----------------------------------------------------------------------------
template <class T>
SLVec3<T> SLQuat4<T>::operator* (const SLVec3<T>& v) const
{
    // nVidia SDK implementation
    SLVec3<T> uv, uuv;
    SLVec3<T> qvec(_x, _y, _z);
    uv = qvec.crossProduct(v);
    uuv = qvec.crossProduct(uv);
    uv *= (2.0f * _w);
    uuv *= 2.0f;
    
    return v + uv + uuv;
}

//-----------------------------------------------------------------------------
template <class T>
bool SLQuat4<T>::operator== (const SLQuat4<T>& q) const
{
    return _x == q._x && _y == q._y && _z == q._z && _w == q._w;
}

//-----------------------------------------------------------------------------
template <class T>
bool SLQuat4<T>::operator!= (const SLQuat4<T>& q) const
{
    return !(*this == q);
}
//-----------------------------------------------------------------------------
template<class T>
SLQuat4<T>& SLQuat4<T>::operator*= (const SLQuat4<T>& q2)
{  
    SLQuat4<T> q;
    SLQuat4<T>& q1 = *this;
    
    q._w = q1._w * q2._w - q1._x * q2._x - q1._y * q2._y - q1._z * q2._z;
    q._x = q1._w * q2._x + q1._x * q2._w + q1._y * q2._z - q1._z * q2._y;
    q._y = q1._w * q2._y + q1._y * q2._w + q1._z * q2._x - q1._x * q2._z;
    q._z = q1._w * q2._z + q1._z * q2._w + q1._x * q2._y - q1._y * q2._x;
    
    q.normalize();
    *this = q;
    return *this;
}
//-----------------------------------------------------------------------------
template<class T>
SLQuat4<T>& SLQuat4<T>::operator*= (const T s)
{  
    _x *= s;
    _y *= s;
    _z *= s;
    _w *= s;
   return *this;
}

//-----------------------------------------------------------------------------
template <class T>
inline T SLQuat4<T>::dot(const SLQuat4<T>& q) const
{
    return _x * q._x + _y * q._y + _z * q._z + _w * q._w;
}

//-----------------------------------------------------------------------------
template <class T>
inline T SLQuat4<T>::length() const
{
    return sqrt(_x*_x + _y*_y + _z*_z + _w*_w);
}

//-----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::normalized() const
{
    T len = length();
    SLQuat4<T> norm;

    if (len > FLT_EPSILON)
    {
        T invLen = ((T)1)/len;
        norm._x = _x * invLen;
        norm._y = _y * invLen;
        norm._z = _z * invLen;
        norm._w = _w * invLen;
    } else
    {  // set invalid result to flag the error.
        norm._x = (T)0;
        norm._y = (T)0;
        norm._z = (T)0;
        norm._w = (T)0;
    }

    return norm;
}

//-----------------------------------------------------------------------------
template <class T>
inline T SLQuat4<T>::normalize()
{
    T len = length();

    if (len > FLT_EPSILON)
    {
        T invLen = ((T)1)/len;
        _x *= invLen;
        _y *= invLen;
        _z *= invLen;
        _w *= invLen;
    } else
    {  // set invalid result to flag the error.
        len = (T)0;
        _x = (T)0;
        _y = (T)0;
        _z = (T)0;
        _w = (T)0;
    }

    return len;
}

//----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::inverted () const
{
    SLQuat4<T> inverse;
    T norm = _x*_x + _y*_y + _z*_z + _w*_w;

    if (norm > (T)0)
    {  
        // for non-unit quaternions we have to normalize
        T invNorm  = ((T)1) / norm;
        inverse._x = -_x * invNorm;
        inverse._y = -_y * invNorm;
        inverse._z = -_z * invNorm;
        inverse._w =  _w * invNorm;
    } else
    {  
        // return an invalid result to flag the error.
        inverse._x = (T)0;
        inverse._y = (T)0;
        inverse._z = (T)0;
        inverse._w = (T)0;
    }

    return inverse;
}

//----------------------------------------------------------------------------
template <class T>
void SLQuat4<T>::invert()
{
    T norm = _x*_x + _y*_y + _z*_z + _w*_w;

    if (norm > (T)0)
    {  
        // for non-unit quaternions we have to normalize
        T invNorm  = ((T)1) / norm;
        _x = -_x * invNorm;
        _y = -_y * invNorm;
        _z = -_z * invNorm;
        _w =  _w * invNorm;
    } else
    {  
        // return an invalid result to flag the error.
        _x = (T)0;
        _y = (T)0;
        _z = (T)0;
        _w = (T)0;
    }
}

//----------------------------------------------------------------------------
template <class T>
SLQuat4<T> SLQuat4<T>::conjugated() const
{  
    // for a unit quaternion the conjugate is equal to the inverse
    return SLQuat4(-_x, -_y, -_z, _w);
}

//----------------------------------------------------------------------------
template <class T>
void SLQuat4<T>::conjugate()
{  
    // for a unit quaternion the conjugate is equal to the inverse
    _x = -_x;
    _y = -_y;
    _z = -_z;
}

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T> SLQuat4<T>::rotated(const SLQuat4<T>& b) const
{
    SLQuat4<T> q;
    q._w = _w*b._w - _x*b._x - _y*b._y - _z*b._z;
    q._x = _w*b._x + _x*b._w + _y*b._z - _z*b._y;
    q._y = _w*b._y + _y*b._w + _z*b._x - _x*b._z;
    q._z = _w*b._z + _z*b._w + _x*b._y - _y*b._x;
    q.normalize();
    return q;
}

//-----------------------------------------------------------------------------
template <class T>
inline void SLQuat4<T>::rotate(const SLQuat4<T>& q2)
{
    SLQuat4<T> q;
    SLQuat4<T>& q1 = *this;
    
    q._w = q1._w * q2._w - q1._x * q2._x - q1._y * q2._y - q1._z * q2._z;
    q._x = q1._w * q2._x + q1._x * q2._w + q1._y * q2._z - q1._z * q2._y;
    q._y = q1._w * q2._y + q1._y * q2._w + q1._z * q2._x - q1._x * q2._z;
    q._z = q1._w * q2._z + q1._z * q2._w + q1._x * q2._y - q1._y * q2._x;
    
    q.normalize();
    *this = q;
}

template <class T>
inline SLVec3<T> SLQuat4<T>::rotate(const SLVec3<T>& vec) const
{
    SLMat3<T> rot = toMat3();
    return rot * vec;
}

//-----------------------------------------------------------------------------
template <class T>
inline SLQuat4<T> SLQuat4<T>::scaled(T s) const
{
    return SLQuat4<T>(_x * s, _y * s, _z * s, _w * s);
}

//-----------------------------------------------------------------------------
template <class T>
inline void SLQuat4<T>::scale(T s)
{
    _x *= s;
    _y *= s;
    _z *= s;
    _w *= s;
}

//-----------------------------------------------------------------------------
//! Linear interpolation
template <class T>
inline SLQuat4<T> SLQuat4<T>::lerp(const SLQuat4<T>& q2, const T t) const
{  
    SLQuat4<T> q = scaled(1-t) + q2.scaled(t);
    q.normalize();
    return q;
}
//-----------------------------------------------------------------------------
//! Linear interpolation
template <class T>
inline void SLQuat4<T>::lerp(const SLQuat4<T>& q1, 
                             const SLQuat4<T>& q2, const T t)
{  
    *this = q1.scaled(1-t) + q2.scaled(t);
    normalize();
}

//-----------------------------------------------------------------------------
//! Spherical linear interpolation
template <class T>
inline SLQuat4<T> SLQuat4<T>::slerp(const SLQuat4<T>& q2, const T t) const
{
    /// @todo clean up the code below and find a working algorithm (or check the original shoemake implementation for errors)
    // Not 100% slerp, uses lerp in case of close angle! note the todo above this line!
    SLfloat factor = t;
	// calc cosine theta
	T cosom = _x * q2._x + _y * q2._y + _z * q2._z + _w * q2._w;

	// adjust signs (if necessary)
	SLQuat4<T> endCpy = q2;
	if( cosom < static_cast<T>(0.0))
	{
		cosom = -cosom;
		endCpy._x = -endCpy._x;   // Reverse all signs
		endCpy._y = -endCpy._y;
		endCpy._z = -endCpy._z;
		endCpy._w = -endCpy._w;
	} 

	// Calculate coefficients
	T sclp, sclq;
	if( (static_cast<T>(1.0) - cosom) > static_cast<T>(0.0001)) // 0.0001 -> some epsillon
	{
		// Standard case (slerp)
		T omega, sinom;
		omega = acos( cosom); // extract theta from dot product's cos theta
		sinom = sin( omega);
		sclp  = sin( (static_cast<T>(1.0) - factor) * omega) / sinom;
		sclq  = sin( factor * omega) / sinom;
	} else
	{
		// Very close, do linear interp (because it's faster)
		sclp = static_cast<T>(1.0) - factor;
		sclq = factor;
	}

    SLQuat4<T> out;
	out._x = sclp * _x + sclq * endCpy._x;
	out._y = sclp * _y + sclq * endCpy._y;
	out._z = sclp * _z + sclq * endCpy._z;
	out._w = sclp * _w + sclq * endCpy._w;
    return out;
    
    /*OLD
    // Ken Shoemake's famous method.
    assert(t>=0 && t<=1 && "Wrong t in SLQuat4::slerp");

    T cosAngle = dot(q2);
    
    if (cosAngle > 1 - FLT_EPSILON) 
    {
        SLQuat4<T> result = q2 + (*this - q2).scaled(t);
        result.normalize();
        return result;
    }
    
    if (cosAngle < 0) cosAngle = 0;
    if (cosAngle > 1) cosAngle = 1;
    
    T theta0 = acos(cosAngle);
    T theta  = theta0 * t;
    
    SLQuat4<T> v2 = (q2 - scaled(cosAngle));
    v2.normalize();
    
    SLQuat4<T> q = scaled(cos(theta)) + v2.scaled(sin(theta));
    q.normalize();
    return q;
    */
}
//-----------------------------------------------------------------------------
//! Spherical linear interpolation
template <class T>
inline void SLQuat4<T>::slerp(const SLQuat4<T>& q1,
                              const SLQuat4<T>& q2, const T t)
{
    // Ken Shoemake's famous method.
    assert(t>=0 && t<=1 && "Wrong t in SLQuat4::slerp");

    T cosAngle = q1.dot(q2);
    
    if (cosAngle > 1 - FLT_EPSILON) 
    {
        *this = q2 + (q1 - q2).scaled(t);
        normalize();
        return;
    }
    
    if (cosAngle < 0) cosAngle = 0;
    if (cosAngle > 1) cosAngle = 1;
    
    T theta0 = acos(cosAngle);
    T theta  = theta0 * t;
    
    SLQuat4<T> v2 = (q2 - q1.scaled(cosAngle));
    v2.normalize();
    
    *this = q1.scaled(cos(theta)) + v2.scaled(sin(theta));
    normalize();
}

//-----------------------------------------------------------------------------
typedef SLQuat4<SLfloat> SLQuat4f;
//-----------------------------------------------------------------------------
#endif
