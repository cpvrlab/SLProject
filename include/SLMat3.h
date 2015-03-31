//#############################################################################
//  File:      math/SLMat3.h
//  Purpose:   3 x 3 Matrix for linear 3D transformations
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMAT3_H
#define SLMAT3_H

#include <SL.h>

//-----------------------------------------------------------------------------
//! 3x3 matrix template class
/*!  
Implements a 3 by 3 matrix template. 9 floats were used instead of the normal 
[3][3] array. The order is columnwise as in OpenGL

| 0  3  6 |
| 1  4  7 |
| 2  5  8 |

\n type definitions for 3x3 matrice:
\n Use SLMat3f for a specific float type 3x3 matrix
\n Use SLMat3d for a specific double type 3x3 matrix
*/
template<class T>
class SLMat3
{
    public:
        T           _m[9];    

                    // Overloaded constructors                        
                    SLMat3      ();                     //!< Sets identity matrix
                    SLMat3      (const SLMat3& A);      //!< Sets mat by other SLMat3
                    SLMat3      (const T *M);           //!< Sets matrix by array
                    SLMat3      (const T M0, const T M3, const T M6,
                                const T M1, const T M4, const T M7,
                                const T M2, const T M5, const T M8);    //!< Sets matrix by components
                    SLMat3      (const T angleDEG, 
                                const T axis_x, 
                                const T axis_y, 
                                const T axis_z);       //!< Sets rotate matrix from axis & angle
                    SLMat3      (const T angleDEG, 
                                const SLVec3<T> axis); //!< Sets rotate matrix
                    SLMat3      (const T scale_xyz);    //!< Sets uniform scaling matrix
                    SLMat3      (const T angleZDEG,
                                const T angleYDEG,
                                const T angleXDEG);    //!< Sets rotation matrix from Euler angles
      
        // Setters
        void        setMatrix   (const SLMat3& A);
        void        setMatrix   (const SLMat3* A);
        void        setMatrix   (const T* M);
        void        setMatrix   (T M0, T M3, T M6,
                                T M1, T M4, T M7,
                                T M2, T M5, T M8);
      
        // Overloaded operators                         
        SLMat3<T>&  operator=   (const SLMat3& A);
        SLMat3<T>&  operator*=  (const SLMat3& A);
        SLMat3<T>&  operator=   (const T* a);
        SLMat3<T>   operator*   (const SLMat3& A) const;
        SLVec3<T>   operator*   (const SLVec3<T>& v) const;
        SLMat3<T>   operator*   (T a) const;           //!< scalar multiplication
        SLMat3<T>&  operator*=  (T a);                 //!< scalar multiplication
        SLMat3<T>   operator/   (T a) const;           //!< scalar division
        SLMat3<T>&  operator/=  (T a);                 //!< scalar division
                    operator const T*() const {return _m;}
                    operator T* (){return _m;};
        T&          operator    ()(SLint row, SLint col)      {return _m[3*col+row];}
        const T&    operator    ()(SLint row, SLint col)const {return _m[3*col+row];};

        //! Sets the rotation components      
        void        rotation    (const T angleDEG, 
                                const SLVec3<T>& axis);
        void        rotation    (const T angleDEG, 
                                const T axisx, const T axisy, const T axisz);
        void        rotation    (const T zAngleRAD, const T yAngleRAD, const T xAngleRAD);
      
        //! Sets the scaling components 
        void        scale       (const T sx, const T sy, const T sz);
        void        scale       (const SLVec3<T>& s);
        void        scale       (const T s);

        // Misc. methods
        void        identity    ();
        void        transpose   ();
        void        invert      ();
        SLMat3<T>   inverse     (); 
        T           trace       () const;
        T           det         () const;

        void        toAngleAxis       (T& angleDEG, SLVec3<T>& axis) const;
        void        toEulerAnglesZYX  (T& zRotRAD, T& yRotRAD, T& xRotRAD);
        void        fromEulerAnglesZXZ(const double angle1RAD,
                                        const double angle2RAD,
                                        const double angle3RAD);
        void        fromEulerAnglesXYZ(const double angle1RAD,
                                        const double angle2RAD,
                                        const double angle3RAD);
        void        print       (const SLchar* str) const;
      
        static void swap        (T& a, T& b) {T t; t=a; a=b; b=t;}
};

//-----------------------------------------------------------------------------
// constructors
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3()
{
    identity(); 
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3(const SLMat3& A) 
{
    setMatrix(A);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3(T M0, T M3, T M6,
                  T M1, T M4, T M7,
                  T M2, T M5, T M8)
{  
    _m[0]=M0; _m[3]=M3; _m[6]=M6; 
    _m[1]=M1; _m[4]=M4; _m[7]=M7; 
    _m[2]=M2; _m[5]=M5; _m[8]=M8;
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3(const T angleDEG, const T axisx, const T axisy, const T axisz)
{
    rotation(angleDEG, axisx, axisy, axisz);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3(const T angleDEG, const SLVec3<T> axis)
{
    rotation(angleDEG, axis);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3(const T scale_xyz)
{
    scale(scale_xyz);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>::SLMat3(const T angleZDEG,
                  const T angleYDEG,
                  const T angleXDEG)
{
    fromEulerAnglesXYZ(angleXDEG*SL_DEG2RAD,
                       angleYDEG*SL_DEG2RAD,
                       angleZDEG*SL_DEG2RAD);
}


//-----------------------------------------------------------------------------
// matrix-matrix and matrix-vector operators
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>& SLMat3<T>::operator =(const SLMat3& A)
{
    _m[0]=A._m[0]; _m[3]=A._m[3]; _m[6]=A._m[6]; 
    _m[1]=A._m[1]; _m[4]=A._m[4]; _m[7]=A._m[7]; 
    _m[2]=A._m[2]; _m[5]=A._m[5]; _m[8]=A._m[8];
    return(*this);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T>& SLMat3<T>::operator =(const T* a) 
{
    for (int i=0; i<9; ++i) _m[i] = a[i];
}
//-----------------------------------------------------------------------------
// matrix multiplication
template<class T>
SLMat3<T> SLMat3<T>::operator *(const SLMat3& A) const
{
    SLMat3<T> NewM(_m[0]*A._m[0] + _m[3]*A._m[1] + _m[6]*A._m[2],      // ROW 1
                   _m[0]*A._m[3] + _m[3]*A._m[4] + _m[6]*A._m[5],
                   _m[0]*A._m[6] + _m[3]*A._m[7] + _m[6]*A._m[8],
                   _m[1]*A._m[0] + _m[4]*A._m[1] + _m[7]*A._m[2],      // ROW 2
                   _m[1]*A._m[3] + _m[4]*A._m[4] + _m[7]*A._m[5],
                   _m[1]*A._m[6] + _m[4]*A._m[7] + _m[7]*A._m[8],
                   _m[2]*A._m[0] + _m[5]*A._m[1] + _m[8]*A._m[2],      // ROW 3
                   _m[2]*A._m[3] + _m[5]*A._m[4] + _m[8]*A._m[5],
                   _m[2]*A._m[6] + _m[5]*A._m[7] + _m[8]*A._m[8]);              
  return(NewM);
}
//-----------------------------------------------------------------------------
// matrix multiplication
template<class T>
SLMat3<T>& SLMat3<T>::operator *=(const SLMat3& A)
{  
    setMatrix(_m[0]*A._m[0] + _m[3]*A._m[1] + _m[6]*A._m[2],      // ROW 1
              _m[0]*A._m[3] + _m[3]*A._m[4] + _m[6]*A._m[5],
              _m[0]*A._m[6] + _m[3]*A._m[7] + _m[6]*A._m[8],
              _m[1]*A._m[0] + _m[4]*A._m[1] + _m[7]*A._m[2],      // ROW 2
              _m[1]*A._m[3] + _m[4]*A._m[4] + _m[7]*A._m[5],
              _m[1]*A._m[6] + _m[4]*A._m[7] + _m[7]*A._m[8],
              _m[2]*A._m[0] + _m[5]*A._m[1] + _m[8]*A._m[2],      // ROW 3
              _m[2]*A._m[3] + _m[5]*A._m[4] + _m[8]*A._m[5],
              _m[2]*A._m[6] + _m[5]*A._m[7] + _m[8]*A._m[8]);  
    return *this;
}
//-----------------------------------------------------------------------------
template<class T>
SLVec3<T> SLMat3<T>::operator *(const SLVec3<T>& v) const
{
    SLVec3<T> NewV;
    NewV.x = _m[0]*v.x + _m[3]*v.y + _m[6]*v.z;
    NewV.y = _m[1]*v.x + _m[4]*v.y + _m[7]*v.z;
    NewV.z = _m[2]*v.x + _m[5]*v.y + _m[8]*v.z;
    return(NewV);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat3<T> SLMat3<T>::operator *(T a) const              
{
    SLMat3<T> NewM(_m[0] * a, _m[3] * a, _m[6] * a, 
                   _m[1] * a, _m[4] * a, _m[7] * a,
                   _m[2] * a, _m[5] * a, _m[8] * a);
    return NewM;
}
//-----------------------------------------------------------------------------
// scalar multiplication
template<class T>
SLMat3<T>& SLMat3<T>::operator *=(T a)
{
    for (int i=0; i<9; ++i) _m[i] *= a;
    return *this;
}
//-----------------------------------------------------------------------------
// scalar division
template<class T>
SLMat3<T> SLMat3<T>::operator /(T a) const              
{
    SLMat3<T> NewM(_m[0] / a, _m[3] / a, _m[6] / a, 
                   _m[1] / a, _m[4] / a, _m[7] / a,
                   _m[2] / a, _m[5] / a, _m[8] / a);
    return NewM;
}
//-----------------------------------------------------------------------------
// scalar division
template<class T>
SLMat3<T>& SLMat3<T>::operator /=(T a)
{
    for (int i=0; i<9; ++i) _m[i] /= a;
    return *this;
}
//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::setMatrix(const SLMat3& A)
{
    for (int i=0; i<9; ++i) _m[i] = A._m[i];
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::setMatrix(const SLMat3* A)
{
    for (int i=0; i<9; ++i) _m[i] = A->_m[i];
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::setMatrix(const T* M)
{
    for (int i=0; i<9; ++i) _m[i] = M[i];
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::setMatrix (T M0, T M3, T M6,
                           T M1, T M4, T M7,
                           T M2, T M5, T M8)
{
    _m[0]=M0; _m[3]=M3; _m[6]=M6; 
    _m[1]=M1; _m[4]=M4; _m[7]=M7; 
    _m[2]=M2; _m[5]=M5; _m[8]=M8;
}
//-----------------------------------------------------------------------------
// Standard Matrix Operations
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::identity()
{
    _m[0]=_m[4]=_m[8]=1;
    _m[1]=_m[2]=_m[3]=_m[5]=_m[6]=_m[7]=0;
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::transpose()
{
    swap(_m[1],_m[3]);
    swap(_m[2],_m[6]);
    swap(_m[5],_m[7]);
}
//-----------------------------------------------------------------------------
//! Inverts the matrix
template<class T>
void SLMat3<T>::invert()
{
    setMatrix(inverse());
}
//-----------------------------------------------------------------------------
//! Returns the inverse of the matrix
template<class T>
SLMat3<T> SLMat3<T>::inverse()
{
    // Compute determinant as early as possible using these cofactors.      
    T d = det();

    if (fabs(d) < FLT_EPSILON) 
    {  cout << "3x3-Matrix is singular. Inversion impossible." << endl;
        exit(0);
    }

    SLMat3<T> i;
    i._m[0] = _m[4]*_m[8] - _m[7]*_m[5];
    i._m[1] = _m[7]*_m[2] - _m[1]*_m[8];
    i._m[2] = _m[1]*_m[5] - _m[4]*_m[2];
    i._m[3] = _m[6]*_m[5] - _m[3]*_m[8];
    i._m[4] = _m[0]*_m[8] - _m[6]*_m[2];
    i._m[5] = _m[3]*_m[2] - _m[0]*_m[5];
    i._m[6] = _m[3]*_m[7] - _m[6]*_m[4];
    i._m[7] = _m[6]*_m[1] - _m[0]*_m[7];
    i._m[8] = _m[0]*_m[4] - _m[3]*_m[1];

    i /= d;
    return i;
}
//-----------------------------------------------------------------------------
// Standard Matrix Transformations
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::rotation(const T angleDEG, const SLVec3<T>& axis)
{
    rotation(angleDEG, axis.x, axis.y, axis.z);
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::rotation(const T angleDEG, 
                         const T axisx, const T axisy, const T axisz)
{
    T angleRAD = (T)angleDEG*SL_DEG2RAD;
    T ca = (T)cos(angleRAD);
    T sa = (T)sin(angleRAD);

    if (axisx==1 && axisy==0 && axisz==0)               // about x-axis
    {   _m[0]=1; _m[3]=0;  _m[6]=0;   
        _m[1]=0; _m[4]=ca; _m[7]=-sa; 
        _m[2]=0; _m[5]=sa; _m[8]=ca; 
    } else 
    if (axisx==0 && axisy==1 && axisz==0)               // about y-axis
    {   _m[0]=ca;  _m[3]=0; _m[6]=sa; 
        _m[1]=0;   _m[4]=1; _m[7]=0;  
        _m[2]=-sa; _m[5]=0; _m[8]=ca;
    } else 
    if (axisx==0 && axisy==0 && axisz==1)               // about z-axis
    {   _m[0]=ca; _m[3]=-sa; _m[6]=0; 
        _m[1]=sa; _m[4]=ca;  _m[7]=0; 
        _m[2]=0;  _m[5]=0;   _m[8]=1;
    } else                                             // arbitrary axis
    {   T l = axisx*axisx + axisy*axisy + axisz*axisz;  // length squared
        T x, y, z;
        x=axisx, y=axisy, z=axisz;
        if ((l > T(1.0001) || l < T(0.9999)) && l!=0)
        {  l=T(1.0)/sqrt(l);
            x*=l; y*=l; z*=l;
        }
        T xy=x*y, yz=y*z, xz=x*z, xx=x*x, yy=y*y, zz=z*z;
        _m[0]=xx + ca*(1-xx);     _m[3]=xy - xy*ca - z*sa;  _m[6]=xz - xz*ca + y*sa;
        _m[1]=xy - xy*ca + z*sa;  _m[4]=yy + ca*(1-yy);     _m[7]=yz - yz*ca - x*sa;
        _m[2]=xz - xz*ca - y*sa;  _m[5]=yz - yz*ca + x*sa;  _m[8]=zz + ca*(1-zz);
    }
}
//-----------------------------------------------------------------------------
/*!
Sets the matrix as a rotation matrix from the 3 euler angles in radians around
the z-axis, y-axis & x-axis.
See Van Verth: Essential Math for Games, chapter 5: Orientation Representation
*/
template<class T>
void SLMat3<T>::rotation(const T zAngleRAD, const T yAngleRAD, const T xAngleRAD)
{
    T Cx = (T)cos(xAngleRAD); T Sx = (T)sin(xAngleRAD);
    T Cy = (T)cos(yAngleRAD); T Sy = (T)sin(yAngleRAD);
    T Cz = (T)cos(zAngleRAD); T Sz = (T)sin(zAngleRAD);

    _m[0] =  (Cy * Cz);
    _m[3] = -(Cy * Sz);
    _m[6] =  Sy;

    _m[1] =  (Sx * Sy * Cz) + (Cx * Sz);
    _m[4] = -(Sx * Sy * Sz) + (Cx * Cz);
    _m[7] = -(Sx * Cy);

    _m[2] = -(Cx * Sy * Cz) + (Sx * Sz);
    _m[5] =  (Cx * Sy * Sz) + (Sx * Cz);
    _m[8] =  (Cx * Cy);
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::scale(T sx, T sy, T sz)
{
    _m[0]=sx; _m[3]=0;  _m[6]=0;
    _m[1]=0;  _m[4]=sy; _m[7]=0;
    _m[2]=0;  _m[5]=0;  _m[8]=sz;
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::scale(const SLVec3<T>& s)
{
    _m[0]=s.x; _m[3]=0;   _m[6]=0;  
    _m[1]=0;   _m[4]=s.y; _m[7]=0;  
    _m[2]=0;   _m[5]=0;   _m[8]=s.z;
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::scale(const T s)
{
    _m[0]=s; _m[3]=0; _m[6]=0;  
    _m[1]=0; _m[4]=s; _m[7]=0;  
    _m[2]=0; _m[5]=0; _m[8]=s;
}
//-----------------------------------------------------------------------------
template<class T>
inline T SLMat3<T>::trace() const
{
    return _m[0] + _m[4] + _m[8];
}
//-----------------------------------------------------------------------------
//! det returns the determinant
template<class T>
inline T SLMat3<T>::det() const
{
    return _m[0]*(_m[4]*_m[8] - _m[7]*_m[5]) -
           _m[3]*(_m[1]*_m[8] - _m[7]*_m[2]) +
           _m[6]*(_m[1]*_m[5] - _m[4]*_m[2]);
}
//-----------------------------------------------------------------------------
//! Conversion to axis and angle in radians
/*!
The matrix must be a rotation matrix for this functions to be valid. The last 
function uses Gram-Schmidt orthonormalization applied to the columns of the 
rotation matrix. The angle must be in radians, not degrees.
*/
template <typename T>
void SLMat3<T>::toAngleAxis(T& angleDEG, SLVec3<T>& axis) const
{
    // Let (x,y,z) be the unit-length axis and let A be an angle of rotation.
    // The rotation matrix is R = I + sin(A)*P + (1-cos(A))*P^2 where
    // I is the identity and
    //
    //       +-        -+
    //       |  0 -z +y |
    //   P = | +z  0 -x |
    //       | -y +x  0 |
    //       +-        -+
    //
    // If A > 0, R represents a counterclockwise rotation about the axis in
    // the sense of looking from the tip of the axis vector towards the
    // origin.  Some algebra will show that
    //
    //   cos(A) = (trace(R)-1)/2  and  R - R^t = 2*sin(A)*P
    //
    // In the event that A = pi, R-R^t = 0 which prevents us from extracting
    // the axis through P.  Instead note that R = I+2*P^2 when A = pi, so
    // P^2 = (R-I)/2.  The diagonal entries of P^2 are x^2-1, y^2-1, and
    // z^2-1.  We can solve these for axis (x,y,z).  Because the angle is pi,
    // it does not matter which sign you choose on the square roots.
    
    T tr       = trace();
    T cs       = ((T)0.5) * (tr - (T)1);
    T angleRAD = acos(cs);  // in [0,PI]

    if (angleRAD > (T)0)
    {
        if (angleRAD < SL_PI)
        {
            axis.x = _m[5] - _m[7];
            axis.y = _m[6] - _m[2];
            axis.z = _m[1] - _m[3];
            axis.normalize();
        }
        else
        {
            // angle is PI
            T halfInverse;
            if (_m[0] >= _m[4])
            {
                // r00 >= r11
                if (_m[0] >= _m[8])
                {
                    // r00 is maximum diagonal term
                    axis.x = ((T)0.5)*sqrt((T)1 + _m[0] - _m[4] - _m[8]);
                    halfInverse = ((T)0.5)/axis.x;
                    axis.y = halfInverse*_m[3];
                    axis.z = halfInverse*_m[6];
                }
                else
                {
                    // r22 is maximum diagonal term
                    axis.z = ((T)0.5)*sqrt((T)1 + _m[8] - _m[0] - _m[4]);
                    halfInverse = ((T)0.5)/axis.z;
                    axis.x = halfInverse*_m[6];
                    axis.y = halfInverse*_m[7];
                }
            }
            else
            {
                // r11 > r00
                if (_m[4] >= _m[8])
                {
                    // r11 is maximum diagonal term
                    axis.y = ((T)0.5)*sqrt((T)1 + _m[4] - _m[0] - _m[8]);
                    halfInverse  = ((T)0.5)/axis.y;
                    axis.x = halfInverse*_m[3];
                    axis.z = halfInverse*_m[7];
                }
                else
                {
                    // r22 is maximum diagonal term
                    axis.z = ((T)0.5)*sqrt((T)1 + _m[8] - _m[0] - _m[4]);
                    halfInverse = ((T)0.5)/axis.z;
                    axis.x = halfInverse*_m[6];
                    axis.y = halfInverse*_m[7];
                }
            }
        }
    }
    else
    {
        // The angle is 0 and the matrix is the identity.  Any axis will
        // work, so just use the x-axis.
        axis.x = (T)1;
        axis.y = (T)0;
        axis.z = (T)0;
    }
    angleDEG = angleRAD * SL_RAD2DEG;
}

//-----------------------------------------------------------------------------
/*!
Gets one set of possible z-y-x euler angles that will generate this matrix
Source: Essential Mathematics for Games and Interactive Applications
A Programmer’s Guide 2nd edition by James M. Van Verth and Lars M. Bishop
*/
template<class T>
void SLMat3<T>::toEulerAnglesZYX(T& zRotRAD, T& yRotRAD, T& xRotRAD)
{
    T Cx, Sx;
    T Cy, Sy;
    T Cz, Sz;

    Sy = _m[6];
    Cy = (T)sqrt(1.0 - Sy*Sy);
    
    // normal case
    if (SL_abs(Cy) > FLT_EPSILON)
    {
        T factor = 1.0 / Cy;
        Sx = -_m[7]*factor;
        Cx =  _m[8]*factor;
        Sz = -_m[3]*factor;
        Cz =  _m[0]*factor;
    }
    else // x and z axes aligned
    {
        Sz = 0.0f;
        Cz = 1.0f;
        Sx = _m[5];
        Cx = _m[4];
    }

    zRotRAD = atan2f(Sz, Cz);
    yRotRAD = atan2f(Sy, Cy);
    xRotRAD = atan2f(Sx, Cx);
}

//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a rotation matrix from the 3 euler angles
in radians around the z-axis, x-axis & z-axis.
See: http://en.wikipedia.org/wiki/Euler_angles
*/
template<class T>
void SLMat3<T>::fromEulerAnglesZXZ(const double angle1RAD,
                                   const double angle2RAD,
                                   const double angle3RAD)
{
    double s1 = sin(angle1RAD), c1 = cos(angle1RAD);
    double s2 = sin(angle2RAD), c2 = cos(angle2RAD);
    double s3 = sin(angle3RAD), c3 = cos(angle3RAD);

    _m[0]=(T)( c1*c3 - s1*c2*s3); _m[3]=(T)( s1*c3 + c1*c2*s3);  _m[6] =(T)( s2*s3);
    _m[1]=(T)(-c1*s3 - s1*c2*c3); _m[4]=(T)( c1*c2*c3 - s1*s3);  _m[7] =(T)( s2*c3);
    _m[2]=(T)( s1*s2);            _m[5]=(T)(-c1*s2);             _m[8] =(T)( c2);
}
//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a rotation matrix from the 3 euler angles
in radians around the z-axis, y-axis & x-axis.
See: http://en.wikipedia.org/wiki/Euler_angles
*/
template<class T>
void SLMat3<T>::fromEulerAnglesXYZ(const double angle1RAD,
                                   const double angle2RAD,
                                   const double angle3RAD)
{
    double s1 = sin(angle1RAD), c1 = cos(angle1RAD);
    double s2 = sin(angle2RAD), c2 = cos(angle2RAD);
    double s3 = sin(angle3RAD), c3 = cos(angle3RAD);

    _m[0]=(T) (c2*c3);              _m[3]=(T)-(c2*s3);              _m[6] =(T) s2;
    _m[1]=(T) (s1*s2*c3) + (c1*s3); _m[4]=(T)-(s1*s2*s3) + (c1*c3); _m[7] =(T)-(s1*c2);
    _m[2]=(T)-(c1*s2*c3) + (s1*s3); _m[5]=(T) (c1*s2*s3) + (s1*c3); _m[8] =(T) (c1*c2);
}

//-----------------------------------------------------------------------------
// Handy matrix printing routine.
//-----------------------------------------------------------------------------
template<class T>
void SLMat3<T>::print(const SLchar* str) const
{
    if (str) SL_LOG("%s",str);
    SL_LOG("% 3.3f % 3.3f % 3.3f\n", _m[0],_m[3],_m[6]);
    SL_LOG("% 3.3f % 3.3f % 3.3f\n", _m[1],_m[4],_m[7]);
    SL_LOG("% 3.3f % 3.3f % 3.3f\n", _m[2],_m[5],_m[8]);
}
//-----------------------------------------------------------------------------
typedef SLMat3<SLfloat>  SLMat3f;
#ifdef SL_HAS_DOUBLE
typedef SLMat3<SLdouble> SLMat3d;
#endif
//-----------------------------------------------------------------------------
#endif
