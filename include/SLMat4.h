//#############################################################################
//  File:      math/SLMat4.h
//  Purpose:   4 x 4 Matrix for affine transformations
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMAT4_H
#define SLMAT4_H

#include <SL.h>
#include <SLMath.h>
#include <SLMat3.h>
#include <SLVec4.h>
#include <Shoemake/EulerAngles.h>
#include <Shoemake/Decompose.h>

extern void decomp_affine(HMatrix A, AffineParts *parts);

//-----------------------------------------------------------------------------
//! 4x4 matrix template class
/*!  
Implements a 4 by 4 matrix template class. An array of 16 floats or double is
used instead of a 2D array [4][4] to be compliant with OpenGL. 
The index layout is as follows:
<PRE>
     | 0  4  8 12 |
     | 1  5  9 13 |
 M = | 2  6 10 14 |
     | 3  7 11 15 |

</PRE>
Vectors are interpreted as column vectors when applying matrix multiplications. 
This means a vector is as a single column, 4-row matrix. The result is that the 
transformations implemented by the matrices happens right-to-left e.g. if 
vector V is to be transformed by M1 then M2 then M3, the calculation would 
be M3 * M2 * M1 * V. The order that matrices are concatenated is vital 
since matrix multiplication is not commutative, i.e. you can get a different 
result if you concatenate in the wrong order.
The use of column vectors and right-to-left ordering is the standard in most 
mathematical texts, and is the same as used in OpenGL. It is, however, the 
opposite of Direct3D, which has inexplicably chosen to differ from the 
accepted standard and uses row vectors and left-to-right matrix multiplication.
*/

template<class T>
class SLMat4
{
    public:
        // Constructors
                    SLMat4      ();                     //!< Sets identity matrix
                    SLMat4      (const SLMat4& A);      //!< Sets mat by other SLMat4
                    SLMat4      (const T* M);           //!< Sets mat by array
                    SLMat4      (const T M0, const T M4, const T M8,  const T M12,
                                 const T M1, const T M5, const T M9,  const T M13,
                                 const T M2, const T M6, const T M10, const T M14,
                                 const T M3, const T M7, const T M11, const T M15);
                    SLMat4      (const T tx,               
                                 const T ty,
                                 const T tz);           //!< Sets translate matrix
                    SLMat4      (const T degAng,           
                                 const T axis_x,
                                 const T axis_y,
                                 const T axis_z);       //!< Sets rotation matrix
                    SLMat4      (const T scale_xyz);    //!< Sets scaling matrix
                    SLMat4      (const SLVec3<T>& fromUnitVec,
                                 const SLVec3<T>& toUnitVec); //!< Sets rotation matrix
                    SLMat4      (const SLVec3<T>& translation,
                                 const SLMat3<T>& rotation,
                                 const SLVec3<T>& scale); //!< Set matrix by translation, rotation & scale
         
        // Setters
        void        setMatrix   (const SLMat4& A);      //!< Set matrix by other matrix
        void        setMatrix   (const SLMat4* A);      //!< Set matrix by other matrix pointer
        void        setMatrix   (const T* M);           //!< Set matrix by float[16] array
        void        setMatrix   (T M0, T M4, T M8 , T M12, 
                                 T M1, T M5, T M9 , T M13,
                                 T M2, T M6, T M10, T M14,
                                 T M3, T M7, T M11, T M15); //!< Set matrix by components
        void        setMatrix   (const SLVec3<T>& translation,
                                 const SLMat3<T>& rotation,
                                 const SLVec3<T>& scale);   //!< Set matrix by translation, rotation & scale

        // Getters
  const T*          m           () const        {return _m;}
        T           m           (int i) const   {assert(i>=0 && i<16); return _m[i];}
        SLMat3<T>   mat3        () const        {SLMat3<T> m3;
                                                 m3.setMatrix(_m[0], _m[4], _m[ 8], 
                                                              _m[1], _m[5], _m[ 9], 
                                                              _m[2], _m[6], _m[10]);
                                                 return m3;}
        // Operators                            
        SLMat4<T>&  operator=   (const SLMat4& A);         //!< assignment operator
        SLMat4<T>   operator*   (const SLMat4& A) const;   //!< matrix-matrix multiplication
        SLMat4<T>&  operator*=  (const SLMat4& A);         //!< matrix-matrix multiplication
        SLMat4<T>   operator+   (const SLMat4& A) const;   //!< matrix-matrix addition
        SLVec3<T>   operator*   (const SLVec3<T>& v) const;//!< SLVec3 mult w. persp div
        SLVec4<T>   operator*   (const SLVec4<T>& v) const;//!< SLVec4 mult
        SLMat4<T>   operator*   (const T a) const;         //!< scalar mult
        SLMat4<T>&  operator*=  (const T a);               //!< scalar mult
        SLMat4<T>   operator/   (const T a) const;         //!< scalar division
        SLMat4<T>&  operator/=  (const T a);               //!< scalar division
        T&          operator    ()(int row, int col)      {return _m[4*col+row];}
  const T&          operator    ()(int row, int col)const {return _m[4*col+row];}
            
        // Transformation corresponding to the equivalent gl* OpenGL function
        // They all set a transformation that is multiplied onto the matrix
        void        multiply    (const SLMat4& A);
        SLVec3<T>   multVec     (const SLVec3<T> v) const;
        SLVec4<T>   multVec     (const SLVec4<T> v) const;
        void        add         (const SLMat4& A);
        void        translate   (const T tx, const T ty, const T tz=0);
        void        translate   (const SLVec2<T>& t);
        void        translate   (const SLVec3<T>& t);
        void        rotate      (const T degAng, 
                                const T axisx, const T axisy, const T axisz);
        void        rotate      (const T degAng, 
                                 const SLVec3<T>& axis);
        void        rotate      (const SLVec3<T>& fromUnitVec,
                                 const SLVec3<T>& toUnitVec);
        void        scale       (const T sxyz);
        void        scale       (const T sx, const T sy, const T sz);
        void        scale       (const SLVec3<T>& sxyz);
         
        // Defines a view frustum projection matrix equivalent to glFrustum
        void        frustum     (const T l, const T r, const T b, const T t, 
                                 const T n, const T f);
        // Defines a perspective projection matrix with a field of view angle 
        void        perspective (const T fov, const T aspect, 
                                 const T n, const T f);
        // Defines a orthographic projection matrix with a field of view angle 
        void        ortho       (const T l, const T r, const T b, const T t, 
                                 const T n, const T f);
        // Defines the viewport matrix
        void        viewport    (const T x, const T y, const T ww, const T wh, 
                                 const T n=0.0f, const T f=1.0f);
        // Defines the a view matrix as the corresponding gluLookAt function
        void        lookAt      (const T EyeX, const T EyeY, const T EyeZ,
                                 const T AtX=0,const T AtY=0,const T AtZ=0,
                                 const T UpX=0,const T UpY=0,const T UpZ=0);
        void        lookAt      (const SLVec3<T>& Eye, 
                                 const SLVec3<T>& At=SLVec3<T>::ZERO, 
                                 const SLVec3<T>& Up=SLVec3<T>::ZERO);
        // Reads out of the matrix the look at parameters
        void        lookAt      (SLVec3<T>* eye, 
                                 SLVec3<T>* at, 
                                 SLVec3<T>* up, 
                                 SLVec3<T>* right) const;
        // Defines the a model matrix for light positioning
        void        lightAt     (const T PosX, const T PosY, const T PosZ,
                                 const T AtX=0,const T AtY=0,const T AtZ=0,
                                 const T UpX=0,const T UpY=0,const T UpZ=0);
        void        lightAt     (const SLVec3<T>& pos, 
                                 const SLVec3<T>& At=SLVec3<T>::ZERO, 
                                 const SLVec3<T>& Up=SLVec3<T>::ZERO);
        void        posAtUp     (const T PosX,    const T PosY,    const T PosZ,
                                 const T dirAtX=0,const T dirAtY=0,const T dirAtZ=0,
                                 const T dirUpX=0,const T dirUpY=0,const T dirUpZ=0);
        void        posAtUp     (const SLVec3<T>& pos, 
                                 const SLVec3<T>& dirAt=SLVec3<T>::ZERO, 
                                 const SLVec3<T>& dirUp=SLVec3<T>::ZERO);
         
         // Returns the axis and translation vectors
   const SLVec3<T>   translation () const {return SLVec3<T>(_m[12], _m[13], _m[14]);}
   const SLVec3<T>   axisX       () const {return SLVec3<T>(_m[ 0], _m[ 1], _m[ 2]);}
   const SLVec3<T>   axisY       () const {return SLVec3<T>(_m[ 4], _m[ 5], _m[ 6]);}
   const SLVec3<T>   axisZ       () const {return SLVec3<T>(_m[ 8], _m[ 9], _m[10]);}

         // Sets the translation with or without overwriting the linear submatrix
         void        translation (const T tx, const T ty, const T tz, 
                                  const SLbool keepLinear=true);
         void        translation (const SLVec3<T>& t, 
                                  const SLbool keepLinear=true);
         // Sets the rotation with or without overwriting the translation
         void        rotation    (const T degAng,
                                  const SLVec3<T>& axis,
                                  const SLbool keepTranslation=true);
         void        rotation    (const T degAng,
                                  const T axisx, const T axisy, const T axisz,
                                  const SLbool keepTranslation=true);
         void        rotation    (const SLVec3<T>& fromUnitVec,
                                  const SLVec3<T>& toUnitVec);
         // Sets the scaling with or without overwriting the translation
         void        scaling     (const T sxyz, 
                                  const bool keepTrans=true);
         void        scaling     (const SLVec3<T>& sxyz, 
                                  const bool keepTrans=true);
         void        scaling     (const T sx, const T sy, const T sz,
                                  const bool keepTranslation=true);
         
         // Set matrix from euler angles or get euler angles from matrix
         void        fromEulerAnglesZXZ(const double angle1RAD,
                                        const double angle2RAD,
                                        const double angle3RAD,
                                        const bool keepTranslation=true);
         void        fromEulerAnglesXYZ(const double angle1RAD,
                                        const double angle2RAD,
                                        const double angle3RAD,
                                        const bool keepTranslation=true);

         // Get euler angles from matrix
         void        toEulerAnglesZYX(T& zRotRAD, T& yRotRAD, T& xRotRAD);
   
         // Misc. methods
         void        identity    ();
         void        transpose   ();
         void        invert      ();
         SLMat4<T>   inverse     () const;
         SLMat3<T>   inverseTransposed();
         T           trace       () const;

         // Composing & decomposing
         void        compose     (const SLVec3f trans, 
                                  const SLVec3f rotEulerRAD, 
                                  const SLVec3f scale);
         void        decompose   (SLVec3f &trans, 
                                  SLVec4f &rotQuat, 
                                  SLVec3f &scale);
         void        decompose   (SLVec3f &trans, 
                                  SLMat3f &rotMat, 
                                  SLVec3f &scale);
         void        decompose   (SLVec3f &trans, 
                                  SLVec3f &rotEulerRAD, 
                                  SLVec3f &scale);

         void        print       (const SLchar* str=0) const;
         SLstring    toString    () const;

  static void        swap        (T& a, T& b) {T t; t=a;a=b;b=t;}

  private:
         T           _m[16];     //!< The 16 elements of the matrix
};

//-----------------------------------------------------------------------------
// Constructors
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4()
{  identity(); 
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const SLMat4& A) 
{  setMatrix(A);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(T M0, T M4, T M8,  T M12,
                  T M1, T M5, T M9,  T M13,
                  T M2, T M6, T M10, T M14,
                  T M3, T M7, T M11, T M15)
{
    setMatrix(M0, M4, M8, M12,
              M1, M5, M9, M13,
              M2, M6, M10,M14,
              M3, M7, M11,M15);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const T* M) 
{
    setMatrix(M);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const T tx, const T ty, const T tz) 
{
    translation(tx, ty, tz, false);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const T degAng, const T ax, const T ay, const T az)
{   
    rotation(degAng, ax, ay, az, false);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const T scale_xyz)
{
    scaling(scale_xyz);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const SLVec3<T>& fromUnitVec, const SLVec3<T>& toUnitVec)
{
    rotation(fromUnitVec, toUnitVec);
}
//-----------------------------------------------------------------------------
template<class T>
SLMat4<T>::SLMat4(const SLVec3<T>& translation,
                  const SLMat3<T>& rotation,
                  const SLVec3<T>& scale)
{
    setMatrix(translation, rotation, scale);
}
//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
template<class T>
void SLMat4<T>::setMatrix(const SLMat4& A)
{
    for (int i=0; i<16; ++i) _m[i] = A._m[i];
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat4<T>::setMatrix(const SLMat4* A)
{
    for (int i=0; i<16; ++i) _m[i] = A->_m[i];
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat4<T>::setMatrix(const T* M)
{
    for (int i=0; i<16; ++i) _m[i] = M[i];
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat4<T>::setMatrix(T M0, T M4, T M8 , T M12,
                          T M1, T M5, T M9 , T M13,
                          T M2, T M6, T M10, T M14,
                          T M3, T M7, T M11, T M15)
{
    _m[0]=M0; _m[4]=M4; _m[ 8]=M8;  _m[12]=M12;
    _m[1]=M1; _m[5]=M5; _m[ 9]=M9;  _m[13]=M13;
    _m[2]=M2; _m[6]=M6; _m[10]=M10; _m[14]=M14;
    _m[3]=M3; _m[7]=M7; _m[11]=M11; _m[15]=M15;
}
//-----------------------------------------------------------------------------
template<class T>
void SLMat4<T>::setMatrix(const SLVec3<T>& translation,
                          const SLMat3<T>& rotation,
                          const SLVec3<T>& scale)
{
    setMatrix(scale.x * rotation[0], scale.y * rotation[3], scale.z * rotation[6], translation.x,
              scale.x * rotation[1], scale.y * rotation[4], scale.z * rotation[7], translation.y,
              scale.x * rotation[2], scale.y * rotation[5], scale.z * rotation[8], translation.z,
              0                    , 0                    , 0                    , 1);
}
//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------
/*!
Matrix assignment with instance
*/
template<class T>
SLMat4<T>& SLMat4<T>::operator =(const SLMat4& A)
{
    _m[0]=A._m[0]; _m[4]=A._m[4]; _m[ 8]=A._m[ 8]; _m[12]=A._m[12];
    _m[1]=A._m[1]; _m[5]=A._m[5]; _m[ 9]=A._m[ 9]; _m[13]=A._m[13];
    _m[2]=A._m[2]; _m[6]=A._m[6]; _m[10]=A._m[10]; _m[14]=A._m[14];
    _m[3]=A._m[3]; _m[7]=A._m[7]; _m[11]=A._m[11]; _m[15]=A._m[15];
    return *this;
}
//-----------------------------------------------------------------------------
/*!
Matrix - matrix multiplication
*/
template<class T>
SLMat4<T> SLMat4<T>::operator *(const SLMat4& A) const
{
    SLMat4<T> newM((T*)this);
    newM.multiply(A);
    return newM;
}
//-----------------------------------------------------------------------------
/*
!Matrix - matrix multiplication
*/
template<class T>
SLMat4<T>& SLMat4<T>::operator *=(const SLMat4& A)
{
    multiply(A);
    return *this;
}
//-----------------------------------------------------------------------------
/*
!Matrix - matrix multiplication
*/
template<class T>
SLMat4<T> SLMat4<T>::operator+(const SLMat4& A) const
{
    SLMat4<T> newM((T*)this);
    newM.add(A);
    return newM;
}

//-----------------------------------------------------------------------------
/*!
Matrix - 3D vector multiplication with perspective division
*/
template<class T>
SLVec3<T> SLMat4<T>::operator *(const SLVec3<T>& v) const  
{
    T W = 1 / (_m[3]*v.x + _m[7]*v.y + _m[11]*v.z + _m[15]);
    return SLVec3<T>((_m[0]*v.x + _m[4]*v.y + _m[ 8]*v.z + _m[12]) * W,
                     (_m[1]*v.x + _m[5]*v.y + _m[ 9]*v.z + _m[13]) * W,
                     (_m[2]*v.x + _m[6]*v.y + _m[10]*v.z + _m[14]) * W);
}
//-----------------------------------------------------------------------------
/*!
Matrix - 4D vector multiplication
*/
template<class T>
SLVec4<T> SLMat4<T>::operator *(const SLVec4<T>& v) const
{
    return SLVec4<T>(_m[0]*v.x + _m[4]*v.y + _m[ 8]*v.z + _m[12]*v.w,
                     _m[1]*v.x + _m[5]*v.y + _m[ 9]*v.z + _m[13]*v.w,
                     _m[2]*v.x + _m[6]*v.y + _m[10]*v.z + _m[14]*v.w,
                     _m[3]*v.x + _m[7]*v.y + _m[11]*v.z + _m[15]*v.w);
}
//-----------------------------------------------------------------------------
/*!
Scalar multiplication.
*/
template<class T>
SLMat4<T> SLMat4<T>::operator *(const T a) const              
{
    return SLMat4<T>(_m[ 0]*a, _m[ 1]*a, _m[ 2]*a, _m[ 3]*a, 
                     _m[ 4]*a, _m[ 5]*a, _m[ 6]*a, _m[ 7]*a,
                     _m[ 8]*a, _m[ 9]*a, _m[10]*a, _m[11]*a, 
                     _m[12]*a, _m[13]*a, _m[14]*a, _m[15]*a);
}
//-----------------------------------------------------------------------------
/*!
Scalar multiplication.
*/
template<class T>
SLMat4<T>& SLMat4<T>::operator *=(const T a)
{
    for (int i=0; i<16; ++i) _m[i] *= a;
    return *this;
}
//-----------------------------------------------------------------------------
/*!
Scalar division.
*/
template<class T>
SLMat4<T> SLMat4<T>::operator /(const T a) const              
{
    T invA = 1 / a;
    SLMat4<T> newM(_m[ 0]*invA, _m[ 1]*invA, _m[ 2]*invA, _m[ 3]*invA, 
                   _m[ 4]*invA, _m[ 5]*invA, _m[ 6]*invA, _m[ 7]*invA,
                   _m[ 8]*invA, _m[ 9]*invA, _m[10]*invA, _m[11]*invA, 
                   _m[12]*invA, _m[13]*invA, _m[14]*invA, _m[15]*invA);
    return newM;
}
//-----------------------------------------------------------------------------
/*!
Scalar division.
*/
template<class T>
SLMat4<T>& SLMat4<T>::operator /=(const T a)
{  
    T invA = 1 / a;
    for (int i=0; i<16; ++i) _m[i] *= invA;
    return *this;
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a another matrix. Corresponds to the OpenGL function
glMultMatrix*.
*/
template<class T>
void SLMat4<T>::multiply(const SLMat4& A)
{                                                       
    setMatrix(_m[0]*A._m[ 0] + _m[4]*A._m[ 1] + _m[8] *A._m[ 2] + _m[12]*A._m[ 3], //row 1
              _m[0]*A._m[ 4] + _m[4]*A._m[ 5] + _m[8] *A._m[ 6] + _m[12]*A._m[ 7],
              _m[0]*A._m[ 8] + _m[4]*A._m[ 9] + _m[8] *A._m[10] + _m[12]*A._m[11],
              _m[0]*A._m[12] + _m[4]*A._m[13] + _m[8] *A._m[14] + _m[12]*A._m[15],
              _m[1]*A._m[ 0] + _m[5]*A._m[ 1] + _m[9] *A._m[ 2] + _m[13]*A._m[ 3], //row 2
              _m[1]*A._m[ 4] + _m[5]*A._m[ 5] + _m[9] *A._m[ 6] + _m[13]*A._m[ 7],
              _m[1]*A._m[ 8] + _m[5]*A._m[ 9] + _m[9] *A._m[10] + _m[13]*A._m[11],
              _m[1]*A._m[12] + _m[5]*A._m[13] + _m[9] *A._m[14] + _m[13]*A._m[15],
              _m[2]*A._m[ 0] + _m[6]*A._m[ 1] + _m[10]*A._m[ 2] + _m[14]*A._m[ 3], //row 3
              _m[2]*A._m[ 4] + _m[6]*A._m[ 5] + _m[10]*A._m[ 6] + _m[14]*A._m[ 7],
              _m[2]*A._m[ 8] + _m[6]*A._m[ 9] + _m[10]*A._m[10] + _m[14]*A._m[11],
              _m[2]*A._m[12] + _m[6]*A._m[13] + _m[10]*A._m[14] + _m[14]*A._m[15],
              _m[3]*A._m[ 0] + _m[7]*A._m[ 1] + _m[11]*A._m[ 2] + _m[15]*A._m[ 3], //row 4
              _m[3]*A._m[ 4] + _m[7]*A._m[ 5] + _m[11]*A._m[ 6] + _m[15]*A._m[ 7],
              _m[3]*A._m[ 8] + _m[7]*A._m[ 9] + _m[11]*A._m[10] + _m[15]*A._m[11],
              _m[3]*A._m[12] + _m[7]*A._m[13] + _m[11]*A._m[14] + _m[15]*A._m[15]);
             
              //     | 0  4  8 12 |   | 0  4  8 12 |
              //     | 1  5  9 13 |   | 1  5  9 13 |
              // M = | 2  6 10 14 | x | 2  6 10 14 |
              //     | 3  7 11 15 |   | 3  7 11 15 |
}

//-----------------------------------------------------------------------------
/*!
Adds the matrix to an other matrix A.
*/
template<class T>
void SLMat4<T>::add(const SLMat4& A)
{
    for (SLint i = 0; i < 16; ++i)
        _m[i] = _m[i] + A._m[i];
}
//-----------------------------------------------------------------------------
/*!
Matrix - 3D vector multiplication with perspective division
*/
template<class T>
SLVec3<T> SLMat4<T>::multVec(const SLVec3<T> v) const
{
    T W = 1 / (_m[3]*v.x + _m[7]*v.y + _m[11]*v.z + _m[15]);
    return SLVec3<T>((_m[0]*v.x + _m[4]*v.y + _m[ 8]*v.z + _m[12]) * W,
                     (_m[1]*v.x + _m[5]*v.y + _m[ 9]*v.z + _m[13]) * W,
                     (_m[2]*v.x + _m[6]*v.y + _m[10]*v.z + _m[14]) * W);
}
//-----------------------------------------------------------------------------
/*!
Matrix - 4D vector multiplication
*/
template<class T>
SLVec4<T> SLMat4<T>::multVec(const SLVec4<T> v) const
{
    return SLVec4<T>(_m[0]*v.x + _m[4]*v.y + _m[ 8]*v.z + _m[12]*v.w,
                     _m[1]*v.x + _m[5]*v.y + _m[ 9]*v.z + _m[13]*v.w,
                     _m[2]*v.x + _m[6]*v.y + _m[10]*v.z + _m[14]*v.w,
                     _m[3]*v.x + _m[7]*v.y + _m[11]*v.z + _m[15]*v.w);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a translation matrix. 
Corresponds to the OpenGL function glTranslate*.
*/
template <class T>
void SLMat4<T>::translate(const T tx, const T ty, const T tz)
{
    SLMat4<T> trans(tx, ty, tz);
    multiply(trans);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a translation matrix. 
Corresponds to the OpenGL function glTranslate*.
*/
template<class T>
void SLMat4<T>::translate(const SLVec2<T>& t)
{
    SLMat4<T> trans(t.x, t.y, 0);
    multiply(trans);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a translation matrix. 
Corresponds to the OpenGL function glTranslate*.
*/
template<class T>
void SLMat4<T>::translate(const SLVec3<T>& t)
{
    SLMat4<T> trans(t.x, t.y, t.z);
    multiply(trans);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a rotation matrix. 
Corresponds to the OpenGL function glRotate*.
*/
template<class T>
void SLMat4<T>::rotate(const T degAng, const SLVec3<T>& axis)
{
    SLMat4<T> R(degAng, axis.x, axis.y, axis.z);
    multiply(R);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a rotation matrix created with a rotation between
two vectors.
*/
template<class T>
void SLMat4<T>::rotate(const SLVec3<T>& fromUnitVec, const SLVec3<T>& toUnitVec)
{
    SLMat4<T> R(fromUnitVec, toUnitVec);
    multiply(R);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a rotation matrix. 
Corresponds to the OpenGL function glRotate*.
*/
template<class T>
void SLMat4<T>::rotate(const T degAng, const T axisx, const T axisy, const T axisz)
{
    SLMat4<T> R(degAng, axisx, axisy, axisz);
    multiply(R);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a scaling matrix. 
Corresponds to the OpenGL function glScale*.
*/
template<class T>
void SLMat4<T>::scale(const T s)
{   SLMat4<T> S(s, 0, 0, 0,
                0, s, 0, 0,
                0, 0, s, 0,
                0, 0, 0, 1);
    multiply(S);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a scaling matrix. 
Corresponds to the OpenGL function glScale*.
*/
template<class T>
void SLMat4<T>::scale(const T sx, const T sy, const T sz)
{
    SLMat4<T> S(sx, 0, 0, 0,
                0,sy, 0, 0,
                0, 0,sz, 0,
                0, 0, 0, 1);
    multiply(S);
}
//-----------------------------------------------------------------------------
/*!
Multiplies the matrix with a scaling matrix. 
Corresponds to the OpenGL function glScale*.
*/
template<class T>
void SLMat4<T>::scale(const SLVec3<T>& s)
{
    SLMat4<T> S(s.x,  0,  0,  0,
                0,s.y,  0,  0,
                0,  0,s.z,  0,
                0,  0,  0,  1);
    multiply(S);
}
//-----------------------------------------------------------------------------
//! Defines the view matrix with an eye position, a look at point and an up vector.
/*!
This method is equivalent to the OpenGL function gluLookAt.
*/
template<class T>
void SLMat4<T>::lookAt( const T EyeX, const T EyeY, const T EyeZ,
                        const T  AtX, const T  AtY, const T  AtZ,
                        const T  UpX, const T  UpY, const T  UpZ)
{
    lookAt(SLVec3<T>(EyeX,EyeY,EyeZ),
           SLVec3<T>( AtX, AtY, AtZ),
           SLVec3<T>( UpX, UpY, UpZ));
}
//-----------------------------------------------------------------------------
//! Defines the view matrix with an eye position, a look at point and an up vector.
/*!
This method is equivalent to the OpenGL function gluLookAt.
\param Eye Vector to the position of the eye (view point).
\param At Vector to the target point.
\param Up Vector that points from the viewpoint upwards. If Up is a zero vector
a default up vector is calculated with a default look-right vector (VZ) that 
lies in the x-z plane.
*/
template<class T>
void SLMat4<T>::lookAt(const SLVec3<T>& Eye,
                       const SLVec3<T>& At,
                       const SLVec3<T>& Up)
{
    SLVec3<T> VX, VY, VZ, VT;
    SLMat3<T> xz(0.0, 0.0, 1.0,         // matrix that transforms YZ into a 
                 0.0, 0.0, 0.0,         // vector that is perpendicular to YZ and 
                -1.0, 0.0, 0.0);        // lies in the x-z plane

    VZ = Eye-At; 
    VZ.normalize(); 

    if (Up==SLVec3<T>::ZERO || Up==VZ)  
    {   VX.set(xz*VZ);    
        VX.normalize();
    } else
    {   VX.cross(Up, VZ); 
        VX.normalize();
    }
    VY.cross(VZ, VX); VY.normalize();
    VT = -Eye;

    setMatrix(VX.x, VX.y, VX.z, VX*VT,
              VY.x, VY.y, VY.z, VY*VT,
              VZ.x, VZ.y, VZ.z, VZ*VT,
              0.0,  0.0,  0.0,  1.0);
}
//-----------------------------------------------------------------------------
/*! 
This method retrieves the eye position the look at, up & right vector out of 
the view matrix. Attention: The look-at is normalized vector, not a point.
*/
template<class T>
void SLMat4<T>::lookAt(SLVec3<T>* eye, 
                       SLVec3<T>* at, 
                       SLVec3<T>* up, 
                       SLVec3<T>* ri) const
{
    SLMat4<T> invRot(_m);               // get the current view matrix
    invRot.translation(0,0,0);          // remove the translation
    invRot.transpose();                 // transpose it to get inverse rot.
    eye->set(invRot.multVec(-translation())); // setMatrix eye
    ri->set( _m[0], _m[4], _m[8]);            // normalized look right vector
    up->set( _m[1], _m[5], _m[9]);            // normalized look up vector
    at->set(-_m[2],-_m[6],-_m[10]);           // normalized look at vector
}
//-----------------------------------------------------------------------------
/*! 
Defines the a model matrix for positioning a light at pos and shining at At
*/
template<class T>
void SLMat4<T>::lightAt(const T PosX, const T PosY, const T PosZ,
                        const T  AtX, const T  AtY, const T  AtZ,
                        const T  UpX, const T  UpY, const T  UpZ)
{
    lightAt(SLVec3<T>(PosX,PosY,PosZ),
            SLVec3<T>( AtX, AtY, AtZ),
            SLVec3<T>( UpX, UpY, UpZ));
}
//-----------------------------------------------------------------------------
//! Defines the a model matrix for positioning a light source.
/*!
Utility method for defining the transformation matrix for a spot light. There 
is no equivalent function for this purpose in OpenGL.
/param pos Vector to the position of the light source.
/param At  Vector to a target point where a spot light shines to.
*/
template<class T>
void SLMat4<T>::lightAt(const SLVec3<T>& pos,
                        const SLVec3<T>& At,
                        const SLVec3<T>& Up)
{
    SLVec3<T> VX, VY, VZ;
    SLMat3<T> xz(0.0, 0.0, 1.0,         // matrix that transforms VZ into a
                0.0, 0.0, 0.0,         // vector that is perpendicular to YZ and 
                -1.0, 0.0, 0.0);        // lies in the x-z plane

    VZ = pos-At;   VZ.normalize();

    if (Up==SLVec3<T>::ZERO)  
    {   VX = xz*VZ;  VX.normalize();
        VY = VZ^VX;  VY.normalize();
    } else
    {   VX = Up^VZ;  VX.normalize();
        VY = VZ^VX;  VY.normalize();
    }

    setMatrix(VX.x, VY.x, VZ.x, pos.x,
                VX.y, VY.y, VZ.y, pos.y,
                VX.z, VY.z, VZ.z, pos.z,
                0.0,  0.0,  0.0, 1.0);
}
//-----------------------------------------------------------------------------
//! Same as lightAt
template<class T>
void SLMat4<T>::posAtUp(const T PosX,   const T PosY,   const T PosZ,
                        const T dirAtX, const T dirAtY, const T dirAtZ,
                        const T dirUpX, const T dirUpY, const T dirUpZ)
{
    lightAt(SLVec3<T>(  PosX,   PosY,   PosZ),
            SLVec3<T>(dirAtX, dirAtY, dirAtZ),
            SLVec3<T>(dirUpX, dirUpY, dirUpZ));
}

//-----------------------------------------------------------------------------
//! Same as lightAt
template<class T>
void SLMat4<T>::posAtUp(const SLVec3<T>& pos,
                        const SLVec3<T>& dirAt,
                        const SLVec3<T>& dirUp)
{
    lightAt(pos, dirAt, dirUp);
}

//---------------------------------------------------------------------------
//! Defines a view frustum projection matrix equivalent to OpenGL's glFrustum
/*!
The view frustum is the truncated view pyramid of a central projection that
is defined with the folowing parameters:
\param l Distance from the center of projection (COP) to the left border on 
the near clipping plane.
\param r Distance from the COP to the right border on the near clipping plane. 
\param b Distance from the COP to the bottom border on the near clipping plane. 
\param t Distance from the COP to the top border on the near clipping plane.
\param n Distance from the eye to near clipping plane of the view frustum.
\param f Distance from the eye to far clipping plane of the view frustum.  
*/
template<class T>
void SLMat4<T>::frustum(const T l, const T r, const T b, const T t, 
                        const T n, const T f)
{
    _m[0]=(2*n)/(r-l); _m[4]=0;           _m[8] = (r+l)/(r-l); _m[12]=0;
    _m[1]=0;           _m[5]=(2*n)/(t-b); _m[9] = (t+b)/(t-b); _m[13]=0;
    _m[2]=0;           _m[6]=0;           _m[10]=-(f+n)/(f-n); _m[14]=(-2*f*n)/(f-n);
    _m[3]=0;           _m[7]=0;           _m[11]=-1;           _m[15]=0;
}
//---------------------------------------------------------------------------
//! Defines a view frustum projection matrix for a perspective projection
/*!
This method is equivalent to the OpenGL function gluPerspective except that
instead of the window aspect the window width and height have to be passed.
\param fov: Vertical field of view angle (zoom angle)
\param aspect: aspect ratio of of the viewport = width / height
\param n: Distance from the eye to near clipping plane of the view frustum.
\param f: Distance from the eye to far clipping plane of the view frustum.  
*/
template<class T>
void SLMat4<T>::perspective(const T fov, const T aspect, 
                            const T n, const T f)
{
   T t = (T)tan(fov*SL_DEG2RAD*0.5)*n;
   T b = -t;
   T r = t*aspect;
   T l = -r;
   frustum(l,r,b,t,n,f);
}
//---------------------------------------------------------------------------
//! Defines a ortographic projection matrix equivalent to OpenGL's glOrtho
/*!
\param l Distance from the center of projection (COP) to the left border on 
the near clipping plane.
\param r Distance from the COP to the right border on the near clipping plane. 
\param b Distance from the COP to the bottom border on the near clipping plane. 
\param t Distance from the COP to the top border on the near clipping plane.
\param n Distance from the eye to near clipping plane of the view frustum.
\param f Distance from the eye to far clipping plane of the view frustum.  
*/
template<class T>
void SLMat4<T>::ortho(const T l, const T r, const T b, const T t, 
                      const T n, const T f)
{
    _m[0]=2/(r-l); _m[4]=0;       _m[8]=0;         _m[12]=-(r+l)/(r-l);
    _m[1]=0;       _m[5]=2/(t-b); _m[9]=0;         _m[13]=-(t+b)/(t-b);
    _m[2]=0;       _m[6]=0;       _m[10]=-2/(f-n); _m[14]=-(f+n)/(f-n);
    _m[3]=0;       _m[7]=0;       _m[11]=0;        _m[15]=1;
}
//---------------------------------------------------------------------------
//! Defines a viewport matrix as it is defined by glViewport
/*!
\param x: left window coord. in px.
\param y: top window coord. in px.
\param ww: window width in px.
\param wh: window height in px.
\param n: near depth range (default 0)
\param f: far depth range (default 1)
*/
template<class T>
void SLMat4<T>::viewport(const T x, const T y, const T ww, const T wh, 
                         const T n, const T f)
{
    T ww2 = ww*0.5f;
    T wh2 = wh*0.5f;
   
    // negate the first wh2 because windows has topdown window coords
    _m[0]=ww2; _m[4]=0;    _m[8] =0;          _m[12]=x+ww2;
    _m[1]=0;   _m[5]=-wh2; _m[9] =0;          _m[13]=y+wh2;
    _m[2]=0;   _m[6]=0;    _m[10]=(f-n)*0.5f; _m[14]=(f+n)*0.5f;
    _m[3]=0;   _m[7]=0;    _m[11]=0;          _m[15]=1;
}
//-----------------------------------------------------------------------------
/*!
Sets the translation components. By default the linear 3x3 submatrix containing
rotations and scaling is reset to identity.
*/
template<class T>
void SLMat4<T>::translation(const SLVec3<T>& t, const SLbool keepLinear)
{
    translation(t.x, t.y, t.z, keepLinear);
}
//-----------------------------------------------------------------------------
/*!
Sets the translation components. By default the linear 3x3 submatrix containing
rotations and scaling is reset to identity.
*/
template<class T>
void SLMat4<T>::translation(const T tx, const T ty, const T tz, 
                            const SLbool keepLinear)
{
    _m[12]=tx;
    _m[13]=ty;
    _m[14]=tz;
    if (!keepLinear)
    {   _m[0]=1; _m[4]=0;  _m[8]=0;
        _m[1]=0; _m[5]=1;  _m[9]=0;  
        _m[2]=0; _m[6]=0;  _m[10]=1; 
        _m[3]=0; _m[7]=0;  _m[11]=0; _m[15]=1;
    }
}
//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a rotation matrix with a rotation of degAng 
degrees around an axis. By default the translation components are set to 0.
*/
template<class T>
void SLMat4<T>::rotation(const T degAng, const SLVec3<T>& axis, 
                         const SLbool keepTrans)
{
    rotation(degAng, axis.x, axis.y, axis.z, keepTrans);
}
//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a rotation matrix with a rotation of degAng 
degrees around an axis. By default the translation components are set to 0.
*/
template<class T>
void SLMat4<T>::rotation(const T degAng, 
                         const T axisx, const T axisy, const T axisz,
                         const SLbool keepTrans)
{
    T RadAng = (T)degAng*SL_DEG2RAD;
    T ca=(T)cos(RadAng), sa=(T)sin(RadAng);
    if (axisx==1 && axisy==0 && axisz==0)              // about x-axis
    {   _m[0]=1; _m[4]=0;  _m[8]=0;   
        _m[1]=0; _m[5]=ca; _m[9]=-sa; 
        _m[2]=0; _m[6]=sa; _m[10]=ca; 
    } else 
    if (axisx==0 && axisy==1 && axisz==0)              // about y-axis
    {   _m[0]=ca;  _m[4]=0; _m[8]=sa; 
        _m[1]=0;   _m[5]=1; _m[9]=0;  
        _m[2]=-sa; _m[6]=0; _m[10]=ca;
    } else 
    if (axisx==0 && axisy==0 && axisz==1)              // about z-axis
    {   _m[0]=ca; _m[4]=-sa; _m[8]=0; 
        _m[1]=sa; _m[5]=ca;  _m[9]=0; 
        _m[2]=0;  _m[6]=0;   _m[10]=1;
    } else                                             // arbitrary axis
    {   T l = axisx*axisx + axisy*axisy + axisz*axisz;  // length squared
        T x, y, z;
        x=axisx, y=axisy, z=axisz;
        if ((l > T(1.0001) || l < T(0.9999)) && l!=0)
        {   l=T(1.0)/sqrt(l);
            x*=l; y*=l; z*=l;
        }
        T xy=x*y, yz=y*z, xz=x*z, xx=x*x, yy=y*y, zz=z*z;
        _m[0]=xx + ca*(1-xx);     _m[4]=xy - xy*ca - z*sa;  _m[8] =xz - xz*ca + y*sa;
        _m[1]=xy - xy*ca + z*sa;  _m[5]=yy + ca*(1-yy);     _m[9] =yz - yz*ca - x*sa;
        _m[2]=xz - xz*ca - y*sa;  _m[6]=yz - yz*ca + x*sa;  _m[10]=zz + ca*(1-zz);
    }
    _m[3]=_m[7]=_m[11]=0; _m[15]=1;

    if (!keepTrans) 
        _m[12] = _m[13] = _m[14] = 0;
}
//-----------------------------------------------------------------------------
/*!
Defines a rotation matrix that rotates the vector from to the vector to.
Code and explanation comes from the paper "Efficiently build
a matrix to ratate one vector to another" from Thomas Möller and John Hughes in
the Journal of Graphic Tools, volume 4.
*/
template<class T>
void SLMat4<T>::rotation(const SLVec3<T>&from, const SLVec3<T> &to)
{
    // Creates a rotation matrix that will rotate the Vector 'from' into
    // the Vector 'to'. The resulting matrix is stored in 'this' Matrix.
    // For this method to work correctly, vector 'from' and vector 'to'
    // must both be unit length vectors.

    T cosAngle = from.dot(to);

    if (fabs(cosAngle-1.0) <= FLT_EPSILON) // cosAngle = 1.0
    {
        // Special case where 'from' is equal to 'to'. In other words,
        // the angle between Vector 'from' and Vector 'to' is zero
        // degrees. In this case the identity matrix is returned in
        // Matrix 'result'.

        identity();
    }
    else if (fabs(cosAngle+1.0) <= FLT_EPSILON) // cosAngle = -1.0
    {
        // Special case where 'from' is directly opposite to 'to'. In
        // other words, the angle between Vector 'from' and Vector 'to'
        // is 180 degrees. In this case, the following matrix is used:
        //
        // Let:
        //   F = from
        //   S = vector perpendicular to F
        //   U = S X F
        //
        // We want to rotate from (F, U, S) to (-F, U, -S)
        //
        // | -FxFx+UxUx-SxSx  -FxFy+UxUy-SxSy  -FxFz+UxUz-SxSz  0 |
        // | -FxFy+UxUy-SxSy  -FyFy+UyUy-SySy  -FyFz+UyUz-SySz  0 |
        // | -FxFz+UxUz-SxSz  -FyFz+UyUz-SySz  -FzFz+UzUz-SzSz  0 |
        // |       0                 0                0         1 |

        SLVec3<T> side(0.f, from.z, -from.y);

        if (fabs(side.dot(side)) <= FLT_EPSILON)
        {   side.x = -from.z;
            side.y = 0.f;
            side.z = from.x;
        }

        side.normalize();

        SLVec3<T> up;
        up.cross(side, from);
        up.normalize();

        _m[ 0] = -(from.x * from.x) + (up.x * up.x) - (side.x * side.x);
        _m[ 4] = -(from.x * from.y) + (up.x * up.y) - (side.x * side.y);
        _m[ 8] = -(from.x * from.z) + (up.x * up.z) - (side.x * side.z);
        _m[12] = 0.f;
        _m[ 1] = -(from.x * from.y) + (up.x * up.y) - (side.x * side.y);
        _m[ 5] = -(from.y * from.y) + (up.y * up.y) - (side.y * side.y);
        _m[ 9] = -(from.y * from.z) + (up.y * up.z) - (side.y * side.z);
        _m[13] = 0.f;
        _m[ 2] = -(from.x * from.z) + (up.x * up.z) - (side.x * side.z);
        _m[ 6] = -(from.y * from.z) + (up.y * up.z) - (side.y * side.z);
        _m[10] = -(from.z * from.z) + (up.z * up.z) - (side.z * side.z);
        _m[14] = 0.f;
        _m[ 3] = 0.f;
        _m[ 7] = 0.f;
        _m[11] = 0.f;
        _m[15] = 1.f;
    }
    else
    {
        // This is the most common case. Creates the rotation matrix:
        //
        //               | E + HVx^2    HVxVy - Vz   HVxVz + Vy   0 |
        // R(from, to) = | HVxVy + Vz   E + HVy^2    HVyVz - Vx   0 |
        //               | HVxVz - Vy   HVyVz + Vx   E + HVz^2    0 |
        //               |     0            0            0        1 |
        //
        // where,
        //   V = from X to
        //   E = from.dot(to)
        //   H = (1 - E) / V.dot(V)

        SLVec3<T> v;
        v.cross(from, to);
        v.normalize();

        T h = (1.f - cosAngle) / v.dot(v);

        _m[ 0] = cosAngle + h * v.x * v.x;
        _m[ 4] = h * v.x * v.y - v.z;
        _m[ 8] = h * v.x * v.z + v.y;
        _m[12] = 0.f;
        _m[ 1] = h * v.x * v.y + v.z;
        _m[ 5] = cosAngle + h * v.y * v.y;
        _m[ 9] = h * v.y * v.z - v.x;
        _m[13] = 0.f;
        _m[ 2] = h * v.x * v.z - v.y;
        _m[ 6] = h * v.y * v.z + v.x;
        _m[10] = cosAngle + h * v.z * v.z;
        _m[14] = 0.f;
        _m[ 3] = 0.f;
        _m[ 7] = 0.f;
        _m[11] = 0.f;
        _m[15] = 1.f;
    }
}
//-----------------------------------------------------------------------------    
/*!
Sets the linear 3x3 submatrix as a scaling matrix with the scaling vector s. 
By default the translation components are set to 0.
*/ 
template<class T>
void SLMat4<T>::scaling(const T sxyz, 
                        const SLbool keepTrans)
{
    scaling(sxyz, sxyz, sxyz);
}
//-----------------------------------------------------------------------------    
/*!
Sets the linear 3x3 submatrix as a scaling matrix with the scaling vector s. 
By default the translation components are set to 0.
*/ 
template<class T>
void SLMat4<T>::scaling(const SLVec3<T>& scale, 
                        const SLbool keepTrans)
{
    scaling(scale.x, scale.y, scale.z);
}
//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a scaling matrix with the scaling f_actors 
sx, sy and sz. By default the translation components are set to 0.
*/ 
template<class T>
void SLMat4<T>::scaling(const T sx, const T sy, const T sz,
                        const SLbool keepTrans)
{
    _m[0]=sx; _m[4]=0;  _m[8]=0;   
    _m[1]=0;  _m[5]=sy; _m[9]=0; 
    _m[2]=0;  _m[6]=0;  _m[10]=sz;
    _m[3]=0;  _m[7]=0;  _m[11]=0; _m[15]=1;

    if (!keepTrans) 
        _m[12] = _m[13] = _m[14] = 0;
}
//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a rotation matrix from the 3 euler angles
in radians around the z-axis, x-axis & z-axis.
By default the translation components are set to 0.
See: http://en.wikipedia.org/wiki/Euler_angles
*/
template<class T>
void SLMat4<T>::fromEulerAnglesZXZ(const double angle1RAD,
                                   const double angle2RAD,
                                   const double angle3RAD,
                                   const SLbool keepTrans)
{
    double s1 = sin(angle1RAD), c1 = cos(angle1RAD);
    double s2 = sin(angle2RAD), c2 = cos(angle2RAD);
    double s3 = sin(angle3RAD), c3 = cos(angle3RAD);

    _m[0]=(T)( c1*c3 - s1*c2*s3); _m[4]=(T)( s1*c3 + c1*c2*s3);  _m[8] =(T)( s2*s3);
    _m[1]=(T)(-c1*s3 - s1*c2*c3); _m[5]=(T)( c1*c2*c3 - s1*s3);  _m[9] =(T)( s2*c3);
    _m[2]=(T)( s1*s2);            _m[6]=(T)(-c1*s2);             _m[10]=(T)( c2);

    _m[3]=_m[7]=_m[11]=0; _m[15]=1;

    if (!keepTrans)
    {  _m[12] = _m[13] = _m[14] = 0;
    }
}
//-----------------------------------------------------------------------------
/*!
Sets the linear 3x3 submatrix as a rotation matrix from the 3 euler angles
in radians around the z-axis, y-axis & x-axis.
By default the translation components are set to 0.
See: http://en.wikipedia.org/wiki/Euler_angles
*/
template<class T>
void SLMat4<T>::fromEulerAnglesXYZ(const double angle1RAD,
                                   const double angle2RAD,
                                   const double angle3RAD,
                                   const SLbool keepTrans)
{
    double s1 = sin(angle1RAD), c1 = cos(angle1RAD);
    double s2 = sin(angle2RAD), c2 = cos(angle2RAD);
    double s3 = sin(angle3RAD), c3 = cos(angle3RAD);

    _m[0]=(T) (c2*c3);              _m[4]=(T)-(c2*s3);              _m[8] =(T) s2;
    _m[1]=(T) (s1*s2*c3) + (c1*s3); _m[5]=(T)-(s1*s2*s3) + (c1*c3); _m[9] =(T)-(s1*c2);
    _m[2]=(T)-(c1*s2*c3) + (s1*s3); _m[6]=(T) (c1*s2*s3) + (s1*c3); _m[10]=(T) (c1*c2);

    _m[3]=_m[7]=_m[11]=0; _m[15]=1;

    if (!keepTrans)
    {  _m[12] = _m[13] = _m[14] = 0;
    }
}
//-----------------------------------------------------------------------------
/*!
Gets one set of possible z-y-x euler angles that will generate this matrix
Assumes that upper 3x3 is a rotation matrix
Source: Essential Mathematics for Games and Interactive Applications
A Programmer’s Guide 2nd edition by James M. Van Verth and Lars M. Bishop
*/
template<class T>
void SLMat4<T>::toEulerAnglesZYX(T& zRotRAD, T& yRotRAD, T& xRotRAD)
{
    T Cx, Sx;
    T Cy, Sy;
    T Cz, Sz;

    Sy = _m[8];
    Cy = (T)sqrt(1.0 - Sy*Sy);
    
    // normal case
    if (SL_abs(Cy) > FLT_EPSILON)
    {
        T factor = (T)(1.0 / Cy);
        Sx = -_m[ 9]*factor;
        Cx =  _m[10]*factor;
        Sz = -_m[ 4]*factor;
        Cz =  _m[ 0]*factor;
    }
    else // x and z axes aligned
    {
        Sz = 0.0;
        Cz = 1.0;
        Sx = _m[6];
        Cx = _m[5];
    }

    zRotRAD = atan2f(Sz, Cz);
    yRotRAD = atan2f(Sy, Cy);
    xRotRAD = atan2f(Sx, Cx);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Misc. methods
//-----------------------------------------------------------------------------
//! Sets the identity matrix
template<class T>
void SLMat4<T>::identity()
{
    _m[0]=_m[5]=_m[10]=_m[15]=1;
    _m[1]=_m[2]=_m[3]=_m[4]=_m[6]=_m[7]=_m[8]=_m[9]=_m[11]=_m[12]=_m[13]=_m[14]=0;
}
//-----------------------------------------------------------------------------
//! Sets the transposed matrix by swaping around the main diagonal
template<class T>
void SLMat4<T>::transpose()
{
    swap(_m[1], _m[ 4]);
    swap(_m[2], _m[ 8]);
    swap(_m[6], _m[ 9]);
    swap(_m[3], _m[12]);
    swap(_m[7], _m[13]);
    swap(_m[11],_m[14]);
}
//-----------------------------------------------------------------------------
//! Inverts the matrix
template<class T>
void SLMat4<T>::invert()
{
    setMatrix(inverse());
}
//-----------------------------------------------------------------------------
//! Computes the inverse of a 4x4 non-singular matrix.
template<class T>
SLMat4<T> SLMat4<T>::inverse() const
{
    SLMat4<T> i;
   
    // Code from Mesa-2.2\src\glu\project.c
    T det, d12, d13, d23, d24, d34, d41;

    // Inverse = adjoint / det. (See linear algebra texts.)
    // pre-compute 2x2 dets for last two rows when computing
    // cof_actors of first two rows.
    d12 = ( _m[2]*_m[ 7] - _m[ 3]*_m[ 6]);
    d13 = ( _m[2]*_m[11] - _m[ 3]*_m[10]);
    d23 = ( _m[6]*_m[11] - _m[ 7]*_m[10]);
    d24 = ( _m[6]*_m[15] - _m[ 7]*_m[14]);
    d34 = (_m[10]*_m[15] - _m[11]*_m[14]);
    d41 = (_m[14]*_m[ 3] - _m[15]*_m[ 2]);

    i._m[0] =  (_m[5]*d34 - _m[9]*d24 + _m[13]*d23);
    i._m[1] = -(_m[1]*d34 + _m[9]*d41 + _m[13]*d13);
    i._m[2] =  (_m[1]*d24 + _m[5]*d41 + _m[13]*d12);
    i._m[3] = -(_m[1]*d23 - _m[5]*d13 + _m[ 9]*d12);

    // Compute determinant as early as possible using these cof_actors.
    det = _m[0]*i._m[0] + _m[4]*i._m[1] + _m[8]*i._m[2] + _m[12]*i._m[3];

    // Run singularity test.
    if (fabs(det) < FLT_EPSILON) 
    {   cout << "4x4-Matrix is singular. Inversion impossible." << endl;
        print("");
        cout << endl;
        exit(-1);
    } else 
    {   T invDet = 1 / det;
        // Compute rest of inverse.
        i._m[0] *= invDet;
        i._m[1] *= invDet;
        i._m[2] *= invDet;
        i._m[3] *= invDet;

        i._m[4] = -(_m[4]*d34 - _m[8]*d24 + _m[12]*d23)*invDet;
        i._m[5] =  (_m[0]*d34 + _m[8]*d41 + _m[12]*d13)*invDet;
        i._m[6] = -(_m[0]*d24 + _m[4]*d41 + _m[12]*d12)*invDet;
        i._m[7] =  (_m[0]*d23 - _m[4]*d13 +  _m[8]*d12)*invDet;

        // Pre-compute 2x2 dets for first two rows when computing
        // cofactors of last two rows.
        d12 = _m[ 0]*_m[ 5] - _m[ 1]*_m[ 4];
        d13 = _m[ 0]*_m[ 9] - _m[ 1]*_m[ 8];
        d23 = _m[ 4]*_m[ 9] - _m[ 5]*_m[ 8];
        d24 = _m[ 4]*_m[13] - _m[ 5]*_m[12];
        d34 = _m[ 8]*_m[13] - _m[ 9]*_m[12];
        d41 = _m[12]*_m[ 1] - _m[13]*_m[ 0];

        i._m[ 8] =  (_m[7]*d34 - _m[11]*d24 + _m[15]*d23)*invDet;
        i._m[ 9] = -(_m[3]*d34 + _m[11]*d41 + _m[15]*d13)*invDet;
        i._m[10] =  (_m[3]*d24 + _m[ 7]*d41 + _m[15]*d12)*invDet;
        i._m[11] = -(_m[3]*d23 - _m[ 7]*d13 + _m[11]*d12)*invDet;
        i._m[12] = -(_m[6]*d34 - _m[10]*d24 + _m[14]*d23)*invDet;
        i._m[13] =  (_m[2]*d34 + _m[10]*d41 + _m[14]*d13)*invDet;
        i._m[14] = -(_m[2]*d24 + _m[ 6]*d41 + _m[14]*d12)*invDet;
        i._m[15] =  (_m[2]*d23 - _m[ 6]*d13 + _m[10]*d12)*invDet;
    }
    return i;
}
//-----------------------------------------------------------------------------
/*! Computes the inverse transposed matrix of the upper left 3x3 matrix for
the transformation of vertex normals.
*/
template<class T>
SLMat3<T> SLMat4<T>::inverseTransposed()
{
    SLMat3<T> i(_m[0], _m[4], _m[8],
                _m[1], _m[5], _m[9],
                _m[2], _m[6], _m[10]);
    i.invert();
    i.transpose();
    return i;
}
//-----------------------------------------------------------------------------
/*!
Returns the trace of the matrix that is the sum of the diagonal components.
*/
template<class T>
inline T SLMat4<T>::trace() const
{
    return _m[0] + _m[5] + _m[10] + _m[15];
}



//-----------------------------------------------------------------------------
/*!
Composes the matrix from a translation vector, a rotation Euler angle vector 
and the scaling factors.
*/
template<class T>
void SLMat4<T>::compose(const SLVec3f trans, 
                        const SLVec3f rotEulerRAD, 
                        const SLVec3f scaleFactors)
{  
    fromEulerAnglesXYZ(rotEulerRAD.x, rotEulerRAD.y, rotEulerRAD.z);
    scale(scaleFactors);
    translate(trans);
}
//-----------------------------------------------------------------------------
/*!
Decomposes the matrix into a translation vector, a rotation quaternion and the
scaling factors using polar decomposition introduced by Ken Shoemake.
See the paper in lib-SLExternal/Shoemake/polar-decomp.pdf
*/
template<class T>
void SLMat4<T>::decompose(SLVec3f &trans, SLVec4f &rotQuat, SLVec3f &scale)
{  
    // Input matrix A
    HMatrix A;
    A[0][0]=_m[0]; A[0][1]=_m[4]; A[0][2]=_m[ 8]; A[0][3]=_m[12]; 
    A[1][0]=_m[1]; A[1][1]=_m[5]; A[1][2]=_m[ 9]; A[1][3]=_m[13]; 
    A[2][0]=_m[2]; A[2][1]=_m[6]; A[2][2]=_m[10]; A[2][3]=_m[14]; 
    A[3][0]=_m[3]; A[3][1]=_m[7]; A[3][2]=_m[11]; A[3][3]=_m[15]; 

    AffineParts parts;
    decomp_affine(A, &parts);

    trans.set(parts.t.x, parts.t.y, parts.t.z);
    scale.set(parts.k.x, parts.k.y, parts.k.z);
    rotQuat.set(parts.q.x, parts.q.y, parts.q.z, parts.q.w);
}
//-----------------------------------------------------------------------------
/*!
Decomposes the matrix into a translation vector, a 3x3 rotation matrix and the
scaling factors using polar decomposition introduced by Ken Shoemake.
See the paper in lib-SLExternal/Shoemake/polar-decomp.pdf
*/
template<class T>
void SLMat4<T>::decompose(SLVec3f &trans, SLMat3f &rotMat, SLVec3f &scale)
{  
    SLVec4f rotQuat;
    decompose(trans, rotQuat, scale);

    // Convert quaternion to 3x3 matrix
    SLfloat qx = rotQuat.x;
    SLfloat qy = rotQuat.y;
    SLfloat qz = rotQuat.z;
    SLfloat qw = rotQuat.w;

    SLfloat  x2 = qx * 2.0f;  
    SLfloat  y2 = qy * 2.0f;  
    SLfloat  z2 = qz * 2.0f;

    SLfloat wx2 = qw * x2;  SLfloat wy2 = qw * y2;  SLfloat wz2 = qw * z2;
    SLfloat xx2 = qx * x2;  SLfloat xy2 = qx * y2;  SLfloat xz2 = qx * z2;
    SLfloat yy2 = qy * y2;  SLfloat yz2 = qy * z2;  SLfloat zz2 = qz * z2;

    rotMat.setMatrix(1.0f-(yy2 + zz2),       xy2 - wz2 ,       xz2 + wy2,
                            xy2 + wz2 , 1.0f-(xx2 + zz2),       yz2 - wx2,
                            xz2 - wy2 ,       yz2 + wx2 , 1.0f-(xx2 + yy2));
}
//-----------------------------------------------------------------------------
/*!
Decomposes the matrix into a translation vector, a rotation Euler angle vector 
and the scaling factors using polar decomposition introduced by Ken Shoemake.
See the paper in lib-SLExternal/Shoemake/polar-decomp.pdf
*/
template<class T>
void SLMat4<T>::decompose(SLVec3f &trans, SLVec3f &rotEulerRAD, SLVec3f &scale)
{  
    SLMat3f rotMat;
    decompose(trans, rotMat, scale);

    SLfloat rotAngleXRAD, rotAngleYRAD, rotAngleZRAD;
    rotMat.toEulerAnglesZYX(rotAngleZRAD, rotAngleYRAD, rotAngleXRAD);

    rotEulerRAD.set(rotAngleXRAD, rotAngleYRAD, rotAngleZRAD);
}



//-----------------------------------------------------------------------------
/*!
Prints out the matrix row by row.
*/
template<class T>
void SLMat4<T>::print(const SLchar* str) const
{
    if (str) SL_LOG("%s\n",str);
    SL_LOG("% 3.3f % 3.3f % 3.3f % 3.3f\n",  _m[0],_m[4],_m[ 8],_m[12]);
    SL_LOG("% 3.3f % 3.3f % 3.3f % 3.3f\n",  _m[1],_m[5],_m[ 9],_m[13]);
    SL_LOG("% 3.3f % 3.3f % 3.3f % 3.3f\n",  _m[2],_m[6],_m[10],_m[14]);
    SL_LOG("% 3.3f % 3.3f % 3.3f % 3.3f\n",  _m[3],_m[7],_m[11],_m[15]);
}
//-----------------------------------------------------------------------------
/*!
Prints out the matrix row by row.
*/
template<class T>
SLstring SLMat4<T>::toString() const
{
    SLchar cstr[500];
    sprintf(cstr,
            "% 3.3f % 3.3f % 3.3f % 3.3f\n% 3.3f % 3.3f % 3.3f % 3.3f\n% 3.3f % 3.3f % 3.3f % 3.3f\n% 3.3f % 3.3f % 3.3f % 3.3f",
            _m[0],_m[4],_m[ 8],_m[12],
            _m[1],_m[5],_m[ 9],_m[13],
            _m[2],_m[6],_m[10],_m[14],
            _m[3],_m[7],_m[11],_m[15]);
    SLstring cppstr = cstr;
    return cppstr;
}
//-----------------------------------------------------------------------------
typedef SLMat4<SLfloat>  SLMat4f;
#ifdef SL_HAS_DOUBLE
typedef SLMat4<SLdouble> SLMat4d;
#endif
typedef std::vector<SLMat4f>  SLVMat4f;
//-----------------------------------------------------------------------------
#endif
