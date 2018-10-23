//#############################################################################
//  File:      Globals/SLMat4f.cs
//  Purpose:   4 x 4 Matrix for affine transformations
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

using System;
using System.Collections;
using System.Text;
using System.Drawing;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;

/// <summary>
/// Implements a 4 by 4 matrix class for affine transformations.
/// 16 floats were used instead of the normal[4][4] to be compliant with OpenGL. 
/// OpenGL uses premultiplication with column vectors. These matrices can be fed 
/// directly into the OpenGL matrix stack with glLoadMatrix or glMultMatrix. The
/// index layout is as follows:
/// 
///     | 0  4  8 12 |
///     | 1  5  9 13 |
/// M = | 2  6 10 14 |
///     | 3  7 11 15 |
///     
/// Vectors are interpreted as column vectors when applying matrix multiplications. 
/// This means a vector is as a single column, 4-row matrix. The result is that the 
/// transformations implemented by the matrices happens right-to-left e.g. if 
/// vector V is to be transformed by M1 then M2 then M3, the calculation would 
/// be M3 * M2 * M1 * V. The order that matrices are concatenated is vital 
/// since matrix multiplication is not commutative, i.e. you can get a different 
/// result if you concatenate in the wrong order.
/// The use of column vectors and right-to-left ordering is the standard in most 
/// mathematical texts, and is the same as used in OpenGL. It is, however, the 
/// opposite of Direct3D, which has inexplicably chosen to differ from the 
/// accepted standard and uses row vectors and left-to-right matrix multiplication.
/// </summary>
public class SLMat4f
{
   /// <summary>matrix elements as a flat float[16] array</summary>
   public float[] m = new float[16];
   
   /// <summary>
   /// Default constructor sets the matrix to the identity matrix
   /// </summary>
   public SLMat4f() {Identity();}
   
   /// <summary>
   /// Copy constructor
   /// </summary>
   public SLMat4f(SLMat4f A) {Set(A);}
   
   /// <summary>
   /// Constructor with 16 float elements row by row
   /// </summary>
   public SLMat4f(float M0, float M4, float M8 , float M12,
                  float M1, float M5, float M9 , float M13,
                  float M2, float M6, float M10, float M14,
                  float M3, float M7, float M11, float M15)
   {  m[0]=M0; m[4]=M4; m[ 8]=M8;  m[12]=M12;
      m[1]=M1; m[5]=M5; m[ 9]=M9;  m[13]=M13;
      m[2]=M2; m[6]=M6; m[10]=M10; m[14]=M14;
      m[3]=M3; m[7]=M7; m[11]=M11; m[15]=M15;
   }
   
   /// <summary>
   /// Set the matrix to identity
   /// </summary>
   public void Identity()
   {  m[0]=m[5]=m[10]=m[15]=1;
      m[1]=m[2]=m[3]=m[4]=m[6]=m[7]=m[8]=m[9]=m[11]=m[12]=m[13]=m[14]=0;
   }
   
   /// <summary>
   /// Access operator to the single matrix elements 0-15
   /// </summary>
   public float this[int index]
   {  get	
      {  if (index>=0 && index<16) return m[index];
         else throw new IndexOutOfRangeException();
      }
      set 
      {  if (index>=0 && index<16) m[index] = value;
         else throw new IndexOutOfRangeException();
      }
   }
   
   /// <summary>
   /// Sets the matrix with the matrix A
   /// </summary>
   public void Set(SLMat4f A)
   {  m[0]=A.m[0]; m[4]=A.m[4]; m[ 8]=A.m[8];  m[12]=A.m[12];
      m[1]=A.m[1]; m[5]=A.m[5]; m[ 9]=A.m[9];  m[13]=A.m[13];
      m[2]=A.m[2]; m[6]=A.m[6]; m[10]=A.m[10]; m[14]=A.m[14];
      m[3]=A.m[3]; m[7]=A.m[7]; m[11]=A.m[11]; m[15]=A.m[15];
   }
   
   /// <summary>
   /// Sets the matrix with the 16 float elements
   /// </summary>
   public void Set(float M0, float M4, float M8 , float M12,
                   float M1, float M5, float M9 , float M13,
                   float M2, float M6, float M10, float M14,
                   float M3, float M7, float M11, float M15)
   {  m[0]=M0; m[4]=M4; m[ 8]=M8;  m[12]=M12;
      m[1]=M1; m[5]=M5; m[ 9]=M9;  m[13]=M13;
      m[2]=M2; m[6]=M6; m[10]=M10; m[14]=M14;
      m[3]=M3; m[7]=M7; m[11]=M11; m[15]=M15;
   }

   /// <summary>
   /// Return the linear top-left 3x3-submatrix
   /// </summary>
   /// <returns></returns>
   public SLMat3f Mat3()
   {
      return new SLMat3f(m[0], m[4], m[8],
                         m[1], m[5], m[9],
                         m[2], m[6], m[10]);
   }
   
   /// <summary>
   /// Post multiplies the matrix by matrix A
   /// </summary>   
   public void Multiply(SLMat4f A)
   {      
      //     | 0  4  8 12 |   | 0  4  8 12 |
      //     | 1  5  9 13 |   | 1  5  9 13 |
      // M = | 2  6 10 14 | x | 2  6 10 14 |
      //     | 3  7 11 15 |   | 3  7 11 15 |
      
      Set(m[0]*A.m[ 0] + m[4]*A.m[ 1] + m[8] *A.m[ 2] + m[12]*A.m[ 3], //row 1
          m[0]*A.m[ 4] + m[4]*A.m[ 5] + m[8] *A.m[ 6] + m[12]*A.m[ 7],
          m[0]*A.m[ 8] + m[4]*A.m[ 9] + m[8] *A.m[10] + m[12]*A.m[11],
          m[0]*A.m[12] + m[4]*A.m[13] + m[8] *A.m[14] + m[12]*A.m[15],
          
          m[1]*A.m[ 0] + m[5]*A.m[ 1] + m[9] *A.m[ 2] + m[13]*A.m[ 3], //row 2
          m[1]*A.m[ 4] + m[5]*A.m[ 5] + m[9] *A.m[ 6] + m[13]*A.m[ 7],
          m[1]*A.m[ 8] + m[5]*A.m[ 9] + m[9] *A.m[10] + m[13]*A.m[11],
          m[1]*A.m[12] + m[5]*A.m[13] + m[9] *A.m[14] + m[13]*A.m[15],
          
          m[2]*A.m[ 0] + m[6]*A.m[ 1] + m[10]*A.m[ 2] + m[14]*A.m[ 3], //row 3
          m[2]*A.m[ 4] + m[6]*A.m[ 5] + m[10]*A.m[ 6] + m[14]*A.m[ 7],
          m[2]*A.m[ 8] + m[6]*A.m[ 9] + m[10]*A.m[10] + m[14]*A.m[11],
          m[2]*A.m[12] + m[6]*A.m[13] + m[10]*A.m[14] + m[14]*A.m[15],
          
          m[3]*A.m[ 0] + m[7]*A.m[ 1] + m[11]*A.m[ 2] + m[15]*A.m[ 3], //row 4
          m[3]*A.m[ 4] + m[7]*A.m[ 5] + m[11]*A.m[ 6] + m[15]*A.m[ 7],
          m[3]*A.m[ 8] + m[7]*A.m[ 9] + m[11]*A.m[10] + m[15]*A.m[11],
          m[3]*A.m[12] + m[7]*A.m[13] + m[11]*A.m[14] + m[15]*A.m[15]);
   }
   
   /// <summary>
   /// Post multiplies the matrix by the vector v and returns a vector
   /// with the additional perspective division.
   /// </summary>
   public SLVec3f Multiply(SLVec3f v) 
   {  float W = m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15];
      return new SLVec3f((m[0]*v.x + m[4]*v.y + m[ 8]*v.z + m[12]) / W,
                         (m[1]*v.x + m[5]*v.y + m[ 9]*v.z + m[13]) / W,
                         (m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]) / W);
   }

   /// <summary>
   /// Post multiplies the matrix by the vector v and returns a vector.
   /// </summary>
   public SLVec4f Multiply(SLVec4f v) 
   {  return new SLVec4f(m[0]*v.x + m[4]*v.y + m[ 8]*v.z + m[12]*v.w,
                         m[1]*v.x + m[5]*v.y + m[ 9]*v.z + m[13]*v.w,
                         m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
                         m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]*v.w);

            
   }
   
   /// <summary>
   /// Post multiplies the matrix by the vector v and writes it into 
   /// a point p with x and y
   /// </summary>
   public void Multiply(SLVec3f v, ref System.Drawing.PointF p) 
   {  float W = m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15];
      p.X = (m[0]*v.x + m[4]*v.y + m[ 8]*v.z + m[12]) / W;
      p.Y = (m[1]*v.x + m[5]*v.y + m[ 9]*v.z + m[13]) / W;
   }
   
   /// <summary>
   /// Post multiplies a translation matrix defined by the vector [tx,ty,tz]
   /// </summary>
   public void Translate(float tx, float ty, float tz)
   {  SLMat4f Tr = new SLMat4f();
      Tr.Translation(tx, ty, tz);
      Multiply(Tr);
   }
   
   /// <summary>
   /// Post multiplies a translation matrix defined by the vector t
   /// </summary>
   public void Translate(SLVec3f t)
   {  SLMat4f Tr = new SLMat4f();
      Tr.Translation(t);
      Multiply(Tr);
   }
   
   /// <summary>
   /// Post multiplies a rotation matrix defined by 
   /// the angle degAng and the rotation axis [axisx,axisy,axisz]
   /// </summary>
   public void Rotate(float degAng, float axisx, float axisy, float axisz)
   {  SLMat4f R = new SLMat4f();
      R.Rotation(degAng, axisx, axisy, axisz);
      Multiply(R);
   }  
   
   /// <summary>
   /// Post multiplies a rotation matrix defined by 
   /// the angle degAng and the rotation axis
   /// </summary>
   public void Rotate(float degAng, SLVec3f axis)
   {  SLMat4f R = new SLMat4f();
      R.Rotation(degAng, axis);
      Multiply(R);
   }
   
   /// <summary>
   /// Post multiplies a scaling matrix defined by the vector [sx,sy,sz]
   /// </summary>
   public void Scale(float sx, float sy, float sz)
   {  SLMat4f S = new SLMat4f();
      S.Scaling(sx, sy, sz);
      Multiply(S);
   }
   
   /// <summary>
   /// Post multiplies a scaling matrix defined by the vector s
   /// </summary>
   public void Scale(SLVec3f s)
   {  SLMat4f S = new SLMat4f();
      S.Scaling(s);
      Multiply(S);
   }
   
   /// <summary>
   /// Returns a copy of the matrix
   /// </summary>
   /// <returns></returns>
   public SLMat4f Clone()
   {  return new SLMat4f(this);
   }
   
   /// <summary>
   /// Inverts the matrix
   /// </summary>
   public void Invert()
   {  Set(Inverse());
   }   
   
   /// <summary>
   /// Returns the inverse of the matrix
   /// </summary>
   /// <returns></returns>
   public SLMat4f Inverse()
   {  
      SLMat4f I = new SLMat4f();
   
      // Code from Mesa-2.2\src\glu\project.c
      float det, d12, d13, d23, d24, d34, d41;

      // Inverse = adjoint / det. (See linear algebra texts.)
      // pre-compute 2x2 dets for last two rows when computing
      // cof_actors of first two rows.
      d12 = ( m[2]*m[ 7] - m[ 3]*m[ 6]);
      d13 = ( m[2]*m[11] - m[ 3]*m[10]);
      d23 = ( m[6]*m[11] - m[ 7]*m[10]);
      d24 = ( m[6]*m[15] - m[ 7]*m[14]);
      d34 = (m[10]*m[15] - m[11]*m[14]);
      d41 = (m[14]*m[ 3] - m[15]*m[ 2]);

      I.m[0] =  (m[5]*d34 - m[9]*d24 + m[13]*d23);
      I.m[1] = -(m[1]*d34 + m[9]*d41 + m[13]*d13);
      I.m[2] =  (m[1]*d24 + m[5]*d41 + m[13]*d12);
      I.m[3] = -(m[1]*d23 - m[5]*d13 + m[ 9]*d12);

      // Compute determinant as early as possible using these cof_actors.
      det = m[0]*I.m[0] + m[4]*I.m[1] + m[8]*I.m[2] + m[12]*I.m[3];

      // Run singularity test.
      if (Math.Abs(det) <= 0.00005) 
      {  throw new DivideByZeroException("Matrix is singular. Inversion impossible.");
      } else 
      {  float invDet = 1 / det;
         // Compute rest of inverse.
         I.m[0] *= invDet;
         I.m[1] *= invDet;
         I.m[2] *= invDet;
         I.m[3] *= invDet;

         I.m[4] = -(m[4]*d34 - m[8]*d24 + m[12]*d23)*invDet;
         I.m[5] =  (m[0]*d34 + m[8]*d41 + m[12]*d13)*invDet;
         I.m[6] = -(m[0]*d24 + m[4]*d41 + m[12]*d12)*invDet;
         I.m[7] =  (m[0]*d23 - m[4]*d13 + m[ 8]*d12)*invDet;

         // Pre-compute 2x2 dets for first two rows when computing
         // cofactors of last two rows.
         d12 = m[ 0]*m[ 5] - m[ 1]*m[ 4];
         d13 = m[ 0]*m[ 9] - m[ 1]*m[ 8];
         d23 = m[ 4]*m[ 9] - m[ 5]*m[ 8];
         d24 = m[ 4]*m[13] - m[ 5]*m[12];
         d34 = m[ 8]*m[13] - m[ 9]*m[12];
         d41 = m[12]*m[ 1] - m[13]*m[ 0];

         I.m[ 8] =  (m[7]*d34 - m[11]*d24 + m[15]*d23)*invDet;
         I.m[ 9] = -(m[3]*d34 + m[11]*d41 + m[15]*d13)*invDet;
         I.m[10] =  (m[3]*d24 + m[ 7]*d41 + m[15]*d12)*invDet;
         I.m[11] = -(m[3]*d23 - m[ 7]*d13 + m[11]*d12)*invDet;
         I.m[12] = -(m[6]*d34 - m[10]*d24 + m[14]*d23)*invDet;
         I.m[13] =  (m[2]*d34 + m[10]*d41 + m[14]*d13)*invDet;
         I.m[14] = -(m[2]*d24 + m[ 6]*d41 + m[14]*d12)*invDet;
         I.m[15] =  (m[2]*d23 - m[ 6]*d13 + m[10]*d12)*invDet;
      }
      return I;
   }
   
   /// <summary>
   /// Transposes the matrix
   /// </summary>
   public void Transpose()
   {  SLUtils.Swap(ref m[1], ref m[ 4]);
      SLUtils.Swap(ref m[2], ref m[ 8]);
      SLUtils.Swap(ref m[6], ref m[ 9]);
      SLUtils.Swap(ref m[3], ref m[12]);
      SLUtils.Swap(ref m[7], ref m[13]);
      SLUtils.Swap(ref m[11],ref m[14]);
   }
   
   /// <summary>
   /// Returns the transposed of the matrix
   /// </summary>
   public SLMat4f Transposed()
   {  SLMat4f t = new SLMat4f(this);
      t.Transpose();
      return t;
   }

   /// <summary>
   /// Returns the inverse transposed linear 3x3 submatrix 
   /// that can be used as a normal matrix.
   /// </summary>
   public SLMat3f InverseTransposed()
   {
      SLMat3f it = new SLMat3f(this.Mat3());
      it.Invert();
      it.Transpose();
      return it;
   }

   /// <summary>
   /// Defines a view frustum projection matrix equivalent to OpenGL's glFrustum
   /// </summary>
   /// <param name="l">Distance from the center of proj. (COP) to the LEFT on the near clip plane</param>
   /// <param name="r">Distance from the COP to the RIGHT border on the near clipping plane</param>
   /// <param name="b">Distance from the COP to the BOTTOM border on the near clipping plane</param>
   /// <param name="t">Distance from the COP to the TOP border on the near clipping plane</param>
   /// <param name="n">Distance from the eye to NEAR clipping plane of the view frustum</param>
   /// <param name="f">Distance from the eye to FAR clipping plane of the view frustum</param>
   public void Frustum(float l, float r, float b, float t, float n, float f)
   {  m[0]=(2*n)/(r-l); m[4]=0;           m[8]=(r+l)/(r-l);   m[12]=0;
      m[1]=0;           m[5]=(2*n)/(t-b); m[9]=(t+b)/(t-b);   m[13]=0;
      m[2]=0;           m[6]=0;           m[10]=-(f+n)/(f-n); m[14]=(-2*f*n)/(f-n);
      m[3]=0;           m[7]=0;           m[11]=-1;           m[15]=0;
   }
   
   /// <summary>
   /// Defines a view frustum projection matrix for a perspective projection
   /// This method is equivalent to the OpenGL function gluPerspective except that
   /// instead of the window aspect the window width and height have to be passed.
   /// </summary>
   /// <param name="fov">Vertical field of view angle (zoom angle)</param>
   /// <param name="apect">aspect ration of viewport = width / height</param>
   /// <param name="n">Distance from the eye to near clipping plane of the view frustum</param>
   /// <param name="f">Distance from the eye to far clipping plane of the view frustum</param>
   public void Perspective(float fov, float aspect, float n, float f)
   {  
      float t = (float)(Math.Tan(fov * SLUtils.DEG2RAD * 0.5) * n);
      float b = -t;
      float r = t*aspect;
      float l = -r;
      Frustum(l,r,b,t,n,f);
   }
   
   /// <summary>
   /// Defines a viewport matrix as it is defined by OpenGL glViewport
   /// </summary>
   /// <param name="x">left window coord. in px.</param>
   /// <param name="y">top window coord. in px.</param>
   /// <param name="ww">window width in px.</param>
   /// <param name="wh">window height in px.</param>
   /// <param name="n">near depth range (default 0)</param>
   /// <param name="f">far depth range (default 1)</param>
   public void Viewport(float x,  float y, float ww, float wh, float n,  float f)
   {  float ww2 = ww*0.5f;
      float wh2 = wh*0.5f;
      // negate the first wh because windows has topdown window coords
      m[0]=ww2; m[4]=0;    m[8] =0;          m[12]=ww2+x;
      m[1]=0;   m[5]=-wh2; m[9] =0;          m[13]=wh2+y;
      m[2]=0;   m[6]=0;    m[10]=(f-n)*0.5f; m[14]=(f+n)*0.5f;
      m[3]=0;   m[7]=0;    m[11]=0;          m[15]=1;
   }
   
   /// <summary>
   /// Defines the view matrix with an eye position, a look at point and an up vector.
   /// This method is equivalent to the OpenGL function gluLookAt.
   /// If Up is a zero vector a default up vector is calculated with a default 
   /// look-right vector (VZ) that lies in the x-z plane.
   /// </summary>
   /// <param name="Eye">Eye Vector to the position of the eye (view point)</param>
   /// <param name="At">At Vector to the target point</param>
   /// <param name="Up">Up Vector that points from the viewpoint upwards.</param>
   public void LookAt(SLVec3f Eye, SLVec3f At, SLVec3f Up)
   {  SLVec3f VX, VY, VZ, VT, ZERO;
      //SLMat3<T> xz(0.0, 0.0, 1.0,         // matrix that transforms YZ into a 
      //             0.0, 0.0, 0.0,         // vector that is perpendicular to YZ and 
      //            -1.0, 0.0, 0.0);        // lies in the x-z plane

      VZ = Eye-At; 
      VZ.Normalize(); 
      VX = new SLVec3f();
      ZERO = new SLVec3f();
      
      if (Up==ZERO)  
      {  VX.x = VZ.z;
         VX.y = 0;
         VX.z =-1*VZ.x;         
      } else
      {  VX = Up.Cross(VZ);
      }
      VY = SLVec3f.CrossProduct(VZ, VX);
      VX.Normalize(); 
      VY.Normalize(); 
      VZ.Normalize(); 
      VT = -Eye;

      Set(VX.x, VX.y, VX.z, SLVec3f.DotProduct(VX,VT),
          VY.x, VY.y, VY.z, SLVec3f.DotProduct(VY,VT),
          VZ.x, VZ.y, VZ.z, SLVec3f.DotProduct(VZ,VT),
          0.0f, 0.0f, 0.0f, 1.0f);
   }
   
   /// <summary>
   /// Defines the view matrix with an eye position, a look at point and an up vector.
   /// This method is equivalent to the OpenGL function gluLookAt.
   /// If Up is a zero vector a default up vector is calculated with a default 
   /// look-right vector (VZ) that lies in the x-z plane.
   /// </summary>
   /// <param name="eyeX"></param>
   /// <param name="eyeY"></param>
   /// <param name="eyeZ"></param>
   /// <param name="atX"></param>
   /// <param name="atY"></param>
   /// <param name="atZ"></param>
   /// <param name="upX"></param>
   /// <param name="upY"></param>
   /// <param name="upZ"></param>
   public void LookAt(float eyeX, float eyeY, float eyeZ,
                      float atX, float atY, float atZ,
                      float upX, float upY, float upZ)
   {  SLVec3f eye = new SLVec3f(eyeX, eyeY,eyeZ);
      SLVec3f at = new SLVec3f(atX, atY, atZ);
      SLVec3f up = new SLVec3f(upX, upY, upZ);
      LookAt(eye, at, up); 
   }
   
   /// <summary>
   /// Retrieves the camera vectors eye, at and up if this matrix would be a view matrix
   /// </summary>
   /// <param name="Eye"></param>
   /// <param name="At"></param>
   /// <param name="Up"></param>
   public void GetLookAt(ref SLVec3f Eye, ref SLVec3f At, ref SLVec3f Up)
   {  
      SLMat4f invRot = new SLMat4f(this);
      SLVec3f translation = new SLVec3f(m[12],m[13],m[14]);
      invRot.m[12] = 0; 
      invRot.m[13] = 0;
      invRot.m[14] = 0;
      invRot.Transpose();
      Eye.Set(invRot.Multiply(-translation)); // vector to the eye
      Up.Set(m[1], m[5], m[9]);     // normalized look up vector
      At.Set(-m[2],-m[6],-m[10]);   // normalized look at vector
   }
   
   /// <summary>
   /// Defines a translation vector with the vector t
   /// </summary>
   void Translation(SLVec3f t)
   {  Translation(t.x, t.y, t.z, false);
   }
   
   /// <summary>
   /// Defines a translation vector with the vector t
   /// </summary>
   /// <param name="t">translation vector</param>
   /// <param name="keepLinear">flag if linear 3x3 submatrix should be kept</param>
   void Translation(SLVec3f t, bool keepLinear)
   {  Translation(t.x, t.y, t.z, keepLinear);
   }
   
   /// <summary>
   /// Defines a translation vector with the vector [tx,tx,tz]
   /// </summary>
   /// <param name="tx"></param>
   /// <param name="ty"></param>
   /// <param name="tz"></param>
   void Translation(float tx, float ty, float tz)
   {  Translation(tx, ty, tz, false);
   }
   
   /// <summary>
   /// Defines a translation vector with the vector [tx,tx,tz]
   /// </summary>
   /// <param name="tx"></param>
   /// <param name="ty"></param>
   /// <param name="tz"></param>
   /// <param name="keepLinear">flag if linear 3x3 submatrix should be kept</param>
   void Translation(float tx, float ty, float tz, bool keepLinear)
   {  m[12]=tx;
      m[13]=ty;
      m[14]=tz;
      if (!keepLinear)
      {  m[0]=1; m[4]=0;  m[8]=0;
         m[1]=0; m[5]=1;  m[9]=0;  
         m[2]=0; m[6]=0;  m[10]=1; 
         m[3]=0; m[7]=0;  m[11]=0; m[15]=1;
      }
   }
   
   /// <summary>
   /// Sets a rotation matrix
   /// </summary>
   /// <param name="degAng">angle of rotation in degrees</param>
   /// <param name="axis">rotation axis</param>
   void Rotation(float degAng, SLVec3f axis)
   {  Rotation(degAng, axis.x, axis.y, axis.z, false);
   }   
   
   /// <summary>
   /// Sets a rotation matrix
   /// </summary>
   /// <param name="degAng">angle of rotation in degrees</param>
   /// <param name="axis">rotation axis</param>
   /// <param name="keepTranslation">flag if the translation should kept</param>
   void Rotation(float degAng, SLVec3f axis, bool keepTranslation)
   {  Rotation(degAng, axis.x, axis.y, axis.z, keepTranslation);
   }
   
   /// <summary>
   /// Sets a rotation matrix
   /// </summary>
   /// <param name="degAng">angle of rotation in degrees</param>
   /// <param name="axisx">rotation axis-x</param>
   /// <param name="axisy">rotation axis-y</param>
   /// <param name="axisz">rotation axis-z</param>
   void Rotation(float degAng, float axisx, float axisy, float axisz)
   {  Rotation(degAng, axisx, axisy, axisz, false);
   }
   
   /// <summary>
   /// Sets a rotation matrix
   /// </summary>
   /// <param name="degAng">angle of rotation in degrees</param>
   /// <param name="axisx">rotation axis-x</param>
   /// <param name="axisy">rotation axis-y</param>
   /// <param name="axisz">rotation axis-z</param>
   /// <param name="keepTranslation">flag if the translation should kept</param>
   void Rotation(float degAng, float axisx, float axisy, float axisz, bool keepTranslation)
   {  float RadAng = degAng*(float)SLUtils.DEG2RAD;
      float ca=(float)Math.Cos(RadAng);
      float sa=(float)Math.Sin(RadAng);
      
      if (axisx==1 && axisy==0 && axisz==0)               // about x-axis
      {  m[0]=1; m[4]=0;  m[8]=0;   
         m[1]=0; m[5]=ca; m[9]=-sa; 
         m[2]=0; m[6]=sa; m[10]=ca; 
      } else 
      if (axisx==0 && axisy==1 && axisz==0)               // about y-axis
      {  m[0]=ca;  m[4]=0; m[8]=sa; 
         m[1]=0;   m[5]=1; m[9]=0;  
         m[2]=-sa; m[6]=0; m[10]=ca;
      } else 
      if (axisx==0 && axisy==0 && axisz==1)               // about z-axis
      {  m[0]=ca; m[4]=-sa; m[8]=0; 
         m[1]=sa; m[5]=ca;  m[9]=0; 
         m[2]=0;  m[6]=0;   m[10]=1;
      } else                                                   // arbitrary axis
      {  float len = axisx*axisx + axisy*axisy + axisz*axisz; // length squared
         float x, y, z;
         x=axisx;
         y=axisy;
         z=axisz;
         if (len > 1.0001 || len < 0.9999 && len!=0)
         {  len = 1/(float)Math.Sqrt(len);
            x*=len; y*=len; z*=len;
         }
         float xy=x*y, yz=y*z, xz=x*z, xx=x*x, yy=y*y, zz=z*z;
         m[0]=xx + ca*(1-xx);     m[4]=xy - xy*ca - z*sa;  m[8] =xz - xz*ca + y*sa;
         m[1]=xy - xy*ca + z*sa;  m[5]=yy + ca*(1-yy);     m[9] =yz - yz*ca - x*sa;
         m[2]=xz - xz*ca - y*sa;  m[6]=yz - yz*ca + x*sa;  m[10]=zz + ca*(1-zz);
      }
      m[3]=m[7]=m[11]=0; m[15]=1;

      if (!keepTranslation) {m[12] = m[13] = m[14] = 0;} 
   }
   
   /// <summary>
   /// Sets a scaling matrix
   /// </summary>
   /// <param name="s">scaling factors</param>
   void Scaling(SLVec3f s)
   {  Scaling(s.x, s.y, s.z, false);
   }
   
   /// <summary>
   /// Sets a scaling matrix
   /// </summary>
   /// <param name="s">scaling factors</param>
   /// <param name="keepTranslation">flag if the translation should kept</param>
   void Scaling(SLVec3f s, bool keepTranslation)
   {  Scaling(s.x, s.y, s.z, keepTranslation);
   }
   
   /// <summary>
   /// Sets a scaling matrix
   /// </summary>
   /// <param name="sx">scaling factor in x-direction</param>
   /// <param name="sy">scaling factor in y-direction</param>
   /// <param name="sz">scaling factor in z-direction</param>
   void Scaling(float sx, float sy, float sz)
   {  Scaling(sx, sy, sz, false);
   }
   
   /// <summary>
   /// Sets a scaling matrix
   /// </summary>
   /// <param name="sx">scaling factor in x-direction</param>
   /// <param name="sy">scaling factor in y-direction</param>
   /// <param name="sz">scaling factor in z-direction</param>
   /// <param name="keepTranslation">flag if translation vector should remain.</param>
   void Scaling(float sx, float sy, float sz, bool keepTranslation)
   {  m[0]=sx; m[4]=0;  m[8]=0;   
      m[1]=0;  m[5]=sy; m[9]=0; 
      m[2]=0;  m[6]=0;  m[10]=sz;
      m[3]=0;  m[7]=0;  m[11]=0; m[15]=1;
      if (!keepTranslation) {m[12] = m[13] = m[14] = 0;}
   }
   
   #region Operators
   /// <summary>
   /// Matrix - Vector multiplication
   /// </summary>
   /// <param name="u">A <see cref="SLMat4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance transormed by the matrix m</returns>
   /// <summary>
   public static SLVec3f operator*(SLMat4f m, SLVec3f v)
   {
      return m.Multiply(v);
   }
   /// <summary>
   /// Matrix - Vector multiplication
   /// </summary>
   /// <param name="u">A <see cref="SLMat4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance transormed by the matrix m</returns>
   /// <summary>
   public static SLVec4f operator*(SLMat4f m, SLVec4f v)
   {
      return m.Multiply(v);
   }
   /// <summary>
   /// Matrix - Matrix multiplication
   /// </summary>
   /// <param name="u">A <see cref="SLMat4f"/> instance.</param>
   /// <param name="v">A <see cref="SLMat4f"/> instance.</param>
   /// <returns>A new <see cref="SLMat4f"/> instance multiplied by the matrix m</returns>
   /// <summary>
   public static SLMat4f operator*(SLMat4f m1, SLMat4f m2)
   {  
      SLMat4f m = new SLMat4f(m1);
      m.Multiply(m2);
      return m;
   }
   #endregion

   #region Overrides
   /// <summary>
   /// Returns the hashcode for this instance.
   /// </summary>
   /// <returns>A 32-bit signed integer hash code.</returns>
   public override int GetHashCode()
   {
      return m[0].GetHashCode() ^ 
             m[1].GetHashCode() ^ 
             m[2].GetHashCode() ^
             m[3].GetHashCode() ^
             m[4].GetHashCode() ^
             m[5].GetHashCode() ^
             m[6].GetHashCode() ^
             m[7].GetHashCode() ^
             m[8].GetHashCode() ^
             m[9].GetHashCode() ^
             m[10].GetHashCode() ^
             m[11].GetHashCode() ^
             m[12].GetHashCode() ^
             m[13].GetHashCode() ^
             m[14].GetHashCode() ^
             m[15].GetHashCode();
   }
   /// <summary>
   /// Returns a string representation of this object.
   /// </summary>
   /// <returns>A string representation of this object.</returns>
   public override string ToString()
   {
      return m[ 0].ToString("0.00") + " " + m[ 4].ToString("0.00") + " " + 
             m[ 8].ToString("0.00") + " " + m[12].ToString("0.00") + "\n" + 
             m[ 1].ToString("0.00") + " " + m[ 5].ToString("0.00") + " " + 
             m[ 9].ToString("0.00") + " " + m[13].ToString("0.00") + "\n" +
             m[ 2].ToString("0.00") + " " + m[ 6].ToString("0.00") + " " + 
             m[10].ToString("0.00") + " " + m[14].ToString("0.00");
   }
   /// <summary>
   /// Returns a value indicating whether this instance is equal to
   /// the specified object.
   /// </summary>
   /// <param name="obj">An object to compare to this instance.</param>
   /// <returns>True if <paramref name="obj"/> is a <see cref="SLMat3f"/> and has the same values as this instance; otherwise, False.</returns>m[3].ToString("0.00") + " " + m[7].ToString("0.00") + " " + m[11].ToString("0.00") + " " + m[15].ToString("0.00"));
   public override bool Equals(object obj)
   {
      if (obj is SLMat4f)
      {
         SLMat4f m = (SLMat4f)obj;
         return (this.m[0] == m.m[0]) && 
                (this.m[1] == m.m[1]) && 
                (this.m[2] == m.m[2]) && 
                (this.m[3] == m.m[3]) && 
                (this.m[4] == m.m[4]) && 
                (this.m[5] == m.m[5]) && 
                (this.m[6] == m.m[6]) && 
                (this.m[7] == m.m[7]) && 
                (this.m[8] == m.m[8]) && 
                (this.m[9] == m.m[9]) && 
                (this.m[10] == m.m[10]) && 
                (this.m[11] == m.m[11]) && 
                (this.m[12] == m.m[12]) && 
                (this.m[13] == m.m[13]) && 
                (this.m[14] == m.m[14]) && 
                (this.m[15] == m.m[15]);
      }
      return false;
   }
   #endregion
}