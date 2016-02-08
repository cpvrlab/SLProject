//#############################################################################
//  File:      Globals/SLMat3f.cs
//  Purpose:   3 x 3 Matrix for linear 3D transformations
//  Author:    Marcus Hudritsch
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
/// Implements a 3 by 3 matrix template. 9 floats were used instead of the normal 
///[3][3] array. The order is columnwise as in OpenGL
///
///     | 0  3  6 |
/// M = | 1  4  7 |
///     | 2  5  8 |
/// </summary>
public class SLMat3f
{
   /// <summary>matrix elements as a flat float[9] array</summary>
   public float[] m = new float[9];
   
   /// <summary>
   /// Default constructor sets the matrix to the identity matrix
   /// </summary>
   public SLMat3f() {Identity();}
   
   /// <summary>
   /// Copy constructor
   /// </summary>
   public SLMat3f(SLMat3f A) {Set(A);}
   
   /// <summary>
   /// Constructor with 9 float elements row by row
   /// </summary>
   public SLMat3f(float M0, float M3, float M6,
                  float M1, float M4, float M7,
                  float M2, float M5, float M8)
   {  m[0]=M0; m[3]=M3; m[6]=M6; 
      m[1]=M1; m[4]=M4; m[7]=M7; 
      m[2]=M2; m[5]=M5; m[8]=M8;
   }
   
   /// <summary>
   /// Set the matrix to identity
   /// </summary>
   public void Identity()
   {  m[0]=m[4]=m[8]=1;
      m[1]=m[2]=m[3]=m[5]=m[6]=m[7]=0;
   }
   
   /// <summary>
   /// Access operator to the single matrix elements 0-8
   /// </summary>
   public float this[int index]
   {  get	
      {  if (index>=0 && index<9) return m[index];
         else throw new IndexOutOfRangeException();
      }
      set 
      {  if (index>=0 && index<9) m[index] = value;
         else throw new IndexOutOfRangeException();
      }
   }
   
   /// <summary>
   /// Sets the matrix with the matrix A
   /// </summary>
   public void Set(SLMat3f A)
   {  m[0]=A.m[0]; m[3]=A.m[3]; m[6]=A.m[6]; 
      m[1]=A.m[1]; m[4]=A.m[4]; m[7]=A.m[7]; 
      m[2]=A.m[2]; m[5]=A.m[5]; m[8]=A.m[8];
   }
   
   /// <summary>
   /// Sets the matrix with the 9 float
   /// </summary>
   public void Set(float M0, float M3, float M6,
                   float M1, float M4, float M7,
                   float M2, float M5, float M8)
   {  m[0]=M0; m[3]=M3; m[6]=M6;
      m[1]=M1; m[4]=M4; m[7]=M7;
      m[2]=M2; m[5]=M5; m[8]=M8;
   }
   
   /// <summary>
   /// Post multiplies the matrix by matrix A
   /// </summary>   
   public void Multiply(SLMat3f A)
   {      
      //     | 0  3  6 |     | 0  3  6 |
      // M = | 1  4  7 |  X  | 1  4  7 |
      //     | 2  5  8 |     | 2  5  8 |
      
      Set(m[0]*A.m[0] + m[3]*A.m[1] + m[6]*A.m[2],
          m[0]*A.m[3] + m[3]*A.m[4] + m[6]*A.m[5],
          m[0]*A.m[6] + m[3]*A.m[7] + m[6]*A.m[8],
          m[1]*A.m[0] + m[4]*A.m[1] + m[7]*A.m[2],
          m[1]*A.m[3] + m[4]*A.m[4] + m[7]*A.m[5],
          m[1]*A.m[6] + m[4]*A.m[7] + m[7]*A.m[8],
          m[2]*A.m[0] + m[5]*A.m[1] + m[8]*A.m[2],
          m[2]*A.m[3] + m[5]*A.m[4] + m[8]*A.m[5],
          m[2]*A.m[6] + m[5]*A.m[7] + m[8]*A.m[8]);
   }
   
   /// <summary>
   /// Post multiplies the matrix by the vector v and returns a vector
   /// </summary>
   public SLVec3f Multiply(SLVec3f v) 
   {  return new SLVec3f(m[0]*v.x + m[3]*v.y + m[6]*v.z,
                         m[1]*v.x + m[4]*v.y + m[7]*v.z,
                         m[2]*v.x + m[5]*v.y + m[8]*v.z);
   }
   
   /// <summary>
   /// Post multiplies a rotation matrix defined by 
   /// the angle degAng and the rotation axis [axisx,axisy,axisz]
   /// </summary>
   public void Rotate(float degAng, float axisx, float axisy, float axisz)
   {  SLMat3f R = new SLMat3f();
      R.Rotation(degAng, axisx, axisy, axisz);
      Multiply(R);
   }  
   
   /// <summary>
   /// Post multiplies a rotation matrix defined by 
   /// the angle degAng and the rotation axis
   /// </summary>
   public void Rotate(float degAng, SLVec3f axis)
   {  SLMat3f R = new SLMat3f();
      R.Rotation(degAng, axis);
      Multiply(R);
   }
   
   /// <summary>
   /// Post multiplies a scaling matrix defined by the vector [sx,sy,sz]
   /// </summary>
   public void Scale(float sx, float sy, float sz)
   {  SLMat3f S = new SLMat3f();
      S.Scaling(sx, sy, sz);
      Multiply(S);
   }
   
   /// <summary>
   /// Post multiplies a scaling matrix defined by the vector s
   /// </summary>
   public void Scale(SLVec3f s)
   {  SLMat3f S = new SLMat3f();
      S.Scaling(s);
      Multiply(S);
   }
   
   /// <summary>
   /// Returns a copy of the matrix
   /// </summary>
   public SLMat3f Clone()
   {  return new SLMat3f(this);
   }

   /// <summary>
   /// Returns the determinant
   /// </summary>
   public float Det()
   {  return m[0]*(m[4]*m[8] - m[7]*m[5]) -
             m[3]*(m[1]*m[8] - m[7]*m[2]) +
             m[6]*(m[1]*m[5] - m[4]*m[2]);
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
   public SLMat3f Inverse()
   {  
      // Compute determinant as early as possible using these cofactors.      
      float d = Det();

      if (Math.Abs(d) <= 0.0000000001f) 
         throw new DivideByZeroException("Matrix is singular. Inversion impossible.");

      SLMat3f I = new SLMat3f();
      I.m[0] = (m[4]*m[8] - m[7]*m[5]) / d;
      I.m[1] = (m[7]*m[2] - m[1]*m[8]) / d;
      I.m[2] = (m[1]*m[5] - m[4]*m[2]) / d;
      I.m[3] = (m[6]*m[5] - m[3]*m[8]) / d;
      I.m[4] = (m[0]*m[8] - m[6]*m[2]) / d;
      I.m[5] = (m[3]*m[2] - m[0]*m[5]) / d;
      I.m[6] = (m[3]*m[7] - m[6]*m[4]) / d;
      I.m[7] = (m[6]*m[1] - m[0]*m[7]) / d;
      I.m[8] = (m[0]*m[4] - m[3]*m[1]) / d;
      return I;
   }
   
   /// <summary>
   /// Transposes the matrix
   /// </summary>
   public void Transpose()
   {  SLUtils.Swap(ref m[1], ref m[ 3]);
      SLUtils.Swap(ref m[2], ref m[ 6]);
      SLUtils.Swap(ref m[5], ref m[ 7]);
   }
   
   /// <summary>
   /// Returns the transposed of the matrix
   /// </summary>
   public SLMat3f Transposed()
   {  SLMat3f t = new SLMat3f(this);
      t.Transpose();
      return t;
   }
   
   /// <summary>
   /// Sets a rotation matrix
   /// </summary>
   /// <param name="degAng">angle of rotation in degrees</param>
   /// <param name="axis">rotation axis</param>
   public void Rotation(float degAng, SLVec3f axis)
   {  Rotation(degAng, axis.x, axis.y, axis.z);
   }
   
   /// <summary>
   /// Sets a rotation matrix
   /// </summary>
   /// <param name="degAng">angle of rotation in degrees</param>
   /// <param name="axisx">rotation axis-x</param>
   /// <param name="axisy">rotation axis-y</param>
   /// <param name="axisz">rotation axis-z</param>
   public void Rotation(float degAng, float axisx, float axisy, float axisz)
   {  float RadAng = degAng*(float)SLUtils.DEG2RAD;
      float ca=(float)Math.Cos(RadAng);
      float sa=(float)Math.Sin(RadAng);

      if (axisx==1 && axisy==0 && axisz==0)               // about x-axis
      {  m[0]=1; m[3]=0;  m[6]=0;   
         m[1]=0; m[4]=ca; m[7]=-sa; 
         m[2]=0; m[5]=sa; m[8]=ca; 
      } else 
      if (axisx==0 && axisy==1 && axisz==0)               // about y-axis
      {  m[0]=ca;  m[3]=0; m[6]=sa; 
         m[1]=0;   m[4]=1; m[7]=0;  
         m[2]=-sa; m[5]=0; m[8]=ca;
      } else 
      if (axisx==0 && axisy==0 && axisz==1)               // about z-axis
      {  m[0]=ca; m[3]=-sa; m[6]=0; 
         m[1]=sa; m[4]=ca;  m[7]=0; 
         m[2]=0;  m[5]=0;   m[8]=1;
      } else                                                // arbitrary axis
      {  float l = axisx*axisx + axisy*axisy + axisz*axisz;  // length squared
         float x=axisx, y=axisy, z=axisz;
         if ((l > 1.0001 || l < 0.9999) && l!=0)
         {  l= 1.0f/(float)Math.Sqrt(l);
            x*=l; y*=l; z*=l;
         }
         float xy=x*y, yz=y*z, xz=x*z, xx=x*x, yy=y*y, zz=z*z;
         m[0]=xx + ca*(1-xx);     m[3]=xy - xy*ca - z*sa;  m[6]=xz - xz*ca + y*sa;
         m[1]=xy - xy*ca + z*sa;  m[4]=yy + ca*(1-yy);     m[7]=yz - yz*ca - x*sa;
         m[2]=xz - xz*ca - y*sa;  m[5]=yz - yz*ca + x*sa;  m[8]=zz + ca*(1-zz);
      }
   }
    
   /// <summary>
   /// Sets a scaling matrix
   /// </summary>
   /// <param name="s">scaling factors</param>
   public void Scaling(SLVec3f s)
   {  Scaling(s.x, s.y, s.z);
   }
   
   /// <summary>
   /// Sets a scaling matrix
   /// </summary>
   /// <param name="sx">scaling factor in x-direction</param>
   /// <param name="sy">scaling factor in y-direction</param>
   /// <param name="sz">scaling factor in z-direction</param>
   public void Scaling(float sx, float sy, float sz)
   {  m[0]=sx; m[3]=0;  m[6]=0;   
      m[1]=0;  m[4]=sy; m[7]=0; 
      m[2]=0;  m[5]=0;  m[8]=sz;
   }

   /// <summary>
   /// Returns the trace of the matrix
   /// </summary>
   float Trace()
   {  
      return m[0] + m[4] + m[8];
   }

   /// <summary>
   /// Conversion to axis and angle in radians
   /// The matrix must be a rotation matrix for this functions to be valid. The last 
   /// function uses Gram-Schmidt orthonormalization applied to the columns of the 
   /// rotation matrix. The angle must be in radians, not degrees.
   /// </summary>
   /// <param name="angleDEG"></param>
   /// <param name="axis"></param>
   public void ToAngleAxis(out float angleDEG, out SLVec3f axis)
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
    
      float  tr       = Trace();
      float  cs       = 0.5f * (tr - 1.0f);
      double angleRAD = Math.Acos(cs);  // in [0,PI]
       
      // Init axis
      axis = SLVec3f.XAxis;

      if (angleRAD > 0)
      {
         if (angleRAD < Math.PI)
         {  axis.x = m[5] - m[7];
            axis.y = m[6] - m[2];
            axis.z = m[1] - m[3];
            axis.Normalize();
         }
         else
         {
            // angle is PI
            float halfInverse;
            if (m[0] >= m[4])
            {
                  // r00 >= r11
                  if (m[0] >= m[8])
                  {  // r00 is maximum diagonal term
                     axis.x = 0.5f * (float)Math.Sqrt(1 + m[0] - m[4] - m[8]);
                     halfInverse = 0.5f / axis.x;
                     axis.y = halfInverse * m[3];
                     axis.z = halfInverse * m[6];
                  }
                  else
                  {  // r22 is maximum diagonal term
                     axis.z = 0.5f * (float)Math.Sqrt(1 + m[8] - m[0] - m[4]);
                     halfInverse = 0.5f / axis.z;
                     axis.x = halfInverse * m[6];
                     axis.y = halfInverse * m[7];
                  }
            }
            else
            {
                  // r11 > r00
                  if (m[4] >= m[8])
                  {  // r11 is maximum diagonal term
                     axis.y = 0.5f * (float)Math.Sqrt(1 + m[4] - m[0] - m[8]);
                     halfInverse  = 0.5f / axis.y;
                     axis.x = halfInverse * m[3];
                     axis.z = halfInverse * m[7];
                  }
                  else
                  {  // r22 is maximum diagonal term
                     axis.z = 0.5f * (float)Math.Sqrt(1 + m[8] - m[0] - m[4]);
                     halfInverse = 0.5f / axis.z;
                     axis.x = halfInverse * m[6];
                     axis.y = halfInverse * m[7];
                  }
            }
         }
      }
      angleDEG = (float)(angleRAD * SLUtils.RAD2DEG);
   }
   
   #region Operators
   /// <summary>
   /// Matrix - Vector multiplication
   /// </summary>
   /// <param name="u">A <see cref="SLMat3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance transormed by the matrix m</returns>
   /// <summary>
   public static SLVec3f operator*(SLMat3f m, SLVec3f v)
   {
      return m.Multiply(v);
   }
   /// <summary>
   /// Matrix - Matrix multiplication
   /// </summary>
   /// <param name="u">A <see cref="SLMat3f"/> instance.</param>
   /// <param name="v">A <see cref="SLMat3f"/> instance.</param>
   /// <returns>A new <see cref="SLMat3f"/> instance multiplied by the matrix m</returns>
   /// <summary>
   public static SLMat3f operator*(SLMat3f m1, SLMat3f m2)
   {  
      SLMat3f m = new SLMat3f(m1);
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
             m[8].GetHashCode();
   }

   /// <summary>
   /// Returns a string representation of this object.
   /// </summary>
   /// <returns>A string representation of this object.</returns>
   public override string ToString()
   {
      return m[0].ToString("0.00") + " " + m[3].ToString("0.00") + " " + m[6].ToString("0.00") + "\n" +
             m[1].ToString("0.00") + " " + m[4].ToString("0.00") + " " + m[7].ToString("0.00") + "\n" +
             m[2].ToString("0.00") + " " + m[5].ToString("0.00") + " " + m[8].ToString("0.00");
   }

   /// <summary>
   /// Returns a value indicating whether this instance is equal to
   /// the specified object.
   /// </summary>
   /// <param name="obj">An object to compare to this instance.</param>
   /// <returns>True if <paramref name="obj"/> is a <see cref="SLMat3f"/> and has the same values as this instance; otherwise, False.</returns>
   public override bool Equals(object obj)
   {
      if (obj is SLMat3f)
      {
         SLMat3f m = (SLMat3f)obj;
         return (this.m[0] == m.m[0]) && 
                (this.m[1] == m.m[1]) && 
                (this.m[2] == m.m[2]) && 
                (this.m[3] == m.m[3]) && 
                (this.m[4] == m.m[4]) && 
                (this.m[5] == m.m[5]) && 
                (this.m[6] == m.m[6]) && 
                (this.m[7] == m.m[7]);
      }
      return false;
   }
   #endregion
}