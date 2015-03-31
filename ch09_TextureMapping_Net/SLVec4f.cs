//#############################################################################
//  File:      Globals/Math/SLVec4f.cs
//  Purpose:   4 Component vector class
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
/// 4D vector class for standard 3D homogeneous vector algebra.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public class SLVec4f
{
   #region public fields
   /// <summary>x-component of vector</summary>
   public float x;
   /// <summary>y-component of vector</summary>
   public float y;
   /// <summary>z-component of vector</summary>
   public float z;
   /// <summary>w-component of vector</summary>
   public float w;
   #endregion

   #region Constructors
   /// <summary>
   /// Default constructor that inits all components to zero
   /// </summary>
   public SLVec4f()
   {
      this.x = 0;
      this.y = 0;
      this.z = 0;
      this.w = 1;
   }
   
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec4f"/> class with the specified coordinates.
   /// </summary>
   /// <param name="x">The vector's x coordinate.</param>
   /// <param name="y">The vector's y coordinate.</param>
   /// <param name="z">The vector's z coordinate.</param>
   public SLVec4f(float x, float y, float z)
   {
      this.x = x;
      this.y = y;
      this.z = z;
      this.w = 1;
   }
   
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec4f"/> class with the specified coordinates.
   /// </summary>
   /// <param name="x">The vector's x coordinate.</param>
   /// <param name="y">The vector's y coordinate.</param>
   /// <param name="z">The vector's z coordinate.</param>
   public SLVec4f(float x, float y, float z, float w)
   {
      this.x = x;
      this.y = y;
      this.z = z;
      this.w = w;
   }

   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec4f"/> class with the specified coordinates.
   /// </summary>
   /// <param name="coordinates">An array containing the coordinate parameters.</param>
   public SLVec4f(float[] coordinates)
   {
      Debug.Assert(coordinates != null);
      Debug.Assert(coordinates.Length >= 4);

      this.x = coordinates[0];
      this.y = coordinates[1];
      this.z = coordinates[2];
      this.w = coordinates[3];
   }
   
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec4f"/> class using coordinates from a given <see cref="SLVec3f"/> instance.
   /// </summary>
   /// <param name="vector">A <see cref="SLVec3f"/> to get the coordinates from.</param>
   public SLVec4f(SLVec3f vector)
   {
      this.x = vector.x;
      this.y = vector.y;
      this.z = vector.z;
      this.w = 1;
   }  
   
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec4f"/> class using coordinates from a given <see cref="SLVec3f"/> instance.
   /// </summary>
   /// <param name="vector">A <see cref="SLVec4f"/> to get the coordinates from.</param>
   public SLVec4f(SLVec4f vector)
   {
      this.x = vector.x;
      this.y = vector.y;
      this.z = vector.z;
      this.w = vector.w;
   }
   #endregion

   #region Constants
   /// <summary>
   /// 3-Dimentional float-precision floating point zero vector.
   /// </summary>
   public static readonly SLVec4f Zero	= new SLVec4f(0.0f, 0.0f, 0.0f);
   /// <summary>
   /// 3-Dimentional float-precision floating point X-Axis vector.
   /// </summary>
   public static readonly SLVec4f XAxis	= new SLVec4f(1.0f, 0.0f, 0.0f);
   /// <summary>
   /// 3-Dimentional float-precision floating point Y-Axis vector.
   /// </summary>
   public static readonly SLVec4f YAxis	= new SLVec4f(0.0f, 1.0f, 0.0f);
   /// <summary>
   /// 3-Dimentional float-precision floating point Y-Axis vector.
   /// </summary>
   public static readonly SLVec4f ZAxis	= new SLVec4f(0.0f, 0.0f, 1.0f);
   #endregion

   #region Public Static Vector Arithmetics
   /// <summary>
   /// Adds two vectors.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the sum.</returns>
   public static SLVec4f Add(SLVec4f v, SLVec4f w)
   {
      return new SLVec4f(v.x + w.x, v.y + w.y, v.z + w.z, v.w + w.w);
   }
   /// <summary>
   /// Adds a vector and a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the sum.</returns>
   public static SLVec4f Add(SLVec4f v, float s)
   {
      return new SLVec4f(v.x + s, v.y + s, v.z + s, v.w + s);
   }
   /// <summary>
   /// Adds two vectors and put the result in the third vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance</param>
   /// <param name="w">A <see cref="SLVec4f"/> instance to hold the result.</param>
   public static void Add(SLVec4f u, SLVec4f v, SLVec4f w)
   {
      w.x = u.x + v.x;
      w.y = u.y + v.y;
      w.z = u.z + v.z;
      w.w = u.w + v.w;
   }
   /// <summary>
   /// Adds a vector and a scalar and put the result into another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance to hold the result.</param>
   public static void Add(SLVec4f u, float s, SLVec4f v)
   {
      v.x = u.x + s;
      v.y = u.y + s;
      v.z = u.z + s;
      v.w = u.w + s;
   }
   /// <summary>
   /// Subtracts a vector from a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the difference.</returns>
   /// <remarks>
   ///	result[i] = m_v[i] - w[i].
   /// </remarks>
   public static SLVec4f Subtract(SLVec4f v, SLVec4f w)
   {
      return new SLVec4f(v.x - w.x, v.y - w.y, v.z - w.z, v.w - w.w);
   }
   /// <summary>
   /// Subtracts a scalar from a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = m_v[i] - s
   /// </remarks>
   public static SLVec4f Subtract(SLVec4f v, float s)
   {
      return new SLVec4f(v.x - s, v.y - s, v.z - s, v.w - s);
   }
   /// <summary>
   /// Subtracts a vector from a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = s - m_v[i]
   /// </remarks>
   public static SLVec4f Subtract(float s, SLVec4f v)
   {
      return new SLVec4f(s - v.x, s - v.y, s - v.z, s - v.w);
   }
   /// <summary>
   /// Subtracts a vector from a second vector and puts the result into a third vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance</param>
   /// <param name="w">A <see cref="SLVec4f"/> instance to hold the result.</param>
   /// <remarks>
   ///	w[i] = m_v[i] - w[i].
   /// </remarks>
   public static void Subtract(SLVec4f u, SLVec4f v, SLVec4f w)
   {
      w.x = u.x - v.x;
      w.y = u.y - v.y;
      w.z = u.z - v.z;
      w.w = u.w - v.w;
   }
   /// <summary>
   /// Subtracts a vector from a scalar and put the result into another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = u[i] - s
   /// </remarks>
   public static void Subtract(SLVec4f u, float s, SLVec4f v)
   {
      v.x = u.x - s;
      v.y = u.y - s;
      v.z = u.z - s;
      v.w = u.w - s;
   }
   /// <summary>
   /// Subtracts a scalar from a vector and put the result into another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = s - u[i]
   /// </remarks>
   public static void Subtract(float s, SLVec4f u, SLVec4f v)
   {
      v.x = s - u.x;
      v.y = s - u.y;
      v.z = s - u.z;
      v.w = s - u.w;
   }
   /// <summary>
   /// Divides a vector by another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the quotient.</returns>
   /// <remarks>
   ///	result[i] = u[i] / m_v[i].
   /// </remarks>
   public static SLVec4f Divide(SLVec4f u, SLVec4f v)
   {
      return new SLVec4f(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
   }
   /// <summary>
   /// Divides a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = m_v[i] / s;
   /// </remarks>
   public static SLVec4f Divide(SLVec4f v, float s)
   {
      return new SLVec4f(v.x / s, v.y / s, v.z / s, v.w / s);
   }
   /// <summary>
   /// Divides a scalar by a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = s / m_v[i]
   /// </remarks>
   public static SLVec4f Divide(float s, SLVec4f v)
   {
      return new SLVec4f(s / v.x, s/ v.y, s / v.z, s / v.w);
   }
   /// <summary>
   /// Divides a vector by another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec4f"/> instance to hold the result.</param>
   /// <remarks>
   /// w[i] = u[i] / m_v[i]
   /// </remarks>
   public static void Divide(SLVec4f u, SLVec4f v, SLVec4f w)
   {
      w.x = u.x / v.x;
      w.y = u.y / v.y;
      w.z = u.z / v.z;
      w.w = u.w / v.w;
   }
   /// <summary>
   /// Divides a vector by a scalar.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = u[i] / s
   /// </remarks>
   public static void Divide(SLVec4f u, float s, SLVec4f v)
   {
      v.x = u.x / s;
      v.y = u.y / s;
      v.z = u.z / s;
      v.w = u.w / s;
   }
   /// <summary>
   /// Divides a scalar by a vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = s / u[i]
   /// </remarks>
   public static void Divide(float s, SLVec4f u, SLVec4f v)
   {
      v.x = s / u.x;
      v.y = s / u.y;
      v.z = s / u.z;
      v.w = s / u.w;
   }
   /// <summary>
   /// Multiplies a vector by a scalar.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the result.</returns>
   public static SLVec4f Multiply(SLVec4f u, float s)
   {
      return new SLVec4f(u.x * s, u.y * s, u.z * s, u.w * s);
   }
   /// <summary>
   /// Multiplies a vector by a scalar and put the result in another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance to hold the result.</param>
   public static void Multiply(SLVec4f u, float s, SLVec4f v)
   {
      v.x = u.x * s;
      v.y = u.y * s;
      v.z = u.z * s;
      v.w = u.w * s;
   }
   /// <summary>
   /// Calculates the dot product of two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>The dot product value.</returns>
   public static float DotProduct(SLVec4f u, SLVec4f v)
   {
      return (u.x * v.x) + (u.y * v.y) + (u.z * v.z) + (u.w * v.w);
   }
   /// <summary>
   /// Calculates the cross product of two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the cross product result.</returns>
   public static SLVec4f CrossProduct(SLVec4f u, SLVec4f v)
   {
      return new SLVec4f( 
         u.y*v.z - u.z*v.y, 
         u.z*v.x - u.x*v.z, 
         u.x*v.y - u.y*v.x,
         1);
   }
   /// <summary>
   /// Calculates the cross product of two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec4f"/> instance to hold the cross product result.</param>
   public static void CrossProduct(SLVec4f u, SLVec4f v, SLVec4f w)
   {
      w.x = u.y*v.z - u.z*v.y;
      w.y = u.z*v.x - u.x*v.z;
      w.z = u.x*v.y - u.y*v.x;
      w.w = 1;
   }
   /// <summary>
   /// Negates a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the negated values.</returns>
   public static SLVec4f Negate(SLVec4f v)
   {
      return new SLVec4f(-v.x, -v.y, -v.z, -v.w);
   }
   /// <summary>
   /// Tests whether two vectors are approximately equal given a tolerance value.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="tolerance">The tolerance value used to test approximate equality.</param>
   /// <returns>True if the two vectors are approximately equal; otherwise, False.</returns>
   public static bool ApproxEqual(SLVec4f v, SLVec4f u, float tolerance)
   {
      return
         (
         (System.Math.Abs(v.x - u.x) <= tolerance) &&
         (System.Math.Abs(v.y - u.y) <= tolerance) &&
         (System.Math.Abs(v.z - u.z) <= tolerance) &&
         (System.Math.Abs(v.w - u.w) <= tolerance)
         );
   }
   #endregion

   #region Public Methods
   /// <summary>
   /// Setter for all components at once
   /// </summary>
   public void Set(SLVec4f v)
   {  x=v.x;
      y=v.y;
      z=v.z;
      w=v.w;
   }   
   /// <summary>
   /// Setter for all components at once
   /// </summary>
   public void Set(SLVec3f v)
   {  x=v.x;
      y=v.y;
      z=v.z;
      w=1;
   }
   /// <summary>
   /// Setter for all components at once
   /// </summary>
   public void Set(float X, float Y, float Z)
   {  x=X;
      y=Y;
      z=Z;
      z=1;
   }
   /// <summary>
   /// Setter for all components at once
   /// </summary>
   public void Set(float X, float Y, float Z, float W)
   {  x=X;
      y=Y;
      z=Z;
      z=W;
   }
   /// <summary>
   /// Scale the vector so that its length is 1.
   /// </summary>
   public void Normalize()
   {
      float length = Length();
      if (length == 0)
      {
         throw new DivideByZeroException("Trying to normalize a vector with length of zero.");
      }

      this.x /= length;
      this.y /= length;
      this.z /= length;
      this.w /= length;
   }
   /// <summary>
   /// Returns the length of the vector.
   /// </summary>
   /// <returns>The length of the vector. (Sqrt(X*X + Y*Y + Z*Z + W*W))</returns>
   public float Length()
   {
      return (float)Math.Sqrt(this.x*this.x + 
                              this.y*this.y + 
                              this.z*this.z + 
                              this.w*this.w);
   }
   /// <summary>
   /// Returns the squared length of the vector.
   /// </summary>
   /// <returns>The squared length of the vector. (X*X + Y*Y + Z*Z + W*W)</returns>
   public float LengthSquared()
   {
      return (this.x*this.x + 
              this.y*this.y + 
              this.z*this.z + 
              this.w*this.w);
   }
   /// <summary>
   /// Sets the minimum values of this and the passed vector v
   /// </summary>
   public void SetMin(SLVec4f v)
   {  if (v.x < x) x=v.x;
      if (v.y < y) y=v.y;
      if (v.z < z) z=v.z;
      if (v.w < w) w=v.w;
   }
   /// <summary>
   /// Sets the maximum values of this and the passed vector v
   /// </summary>
   public void SetMax(SLVec4f v)
   {  if (v.x > x) x=v.x;
      if (v.y > y) y=v.y;
      if (v.z > z) z=v.z;
      if (v.w > w) w=v.w;
   }
   /// <summary>
   /// Returns the dot product of this with vector v
   /// </summary>
   public float Dot(SLVec4f v)
   {  return (x * v.x) + (y * v.y) + (z * v.z) + (w * v.w);
   }
   /// <summary>
   /// Returns the cross product of this with vector v
   /// </summary>
   public SLVec4f Cross(SLVec4f v)
   {  return new SLVec4f(y*v.z - z*v.y, 
                         z*v.x - x*v.z, 
                         x*v.y - y*v.x,
                         1);
   }
   #endregion

   #region Overrides
   /// <summary>
   /// Returns the hashcode for this instance.
   /// </summary>
   /// <returns>A 32-bit signed integer hash code.</returns>
   public override int GetHashCode()
   {
      return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
   }
   /// <summary>
   /// Returns a value indicating whether this instance is equal to
   /// the specified object.
   /// </summary>
   /// <param name="obj">An object to compare to this instance.</param>
   /// <returns>True if <paramref name="obj"/> is a <see cref="SLVec4f"/> and has the same values as this instance; otherwise, False.</returns>
   public override bool Equals(object obj)
   {
      if (obj is SLVec4f)
      {
         SLVec4f v = (SLVec4f)obj;
         return (this.x == v.x) && (this.y == v.y) && (this.z == v.z) && (this.w == v.w);
      }
      return false;
   }

   /// <summary>
   /// Returns a string representation of this object.
   /// </summary>
   /// <returns>A string representation of this object.</returns>
   public override string ToString()
   {
      return string.Format("({0}, {1}, {2}, {3})", this.x, this.y, this.z, this.w);
   }
   #endregion
   
   #region Comparison Operators
   /// <summary>
   /// Tests whether two specified vectors are equal.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the two vectors are equal; otherwise, False.</returns>
   public static bool operator==(SLVec4f u, SLVec4f v)
   {
      if (Object.Equals(u, null))
      {
         return Object.Equals(v, null);
      }

      if (Object.Equals(v, null))
      {
         return Object.Equals(u, null);
      }

      return (u.x == v.x) && (u.y == v.y) && (u.z == v.z) && (u.w == v.w);
   }
   /// <summary>
   /// Tests whether two specified vectors are not equal.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the two vectors are not equal; otherwise, False.</returns>
   public static bool operator!=(SLVec4f u, SLVec4f v)
   {
      if (Object.Equals(u, null))
      {
         return !Object.Equals(v, null);
      }

      if (Object.Equals(v, null))
      {
         return !Object.Equals(u, null);
      }

      return !((u.x == v.x) && (u.y == v.y) && (u.z == v.z) && (u.w == v.w));
   }
   /// <summary>
   /// Tests if a vector's components are greater than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are greater than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator>(SLVec4f u, SLVec4f v)
   {
      return ((u.x > v.x) && 
              (u.y > v.y) && 
              (u.z > v.z) && 
              (u.w > v.w));
   }
   /// <summary>
   /// Tests if a vector's components are smaller than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are smaller than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator<(SLVec4f u, SLVec4f v)
   {
      return ((u.x < v.x) && 
              (u.y < v.y) && 
              (u.z < v.z) && 
              (u.w < v.w));
   }
   /// <summary>
   /// Tests if a vector's components are greater or equal than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are greater or equal than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator>=(SLVec4f u, SLVec4f v)
   {
      return ((u.x >= v.x) && 
              (u.y >= v.y) && 
              (u.z >= v.z) && 
              (u.w >= v.w));
   }
   /// <summary>
   /// Tests if a vector's components are smaller or equal than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are smaller or equal than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator<=(SLVec4f u, SLVec4f v)
   {
      return ((u.x <= v.x) && 
              (u.y <= v.y) && 
              (u.z <= v.z) && 
              (u.w <= v.w));
   }
   #endregion

   #region Unary Operators
   /// <summary>
   /// Negates the values of the vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the negated values.</returns>
   public static SLVec4f operator-(SLVec4f v)
   {
      return SLVec4f.Negate(v);
   }
   #endregion

   #region Binary Operators
   /// <summary>
   /// Adds two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the sum.</returns>
   public static SLVec4f operator+(SLVec4f u, SLVec4f v)
   {
      return SLVec4f.Add(u,v);
   }
   /// <summary>
   /// Adds a vector and a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the sum.</returns>
   public static SLVec4f operator+(SLVec4f v, float s)
   {
      return SLVec4f.Add(v,s);
   }
   /// <summary>
   /// Adds a vector and a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the sum.</returns>
   public static SLVec4f operator+(float s, SLVec4f v)
   {
      return SLVec4f.Add(v,s);
   }
   /// <summary>
   /// Subtracts a vector from a vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the difference.</returns>
   /// <remarks>
   ///	result[i] = m_v[i] - w[i].
   /// </remarks>
   public static SLVec4f operator-(SLVec4f u, SLVec4f v)
   {
      return SLVec4f.Subtract(u,v);
   }
   /// <summary>
   /// Subtracts a scalar from a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = m_v[i] - s
   /// </remarks>
   public static SLVec4f operator-(SLVec4f v, float s)
   {
      return SLVec4f.Subtract(v,s);
   }
   /// <summary>
   /// Subtracts a vector from a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = s - m_v[i]
   /// </remarks>
   public static SLVec4f operator-(float s, SLVec4f v)
   {
      return SLVec4f.Subtract(s,v);
   }
   /// <summary>
   /// Multiplies a vector by a scaar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the result.</returns>
   public static SLVec4f operator*(SLVec4f v, float s)
   {
      return SLVec4f.Multiply(v,s);
   }
   /// <summary>
   /// Multiplies a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the result.</returns>
   public static SLVec4f operator*(float s, SLVec4f v)
   {
      return SLVec4f.Multiply(v,s);
   }
   /// <summary>
   /// Divides a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = m_v[i] / s;
   /// </remarks>
   public static SLVec4f operator/(SLVec4f v, float s)
   {
      return SLVec4f.Divide(v,s);
   }
   /// <summary>
   /// Divides a scalar by a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec4f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = s / m_v[i]
   /// </remarks>
   public static SLVec4f operator/(float s, SLVec4f v)
   {
      return SLVec4f.Divide(s,v);
   }
   #endregion

   #region Array Indexing Operator
   /// <summary>
   /// Indexer ( [x, y] ).
   /// </summary>
   public float this[int index]
   {
      get	
      {
         switch( index ) 
         {
            case 0:
               return x;
            case 1:
               return y;
            case 2:
               return z;
            case 3:
               return w;
            default:
               throw new IndexOutOfRangeException();
         }
      }
      set 
      {
         switch( index ) 
         {
            case 0:
               x = value;
               break;
            case 1:
               y = value;
               break;
            case 2:
               z = value;
               break;
            case 3:
               w = value;
               break;
            default:
               throw new IndexOutOfRangeException();
         }
      }

   }

   #endregion

   #region Conversion Operators
   /// <summary>
   /// Converts the vector to an array of float-precision floating point values.
   /// </summary>
   /// <param name="v">A <see cref="SLVec4f"/> instance.</param>
   /// <returns>An array of float-precision floating point values.</returns>
   public static explicit operator float[](SLVec4f v)
   {
      float[] array = new float[4];
      array[0] = v.x;
      array[1] = v.y;
      array[2] = v.z;
      array[3] = v.z;
      return array;
   }
   #endregion

}
