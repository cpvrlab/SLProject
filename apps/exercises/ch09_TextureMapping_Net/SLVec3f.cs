//#############################################################################
//  File:      Globals/Math/SLVec3f.cs
//  Purpose:   3 Component vector class
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
/// 3D vector class for standard 3D vector algebra.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public class SLVec3f
{
   #region public fields
   /// <summary>x-component of vector</summary>
   public float x;
   /// <summary>y-component of vector</summary>
   public float y;
   /// <summary>z-component of vector</summary>
   public float z;
   #endregion

   #region Constructors
   /// <summary>
   /// Default constructor that inits all components to zero
   /// </summary>
   public SLVec3f()
   {
      this.x = 0;
      this.y = 0;
      this.z = 0;
   }
   
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec3f"/> class with the specified coordinates.
   /// </summary>
   /// <param name="x">The vector's x coordinate.</param>
   /// <param name="y">The vector's y coordinate.</param>
   /// <param name="z">The vector's z coordinate.</param>
   public SLVec3f(float x, float y, float z)
   {
      this.x = x;
      this.y = y;
      this.z = z;
   }
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec3f"/> class with the specified coordinates.
   /// </summary>
   /// <param name="coordinates">An array containing the coordinate parameters.</param>
   public SLVec3f(float[] coordinates)
   {
      Debug.Assert(coordinates != null);
      Debug.Assert(coordinates.Length >= 3);

      this.x = coordinates[0];
      this.y = coordinates[1];
      this.z = coordinates[2];
   }
   /// <summary>
   /// Initializes a new instance of the <see cref="SLVec3f"/> class using coordinates from a given <see cref="SLVec3f"/> instance.
   /// </summary>
   /// <param name="vector">A <see cref="SLVec3f"/> to get the coordinates from.</param>
   public SLVec3f(SLVec3f vector)
   {
      this.x = vector.x;
      this.y = vector.y;
      this.z = vector.z;
   }
   #endregion

   #region Constants
   /// <summary>
   /// 3-Dimentional float-precision floating point zero vector.
   /// </summary>
   public static readonly SLVec3f Zero	= new SLVec3f(0.0f, 0.0f, 0.0f);
   /// <summary>
   /// 3-Dimentional float-precision floating point X-Axis vector.
   /// </summary>
   public static readonly SLVec3f XAxis	= new SLVec3f(1.0f, 0.0f, 0.0f);
   /// <summary>
   /// 3-Dimentional float-precision floating point Y-Axis vector.
   /// </summary>
   public static readonly SLVec3f YAxis	= new SLVec3f(0.0f, 1.0f, 0.0f);
   /// <summary>
   /// 3-Dimentional float-precision floating point Y-Axis vector.
   /// </summary>
   public static readonly SLVec3f ZAxis	= new SLVec3f(0.0f, 0.0f, 1.0f);
   #endregion

   #region Public Static Vector Arithmetics
   /// <summary>
   /// Adds two vectors.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the sum.</returns>
   public static SLVec3f Add(SLVec3f v, SLVec3f w)
   {
      return new SLVec3f(v.x + w.x, v.y + w.y, v.z + w.z);
   }
   /// <summary>
   /// Adds a vector and a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the sum.</returns>
   public static SLVec3f Add(SLVec3f v, float s)
   {
      return new SLVec3f(v.x + s, v.y + s, v.z + s);
   }
   /// <summary>
   /// Adds two vectors and put the result in the third vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance</param>
   /// <param name="w">A <see cref="SLVec3f"/> instance to hold the result.</param>
   public static void Add(SLVec3f u, SLVec3f v, SLVec3f w)
   {
      w.x = u.x + v.x;
      w.y = u.y + v.y;
      w.z = u.z + v.z;
   }
   /// <summary>
   /// Adds a vector and a scalar and put the result into another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance to hold the result.</param>
   public static void Add(SLVec3f u, float s, SLVec3f v)
   {
      v.x = u.x + s;
      v.y = u.y + s;
      v.z = u.z + s;
   }
   /// <summary>
   /// Subtracts a vector from a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the difference.</returns>
   /// <remarks>
   ///	result[i] = m_v[i] - w[i].
   /// </remarks>
   public static SLVec3f Subtract(SLVec3f v, SLVec3f w)
   {
      return new SLVec3f(v.x - w.x, v.y - w.y, v.z - w.z);
   }
   /// <summary>
   /// Subtracts a scalar from a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = m_v[i] - s
   /// </remarks>
   public static SLVec3f Subtract(SLVec3f v, float s)
   {
      return new SLVec3f(v.x - s, v.y - s, v.z - s);
   }
   /// <summary>
   /// Subtracts a vector from a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = s - m_v[i]
   /// </remarks>
   public static SLVec3f Subtract(float s, SLVec3f v)
   {
      return new SLVec3f(s - v.x, s - v.y, s - v.z);
   }
   /// <summary>
   /// Subtracts a vector from a second vector and puts the result into a third vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance</param>
   /// <param name="w">A <see cref="SLVec3f"/> instance to hold the result.</param>
   /// <remarks>
   ///	w[i] = m_v[i] - w[i].
   /// </remarks>
   public static void Subtract(SLVec3f u, SLVec3f v, SLVec3f w)
   {
      w.x = u.x - v.x;
      w.y = u.y - v.y;
      w.z = u.z - v.z;
   }
   /// <summary>
   /// Subtracts a vector from a scalar and put the result into another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = u[i] - s
   /// </remarks>
   public static void Subtract(SLVec3f u, float s, SLVec3f v)
   {
      v.x = u.x - s;
      v.y = u.y - s;
      v.z = u.z - s;
   }
   /// <summary>
   /// Subtracts a scalar from a vector and put the result into another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = s - u[i]
   /// </remarks>
   public static void Subtract(float s, SLVec3f u, SLVec3f v)
   {
      v.x = s - u.x;
      v.y = s - u.y;
      v.z = s - u.z;
   }
   /// <summary>
   /// Divides a vector by another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the quotient.</returns>
   /// <remarks>
   ///	result[i] = u[i] / m_v[i].
   /// </remarks>
   public static SLVec3f Divide(SLVec3f u, SLVec3f v)
   {
      return new SLVec3f(u.x / v.x, u.y / v.y, u.z / v.z);
   }
   /// <summary>
   /// Divides a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = m_v[i] / s;
   /// </remarks>
   public static SLVec3f Divide(SLVec3f v, float s)
   {
      return new SLVec3f(v.x / s, v.y / s, v.z / s);
   }
   /// <summary>
   /// Divides a scalar by a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = s / m_v[i]
   /// </remarks>
   public static SLVec3f Divide(float s, SLVec3f v)
   {
      return new SLVec3f(s / v.x, s/ v.y, s / v.z);
   }
   /// <summary>
   /// Divides a vector by another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec3f"/> instance to hold the result.</param>
   /// <remarks>
   /// w[i] = u[i] / m_v[i]
   /// </remarks>
   public static void Divide(SLVec3f u, SLVec3f v, SLVec3f w)
   {
      w.x = u.x / v.x;
      w.y = u.y / v.y;
      w.z = u.z / v.z;
   }
   /// <summary>
   /// Divides a vector by a scalar.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = u[i] / s
   /// </remarks>
   public static void Divide(SLVec3f u, float s, SLVec3f v)
   {
      v.x = u.x / s;
      v.y = u.y / s;
      v.z = u.z / s;
   }
   /// <summary>
   /// Divides a scalar by a vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance to hold the result.</param>
   /// <remarks>
   /// m_v[i] = s / u[i]
   /// </remarks>
   public static void Divide(float s, SLVec3f u, SLVec3f v)
   {
      v.x = s / u.x;
      v.y = s / u.y;
      v.z = s / u.z;
   }
   /// <summary>
   /// Multiplies a vector by a scalar.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the result.</returns>
   public static SLVec3f Multiply(SLVec3f u, float s)
   {
      return new SLVec3f(u.x * s, u.y * s, u.z * s);
   }
   /// <summary>
   /// Multiplies a vector by a scalar and put the result in another vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance to hold the result.</param>
   public static void Multiply(SLVec3f u, float s, SLVec3f v)
   {
      v.x = u.x * s;
      v.y = u.y * s;
      v.z = u.z * s;
   }
   /// <summary>
   /// Calculates the dot product of two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>The dot product value.</returns>
   public static float DotProduct(SLVec3f u, SLVec3f v)
   {
      return (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
   }
   /// <summary>
   /// Calculates the cross product of two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the cross product result.</returns>
   public static SLVec3f CrossProduct(SLVec3f u, SLVec3f v)
   {
      return new SLVec3f( 
         u.y*v.z - u.z*v.y, 
         u.z*v.x - u.x*v.z, 
         u.x*v.y - u.y*v.x );
   }
   /// <summary>
   /// Calculates the cross product of two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="w">A <see cref="SLVec3f"/> instance to hold the cross product result.</param>
   public static void CrossProduct(SLVec3f u, SLVec3f v, SLVec3f w)
   {
      w.x = u.y*v.z - u.z*v.y;
      w.y = u.z*v.x - u.x*v.z;
      w.z = u.x*v.y - u.y*v.x;
   }
   /// <summary>
   /// Negates a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the negated values.</returns>
   public static SLVec3f Negate(SLVec3f v)
   {
      return new SLVec3f(-v.x, -v.y, -v.z);
   }
   /// <summary>
   /// Tests whether two vectors are approximately equal given a tolerance value.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="tolerance">The tolerance value used to test approximate equality.</param>
   /// <returns>True if the two vectors are approximately equal; otherwise, False.</returns>
   public static bool ApproxEqual(SLVec3f v, SLVec3f u, float tolerance)
   {
      return
         (
         (System.Math.Abs(v.x - u.x) <= tolerance) &&
         (System.Math.Abs(v.y - u.y) <= tolerance) &&
         (System.Math.Abs(v.z - u.z) <= tolerance)
         );
   }
   #endregion

   #region Public Methods
      /// <summary>
   /// Setter for all components at once
   /// </summary>
   public void Set(SLVec3f v)
   {  x=v.x;
      y=v.y;
      z=v.z;
   }
   
   /// <summary>
   /// Setter for all components at once
   /// </summary>
   public void Set(float X, float Y, float Z)
   {  x=X;
      y=Y;
      z=Z;
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
   }
   /// <summary>
   /// Returns the length of the vector.
   /// </summary>
   /// <returns>The length of the vector. (Sqrt(X*X + Y*Y + Z*Z))</returns>
   public float Length()
   {
      return (float)Math.Sqrt(this.x*this.x + this.y*this.y + this.z*this.z);
   }
   /// <summary>
   /// Returns the squared length of the vector.
   /// </summary>
   /// <returns>The squared length of the vector. (X*X + Y*Y + Z*Z)</returns>
   public float LengthSquared()
   {
      return (this.x*this.x + this.y*this.y + this.z*this.z);
   }
   /// <summary>
   /// Sets the minimum values of this and the passed vector v
   /// </summary>
   public void SetMin(SLVec3f v)
   {  if (v.x < x) x=v.x;
      if (v.y < y) y=v.y;
      if (v.z < z) z=v.z;
   }
   /// <summary>
   /// Sets the maximum values of this and the passed vector v
   /// </summary>
   public void SetMax(SLVec3f v)
   {  if (v.x > x) x=v.x;
      if (v.y > y) y=v.y;
      if (v.z > z) z=v.z;
   }
   /// <summary>
   /// Returns the dot product of this with vector v
   /// </summary>
   public float Dot(SLVec3f v)
   {  return (x * v.x) + (y * v.y) + (z * v.z);
   }
   /// <summary>
   /// Returns the cross product of this with vector v
   /// </summary>
   public SLVec3f Cross(SLVec3f v)
   {  return new SLVec3f(y*v.z - z*v.y, 
                             z*v.x - x*v.z, 
                             x*v.y - y*v.x );
   }
   #endregion

   #region Overrides
   /// <summary>
   /// Returns the hashcode for this instance.
   /// </summary>
   /// <returns>A 32-bit signed integer hash code.</returns>
   public override int GetHashCode()
   {
      return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
   }
   /// <summary>
   /// Returns a value indicating whether this instance is equal to
   /// the specified object.
   /// </summary>
   /// <param name="obj">An object to compare to this instance.</param>
   /// <returns>True if <paramref name="obj"/> is a <see cref="SLVec3f"/> and has the same values as this instance; otherwise, False.</returns>
   public override bool Equals(object obj)
   {
      if (obj is SLVec3f)
      {
         SLVec3f v = (SLVec3f)obj;
         return (this.x == v.x) && (this.y == v.y) && (this.z == v.z);
      }
      return false;
   }

   /// <summary>
   /// Returns a string representation of this object.
   /// </summary>
   /// <returns>A string representation of this object.</returns>
   public override string ToString()
   {
      return string.Format("({0}, {1}, {2})", this.x, this.y, this.z);
   }
   #endregion
   
   #region Comparison Operators
   /// <summary>
   /// Tests whether two specified vectors are equal.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the two vectors are equal; otherwise, False.</returns>
   public static bool operator==(SLVec3f u, SLVec3f v)
   {
      if (Object.Equals(u, null))
      {
         return Object.Equals(v, null);
      }

      if (Object.Equals(v, null))
      {
         return Object.Equals(u, null);
      }

      return (u.x == v.x) && (u.y == v.y) && (u.z == v.z);
   }
   /// <summary>
   /// Tests whether two specified vectors are not equal.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the two vectors are not equal; otherwise, False.</returns>
   public static bool operator!=(SLVec3f u, SLVec3f v)
   {
      if (Object.Equals(u, null))
      {
         return !Object.Equals(v, null);
      }

      if (Object.Equals(v, null))
      {
         return !Object.Equals(u, null);
      }

      return !((u.x == v.x) && (u.y == v.y) && (u.z == v.z));
   }
   /// <summary>
   /// Tests if a vector's components are greater than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are greater than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator>(SLVec3f u, SLVec3f v)
   {
      return (
         (u.x > v.x) && 
         (u.y > v.y) && 
         (u.z > v.z));
   }
   /// <summary>
   /// Tests if a vector's components are smaller than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are smaller than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator<(SLVec3f u, SLVec3f v)
   {
      return (
         (u.x < v.x) && 
         (u.y < v.y) && 
         (u.z < v.z));
   }
   /// <summary>
   /// Tests if a vector's components are greater or equal than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are greater or equal than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator>=(SLVec3f u, SLVec3f v)
   {
      return (
         (u.x >= v.x) && 
         (u.y >= v.y) && 
         (u.z >= v.z));
   }
   /// <summary>
   /// Tests if a vector's components are smaller or equal than another vector's components.
   /// </summary>
   /// <param name="u">The left-hand vector.</param>
   /// <param name="v">The right-hand vector.</param>
   /// <returns>True if the left-hand vector's components are smaller or equal than the right-hand vector's component; otherwise, False.</returns>
   public static bool operator<=(SLVec3f u, SLVec3f v)
   {
      return (
         (u.x <= v.x) && 
         (u.y <= v.y) && 
         (u.z <= v.z));
   }
   #endregion

   #region Unary Operators
   /// <summary>
   /// Negates the values of the vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the negated values.</returns>
   public static SLVec3f operator-(SLVec3f v)
   {
      return SLVec3f.Negate(v);
   }
   #endregion

   #region Binary Operators
   /// <summary>
   /// Adds two vectors.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the sum.</returns>
   public static SLVec3f operator+(SLVec3f u, SLVec3f v)
   {
      return SLVec3f.Add(u,v);
   }
   /// <summary>
   /// Adds a vector and a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the sum.</returns>
   public static SLVec3f operator+(SLVec3f v, float s)
   {
      return SLVec3f.Add(v,s);
   }
   /// <summary>
   /// Adds a vector and a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the sum.</returns>
   public static SLVec3f operator+(float s, SLVec3f v)
   {
      return SLVec3f.Add(v,s);
   }
   /// <summary>
   /// Subtracts a vector from a vector.
   /// </summary>
   /// <param name="u">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the difference.</returns>
   /// <remarks>
   ///	result[i] = m_v[i] - w[i].
   /// </remarks>
   public static SLVec3f operator-(SLVec3f u, SLVec3f v)
   {
      return SLVec3f.Subtract(u,v);
   }
   /// <summary>
   /// Subtracts a scalar from a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = m_v[i] - s
   /// </remarks>
   public static SLVec3f operator-(SLVec3f v, float s)
   {
      return SLVec3f.Subtract(v, s);
   }
   /// <summary>
   /// Subtracts a vector from a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> instance containing the difference.</returns>
   /// <remarks>
   /// result[i] = s - m_v[i]
   /// </remarks>
   public static SLVec3f operator-(float s, SLVec3f v)
   {
      return SLVec3f.Subtract(s, v);
   }

   /// <summary>
   /// Multiplies a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the result.</returns>
   public static SLVec3f operator*(SLVec3f v, float s)
   {
      return SLVec3f.Multiply(v,s);
   }
   /// <summary>
   /// Multiplies a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar.</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the result.</returns>
   public static SLVec3f operator*(float s, SLVec3f v)
   {
      return SLVec3f.Multiply(v,s);
   }
   /// <summary>
   /// Divides a vector by a scalar.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = m_v[i] / s;
   /// </remarks>
   public static SLVec3f operator/(SLVec3f v, float s)
   {
      return SLVec3f.Divide(v,s);
   }
   /// <summary>
   /// Divides a scalar by a vector.
   /// </summary>
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <param name="s">A scalar</param>
   /// <returns>A new <see cref="SLVec3f"/> containing the quotient.</returns>
   /// <remarks>
   /// result[i] = s / m_v[i]
   /// </remarks>
   public static SLVec3f operator/(float s, SLVec3f v)
   {
      return SLVec3f.Divide(s,v);
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
   /// <param name="v">A <see cref="SLVec3f"/> instance.</param>
   /// <returns>An array of float-precision floating point values.</returns>
   public static explicit operator float[](SLVec3f v)
   {
      float[] array = new float[3];
      array[0] = v.x;
      array[1] = v.y;
      array[2] = v.z;
      return array;
   }
   #endregion

}
