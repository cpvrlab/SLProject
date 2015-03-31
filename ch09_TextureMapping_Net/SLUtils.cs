//#############################################################################
//  File:      Globals/SL/SLUtils.cs
//  Purpose:   General utility functions not found anywhere else
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

using System;
using System.Drawing;

/// <summary>
/// Utility stuff
/// </summary>
public class SLUtils
{

   #region Constants
   public const double TWOPI = Math.PI * 2; 
   public const double FOURPI = Math.PI * 4; 
   public const double HALFPI = Math.PI / 2; 
   public const double DEG2RAD = Math.PI / 180; 
   public const double RAD2DEG = 180 / Math.PI; 
   public const double EPSILON = 0.000001;
   #endregion 
   
   /// <summary>
   /// assure that the value "a" is between min and max
   /// </summary>
   /// <param name="a">value to check</param>
   /// <param name="min">minimum value</param>
   /// <param name="max">maximum value</param>
   /// <returns>value a between min and max</returns>
   public static int Clamp(int a, int min, int max)
   {
      return (a<min) ? min : (a>max) ? max : a;
   }

   /// <summary>
   /// assure that the value "a" is between min and max
   /// </summary>
   /// <param name="a">value to check</param>
   /// <param name="min">minimum value</param>
   /// <param name="max">maximum value</param>
   /// <returns>value a between min and max</returns>
   public static float Clamp(float a, float min, float max)
   {
      return (a<min) ? min : (a>max) ? max : a;
   }

   /// <summary>
   /// assure that the value "a" is between min and max
   /// </summary>
   /// <param name="a">value to check</param>
   /// <param name="min">minimum value</param>
   /// <param name="max">maximum value</param>
   /// <returns>value a between min and max</returns>
   public static double Clamp(double a, double min, double max)
   {
      return (a<min) ? min : (a>max) ? max : a;
   }
   
   /// <summary>
   /// Swap a with b
   /// </summary>
   /// <param name="a"></param>
   /// <param name="b"></param>
   public static void Swap(ref double a, ref double b) {double t; t=a; a=b; b=t;}
   
   /// <summary>
   /// Swap a with b
   /// </summary>
   /// <param name="a"></param>
   /// <param name="b"></param>
   public static void Swap(ref float a, ref float b) {float t; t=a; a=b; b=t;}
   
   /// <summary>
   /// Swap a with b
   /// </summary>
   /// <param name="a"></param>
   /// <param name="b"></param>
   public static void Swap(ref int a, ref int b) {int t; t=a; a=b; b=t;}
   
   /// <summary>
   /// assure that the value "a" is between 0 and 255
   /// </summary>
   /// <param name="a">value to check</param>
   /// <returns>value a between 0 and 255</returns>
   public static Byte ByteClamp(int a)
   {
      return (((uint)a & 0xffffff00) == 0) ? (Byte)a : ((a < 0) ? (Byte)0 : (Byte)255);
   }
}
