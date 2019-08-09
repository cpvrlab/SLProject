using System;
using System.Runtime.InteropServices;

/// <summary>
/// High Performance counter using the windows QueryPerformanceCounter
/// </summary>
public class HiResTimer
{
   private bool  m_isPerfCounterSupported = false;
   private long  m_frequency = 0;
   private long  m_counterStart;
   
   [System.Security.SuppressUnmanagedCodeSecurity]
   [DllImport("kernel32", CharSet=CharSet.Auto)]
   public static extern bool QueryPerformanceFrequency(out long PerformanceFrequency);

   [System.Security.SuppressUnmanagedCodeSecurity]
   [DllImport("kernel32", CharSet=CharSet.Auto)]
   public static extern bool QueryPerformanceCounter(out long PerformanceCount);

   public HiResTimer()
   {
      // Query the high-resolution timer only if it is supported.
      // A returned frequency of 1000 typically indicates that it is not
      // supported and is emulated by the OS using the same value that is
      // returned by Environment.TickCount.
      // A return value of 0 indicates that the performance counter is
      // not supported.
      bool returnVal = QueryPerformanceFrequency(out m_frequency);

      if (returnVal && m_frequency != 1000)
      {
         // The performance counter is supported.
         m_isPerfCounterSupported = true;
      }
      else
      {
         // The performance counter is not supported. Use
         // Environment.TickCount instead.
         m_frequency = 1000;
      }
   }

   public Int64 Frequency
   {
      get
      {
         return m_frequency;
      }
   }

   public Int64 Value
   {
      get
      {
         Int64 tickCount = 0;

         if (m_isPerfCounterSupported)
         {
               // Get the value here if the counter is supported.
               QueryPerformanceCounter(out tickCount);
               return tickCount;
         }
         else
         {
               // Otherwise, use Environment.TickCount.
               return (Int64)Environment.TickCount;
         }
      }
   }

   public void Start()
   {
      QueryPerformanceCounter(out m_counterStart);
   }

   public Double GetMiliSeconds()
   {
      Int64 counterEnd = 0;
      QueryPerformanceCounter(out counterEnd);
      Int64 timeElapsedInTicks = counterEnd - m_counterStart;
      double ms = (double)(timeElapsedInTicks * 1000) / (double)m_frequency;
      return ms;
   }

   public Double GetSeconds()
   {
      Int64 counterEnd = 0;
      QueryPerformanceCounter(out counterEnd);      
      Int64 timeElapsedInTicks = counterEnd - m_counterStart;
      double s = (double)timeElapsedInTicks / (double)m_frequency;
      return s;
   }
}