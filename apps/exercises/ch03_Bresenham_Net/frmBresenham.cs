using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Diagnostics;

/// <summary>
/// Form for Bresenham-, Pixel- and Voxel-Traversal Demo
/// </summary>
public partial class frmBresenham : Form
{  
   #region Members
   /// <summary>Enlarged pixel size in pixels</summary>
   private int m_pixelSize;

   /// <summary>Enlarged gird size in x-direction</summary>
   private int m_gridX;

   /// <summary>Enlarged gird size in y-direction</summary>
   private int m_gridY;

   /// <summary>Enlarged gird size in z-direction</summary>
   private int m_gridZ;

   /// <summary>X-Offset of the depth grid visualization</summary>
   private int m_offsetX;

   /// <summary>Top coord. blow menu in pixels</summary>
   private int m_topY;

   /// <summary>Hires timer</summary>
   private Stopwatch m_timer;

   /// <summary>Counter for frames for benchmark</summary>
   private int m_cntFrame = 0;

   /// <summary>Average time ratio for benchmark</summary>
   private double m_avgRatio = 0;
   #endregion
   
   /// <summary>1 x 1 pixel bitmap for single pixel drawing</summary>
   /// <remarks>
   /// There is no single pixel drawin in .NET GDI+ rendering!
   /// It has to be drawn by a 1x1 bitmap with DrawImageUnscaled!
   /// </remarks>
   private Bitmap m_point;
   
   /// <summary>Constructor initiallizing all members</summary>
   public frmBresenham()
   {
      InitializeComponent();

      m_pixelSize = 30;
      m_gridX = 21;
      m_gridY = 8;
      m_gridZ = 5;
      m_offsetX = (m_gridX + 2) * m_pixelSize;
      m_topY = menu1.Height;

      
      m_point = new Bitmap(1, 1);
      m_point.SetPixel(0, 0, Color.Black);

      m_timer = new Stopwatch();
         
      this.DoubleBuffered = true;
   }

   /// <summary>The forms paint routine</summary>
   private void frmBresenham_Paint(object sender, PaintEventArgs e)
   {
      Graphics g = e.Graphics;

      if (mnuViewCompare.Checked)
      {  
         m_cntFrame++;

         // Draw 400 lines with GDI+ DrawLine         
         m_timer.Start();
         int top = menu1.Height;
         for (int x=400; x<=800; x+=5) g.DrawLine(Pens.Black, 400, top, x  , top+400);
         for (int y=0;   y<=400; y+=5) g.DrawLine(Pens.Black, 400, top, 800, top+y  );
         double ms_1 = m_timer.Elapsed.Milliseconds;

         // Draw 400 lines with our Bresenham implementation
         m_timer.Start();
         for (int x=0;   x<=400; x+=5)     DrawLineBresenham(g, 0, top, x  ,top+400);
         for (int y=400+top; y>=top; y-=5) DrawLineBresenham(g, 0, top, 400,  y);
         double ms_2 = m_timer.Elapsed.Milliseconds;

            double ratio = ms_2 / ms_1;
         Console.WriteLine("DrawLine: " + ms_1.ToString("0.00ms") + 
                           ", DrawMyLine: " + ms_2.ToString("0.00ms") + 
                           ", Ratio: " + ratio.ToString("0.000"));

         if (m_cntFrame >= 20)
         {  m_avgRatio /= 20;
            Console.WriteLine("Avg. Ratio: " + m_avgRatio.ToString("0.000"));
            Console.WriteLine("-----------------------------------------------------");
            m_cntFrame = 0;
         }
         else
         {  m_avgRatio += ratio;
            this.Invalidate();
         }
      }
      else
      {
         // draw raster grid of x-y plane
         for (int x = 0; x <= m_gridX + 1; ++x)
         {  g.DrawLine(Pens.Black, 
                       x * m_pixelSize, m_topY,
                       x * m_pixelSize, m_topY + (m_gridY + 1) * m_pixelSize);
         }
         for (int y = 0; y <= m_gridY + 1; ++y)
         {  g.DrawLine(Pens.Black, 
                       0, m_topY + y * m_pixelSize, 
                       (m_gridX + 1) * m_pixelSize, m_topY + y * m_pixelSize);
         }

         // draw raster grid of z-y plane
         for (int z = 0; z <= m_gridZ + 1; ++z)
         {  g.DrawLine(Pens.Black, 
                       m_offsetX + z * m_pixelSize, m_topY, 
                       m_offsetX + z * m_pixelSize, m_topY + (m_gridY + 1) * m_pixelSize);
         }
         for (int y = 0; y <= m_gridY + 1; ++y)
         {  g.DrawLine(Pens.Black, 
                       m_offsetX + 0, m_topY + y * m_pixelSize, 
                       m_offsetX + (m_gridZ + 1) * m_pixelSize, m_topY + y * m_pixelSize);
         }

         if (mnuViewBresenham.Checked) TraverseBresenham(g, 0, 0,    m_gridX, m_gridY);
         if (mnuViewPixels.Checked)    TraversePixel    (g, 0, 0,    m_gridX, m_gridY);
         if (mnuViewVoxels.Checked)    TraverseVoxel    (g, 0, 0, 0, m_gridX, m_gridY, m_gridZ);
      }
   }

   /// <summary>
   /// Classsic Bresenham pixel traversal (see drawLineBresenham for details.
   /// This routine draws an enlarged line drawing for demonstration.
   /// </summary>
   /// <param name="g">Graphics device</param>
   /// <param name="x0">x-coord. of first point</param>
   /// <param name="y0">y-coord. of first point</param>
   /// <param name="x1">x-coord. of second point</param>
   /// <param name="y1">y-coord. of second point</param>
   void TraverseBresenham(Graphics g, int x0, int y0, int x1, int y1) 
   {  int x = x0, y = y0;
      int dx = x1 - x0;
      int dy = y1 - y0;
      int stepx=1, stepy=1;

      if (dy < 0) {dy = -dy; stepy = -1;} // mirroring on y
      if (dx < 0) {dx = -dx; stepx = -1;} // mirroring on x

      if (dx > dy)                 // slope < 1 (x is driving)
      {  int dE  = (dy<<1);        // = 2*dy
         int dNE = (dy-dx)<<1;     // = 2*(dy-dx)
         int e   = (dy<<1) - dx;   // = 2*dy - dx;
         
         while (x != x1) 
         {  DrawPixel(g, x, y);
            x += stepx;
            if (e < 0)             // choose E
            {  e += dE;
            } else                 // choose NE
            {  e += dNE; 
               y += stepy;
            }
         }
      } else                       // slope > 1 (y is driving)
      {  int dE  = (dx<<1);        // = 2*dx
         int dNE = (dx-dy)<<1;     // = 2*(dx-dy)
         int e   = (dx<<1) - dy;   // = 2*dx - dy; 
         
         while (y != y1) 
         {  DrawPixel(g, x, y);
            y += stepy; 
            if (e < 0)             // choose E
            {  e += dE; 
            } else                 // choose NE
            {  e += dNE;
               x += stepx; 
            }
         }
      }
      DrawPixel(g, x,y);         // draw last pixel

      // draw center line in x-y plane
      g.DrawLine(Pens.Red,
                 x0 * m_pixelSize + m_pixelSize * 0.5f,
                 y0 * m_pixelSize + m_pixelSize * 0.5f + m_topY,
                 x1 * m_pixelSize + m_pixelSize * 0.5f,
                 y1 * m_pixelSize + m_pixelSize * 0.5f + m_topY);
   }

   /// <summary>
   /// Pixel traversal with all pixels that the central line pierces.
   /// </summary>
   /// <param name="g">Graphics device</param>
   /// <param name="x0">x-coord. of first point</param>
   /// <param name="y0">y-coord. of first point</param>
   /// <param name="x1">x-coord. of second point</param>
   /// <param name="y1">y-coord. of second point</param>
   void TraversePixel(Graphics g, int x0, int y0, int x1, int y1) 
   {  int x = x0, y = y0;
      int dx = x1 - x0;
      int dy = y1 - y0;
      int stepx=1, stepy=1;

      if (dy < 0) {dy = -dy; stepy = -1;} // mirroring on y
      if (dx < 0) {dx = -dx; stepx = -1;} // mirroring on x

      if (dx > dy)                   // slope < 1 (x is driving)
      {  int dE  = (dy<<1);          // = 2*dy
         int dNE = (dy-dx)<<1;       // = 2*(dy-dx)
         int e   = 3*dy - dx; 
         
         while (x != x1) 
         {  DrawPixel(g, x, y);
            x += stepx;
            if (e < 0)               // choose E
            {  e += dE;
            } else                   // choose NE
            {  e += dNE; 
               DrawPixel(g, x, y);
               y += stepy;
            }
         }
      } 
      else                           // slope > 1 (y is driving)
      {  int dE  = (dx<<1);          // = 2*dx
         int dNE = (dx-dy)<<1;       // = 2*(dx-dy)
         int e   = 3*dx - dy;
         
         while (y != y1) 
         {  DrawPixel(g, x, y);
            y += stepy;          
            if (e < 0)               // choose E
            {  e += dE; 
            } else                   // choose NE
            {  e += dNE;
               DrawPixel(g, x, y);
               x += stepx; 
            }
         }
      }
      DrawPixel(g, x, y);          // draw last pixel
      
      g.SmoothingMode = SmoothingMode.AntiAlias;

      // draw center line in x-y plane
      g.DrawLine(Pens.Red,
                 x0 * m_pixelSize + m_pixelSize * 0.5f,
                 y0 * m_pixelSize + m_pixelSize * 0.5f + m_topY,
                 x1 * m_pixelSize + m_pixelSize * 0.5f,
                 y1 * m_pixelSize + m_pixelSize * 0.5f + m_topY);

      g.SmoothingMode = SmoothingMode.None;
   }
   
   /// <summary>
   /// Voxel traversal (same as pixel traversal but in 3D).
   /// </summary>
   /// <param name="g">Graphics device</param>
   /// <param name="x0">x-coord. of first point</param>
   /// <param name="y0">y-coord. of first point</param>
   /// <param name="x1">x-coord. of second point</param>
   /// <param name="y1">y-coord. of second point</param>
   void TraverseVoxel(Graphics g, int x0, int y0, int z0, int x1, int y1, int z1) 
   {  int x = x0, y = y0, z = z0;
      int dx = x1 - x0;
      int dy = y1 - y0;
      int dz = z1 - z0;
      int stepx=1, stepy=1, stepz=1;

      if (dy < 0) {dy = -dy; stepy = -1;}  // mirroring on y
      if (dx < 0) {dx = -dx; stepx = -1;}  // mirroring on x
      if (dz < 0) {dz = -dz; stepz = -1;}  // mirroring on z

      DrawVoxel(g, x, y, z);           // draw first voxel

      if (dx > dy && dx > dz)        // x is driving w. smallest slope
      {  int dEy  = (dy<<1);         // = 2*dy
         int dEz  = (dz<<1);         // = 2*dz
         int dNEy = (dy-dx)<<1;      // = 2*(dy-dx)
         int dNEz = (dz-dx)<<1;      // = 2*(dz-dx)
         int ey  = 3*dy - dx; 
         int ez  = 3*dz - dx; 
         
         while (x != x1) 
         {  x += stepx;
            DrawVoxel(g, x, y, z);
            if (ey > 0 && ez > 0)
            {  if (ey*dz > ez*dy)
               {  y += stepy;
                  DrawVoxel(g, x, y, z);
                  z += stepz;
                  DrawVoxel(g, x, y, z);
               } else
               {  z += stepz;
                  DrawVoxel(g, x, y, z);
                  y += stepy;
                  DrawVoxel(g, x, y, z);
               }
               ey += dNEy;
               ez += dNEz;
            } else
            {  if (ey > 0)
               {  y += stepy;
                  DrawVoxel(g, x, y, z);
                  ey += dNEy;
                  ez += dEz;
               } else
               {  ey += dEy;
                  if (ez > 0)
                  {  z += stepz;
                     DrawVoxel(g, x, y, z);
                     ez += dNEz;
                  } else
                  {  ez += dEz;
                  }
               }
            }
         }
      } 
      else if (dy > dx && dy > dz)   // y is driving w. smallest slope
      {  int dEx  = (dx<<1);         // = 2*dx
         int dEz  = (dz<<1);         // = 2*dz
         int dNEx = (dx-dy)<<1;      // = 2*(dx-dy)
         int dNEz = (dz-dy)<<1;      // = 2*(dz-dy)
         int ex  = 3*dx - dy; 
         int ez  = 3*dz - dy; 
         
         while (y != y1) 
         {  y += stepy;
            DrawVoxel(g, x, y, z);
            if (ex > 0 && ez > 0)
            {  if (ex*dz > ez*dx)
               {  x += stepx;
                  DrawVoxel(g, x, y, z);
                  z += stepz;
                  DrawVoxel(g, x, y, z);
               } else
               {  z += stepz;
                  DrawVoxel(g, x, y, z);
                  x += stepx;
                  DrawVoxel(g, x, y, z);
               }
               ex += dNEx;
               ez += dNEz;
            } else
            {  if (ex > 0)
               {  x += stepx;
                  DrawVoxel(g, x, y, z);
                  ex += dNEx;
                  ez += dEz;
               } else
               {  ex += dEx;
                  if (ez > 0)
                  {  z += stepz;
                     DrawVoxel(g, x, y, z);
                     ez += dNEz;
                  } else
                  {  ez += dEz;
                  }
               }
            }
         }
      }
      else                           // z is driving w. smallest slope
      {  int dEx  = (dx<<1);         // = 2*dx
         int dEy  = (dy<<1);         // = 2*dy
         int dNEx = (dx-dz)<<1;      // = 2*(dx-dz)
         int dNEy = (dy-dz)<<1;      // = 2*(dy-dz)
         int ex  = 3*dx - dz; 
         int ey  = 3*dy - dz; 
         
         while (z != z1) 
         {  z += stepz;
            DrawVoxel(g, x, y, z);
            if (ex > 0 && ey > 0)
            {  if (ex*dy > ey*dx)
               {  x += stepx;
                  DrawVoxel(g, x, y, z);
                  y += stepy;
                  DrawVoxel(g, x, y, z);
               } else
               {  y += stepy;
                  DrawVoxel(g, x, y, z);
                  x += stepx;
                  DrawVoxel(g, x, y, z);
               }
               ex += dNEx;
               ey += dNEy;
            } else
            {  if (ex > 0)
               {  x += stepx;
                  DrawVoxel(g, x, y, z);
                  ex += dNEx;
                  ey += dEy;
               } else
               {  ex += dEx;
                  if (ey > 0)
                  {  y += stepy;
                     DrawVoxel(g, x, y, z);
                     ey += dNEy;
                  } else
                  {  ey += dEy;
                  }
               }
            }
         }
      }
      
      // draw center line in x-y plane
      g.DrawLine(Pens.Red,
                 x0 * m_pixelSize + m_pixelSize * 0.5f,
                 y0 * m_pixelSize + m_pixelSize * 0.5f + m_topY,
                 x1 * m_pixelSize + m_pixelSize * 0.5f,
                 y1 * m_pixelSize + m_pixelSize * 0.5f + m_topY);
      
      // draw center line in z-y plane
      g.DrawLine(Pens.Red,
                 m_offsetX + z0 * m_pixelSize + m_pixelSize*0.5f,
                 y0 * m_pixelSize + m_pixelSize * 0.5f + m_topY,
                 m_offsetX + z1 * m_pixelSize + m_pixelSize * 0.5f,
                 y1 * m_pixelSize + m_pixelSize * 0.5f + m_topY);
   }

   /// <summary>
   /// Draws an enlarged pixel for the demo versions
   /// </summary>
   /// <param name="g">Graphics device</param>
   /// <param name="x">Enlarged x-coord.</param>
   /// <param name="y">Enlarged y-coord.</param>
   void DrawPixel(Graphics g, int x, int y)
   {  g.FillRectangle(Brushes.Gray, 
                      x*m_pixelSize+1,
                      m_topY + y * m_pixelSize + 1,
                      m_pixelSize-1,
                      m_pixelSize-1);
   }

      /// <summary>
   /// Draws an enlarged voxel for traverseVoxel method.
   /// </summary>
   /// <param name="g">Graphics device</param>
   /// <param name="x">Enlarged x-coord.</param>
   /// <param name="y">Enlarged y-coord.</param>
   /// <param name="y">Enlarged z-coord.</param>
   void DrawVoxel(Graphics g, int x, int y, int z)
   {  // draw in x-y grid
      g.FillRectangle(Brushes.Gray,
                      x * m_pixelSize + 1,
                      m_topY + y * m_pixelSize + 1,
                      m_pixelSize-1,
                      m_pixelSize-1);
      // draw in z-y grid
      g.FillRectangle(Brushes.Gray,
                      m_offsetX + z * m_pixelSize + 1,
                      m_topY + y * m_pixelSize + 1,
                      m_pixelSize-1,
                      m_pixelSize-1);
   }
   
   /// <summary>
   /// Classic line drawing algorithm from Jack Elton Bresenham (born 1937).
   /// The Bresenham line algorithm is an algorithm which determines which points in 
   /// an n-dimensional raster should be plotted in order to form a close 
   /// approximation to a straight line between two given points. It is commonly used 
   /// to draw lines on a computer screen, as it uses only integer addition, 
   /// subtraction and bit shifting, all of which are very cheap operations in 
   /// standard computer architectures. It is one of the earliest algorithms 
   /// developed in the field of computer graphics. For more info see:
   /// http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
   /// </summary>
   /// <param name="g">Graphics device</param>
   /// <param name="x0">x-coord. of first point</param>
   /// <param name="y0">y-coord. of first point</param>
   /// <param name="x1">x-coord. of second point</param>
   /// <param name="y1">y-coord. of second point</param>
   void DrawLineBresenham(Graphics g, int x0, int y0, int x1, int y1) 
   {  
      unsafe 
      {
         int x = x0, y = y0;
         int dx = x1 - x0;
         int dy = y1 - y0;
         int stepx=1, stepy=1;

         if (dy < 0) {dy = -dy; stepy = -1;} // mirroring on y
         if (dx < 0) {dx = -dx; stepx = -1;} // mirroring on x

         if (dx > dy)                 // slope < 1 (x is driving)
         {  int dE  = (dy<<1);        // = 2*dy
            int dNE = (dy-dx)<<1;     // = 2*(dy-dx)
            int e   = (dy<<1) - dx;   // = 2*dy - dx;
         
            while (x != x1) 
            {  g.DrawImageUnscaled(m_point, x, y);
               x += stepx;
               if (e < 0)             // choose E
               {  e += dE;
               } else                 // choose NE
               {  e += dNE; 
                  y += stepy;
               }
            }
         } else                       // slope > 1 (y is driving)
         {  int dE  = (dx<<1);        // = 2*dx
            int dNE = (dx-dy)<<1;     // = 2*(dx-dy)
            int e   = (dx<<1) - dy;   // = 2*dx - dy; 
         
            while (y != y1) 
            {  g.DrawImageUnscaled(m_point, x, y);
               y += stepy; 
               if (e < 0)             // choose E
               {  e += dE; 
               } else                 // choose NE
               {  e += dNE;
                  x += stepx; 
               }
            }
         }
         g.DrawImageUnscaled(m_point, x, y); // draw last pixel
      }
   }

   #region Menu Handler
   private void mnuViewBresenham_Click(object sender, EventArgs e)
   {
      if (!mnuViewBresenham.Checked) mnuViewBresenham.Checked = true;
      mnuViewPixels.Checked = false;
      mnuViewVoxels.Checked = false;
      mnuViewCompare.Checked = false;
      this.Invalidate();
   }

   private void mnuViewPixels_Click(object sender, EventArgs e)
   {
      mnuViewBresenham.Checked = false;
      if (!mnuViewPixels.Checked) mnuViewPixels.Checked = true;
      mnuViewVoxels.Checked = false;
      mnuViewCompare.Checked = false;
      this.Invalidate();
   }

   private void mnuViewVoxels_Click(object sender, EventArgs e)
   {
      mnuViewBresenham.Checked = false;
      mnuViewPixels.Checked = false;
      if (!mnuViewVoxels.Checked) mnuViewVoxels.Checked = true;
      mnuViewCompare.Checked = false;
      this.Invalidate();
   }

   private void mnuViewCompare_Click(object sender, EventArgs e)
   {
      mnuViewBresenham.Checked = false;
      mnuViewPixels.Checked = false;
      mnuViewVoxels.Checked = false;
      if (!mnuViewCompare.Checked) mnuViewCompare.Checked = true;
      this.Invalidate();
   }

   private void mnuFileExit_Click(object sender, EventArgs e)
   {  
      this.Close();
   }
   #endregion

   /// <summary>
   /// The main entry point for the application.
   /// </summary>
   [STAThread]
   static void Main()
   {
      Application.EnableVisualStyles();
      Application.SetCompatibleTextRenderingDefault(false);
      Application.Run(new frmBresenham());
   }
}
