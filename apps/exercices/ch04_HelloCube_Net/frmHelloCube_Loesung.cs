using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Windows.Forms;

public partial class frmHelloCube : Form
{
   #region Members
   private SLMat4f   m_modelViewMatrix;   // combined model & view matrix
   private SLMat4f   m_projectionMatrix;  // projection matrix
   private SLMat4f   m_viewportMatrix;    // viewport matrix
   private SLVec3f[] m_v;                 // array for vertices for the cube
   private float     m_camZ;              // z-distance of camera

   private float     m_rotx, m_roty;      // rotation angles around x & y axis
   private float     m_dx, m_dy;          // delta mouse motion
   private int       m_startx, m_starty;  // x,y mouse start positions
   private bool      m_mouseLeftDown;     // flag if mouse is down
   #endregion

   /// <summary>
   /// We intialize the matrices the the vertices foth the wireframe cube
   /// </summary>
   public frmHelloCube()
   {
      InitializeComponent();

      // Create matrices
      m_modelViewMatrix  = new SLMat4f();
      m_projectionMatrix = new SLMat4f();
      m_viewportMatrix   = new SLMat4f();

      // define the 8 vertices of a cube
      m_v = new SLVec3f[8];
      m_v[0] = new SLVec3f(-0.5f,-0.5f, 0.5f); // front lower left
      m_v[1] = new SLVec3f( 0.5f,-0.5f, 0.5f); // front lower right
      m_v[2] = new SLVec3f( 0.5f, 0.5f, 0.5f); // front upper right
      m_v[3] = new SLVec3f(-0.5f, 0.5f, 0.5f); // front upper left
      m_v[4] = new SLVec3f(-0.5f,-0.5f,-0.5f); // back lower left
      m_v[5] = new SLVec3f( 0.5f,-0.5f,-0.5f); // back lower right
      m_v[6] = new SLVec3f( 0.5f, 0.5f,-0.5f); // back upper right
      m_v[7] = new SLVec3f(-0.5f, 0.5f,-0.5f); // back upper left

      m_camZ = -2;      // backwards movment of the camera
      m_rotx = 0;
      m_roty = 0;
      m_dx   = 0;
      m_dy   = 0;
      m_mouseLeftDown = false;

      // Without double buffering it would flicker
      this.DoubleBuffered = true;
   }

   /// <summary>
   /// The forms load handler is used to call resize before the first paint
   /// </summary>
   private void frmHelloCube_Load(object sender, EventArgs e)
   {  
      this.Text = "Hello Cube with .NET";
      Console.WriteLine("");
      Console.WriteLine("--------------------------------------------------------------");
      Console.WriteLine("Spinning cube without with .Net ...");
   
      frmHelloCube_Resize(null, null);
   }

   /// <summary>
   /// The forms resize handler get called whenever the form is resized.
   /// When the form resizes we have to redefine the projection matrix
   /// as well as the viewport matrix.
   /// </summary>
   private void frmHelloCube_Resize(object sender, EventArgs e)
   {  
      float aspect = (float)ClientRectangle.Width / (float)ClientRectangle.Height;
      m_projectionMatrix.Perspective(50, aspect, 1.0f, 3.0f);
      m_viewportMatrix.Viewport(0, 0, 
                                ClientRectangle.Width, 
                                ClientRectangle.Height, 
                                0, 1);
      this.Invalidate();
   }

   /// <summary>
   /// The forms paint routine where all drawing happens.
   /// </summary>
   private void frmHelloCube_Paint(object sender, PaintEventArgs e)
   {   
      // start with identity every frame
      m_modelViewMatrix.Identity();
   
      // view transform: move the coordiante system away from the camera
      m_modelViewMatrix.Translate(0, 0, m_camZ);
   
      // model transform: rotate the coordinate system increasingly
      m_modelViewMatrix.Rotate(m_rotx+m_dx, 1,0,0);
      m_modelViewMatrix.Rotate(m_roty+m_dy, 0,1,0);

      // build combined matrix out of viewport, projection & modelview matrix
      SLMat4f m = new SLMat4f();
      m.Multiply(m_viewportMatrix);
      m.Multiply(m_projectionMatrix);
      m.Multiply(m_modelViewMatrix);

      // transform all vertices into screen space (x & y in pixels and z as the depth) 
      SLVec3f[] v2 = new SLVec3f[8];
      for (int i=0; i < m_v.Count(); ++i)
      {  v2[i] = m.Multiply(m_v[i]);
      }

      SLVec3f nZ = new SLVec3f(0, 0, 1);

      // Calculate the cubes plane normals in screen space
      // Be aware that y is inverse due to MS top-left zero coord.
      SLVec3f nN = (v2[1] - v2[0]).cross(v2[0] - v2[3]);
      SLVec3f nF = (v2[5] - v2[4]).cross(v2[7] - v2[4]);
      SLVec3f nL = (v2[4] - v2[0]).cross(v2[3] - v2[0]);
      SLVec3f nR = (v2[5] - v2[1]).cross(v2[1] - v2[2]);
      SLVec3f nT = (v2[7] - v2[3]).cross(v2[2] - v2[3]);
      SLVec3f nB = (v2[4] - v2[0]).cross(v2[0] - v2[1]);
      bool visibleN = nN.dot(nZ) >= 0; //near
      bool visibleF = nF.dot(nZ) >= 0; //far
      bool visibleL = nL.dot(nZ) >= 0; //left
      bool visibleR = nR.dot(nZ) >= 0; //right
      bool visibleT = nT.dot(nZ) >= 0; //top
      bool visibleB = nB.dot(nZ) >= 0; //bottom

      Graphics g = e.Graphics;
      g.SmoothingMode = SmoothingMode.AntiAlias;
      
      // draw front square
      if (visibleN)
      {  g.DrawLine(Pens.Red, v2[0].x, v2[0].y, v2[1].x, v2[1].y);
         g.DrawLine(Pens.Red, v2[1].x, v2[1].y, v2[2].x, v2[2].y);
         g.DrawLine(Pens.Red, v2[2].x, v2[2].y, v2[3].x, v2[3].y);
         g.DrawLine(Pens.Red, v2[3].x, v2[3].y, v2[0].x, v2[0].y);
      }
      // draw back square
      if (visibleF)
      {  g.DrawLine(Pens.Green, v2[4].x, v2[4].y, v2[5].x, v2[5].y);
         g.DrawLine(Pens.Green, v2[5].x, v2[5].y, v2[6].x, v2[6].y);
         g.DrawLine(Pens.Green, v2[6].x, v2[6].y, v2[7].x, v2[7].y);
         g.DrawLine(Pens.Green, v2[7].x, v2[7].y, v2[4].x, v2[4].y);
      }
      // draw left square
      if (visibleL)
      {  g.DrawLine(Pens.Blue, v2[0].x, v2[0].y, v2[4].x, v2[4].y);
         g.DrawLine(Pens.Blue, v2[3].x, v2[3].y, v2[7].x, v2[7].y);
         g.DrawLine(Pens.Red,  v2[3].x, v2[3].y, v2[0].x, v2[0].y);
         g.DrawLine(Pens.Green,v2[7].x, v2[7].y, v2[4].x, v2[4].y);
      }
      // draw right square
      if (visibleR)
      {  g.DrawLine(Pens.Blue, v2[1].x, v2[1].y, v2[5].x, v2[5].y);
         g.DrawLine(Pens.Blue, v2[2].x, v2[2].y, v2[6].x, v2[6].y);
         g.DrawLine(Pens.Red,  v2[1].x, v2[1].y, v2[2].x, v2[2].y);
         g.DrawLine(Pens.Green,v2[6].x, v2[6].y, v2[5].x, v2[5].y);
      }
      // draw top square
      if (visibleT)
      {  g.DrawLine(Pens.Blue, v2[2].x, v2[2].y, v2[6].x, v2[6].y);
         g.DrawLine(Pens.Blue, v2[3].x, v2[3].y, v2[7].x, v2[7].y);
         g.DrawLine(Pens.Red,  v2[3].x, v2[3].y, v2[2].x, v2[2].y);
         g.DrawLine(Pens.Green,v2[6].x, v2[6].y, v2[7].x, v2[7].y);
      }
      // draw bottom square
      if (visibleB)
      {  g.DrawLine(Pens.Blue, v2[0].x, v2[0].y, v2[4].x, v2[4].y);
         g.DrawLine(Pens.Blue, v2[1].x, v2[1].y, v2[5].x, v2[5].y);
         g.DrawLine(Pens.Red,  v2[0].x, v2[0].y, v2[1].x, v2[1].y);
         g.DrawLine(Pens.Green,v2[4].x, v2[4].y, v2[5].x, v2[5].y);
      }

      // Tell the system that the window should be repaint again
      this.Invalidate();
   }

   /// <summary>Handles the mouse down event</summary>
   private void frmHelloCube_MouseDown(object sender, MouseEventArgs e)
   {  
      if (e.Button == MouseButtons.Left)
      {  m_mouseLeftDown = true;
         m_startx = e.X;
         m_starty = e.Y;
         this.Invalidate();
      }
   }
   
   /// <summary>Handles the mouse move event</summary>
   private void frmHelloCube_MouseMove(object sender, MouseEventArgs e)
   {
      if (m_mouseLeftDown)
      {  m_dy = e.X - m_startx;
         m_dx = e.Y - m_starty;
         this.Invalidate();
      }
   }
   
   /// <summary>Handles the mouse up event</summary>
   private void frmHelloCube_MouseUp(object sender, MouseEventArgs e)
   {
      if (e.Button == MouseButtons.Left) 
      {  m_mouseLeftDown = false;
         m_rotx += m_dx;
         m_roty += m_dy;
         m_dx = 0;
         m_dy = 0;
         this.Invalidate();
      }
   }
   
   /// <summary>Handles the mouse wheel event</summary>
   private void frmHelloCube_MouseWheel(object sender, MouseEventArgs e)
   {
      if (e.Delta > 0) m_camZ += 0.1f;
      else m_camZ -= 0.1f;
      this.Invalidate();
   }

   /// <summary>
   /// The main entry point for the application.
   /// </summary>
   [STAThread]
   static void Main()
   {
      Application.EnableVisualStyles();
      Application.SetCompatibleTextRenderingDefault(false);
      Application.Run(new frmHelloCube());
   }
}

