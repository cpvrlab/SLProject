using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

public partial class frmHelloCube : Form
{
   #region Members
   private SLMat4f   m_modelViewMatrix;   // combined model & view matrix
   private SLMat4f   m_projectionMatrix;  // projection matrix
   private SLMat4f   m_viewportMatrix;    // viewport matrix
   private SLVec3f[] m_v;                 // array for vertices for the cube
   private float     m_camZ;              // z-distance of camera
   private float     m_rotAngle;          // angle of cube rotation
   #endregion

   /// <summary>
   /// We intialize the matrices the the vertices foth the wireframe cube
   /// </summary>
   public frmHelloCube()
   {
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
      m_v[6] = new SLVec3f( 0.5f, 0.5f,-0.5f); // back upper left
      m_v[7] = new SLVec3f(-0.5f, 0.5f,-0.5f); // back upper right

      m_camZ = -4;      // backwards movment of the camera
      m_rotAngle = 0;   // initial rotation angle

      // Without double buffering it would flicker
      this.DoubleBuffered = true;

      InitializeComponent();
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
      //m_modelViewMatrix.Translate(1, 0, 0);
      m_modelViewMatrix.Rotate(m_rotAngle+=0.05f, 0,1,0);
      m_modelViewMatrix.Scale(2, 2, 2);
      
      // build combined matrix out of viewport, projection & modelview matrix
      SLMat4f m = new SLMat4f();
      m.Multiply(m_viewportMatrix);
      m.Multiply(m_projectionMatrix);
      m.Multiply(m_modelViewMatrix);

      // transform all vertices into screen space (x & y in pixels and z as the depth) 
      SLVec3f[] v2 = new SLVec3f[8];
      for (int i=0; i < m_v.Length; ++i)
      {  v2[i] = m.Multiply(m_v[i]);
      }

      Graphics g = e.Graphics;
      g.SmoothingMode = SmoothingMode.AntiAlias;
      
      // draw front square
      g.DrawLine(Pens.Red, v2[0].x, v2[0].y, v2[1].x, v2[1].y);
      g.DrawLine(Pens.Red, v2[1].x, v2[1].y, v2[2].x, v2[2].y);
      g.DrawLine(Pens.Red, v2[2].x, v2[2].y, v2[3].x, v2[3].y);
      g.DrawLine(Pens.Red, v2[3].x, v2[3].y, v2[0].x, v2[0].y);
      // draw back square
      g.DrawLine(Pens.Green, v2[4].x, v2[4].y, v2[5].x, v2[5].y);
      g.DrawLine(Pens.Green, v2[5].x, v2[5].y, v2[6].x, v2[6].y);
      g.DrawLine(Pens.Green, v2[6].x, v2[6].y, v2[7].x, v2[7].y);
      g.DrawLine(Pens.Green, v2[7].x, v2[7].y, v2[4].x, v2[4].y);
      // draw from front corners to the back corners
      g.DrawLine(Pens.Blue, v2[0].x, v2[0].y, v2[4].x, v2[4].y);
      g.DrawLine(Pens.Blue, v2[1].x, v2[1].y, v2[5].x, v2[5].y);
      g.DrawLine(Pens.Blue, v2[2].x, v2[2].y, v2[6].x, v2[6].y);
      g.DrawLine(Pens.Blue, v2[3].x, v2[3].y, v2[7].x, v2[7].y);
      
      // Tell the system that the window should be repaint again
      this.Invalidate();
   }

   /// <summary>Handles the mouse down event</summary>
   private void frmHelloCube_MouseDown(object sender, MouseEventArgs e)
   {
      //???
   }
   
   /// <summary>Handles the mouse move event</summary>
   private void frmHelloCube_MouseMove(object sender, MouseEventArgs e)
   {
      //???
   }
   
   /// <summary>Handles the mouse up event</summary>
   private void frmHelloCube_MouseUp(object sender, MouseEventArgs e)
   {
      //???
   }
   
   /// <summary>Handles the mouse wheel event</summary>
   private void frmHelloCube_MouseWheel(object sender, MouseEventArgs e)
   {
      //???
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

