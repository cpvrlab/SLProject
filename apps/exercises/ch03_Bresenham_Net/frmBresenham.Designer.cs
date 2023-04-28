partial class frmBresenham
{
   /// <summary>
   /// Required designer variable.
   /// </summary>
   private System.ComponentModel.IContainer components = null;

   /// <summary>
   /// Clean up any resources being used.
   /// </summary>
   /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
   protected override void Dispose(bool disposing)
   {
      if (disposing && (components != null))
      {
         components.Dispose();
      }
      base.Dispose(disposing);
   }

   #region Windows Form Designer generated code

   /// <summary>
   /// Required method for Designer support - do not modify
   /// the contents of this method with the code editor.
   /// </summary>
   private void InitializeComponent()
   {
         this.menu1 = new System.Windows.Forms.MenuStrip();
         this.mnuFile = new System.Windows.Forms.ToolStripMenuItem();
         this.mnuFileExit = new System.Windows.Forms.ToolStripMenuItem();
         this.mnuView = new System.Windows.Forms.ToolStripMenuItem();
         this.mnuViewBresenham = new System.Windows.Forms.ToolStripMenuItem();
         this.mnuViewPixels = new System.Windows.Forms.ToolStripMenuItem();
         this.mnuViewVoxels = new System.Windows.Forms.ToolStripMenuItem();
         this.mnuViewCompare = new System.Windows.Forms.ToolStripMenuItem();
         this.menu1.SuspendLayout();
         this.SuspendLayout();
         // 
         // menu1
         // 
         this.menu1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuFile,
            this.mnuView});
         this.menu1.Location = new System.Drawing.Point(0, 0);
         this.menu1.Name = "menu1";
         this.menu1.Size = new System.Drawing.Size(919, 28);
         this.menu1.TabIndex = 0;
         this.menu1.Text = "menuStrip1";
         // 
         // mnuFile
         // 
         this.mnuFile.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuFileExit});
         this.mnuFile.Name = "mnuFile";
         this.mnuFile.Size = new System.Drawing.Size(44, 24);
         this.mnuFile.Text = "&File";
         // 
         // mnuFileExit
         // 
         this.mnuFileExit.Name = "mnuFileExit";
         this.mnuFileExit.Size = new System.Drawing.Size(102, 24);
         this.mnuFileExit.Text = "E&xit";
         this.mnuFileExit.Click += new System.EventHandler(this.mnuFileExit_Click);
         // 
         // mnuView
         // 
         this.mnuView.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.mnuViewBresenham,
            this.mnuViewPixels,
            this.mnuViewVoxels,
            this.mnuViewCompare});
         this.mnuView.Name = "mnuView";
         this.mnuView.Size = new System.Drawing.Size(53, 24);
         this.mnuView.Text = "View";
         // 
         // mnuViewBresenham
         // 
         this.mnuViewBresenham.Name = "mnuViewBresenham";
         this.mnuViewBresenham.Size = new System.Drawing.Size(378, 24);
         this.mnuViewBresenham.Text = "Draw line with Bresenham ";
         this.mnuViewBresenham.Click += new System.EventHandler(this.mnuViewBresenham_Click);
         // 
         // mnuViewPixels
         // 
         this.mnuViewPixels.Name = "mnuViewPixels";
         this.mnuViewPixels.Size = new System.Drawing.Size(378, 24);
         this.mnuViewPixels.Text = "Draw line of all traversed pixels";
         this.mnuViewPixels.Click += new System.EventHandler(this.mnuViewPixels_Click);
         // 
         // mnuViewVoxels
         // 
         this.mnuViewVoxels.Name = "mnuViewVoxels";
         this.mnuViewVoxels.Size = new System.Drawing.Size(378, 24);
         this.mnuViewVoxels.Text = "Draw line of all traversed voxels";
         this.mnuViewVoxels.Click += new System.EventHandler(this.mnuViewVoxels_Click);
         // 
         // mnuViewCompare
         // 
         this.mnuViewCompare.Checked = true;
         this.mnuViewCompare.CheckState = System.Windows.Forms.CheckState.Checked;
         this.mnuViewCompare.Name = "mnuViewCompare";
         this.mnuViewCompare.Size = new System.Drawing.Size(378, 24);
         this.mnuViewCompare.Text = "Compare Bresenham with GDI+ Line Drawing";
         this.mnuViewCompare.Click += new System.EventHandler(this.mnuViewCompare_Click);
         // 
         // frmBresenham
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.ClientSize = new System.Drawing.Size(919, 583);
         this.Controls.Add(this.menu1);
         this.MainMenuStrip = this.menu1;
         this.Name = "frmBresenham";
         this.Text = "Bresenham-, Pixel- and Voxel-Traversal";
         this.Paint += new System.Windows.Forms.PaintEventHandler(this.frmBresenham_Paint);
         this.menu1.ResumeLayout(false);
         this.menu1.PerformLayout();
         this.ResumeLayout(false);
         this.PerformLayout();

   }

   #endregion

   private System.Windows.Forms.MenuStrip menu1;
   private System.Windows.Forms.ToolStripMenuItem mnuFile;
   private System.Windows.Forms.ToolStripMenuItem mnuView;
   private System.Windows.Forms.ToolStripMenuItem mnuViewBresenham;
   private System.Windows.Forms.ToolStripMenuItem mnuViewPixels;
   private System.Windows.Forms.ToolStripMenuItem mnuViewVoxels;
   private System.Windows.Forms.ToolStripMenuItem mnuFileExit;
   private System.Windows.Forms.ToolStripMenuItem mnuViewCompare;
}

