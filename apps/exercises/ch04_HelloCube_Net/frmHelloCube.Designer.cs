﻿
partial class frmHelloCube
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
            this.SuspendLayout();
            // 
            // frmHelloCube
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(384, 262);
            this.Name = "frmHelloCube";
            this.Text = "Hello Cube in C#";
            this.Load += new System.EventHandler(this.frmHelloCube_Load);
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.frmHelloCube_Paint);
            this.MouseDown += new System.Windows.Forms.MouseEventHandler(this.frmHelloCube_MouseDown);
            this.MouseMove += new System.Windows.Forms.MouseEventHandler(this.frmHelloCube_MouseMove);
            this.MouseUp += new System.Windows.Forms.MouseEventHandler(this.frmHelloCube_MouseUp);
            this.MouseWheel += new System.Windows.Forms.MouseEventHandler(this.frmHelloCube_MouseWheel);
            this.Resize += new System.EventHandler(this.frmHelloCube_Resize);
            this.ResumeLayout(false);

   }

   #endregion
}

