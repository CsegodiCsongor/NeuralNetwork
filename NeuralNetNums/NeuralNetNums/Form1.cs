using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetNums
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        NeuralNetwork nn;

        private void Form1_Load(object sender, EventArgs e)
        {
            nn = new NeuralNetwork(new double[,] { { 1, 2, 3 }, { 2, 4, 5 } }, new double[,] { { 0.1, 0.3, 0.76 }, { 0.3, 0.1, 0.2 } }, new int[] { 2, 3, 3, 2 });
            nn.Run();
            nn.ToString();

            for (int i = 1; i < nn.layers.Count; i++)
            {
                richTextBox1.Text += nn.layers[i].z.ToString();
                richTextBox1.Text += "\n";
                richTextBox1.Text += nn.layers[i].a.ToString();
                richTextBox1.Text += "\n";
                if (nn.layers[i].w != null)
                {
                    richTextBox1.Text += nn.layers[i].w.ToString();
                }
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";
                richTextBox1.Text += "\n";

            }
        }

        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            richTextBox1.Text = nn.GetCost().ToString();
            for (int i = 0; i < 1000; i++)
            {
                nn.Run();
            }
        }
    }
}
