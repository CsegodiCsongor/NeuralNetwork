using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace NeuralNetNums
{
    public partial class Form1 : Form
    {
        Bitmap bmp;
        Graphics grp;

        bool draw = false;

        public Form1()
        {
            InitializeComponent();
        }

        NeuralNetwork nn;

        private void Form1_Load(object sender, EventArgs e)
        {
            bmp = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            grp = Graphics.FromImage(bmp);


            for (int i = 0; i < 280; i++)
            {
                for (int j = 0; j < 280; j++)
                {
                    bmp.SetPixel(i, j, Color.Black);
                }
            }
            nn = new NeuralNetwork(inp, exp, new int[] { 784, 16, 16, 10 });

            pictureBox1.Image = bmp;
            //nn = new NeuralNetwork(new double[,] { { 1, 2, 3 }, 
            //                                       { 2, 4, 5 } }, 
            //                                       new double[,] { { 0.1, 0.3, 0.76 }, 
            //                                                       { 0.3, 0.1, 0.2 } },
            //                                       new int[] { 2, 4, 11, 4, 2 });

            //nn = new NeuralNetwork(new double[,] { { 1 },
            //                                       { 2 } },
            //                                       new double[,] { { 0.1 } },
            //                                       new int[] { 2, 1 });

            //nn.Run();
            //nn.ToString();

            //for (int i = 1; i < nn.layers.Count; i++)
            //{
            //    richTextBox1.Text += nn.layers[i].z.ToString();
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += nn.layers[i].a.ToString();
            //    richTextBox1.Text += "\n";
            //    if (nn.layers[i].w != null)
            //    {
            //        richTextBox1.Text += nn.layers[i].w.ToString();
            //    }
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";
            //    richTextBox1.Text += "\n";

            //}
            //Smth();

        }
        double[,] inp = new double[100, 784];
        double[,] exp = new double[100, 10];
        int nrs = 5000;
        int bachSize = 100;
        double[,] inpAll;
        double[,] expAll;

        public void Smth()
        {
            inp = new double[bachSize, 784];
            exp = new double[bachSize, 10];

            inpAll = new double[nrs, 784];
            //expAll = new double[nrs, 1];
            expAll = new double[nrs, 10];

            int k = 0;
            //int g = MnistReader.ReadTrainingData().Count();
            foreach (var image in MnistReader.ReadTrainingData())
            {
                for (int i = 0; i < image.Data.GetLength(0); i++)
                {
                    for (int j = 0; j < image.Data.GetLength(1); j++)
                    {
                        //inp[i * image.Data.GetLength(0) + j, k] = image.Data[i, j];
                        double v = image.Data[i, j];
                        double val = (v / 255);

                        //inp[k % bachSize, i * image.Data.GetLength(0) + j] = val;

                        inpAll[k, i * image.Data.GetLength(0) + j] = val;
                        //expAll[k, 0] = image.Label;

                        expAll[k, image.Label] = 1;
                        //bmp.SetPixel(i, j, Color.FromArgb(255, image.Data[i, j], image.Data[i, j], image.Data[i, j]));
                        //pictureBox1.Image = bmp;

                    }
                }
                k++;
                if (k >= nrs) { break; }
            }


            nn = new NeuralNetwork(inpAll, expAll, new int[] { 784, 16, 16, 10 });

            nn.rtb = richTextBox1;

        }


        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            //string fv = nn.GetCost().ToString() + "\n";
            //richTextBox1.Text = fv;
            //for (int i = 0; i < int.Parse(richTextBox3.Text); i++)
            // {
            //    nn.TrainALL();
            //}
            //string sv = nn.GetCost().ToString() + "\n";
            // richTextBox1.Text += sv;
            //if (double.Parse(fv) > double.Parse(sv))
            //{
            //    richTextBox1.Text += "GUT!!";
            //}
            //else
            //{
            //    richTextBox1.Text += "NOT GUT!!";
            //}
            double cost = 0;
            do
            {
                richTextBox1.Text = "";
                cost = 0;
                for (int q = 0; q < nrs; q++)
                {
                    for (int j = 0; j < inpAll.GetLength(1); j++)
                    {
                        inp[q % bachSize, j] = inpAll[q, j];
                    }

                    exp[q % bachSize, (int)expAll[q, 0]] = 1;
                    richTextBox2.Text += (int)expAll[q, 0] + "\n";

                    if (q % bachSize == 0)
                    {
                        nn = new NeuralNetwork(inp, exp, new int[] { 784, 16, 16, 10 });

                        nn.layers[0].a = new Matrix(inp);
                        nn.expectedOutputs = new Matrix(exp);
                        inp = new double[bachSize, 784];
                        exp = new double[bachSize, 10];

                        for (int i = 0; i < 1; i++)
                        {
                            nn.TrainAllA();
                        }
                        //richTextBox1.Text += q / bachSize + " -th bach ready. \n";
                        //richTextBox1.Text += "Cost is " + nn.GetCost() + "\n \n";
                        //label1.Text = "Bach: " + q/bachSize;
                        cost += nn.GetCost();
                    }
                }
                cost /= (nrs / bachSize);
            } while (cost > 50);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Smth();
        }


        private void pictureBox1_Click(object sender, EventArgs e)
        {
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            draw = true;
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            draw = false;
        }

        private void pictureBox1_MouseCaptureChanged(object sender, EventArgs e)
        {
        }

        private void timer1_Tick(object sender, EventArgs e)
        {

        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (draw)
            {
                Point c = new Point(((MouseEventArgs)e).X, ((MouseEventArgs)e).Y);

                //bmp.SetPixel(c.X, c.Y, Color.Black);

                grp.DrawEllipse(new Pen(Color.White, 10), c.X - 2, c.Y - 2, 4, 4);

                pictureBox1.Image = bmp;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            /*double[,] i = new double[1, 3];
            i[0, 0] = double.Parse(richTextBox2.Text.Split(' ')[0]);
            i[0, 1] = double.Parse(richTextBox2.Text.Split(' ')[1]);
            i[0, 2] = double.Parse(richTextBox2.Text.Split(' ')[2]);

            Matrix q = nn.Predict(i);

            MessageBox.Show("The predicted nr is: ");*/

            double[,] nr = new double[784, 1];

            byte[,] pixels = new byte[280, 280];

            for (int i = 0; i < 280; i++)
            {
                for (int j = 0; j < 280; j++)
                {
                    Color c = bmp.GetPixel(i, j);
                    pixels[i, j] = (byte)((c.R + c.G + c.B) / 3);
                }
            }


            byte[,] miniPix = new byte[28, 28];

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    int val = 0;
                    for (int q = i * 10; q < i * 10 + 10; q++)
                    {
                        for (int w = j * 10; w < j * 10 + 10; w++)
                        {
                            val += pixels[q, w];
                        }
                    }

                    miniPix[i, j] = (byte)((val / 100) / 255);

                }
            }

            Bitmap bmp2 = new Bitmap(pictureBox2.Width, pictureBox2.Height);
            Graphics grp2 = Graphics.FromImage(bmp2);

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    //bmp2.SetPixel(i, j, Color.FromArgb(miniPix[i, j]));
                    bmp2.SetPixel(i, j, Color.FromArgb(255, miniPix[i, j] * 255, miniPix[i, j] * 255, miniPix[i, j] * 255));
                }
            }

            pictureBox2.Image = bmp2;

            Color r = bmp.GetPixel(200, 200);
            //MessageBox.Show(" ");

            for (int i = 0; i < 280; i++)
            {
                for (int j = 0; j < 280; j++)
                {
                    bmp.SetPixel(i, j, Color.Black);
                }
            }
            pictureBox1.Image = bmp;


            double[,] input = new double[1, 784];

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    input[0, i * 28 + j] = miniPix[j, i];
                }
            }

            Matrix p = nn.Predict(input);

            int m = 0;
            for (int i = 1; i < 10; i++)
            {
                if (p[0, i] > p[0, m])
                {
                    m = i;
                }
            }

            richTextBox4.Text = "";
            for (int i = 0; i < 10; i++)
            {
                richTextBox4.Text += p[0, i] + "\n";
            }

            richTextBox4.Text += m;

            //MessageBox.Show("The predicted nr is: " + m);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            richTextBox2.Text = "";

            for (int i = 0; i < inp.GetLength(0); i++)
            {
                Matrix expected = new Matrix(1, exp.GetLength(1));
                Matrix ing = new Matrix(1, inp.GetLength(1));

                for (int j = 0; j < inp.GetLength(1); j++)
                {
                    ing[0, j] = inp[i, j];
                }

                for (int j = 0; j < exp.GetLength(1); j++)
                {
                    expected[0, j] = exp[i, j];
                }

                Matrix p = new Matrix(nn.Predict(ing));

                int m = 0;
                for (int q = 1; q < 10; q++)
                {
                    if (p[0, q] > p[0, m])
                    {
                        m = q;
                    }
                }
                richTextBox2.Text += "The net predicted: " + m + " And it should have predicted ";
                m = 0;
                for (int q = 1; q < 10; q++)
                {
                    if (expected[0, q] > expected[0, m])
                    {
                        m = q;
                    }
                }
                richTextBox2.Text += m + "\n";
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            double[,] nr = new double[784, 1];

            byte[,] pixels = new byte[280, 280];

            for (int i = 0; i < 280; i++)
            {
                for (int j = 0; j < 280; j++)
                {
                    Color c = bmp.GetPixel(i, j);
                    pixels[i, j] = (byte)((c.R + c.G + c.B) / 3);
                }
            }


            byte[,] miniPix = new byte[28, 28];

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    int val = 0;
                    for (int q = i * 10; q < i * 10 + 10; q++)
                    {
                        for (int w = j * 10; w < j * 10 + 10; w++)
                        {
                            val += pixels[q, w];
                        }
                    }

                    miniPix[i, j] = (byte)((val / 100) / 255);

                }
            }

            Bitmap bmp2 = new Bitmap(pictureBox2.Width, pictureBox2.Height);
            Graphics grp2 = Graphics.FromImage(bmp2);

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    //bmp2.SetPixel(i, j, Color.FromArgb(miniPix[i, j]));
                    bmp2.SetPixel(i, j, Color.FromArgb(255, miniPix[i, j] * 255, miniPix[i, j] * 255, miniPix[i, j] * 255));
                }
            }

            pictureBox2.Image = bmp2;

            double[,] input = new double[1, 784];

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    input[0, i * 28 + j] = miniPix[j, i];
                }
            }

            nn.layers[0].a = new Matrix(input);
            nn.inputs = new Matrix(input);
            nn.expectedOutputs = new Matrix(new double[,] { { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 } });
        }

        private void button6_Click(object sender, EventArgs e)
        {
            nn.TrainAllA();
            richTextBox1.Text += nn.GetCost() + "\n";
        }

        private void button7_Click(object sender, EventArgs e)
        {
            nn.BGD(inpAll, expAll, 100, 2000);
        }
    }
}
