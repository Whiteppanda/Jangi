#include"fpga_api.h"
#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/mman.h>
#include<cstring>

#define min(x,y) (((x)<(y))?(x):(y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;

  m1_size_ = v_size * v_size;

  data_size_ = (m_size_+1)*v_size_; // fpga bram data size
  data_size_M = (2*v_size_)*v_size_*sizeof(float);

  fd_ = open("/dev/mem", O_RDWR);
  data_M = static_cast<float*>(mmap(NULL, data_size_M, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, data_addr));
  data_ = new float[data_size_];	

  output_ = static_cast<unsigned int*>(mmap(NULL, sizeof(unsigned int), PROT_READ|PROT_WRITE, MAP_SHARED,fd_, output_addr));
  output_MV = new unsigned int[m_size_];
  // output_M = static_cast<unsigned int*>(NULL);

  num_block_call_ = 0;
}

FPGA::~FPGA()
{
  munmap(data_M, data_size_M);
  munmap(output_, sizeof(unsigned int));
  close(fd_);

  delete[] data_;
}

float* FPGA::matrix(void)
{
  return data_ + v_size_;
}

float* FPGA::vector(void)
{
  return data_;
}

float* FPGA::matrix_M1(void)
{
  return data_M;
}

float* FPGA::matrix_M2(void)
{
  return data_M + m1_size_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

const float* FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float* vec = this->vector();
  float* mat = this->matrix();
  float* out  = reinterpret_cast<float*>(output_MV);  

  for(int i = 0; i < m_size_; ++i)
  {
    out[i] = 0;
    for(int j = 0; j < v_size_; ++j)
      out[i] += vec[j] * mat[v_size_*i + j];
  }

  for(int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;    
}

const float* __attribute__((optimize("O0"))) FPGA::blockMM()
{
  num_block_call_ += 1;

  // fpga version
  *output_ = 0x5555;
  while(*output_ == 0x5555);

  return data_M;    
}

void FPGA::largeMV(const float* large_mat, const float* input, float* output, int num_input, int num_output)
{
  float* vec = this->vector();
  float* mat = this->matrix();

  // 0) Initialize output vector		
  for(int i = 0; i < num_output; ++i) 
    output[i] = 0;

  for(int i = 0; i < num_output; i += m_size_)
  {
    for(int j = 0; j < num_input; j += v_size_)
    {			
      // 0) Initialize input vector
      int block_row = min(m_size_, num_output-i);
      int block_col = min(v_size_, num_input-j);

      // 1) Assign a vector
      memcpy(vec, input + j, sizeof(float) * block_col);
      if (block_col < v_size_)
        memset(vec + block_col, 0, sizeof(float) * (v_size_ - block_col));
      // 2) Assign a matrix
      /* IMPLEMENT */
      for (int k = 0; k < block_row; k++)
      {
        memcpy(mat + k * v_size_, large_mat + (i + k) * num_input + j, sizeof(float) * block_col);

        if (block_col < v_size_)
          memset(mat + k * v_size_ + block_col, 0, sizeof(float) * (v_size_ - block_col));
      }

      for (int k = block_row; k < m_size_; k++)
      {
        memset(mat + k * v_size_, 0, sizeof(float) * (v_size_));
      }

      // 3) Call a function `blockMV() to execute MV multiplication
      const float* ret = this->blockMV();

      // 4) Accumulate intermediate results
      for(int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    } 
  }
}

void FPGA::largeMM(const float* weight_mat, const float* input_mat, float* output, int num_input, int num_output, int num_matrix2)
{
  float* m1 = this->matrix_M1();
  float* m2 = this->matrix_M2();

  // 0) Initialize output vector		
  for(int i = 0; i < num_output*num_matrix2; ++i)
    output[i] = 0;

  for(int i = 0; i < num_output; i += v_size_)
  {
    for(int j = 0; j < num_input; j += v_size_)
    {			
      for(int k = 0; k < num_matrix2; k += v_size_)
      {
        // 0) Initialize input vector
        int block_row = min(v_size_, num_output - i);
        int block_col_1 = min(v_size_, num_input - j);
        int block_col_2 = min(v_size_, num_matrix2 - k);

        // 1) Assign a m1
        // IMPLEMENT THIS
        for (int row1 = 0; row1 < block_row; k++)
        {
          float *ps_dram = mmap(NULL, sizeof(float) * block_col_1, PROT_READ|PROT_WRITE, MAP_SHARE, fd_, 0x10000000);
          memcpy(ps_dram, weight_mat + (i + row1) * num_input + j, sizeof(float) * block_col_1);
          unsigned int *fpga_dma = mmap(NULL, 16*sizeof(unsigned int), PROT_READ|PROT_WRITE, MAP_SHARED, foo, 0x7E200000); // register address
          *(fpga_dma + 6) = 0x10000000;
          *(fpga_dma + 8) = m1 + row1 * v_size_; // should change this to adress which is seen by dma
          *(fpga_dma + 10) = sizeof(float) * block_col_1;
          while((*(fpga_dma + 1) && 0x00000002) == 0);
          
          //memcpy(m1 + row1 * v_size_, weight_mat + (i + row1) * num_input + j, sizeof(float) * block_col_1);

          if (block_col_1 < v_size_)
            memset(m1 + row1 * v_size_ + block_col_1, 0, sizeof(float) * (v_size_ - block_col_1));
        }

        for (int row1 = block_row; row1 < v_size_; row1++)
        {
          memset(m1 + row1 * v_size_, 0, sizeof(float) * (v_size_));
        }

        // 2) Assign a m2
        // IMPLEMENT THIS
        for (int row2 = 0; row2 < block_col_1; row2++)
        {
          float *ps_dram = mmap(NULL, sizeof(float) * block_col_2, PROT_READ|PROT_WRITE, MAP_SHARE, fd_, 0x10000000);
          memcpy(ps_dram, input_mat + (j + row2) * num_matrix2 + k, sizeof(float) * block_col_2);
          unsigned int *fpga_dma = mmap(NULL, 16*sizeof(unsigned int), PROT_READ|PROT_WRITE, MAP_SHARED, foo, 0x43C00000); // register address
          *(fpga_dma + 6) = 0x10000000;
          *(fpga_dma + 8) = m2 + row2 * v_size_; // should change this to adress which is seen by dma
          *(fpga_dma + 10) = sizeof(float) * block_col_2;
          while((*(fpga_dma + 1) && 0x00000002) == 0);


          //memcpy(m2 + row2 * v_size_, input_mat + (j + row2) * num_matrix2 + k, sizeof(float) * block_col_2);

          if (block_col_2 < v_size_)
            memset(m2 + row2 * v_size_ + block_col_2, 0, sizeof(float) * (v_size_ - block_col_2));
        }

        for (int row2 = block_col_1; row2 < v_size_; row2++)
        {
          memset(m2 + row2 * v_size_, 0, sizeof(float) * (v_size_));
        }
        // IMPLEMENT THIS

        // 3) Call a function `blockMM() to execute Matrix matrix multiplication
        const float* ret = this->blockMM();

        // 4) Accumulate intermediate results
        for (int n = 0; n < block_row; ++n)
        {
          for (int m = 0; m < block_col_2; ++m)
          {
            output[(i + n) + (k + m) * num_output] += ret[n * v_size_ + m];
          }
        }
        
      }
    } 
  }
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>>& cnn_weights,
    std::vector<std::vector<float>>& new_weights,
    const std::vector<std::vector<std::vector<float>>>& inputs,
    std::vector<std::vector<float>>& new_inputs) {
  /*
   * Arguments:
   *
   * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
   * new_weights: [?, ?]
   * inputs: [input_channel, input_height, input_width]
   * new_inputs: [?, ?]
   *
   */

  int conv_channel = cnn_weights.size();
  int input_channel = cnn_weights[0].size();
  int conv_height = cnn_weights[0][0].size();
  int conv_width = cnn_weights[0][0][0].size();
  //int input_channel = cnn_weights.size();
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];
  for (int i = 0; i < conv_channel; i++)
  {
    for (int j = 0; j < input_channel; j++)
    {
      for (int k = 0; k < conv_height; k++)
      {
        for (int l = 0; l < conv_width; l++)
        {
          int col = j * conv_height * conv_width + k * conv_width + l;
          new_weights[i][col] = cnn_weights[i][j][k][l];
        }
      }
    }
  }

  // make new_inputs
  for (int i = 0; i < input_channel; i++)
  {
    for (int j = 0; j < input_height - conv_height + 1; j++)
    {
      for (int k = 0; k < input_width - conv_width + 1; k++)
      {
        for (int l = 0; l < conv_height; l++)
        {
          for (int m = 0; m < conv_width; m++)
          {
            int row = i * conv_height * conv_width + l * conv_width + m;
            int col = j * (input_width - conv_width + 1) + k;
            new_inputs[row][col] = inputs[i][j + l][k + m];
          }
        }
      }
    }
  }
}
