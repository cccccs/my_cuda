#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

// 核函数：Reduce求和（树状规约）
__global__ void reduce_sum_kernel_1(int *input, int *output, int n) {
  extern __shared__ int a[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[tid] = input[i];
  __syncthreads();
  for (int i = blockDim.x/2; i >0; i/=2) {
    if(tid < i) {
        a[tid] += a[tid + i];
        __syncthreads();
    }
  }
  if (tid == 0) {
    output[blockIdx.x] = a[tid];
  }
}
__global__ void reduce_sum_kernel(int *input, int *output, int n) {
  extern __shared__ int a[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[tid] = input[i];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i*=2) {
    if(tid %(2*i) == 0) {
        a[tid] += a[tid + i];
        __syncthreads();
    }
  }
  if (tid == 0) {
    output[blockIdx.x] = a[tid];
  }
}
// 主机端函数：验证结果
int verify_result(int *array, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += array[i];
    }
    return sum;
}

// 辅助函数：检查CUDA错误
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA错误 (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 使用CUDA事件计时的版本
void run_with_cuda_events(int *h_input, int *d_input, int *d_output, int *h_output,
                          int n, int threads_per_block, int blocks_per_grid,
                          int shared_mem_size, int iterations) {
    cudaEvent_t start, stop;
    float total_kernel_time = 0.0f;
    float total_memcpy_h2d_time = 0.0f;
    float total_memcpy_d2h_time = 0.0f;
    float total_full_time = 0.0f;
    
    // 创建CUDA事件
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("使用CUDA事件计时 (执行 %d 次取平均值):\n", iterations);
    printf("========================================\n");
    
    for (int iter = 0; iter < iterations; iter++) {
        // 完整过程计时
        cudaEventRecord(start, 0);
        
        // 主机到设备内存传输计时
        cudaEventRecord(start, 0);
        cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float memcpy_h2d_time;
        cudaEventElapsedTime(&memcpy_h2d_time, start, stop);
        total_memcpy_h2d_time += memcpy_h2d_time;
        
        // 核函数执行计时
        cudaEventRecord(start, 0);
        reduce_sum_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float kernel_time;
        cudaEventElapsedTime(&kernel_time, start, stop);
        total_kernel_time += kernel_time;
        
        // 设备到主机内存传输计时
        cudaEventRecord(start, 0);
        cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float memcpy_d2h_time;
        cudaEventElapsedTime(&memcpy_d2h_time, start, stop);
        total_memcpy_d2h_time += memcpy_d2h_time;
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float full_time;
        cudaEventElapsedTime(&full_time, start, stop);
        total_full_time += full_time;
        
        // 打印每次迭代的详细信息
        printf("迭代 %2d: 完整时间=%.3fms (H2D=%.3fms, Kernel=%.3fms, D2H=%.3fms)\n",
               iter + 1, full_time, memcpy_h2d_time, kernel_time, memcpy_d2h_time);
    }
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 打印平均时间
    printf("\n平均时间 (执行 %d 次):\n", iterations);
    printf("  完整过程: %.3f ms\n", total_full_time / iterations);
    printf("  主机到设备传输: %.3f ms\n", total_memcpy_h2d_time / iterations);
    printf("  核函数执行: %.3f ms\n", total_kernel_time / iterations);
    printf("  设备到主机传输: %.3f ms\n", total_memcpy_d2h_time / iterations);
    printf("  核函数占比: %.1f%%\n", 
           (total_kernel_time / total_full_time) * 100);
}

// 使用C++高精度时钟的版本
void run_with_high_res_clock(int *h_input, int *d_input, int *d_output, int *h_output,
                            int n, int threads_per_block, int blocks_per_grid,
                            int shared_mem_size, int iterations) {
    double total_kernel_time = 0.0;
    double total_memcpy_time = 0.0;
    double total_full_time = 0.0;
    
    printf("\n使用高精度时钟计时 (执行 %d 次取平均值):\n", iterations);
    printf("========================================\n");
    
    for (int iter = 0; iter < iterations; iter++) {
        // 完整过程计时
        auto start_full = std::chrono::high_resolution_clock::now();
        
        // 内存传输计时
        auto start_mem = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
        auto end_mem = std::chrono::high_resolution_clock::now();
        
        // 核函数执行计时
        auto start_kernel = std::chrono::high_resolution_clock::now();
        reduce_sum_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
        auto end_kernel = std::chrono::high_resolution_clock::now();
        
        // 结果传回计时
        auto start_result = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
        auto end_result = std::chrono::high_resolution_clock::now();
        
        auto end_full = std::chrono::high_resolution_clock::now();
        
        // 计算时间
        std::chrono::duration<double, std::milli> memcpy_time = 
            (end_mem - start_mem) + (end_result - start_result);
        std::chrono::duration<double, std::milli> kernel_time = end_kernel - start_kernel;
        std::chrono::duration<double, std::milli> full_time = end_full - start_full;
        
        total_memcpy_time += memcpy_time.count();
        total_kernel_time += kernel_time.count();
        total_full_time += full_time.count();
        
        printf("迭代 %2d: 完整时间=%.3fms (内存传输=%.3fms, 核函数=%.3fms)\n",
               iter + 1, full_time.count(), memcpy_time.count(), kernel_time.count());
    }
    
    // 打印平均时间
    printf("\n平均时间 (执行 %d 次):\n", iterations);
    printf("  完整过程: %.3f ms\n", total_full_time / iterations);
    printf("  内存传输: %.3f ms\n", total_memcpy_time / iterations);
    printf("  核函数执行: %.3f ms\n", total_kernel_time / iterations);
    printf("  核函数占比: %.1f%%\n", 
           (total_kernel_time / total_full_time) * 100);
}

int main(int argc, char *argv[]) {
    // 设置参数
    int n = 1024 * 1024;  // 增加到1M个元素，以便更好地测量性能
    int size = n * sizeof(int);
    int iterations = 10;   // 执行次数
    int warmup_iterations = 3; // 预热次数
    
    printf("CUDA Reduce算子性能测试\n");
    printf("=======================\n");
    printf("数组大小: %d (%.2f MB)\n", n, size / (1024.0 * 1024.0));
    printf("总执行次数: %d (包含预热 %d 次)\n", iterations, warmup_iterations);
    
    // 分配主机内存
    int *h_input = (int *)malloc(size);
    int *h_output = (int *)malloc(sizeof(int));
    
    // 初始化输入数组
    printf("初始化输入数组...\n");
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 100 + 1;  // 填充1-100的循环值
    }
    
    // 分配设备内存
    int *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, sizeof(int));
    int *h_block_sums = NULL;
    // 设置线程块和网格大小
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 1) {
        // 分配额外的内存来存储块结果
        cudaFree(d_output);
        cudaMalloc((void **)&d_output, blocks_per_grid * sizeof(int));
    }
    
    // 计算共享内存大小
    int shared_mem_size = threads_per_block * sizeof(int);
    
    printf("配置: 线程块大小=%d, 网格大小=%d, 共享内存=%d字节\n", 
           threads_per_block, blocks_per_grid, shared_mem_size);
    
    // 预热执行（不测量时间）
    printf("\n预热执行 (%d 次)...\n", warmup_iterations);
    for (int i = 0; i < warmup_iterations; i++) {
        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        reduce_sum_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, n);
        cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    printf("预热完成\n");
    
    // 方法1：使用CUDA事件计时
    run_with_cuda_events(h_input, d_input, d_output, h_output, n, 
                         threads_per_block, blocks_per_grid, 
                         shared_mem_size, iterations);
    
    // 方法2：使用高精度时钟计时
    run_with_high_res_clock(h_input, d_input, d_output, h_output, n,
                           threads_per_block, blocks_per_grid,
                           shared_mem_size, iterations);
    
    // 验证结果
    printf("\n结果验证:\n");
    printf("=========\n");
    // 将结果拷贝回主机
    if (blocks_per_grid > 1) {
        // 多块情况：先拷贝各块的和
        h_block_sums = (int *)malloc(blocks_per_grid * sizeof(int));
        cudaMemcpy(h_block_sums, d_output, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);
        
        // 在主机上完成最终归约
        int final_sum = 0;
        for (int i = 0; i < blocks_per_grid; i++) {
            final_sum += h_block_sums[i];
        }
        *h_output = final_sum;
    } else {
        // 单块情况：直接拷贝结果
        cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    }

    int cpu_result = verify_result(h_input, n);
    printf("GPU计算结果: %d\n", *h_output);
    printf("CPU验证结果: %d\n", cpu_result);
    printf("结果是否一致: %s\n", (*h_output == cpu_result) ? "是 ✓" : "否 ✗");
    
    // 性能总结
    printf("\n性能总结:\n");
    printf("=========\n");
    printf("数据大小: %d 个整数 (%.2f MB)\n", n, size / (1024.0 * 1024.0));
    printf("理论计算量: %d 次加法\n", n - 1);
    printf("并行度: %d 个线程块 × %d 个线程/块 = %d 个并发线程\n",
           blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block);
    
    // 清理内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n测试完成！\n");
    
    return 0;
}