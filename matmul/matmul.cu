#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

// 基础矩阵乘法核函数 - 每个线程计算C矩阵的一个元素
__global__ void matmul_kernel_basic(const float *A, const float *B, float *C, 
                                    int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; // 512
    printf("blockDim.x:%d\n blockDim.y:%d\n gridDim.x:%d\n gridDim.y:%d\n", blockDim.x, blockDim.y,
           gridDim.x, gridDim.y);
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    float res = 0;
    for (int i = 0; i < K;i++) {
        res += A[m*K + i] * B[i*N + n];
    }
    C[m*N + n] = res;
}

// 使用共享内存的矩阵乘法核函数（优化版本）- 每个线程计算C矩阵一个
__global__ void matmul_kernel_shared(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    constexpr int TILE_LEN = 32;
    extern __shared__ float a[TILE_LEN][TILE_LEN];
    extern __shared__ float b[TILE_LEN][TILE_LEN];
    int m = blockIdx.x * TILE_LEN + threadIdx.x;
    int n = blockIdx.y * TILE_LEN + threadIdx.y;
    float sum = 0;
    for (int i = 0; i < (K+TILE_LEN-1) / TILE_LEN; i++) {
        a[threadIdx.x][threadIdx.y] = A[m*K+i*TILE_LEN+threadIdx.y];
        b[threadIdx.x][threadIdx.y] = B[(i*TILE_LEN+threadIdx.x)*N+n];
        __syncthreads();
        for (int j = 0; j < TILE_LEN;j++) {
            sum += a[threadIdx.x][j] * b[j][threadIdx.y];
        }
        __syncthreads();
    }
    C[m*N+n] = sum;
}

// 主机端CPU矩阵乘法，用于验证结果
void matmul_cpu(const float *A, const float *B, float *C, 
                int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 验证两个矩阵是否相等（允许浮点数误差）
bool verify_result(const float *C_gpu, const float *C_cpu, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > epsilon) {
            printf("验证失败! 索引 %d: GPU=%f, CPU=%f, 差值=%f\n", 
                   i, C_gpu[i], C_cpu[i], fabs(C_gpu[i] - C_cpu[i]));
            return false;
        }
    }
    return true;
}

// 初始化矩阵
void init_matrix(float *matrix, int rows, int cols, int seed = 0) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (rand() % 100) / 100.0f;  // 0.0 ~ 0.99
    }
}

// 打印矩阵的一部分
void print_matrix_part(const float *matrix, int rows, int cols, 
                       int start_row, int start_col, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows && i + start_row < rows; i++) {
        for (int j = 0; j < num_cols && j + start_col < cols; j++) {
            printf("%6.2f ", matrix[(i + start_row) * cols + (j + start_col)]);
        }
        printf("...\n");
    }
    printf("...\n");
}

// 性能测试函数（使用CUDA事件）
void run_matmul_test(const float *h_A, const float *h_B, float *h_C,
                     float *d_A, float *d_B, float *d_C,
                     int M, int N, int K, int block_size,
                     int iterations, bool use_shared_memory = false) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_kernel_time = 0.0f;
    float total_memcpy_h2d_time = 0.0f;
    float total_memcpy_d2h_time = 0.0f;
    float total_full_time = 0.0f;
    
    // 设置线程块和网格
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N + block_size - 1) / block_size,
                 (M + block_size - 1) / block_size);
    
    printf("配置: 线程块=(%dx%d), 网格=(%dx%d)\n",
           blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    printf("使用%s内存版本\n", use_shared_memory ? "共享" : "全局");
    
    for (int iter = 0; iter < iterations; iter++) {
        // 完整过程计时
        cudaEventRecord(start, 0);
        
        // 主机到设备内存传输计时
        cudaEventRecord(start, 0);
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float memcpy_h2d_time;
        cudaEventElapsedTime(&memcpy_h2d_time, start, stop);
        total_memcpy_h2d_time += memcpy_h2d_time;
        
        // 核函数执行计时
        cudaEventRecord(start, 0);
        if (use_shared_memory) {
            matmul_kernel_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else {
            matmul_kernel_basic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float kernel_time;
        cudaEventElapsedTime(&kernel_time, start, stop);
        total_kernel_time += kernel_time;
        
        // 设备到主机内存传输计时
        cudaEventRecord(start, 0);
        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
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
        
        printf("迭代 %2d: 完整时间=%.3fms (H2D=%.3fms, Kernel=%.3fms, D2H=%.3fms)\n",
               iter + 1, full_time, memcpy_h2d_time, kernel_time, memcpy_d2h_time);
    }
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 计算性能指标
    float avg_full_time = total_full_time / iterations;
    float avg_kernel_time = total_kernel_time / iterations;
    
    // 计算GFLOPS（浮点运算次数/秒）
    // 矩阵乘法浮点运算次数: 2 * M * N * K（每次乘加算2次浮点运算）
    long long flops = 2LL * M * N * K;
    float gflops = (flops / (avg_kernel_time * 1e-3f)) / 1e9f; // GFLOPS
    float memory_bandwidth = (total_memcpy_h2d_time + total_memcpy_d2h_time) / iterations;
    
    printf("\n平均性能统计 (执行 %d 次):\n", iterations);
    printf("  完整过程: %.3f ms\n", avg_full_time);
    printf("  主机到设备传输: %.3f ms\n", total_memcpy_h2d_time / iterations);
    printf("  核函数执行: %.3f ms\n", avg_kernel_time);
    printf("  设备到主机传输: %.3f ms\n", total_memcpy_d2h_time / iterations);
    printf("  核函数占比: %.1f%%\n", (avg_kernel_time / avg_full_time) * 100);
    printf("  计算性能: %.2f GFLOPS\n", gflops);
    printf("  内存传输时间占比: %.1f%%\n", 
           (memory_bandwidth / avg_full_time) * 100);
}

int main(int argc, char *argv[]) {
    // 设置矩阵维度
    int M = 512;  // A矩阵行数，C矩阵行数
    int K = 256;  // A矩阵列数，B矩阵行数
    int N = 1024;  // B矩阵列数，C矩阵列数
    
    int iterations = 10;          // 执行次数

    int warmup_iterations = 3;    // 预热次数
    int block_size = 16;          // 线程块大小
    
    printf("CUDA矩阵乘法算子性能测试\n");
    printf("=======================\n");
    printf("矩阵维度: A[%d×%d] * B[%d×%d] = C[%d×%d]\n", M, K, K, N, M, N);
    printf("矩阵大小: A=%.2f MB, B=%.2f MB, C=%.2f MB\n",
           M * K * sizeof(float) / (1024.0f * 1024.0f),
           K * N * sizeof(float) / (1024.0f * 1024.0f),
           M * N * sizeof(float) / (1024.0f * 1024.0f));
    printf("总执行次数: %d (包含预热 %d 次)\n", iterations, warmup_iterations);
    
    // 分配主机内存
    float *h_A = (float *)malloc(M * K * sizeof(float));
    float *h_B = (float *)malloc(K * N * sizeof(float));
    float *h_C_gpu = (float *)malloc(M * N * sizeof(float));
    float *h_C_cpu = (float *)malloc(M * N * sizeof(float));
    
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        printf("主机内存分配失败!\n");
        return -1;
    }
    
    // 初始化矩阵
    printf("\n初始化矩阵...\n");
    init_matrix(h_A, M, K, 123);
    init_matrix(h_B, K, N, 456);
    
    // 打印部分矩阵内容
    printf("\n矩阵A的前4x4元素:\n");
    print_matrix_part(h_A, M, K, 0, 0, 4, 4);
    
    printf("矩阵B的前4x4元素:\n");
    print_matrix_part(h_B, K, N, 0, 0, 4, 4);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    
    if (!d_A || !d_B || !d_C) {
        printf("设备内存分配失败!\n");
        return -1;
    }
    
    // 预热执行（不测量时间）
    // printf("\n预热执行 (%d 次)...\n", warmup_iterations);
    // for (int i = 0; i < warmup_iterations; i++) {
    //     cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
    //     dim3 blockDim(block_size, block_size);
    //     dim3 gridDim((N + block_size - 1) / block_size,
    //                  (M + block_size - 1) / block_size);
        
    //     matmul_kernel_basic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    //     cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // }
    cudaDeviceSynchronize();
    printf("预热完成\n");
    
    // 测试基础版本
    printf("\n[测试1] 基础版本矩阵乘法（全局内存）\n");
    printf("========================================\n");
    run_matmul_test(h_A, h_B, h_C_gpu, d_A, d_B, d_C,
                    M, N, K, block_size, iterations, false);
    
    // // 测试共享内存版本
    // printf("\n[测试2] 优化版本矩阵乘法（共享内存）\n");
    // printf("========================================\n");
    // run_matmul_test(h_A, h_B, h_C_gpu, d_A, d_B, d_C,
    //                 M, N, K, 32, iterations, true);  // 共享内存版本使用32x32的块
    
    // 在CPU上计算用于验证
    printf("\n在CPU上计算用于验证...\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    printf("CPU计算时间: %.3f ms\n", cpu_time.count());
    
    // 验证结果
    printf("\n验证结果:\n");
    printf("=========\n");
    
    bool is_correct = verify_result(h_C_gpu, h_C_cpu, M * N);
    if (is_correct) {
        printf("✓ 矩阵乘法结果正确！\n");
        
        // 打印结果矩阵的部分内容
        printf("\n结果矩阵C的前4x4元素:\n");
        printf("GPU结果:\n");
        print_matrix_part(h_C_gpu, M, N, 0, 0, 4, 4);
        
        printf("CPU结果:\n");
        print_matrix_part(h_C_cpu, M, N, 0, 0, 4, 4);
    } else {
        printf("✗ 矩阵乘法结果错误！\n");
    }
    
    // 性能总结
    printf("\n性能总结:\n");
    printf("=========\n");
    printf("矩阵维度: %d x %d x %d\n", M, K, N);
    printf("浮点运算总数: %.2f GFLOPS\n", 2.0 * M * N * K / 1e9);
    printf("计算访存比: %.2f FLOP/Byte\n", 
           (2.0 * M * N * K) / ((M * K + K * N + M * N) * sizeof(float)));
    
    // 清理内存
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n测试完成！\n");
    
    return 0;
}