// 测试分阶段GPU任务的编译
#include "guessing_cuda.h"
#include <iostream>

// 简单的编译测试
int main() {
    std::cout << "Testing staged GPU task compilation..." << std::endl;
    
    // 测试创建StagedGpuTask
    TaskManager tm;
    tm.taskcount = 1;
    tm.guesscount = 100;
    tm.seg_types.push_back(1);
    tm.seg_ids.push_back(0);
    tm.seg_lens.push_back(5);
    tm.prefixs.push_back("test");
    tm.prefix_lens.push_back(4);
    tm.seg_value_count.push_back(10);
    
    StagedGpuTask* task = new StagedGpuTask(std::move(tm));
    
    std::cout << "StagedGpuTask created successfully!" << std::endl;
    std::cout << "Current stage: " << (int)task->current_stage << std::endl;
    
    // 清理
    delete task;
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
