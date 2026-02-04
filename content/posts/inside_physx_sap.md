+++
date = '2026-02-03T17:52:00+08:00'
draft = false
title = "Physx的GPU粗相碰撞（Broadphase）"
+++

[NVIDIA Physx](https://github.com/NVIDIA-Omniverse/PhysX)使用SAP算法（Sweep-and-Prune）进行粗相碰撞检测。我们主要关注如下源码：
- [PxgCudaBroadPhaseSap.cpp]([physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp))：`PxgCUDABroadPhaseSap::update`每帧根据AABB的更新执行一次更新，输出新产生的和消失的碰撞对。
- [broadphase.cu]([physx/source/gpubroadphase/src/CUDA/broadphase.cu](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpubroadphase/src/CUDA/broadphase.cu))：SAP算法以及相关算子的实现。

## 更新逻辑

```cpp
PxgCudaBroadPhaseSap::update(...)
{   
    // 将用浮点数的AABB转化为用整数表示的AABB
    translateAABBsKernel();

    
}
```

## 算子解析

