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

    // 标记已经删除的碰撞对
    markRemovedPairsKernel();

    // 标记已更新的碰撞对并更新投影
    markUpdatedPairsKernel();
}
```

## 数据结构
### `mBPDescBuf: PxgTypedCudaBuffer<PxgBroadPhaseDesc>`

宽相碰撞所需要的大部分数据buffer的入口和控制参数，并不存储实际数据。可以在`PxgCudaBroadPhaseSap::updateDescriptor`查看描述符各个成员和`PxgCUDABroadPhaseSap`对象成员的对应关系。

某些算子会接收一个描述符从描述符里面拿到需要的buffer的指针，有些算子会直接从接收buffer的指针，有些会混用这两种参数传递策略。主要因为以下原因：
- CUDA对于传入参数的总大小有限制，较新架构为32K，而老架构仅为4K。
- 描述符填好一次可以复用，避免每个kernel都从host传入大量指针。
- 接口稳定，新增加的参数可以通过描述符访问，避免频繁修改接口和参数爆炸。

对于之后的buffer，我们同时标注它的指针名称以及通过描述符间接访问的形式。

### `updateData_fpBounds -> mBoxFpBoundsBuf: PxgTypedCudaBuffer<PxBounds3>`

所有物体的AABB盒子，每个盒子由6个float组成（min坐标和max坐标）。

### `newIntegerBounds -> mNewIntegerBoundsBuf: PxgTypedCudaBuffer<PxgIntegerAABB>`

所有物体量化后的AABB盒子，每个盒子由6个无符号32位整数组成。

### `desc.boxHandles -> mBoxPtHandlesBuf: PxgCudaBufferN<6>`

这是一个双缓冲结构，0..2存储当前帧的xyz轴handle数组，3..5存储上一帧的xyz轴打包handle数组。每个打包handle数组是按照投影顺序排列好的。

每个打包handle是一个32位数据，打包handle可以理解为在坐标轴上投影端点的元信息，表示如下：
- bit 3..31： 端点对应物体的handle（就是id）。
- bit 2：该端点是否被删除。
- bit 1：该端点对应的投影是否为本帧新创建的。
- bit 0：投影的min=1，max=0。

### `desc.boxSapBox1D -> mBoxSapBox1DBuf: PxgCudaBufferN<3>`

分别存储xyz轴上投影的端点下标数组。数组元素的类型是`PxgSapBox`，在`handle`位置的元素代表对应物体在坐标轴上min/max端点在排序后投影数组里的下标。

我们发现 `mBoxPtHandlesBuf`和`mBoxSapBox1DBuf`可以实现双向查询。

### `desc.updateData_removedHandles -> mRemovedHandlesBuf: PxgTypedCudaBuffer<PxU32> `

要被移除物体的handle组成的数组。


## 算子解析

### `translateAABBsKernel() -> translateAABBsLaunch()`

这个算子将float表示的AABB（`mBoxFpBoundsBuf`）映射到uint32表示的AABB（`mNewIntegerBoundsBuf`），并保持空间内相对位置不变，方便之后步骤的基数排序。一个AABB包围盒由6个float组成，每8个线程负责一个AABB，多余的2个线程闲置。这个算子通过如下步骤转化为6个uint32_t组成的包围盒。

1. 对每个float执行`encodeFloat()`：首先，直接把float的位当作uint32_t。正浮点数的二进制表示对应的整数保留了原本的大小关系。对于负浮点数，我们按位取反，得到后二进制对应的整数也维持了负浮点数的大小关系。因为正浮点数总是大于负浮点数的，我们将正浮点数的二进制表示最高位设为1.

    ```cpp
    // Step 1: if x < y, encodeFloat(x) < encodeFloat(y)
    PxU32 encodeFloat(PxReal f)
    {
        const PxU32 i = __float_as_int(f);
        return i & PX_SIGN_BITMASK ? ~i : i | PX_SIGN_BITMASK;
    }
    ```
2. 对每个float右移`shift`位，把坐标量化到更粗的网络，避免两个相近物体因为浮点数误差在“刚好重叠”和“刚好不重叠”之间来回抖动，从而减少碰撞对反复创建和删除。`shift`通常设为4。
3. 将AABB的max坐标三个float的最低位设为1，min坐标的设为0。这个操作确保一个AABB包围盒内部两个坐标在空间的相对位置关系不变，即min坐标总是小于max的。
    ```cpp
    // Step 2 & 3
    const PxU32 shift = shifts[a];
    const PxU32 b = (encodeFloat(projection) >> shift) & (~1);
    PxU32 out = b | mask;
    ```
4. 此外，该算子还支持设置contact distance来扩张边界和envIDShift设置环境ID，先略过。

### `markRemovedPairsKernel() -> markRemovedPairsLaunch()` 