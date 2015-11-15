#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from mpi4py import MPI
import threading
import Queue


DIMENSIONS = 10000
CHECK_CPU = True
CLUSTER_CPU_COMPUTATION = True
BLOCK_SIZE = 16

def multiply_on_cpu(a, b, queue=None):
    c = np.dot(a, b)
    if queue:
        queue.put(c)
    return c

def multiply_on_device(a, b):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)    
    a_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    with open("kernel.cl", "r") as kernel_source_file:
        kernel_source = kernel_source_file.read()
        kernel_source = "#define BLOCK_SIZE %s" % BLOCK_SIZE + kernel_source
        prg = cl.Program(ctx, kernel_source).build()
    c_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a.nbytes)
    local_shape = [BLOCK_SIZE, BLOCK_SIZE]    
    hA, wA = a.shape
    hB, wB = b.shape
    if (wA  % BLOCK_SIZE):
        wGlobal = (wA/BLOCK_SIZE + 1) * BLOCK_SIZE
    else:
        wGlobal = wA
    if (hA  % BLOCK_SIZE):
        hGlobal = ((hA/BLOCK_SIZE) + 1) * BLOCK_SIZE
    else:
        hGlobal = hA
    global_shape = (hGlobal, wGlobal) 
    print(a.shape, global_shape)
    event = prg.cdot(
        queue, 
        global_shape, 
        local_shape,
        a_g, 
        np.int32(hA),
        np.int32(wA),
        b_g,
        np.int32(hB),
        np.int32(wB),
        c_g
    )
    # event.wait()
    # queue.finish()
    c = np.empty_like(a)
    # res_np_part = np.dot(a_np_part, b_np)
    cl.enqueue_copy(queue, c, c_g)
    return c


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    q = Queue.Queue()
    if rank == 0:
        # a_np = np.random.rand(DIMENSIONS, DIMENSIONS).astype(np.float32)
        # b_np = np.random.rand(DIMENSIONS, DIMENSIONS).astype(np.float32)
        # a_np = np.arange(DIMENSIONS ** 2).reshape((DIMENSIONS, DIMENSIONS)).astype(np.float32)
        a_np = np.random.rand(DIMENSIONS, DIMENSIONS).astype(np.float32)
        b_np = np.ones((DIMENSIONS, DIMENSIONS)).astype(np.float32)
        a_np_parts = np.split(a_np, size)
        c_np_parts = []
    else:
        a_np_parts = []
        b_np = None
    a_np_part = comm.scatter(a_np_parts, root=0)
    b_np = comm.bcast(b_np, root=0)
    if CHECK_CPU and CLUSTER_CPU_COMPUTATION:
        t1 = threading.Thread(target=multiply_on_cpu, args=(a_np_part, b_np, q))
        t1.start()
    c_np_part = multiply_on_device(a_np_part, b_np)
    print("GPU calculated. Rank %s" % rank)
    c_np_parts = comm.gather(c_np_part, root=0)
    if CHECK_CPU and CLUSTER_CPU_COMPUTATION:
        t1.join()
        c_np_cpu_parts = comm.gather(q.get(), root=0)
    if rank == 0:
            c_np = np.concatenate(c_np_parts)
            if CHECK_CPU and CLUSTER_CPU_COMPUTATION:
                c_np_cpu = np.concatenate(c_np_cpu_parts)
            if CHECK_CPU and not CLUSTER_CPU_COMPUTATION:
                print("Calculate with CPU")
                c_np_cpu = np.dot(a_np, b_np)
            print("GPU calculated")
            # Check on CPU with Numpy:
            # print(a_np)
            # print(b_np)
            print(c_np)
            if CHECK_CPU:
                print("Checking result with CPU")
                print("+"*10)
                print(c_np_cpu)
                print(np.array_equal(c_np, c_np_cpu))


if __name__ == "__main__":
    main()
