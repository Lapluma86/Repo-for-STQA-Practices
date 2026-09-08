#!/usr/bin/env python3
"""
GPU监控并自动启动训练脚本

功能：
1. 监控指定GPU的利用率和显存占用
2. 当GPU连续空闲超过阈值时，启动训练
3. 可选：先占位显存，防止被抢占
"""

import subprocess
import time
import argparse
import sys
import os
import signal
from typing import List, Tuple, Optional


class GPUMonitor:
    def __init__(
        self,
        gpu_ids: List[int],
        util_threshold: float = 5.0,
        memory_threshold: int = 1000,
        idle_duration: int = 30,
        check_interval: int = 5,
        reserve_memory: bool = False,
        reserve_mb: int = 1024,
    ):
        """
        Args:
            gpu_ids: 要监控的GPU ID列表
            util_threshold: GPU利用率阈值（%），低于此值视为空闲
            memory_threshold: 显存占用阈值（MB），低于此值视为空闲
            idle_duration: 连续空闲时长（秒）
            check_interval: 检查间隔（秒）
            reserve_memory: 是否在启动训练前先占位显存
            reserve_mb: 占位显存大小（MB）
        """
        self.gpu_ids = gpu_ids
        self.util_threshold = util_threshold
        self.memory_threshold = memory_threshold
        self.idle_duration = idle_duration
        self.check_interval = check_interval
        self.reserve_memory = reserve_memory
        self.reserve_mb = reserve_mb

        self.idle_start_time = None
        self.placeholder_process = None

    def get_gpu_status(self) -> List[Tuple[int, float, int]]:
        """
        获取GPU状态

        Returns:
            List of (gpu_id, utilization%, memory_used_mb)
        """
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            gpu_status = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                if gpu_id in self.gpu_ids:
                    util = float(parts[1].strip())
                    memory = int(parts[2].strip())
                    gpu_status.append((gpu_id, util, memory))

            return gpu_status
        except Exception as e:
            print(f"Error getting GPU status: {e}", file=sys.stderr)
            return []

    def is_idle(self, gpu_status: List[Tuple[int, float, int]]) -> bool:
        """检查所有目标GPU是否空闲"""
        if len(gpu_status) != len(self.gpu_ids):
            return False

        for gpu_id, util, memory in gpu_status:
            if util > self.util_threshold or memory > self.memory_threshold:
                return False

        return True

    def reserve_gpu_memory(self):
        """占位GPU显存，防止被抢占"""
        if not self.reserve_memory:
            return

        print(f"Reserving {self.reserve_mb}MB on GPUs {self.gpu_ids}...")

        # 创建一个简单的CUDA程序来占位显存
        placeholder_code = f"""
import torch
import time
import signal
import sys

def signal_handler(sig, frame):
    print('Releasing GPU memory...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# 在每个GPU上分配显存
tensors = []
for gpu_id in {self.gpu_ids}:
    device = torch.device(f'cuda:{{gpu_id}}')
    # 分配约{self.reserve_mb}MB显存
    size = {self.reserve_mb} * 1024 * 1024 // 4  # float32 = 4 bytes
    tensor = torch.zeros(size, device=device)
    tensors.append(tensor)
    print(f'Reserved memory on GPU {{gpu_id}}')

print('GPU memory reserved. Waiting for training to start...')
print('PID:', os.getpid())

# 保持进程运行
while True:
    time.sleep(1)
"""

        # 启动占位进程
        try:
            self.placeholder_process = subprocess.Popen(
                [sys.executable, "-c", placeholder_code],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpu_ids))},
            )
            time.sleep(3)  # 等待显存分配完成
            print(f"Placeholder process started (PID: {self.placeholder_process.pid})")
        except Exception as e:
            print(f"Failed to reserve GPU memory: {e}", file=sys.stderr)
            self.placeholder_process = None

    def release_placeholder(self):
        """释放占位进程"""
        if self.placeholder_process:
            print("Releasing placeholder process...")
            self.placeholder_process.terminate()
            try:
                self.placeholder_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.placeholder_process.kill()
            self.placeholder_process = None

    def wait_for_idle(self) -> bool:
        """等待GPU空闲"""
        print(f"Monitoring GPUs {self.gpu_ids}...")
        print(f"Idle criteria: util < {self.util_threshold}%, memory < {self.memory_threshold}MB")
        print(f"Required idle duration: {self.idle_duration}s")
        print()

        while True:
            gpu_status = self.get_gpu_status()

            if not gpu_status:
                print("Failed to get GPU status, retrying...", file=sys.stderr)
                time.sleep(self.check_interval)
                continue

            # 打印当前状态
            status_str = " | ".join([
                f"GPU{gpu_id}: {util:.1f}% {memory}MB"
                for gpu_id, util, memory in gpu_status
            ])
            print(f"\r{status_str}", end="", flush=True)

            if self.is_idle(gpu_status):
                if self.idle_start_time is None:
                    self.idle_start_time = time.time()
                    print(f"\nGPUs idle, waiting for {self.idle_duration}s...")
                else:
                    elapsed = time.time() - self.idle_start_time
                    if elapsed >= self.idle_duration:
                        print(f"\nGPUs have been idle for {elapsed:.1f}s, starting training!")
                        return True
                    else:
                        remaining = self.idle_duration - elapsed
                        print(f"\rIdle for {elapsed:.1f}s, {remaining:.1f}s remaining...", end="", flush=True)
            else:
                if self.idle_start_time is not None:
                    print("\nGPUs no longer idle, resetting timer...")
                self.idle_start_time = None

            time.sleep(self.check_interval)

    def run_training(self, train_command: List[str]) -> int:
        """启动训练"""
        print(f"\nStarting training: {' '.join(train_command)}")
        print("=" * 80)

        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        try:
            # 启动训练进程
            process = subprocess.Popen(
                train_command,
                env=env,
            )

            # 等待训练完成
            return_code = process.wait()

            print("=" * 80)
            print(f"Training finished with return code: {return_code}")

            return return_code

        except KeyboardInterrupt:
            print("\nInterrupted by user, terminating training...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            return 1
        except Exception as e:
            print(f"Error running training: {e}", file=sys.stderr)
            return 1
        finally:
            self.release_placeholder()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU and auto-start training when idle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 监控GPU 0，空闲30秒后启动训练
  %(prog)s --gpu 0 --command "python train.py --config tasks/a2/default.yaml"

  # 监控GPU 0和1，空闲60秒后启动，先占位1GB显存
  %(prog)s --gpu 0 1 --idle-duration 60 --reserve-memory --reserve-mb 1024 \\
    --command "python train.py --config tasks/a2/default.yaml"

  # 自定义空闲阈值
  %(prog)s --gpu 0 --util-threshold 10 --memory-threshold 2000 \\
    --command "python train.py --config tasks/a2/default.yaml"
        """
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        required=True,
        help="GPU IDs to monitor (e.g., 0 or 0 1)",
    )
    parser.add_argument(
        "--util-threshold",
        type=float,
        default=5.0,
        help="GPU utilization threshold (%%) for idle detection (default: 5.0)",
    )
    parser.add_argument(
        "--memory-threshold",
        type=int,
        default=1000,
        help="GPU memory threshold (MB) for idle detection (default: 1000)",
    )
    parser.add_argument(
        "--idle-duration",
        type=int,
        default=30,
        help="Required idle duration (seconds) before starting training (default: 30)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=5,
        help="GPU status check interval (seconds) (default: 5)",
    )
    parser.add_argument(
        "--reserve-memory",
        action="store_true",
        help="Reserve GPU memory before training to prevent preemption",
    )
    parser.add_argument(
        "--reserve-mb",
        type=int,
        default=1024,
        help="Amount of memory to reserve (MB) (default: 1024)",
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="Training command to execute (e.g., 'python train.py --config tasks/a2/default.yaml')",
    )

    args = parser.parse_args()

    # 解析训练命令
    train_command = args.command.split()

    # 创建监控器
    monitor = GPUMonitor(
        gpu_ids=args.gpu,
        util_threshold=args.util_threshold,
        memory_threshold=args.memory_threshold,
        idle_duration=args.idle_duration,
        check_interval=args.check_interval,
        reserve_memory=args.reserve_memory,
        reserve_mb=args.reserve_mb,
    )

    try:
        # 等待GPU空闲
        if not monitor.wait_for_idle():
            return 1

        # 可选：占位显存
        monitor.reserve_gpu_memory()

        # 启动训练
        return_code = monitor.run_training(train_command)

        return return_code

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        monitor.release_placeholder()
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        monitor.release_placeholder()
        return 1


if __name__ == "__main__":
    sys.exit(main())
