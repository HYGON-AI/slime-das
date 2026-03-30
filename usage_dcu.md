#slime在dcu上训练示例
本文档主要介绍在dcu上环境搭建至模型训练的整体流程。
##基础环境搭建
由于slime对环境版本要求较严格，因此建议使用我们已经配置验证好的docker镜像，避免潜在环境影响；同时也可以根据我们的验证过的依赖清单，自行进行conda环境配置。

```shell
# 拉取镜像
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk25.04.2-1226-das1.7-py3.10-20251226
# 启动容器（按需修改正确挂载目录）
docker run -it --shm-size=64G --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc=host --network=host --workdir=/workspace --privileged --name dcu_verl_test -v /opt/hyhal:/opt/hyhal:ro -v /home:/home harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk25.04.2-1226-das1.7-py3.10-20251226 /bin/bash
# 如果想自行配置conda环境
cd dcu_slime
pip install -r requirements
```

##数据准备
目前 slime 只支持加载 .jsonl 格式文件，即文件的每一行都是一个 json，一行数据的样例（展开后）为：
```json
{
  "prompt": [
    {
      "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
      "role": "user",
      "step_loss_mask": 1,
    }
  ],
  "label": "34"
}
```
此处以从huggingface_hub下载的资源示例：
```shell
# 下载训练数据集 (dapo-math-17k),注意修改下载目录，
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```
##模型转换
由于megatron无法直接加载huggingface ckpt，当用户采用huggingface模型资源时需要先将模型权重转换为megatron torch_dist格式，此处以qwen3-4B为例。
```shell script
# 下载模型权重
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
# 权重转换
cd dcu_slime/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/dapo-math-17k \
    --save /root/qwen3_4B_torch_dist \
    --use-cpu-initialization
```
其中PYTHONPATH为pip list中megatron-core的路径, --use-cpu-initializationz主要是在利用AMD架构gpu时采用。
##模型训练
此处以qwen3-4B为例。
```shell script
cd dcu_slime/examples
# 填写run_conda.sh脚本文件，在conda环境中启动
PORT="63792" # docker环境需要设定，conda环境则忽略
HOST_FILE="/public/home/linxi1/dcu_slime/hostfile"
DATA_PATH="/public/home/linxi1/dcu_slime/data"
MODEL_PATH="/public/home/linxi1/dcu_slime/data"
PROFILING="" # If you want to profiling, please choose one of {torch}
DTK_ENV="/public/home/linxi1/dtk-25.04.2-rc1-centos8/env.sh"
```
配置hostfile文件，按如下方式配置，可以填写节点ip或者节点主机名，多节点则填写多个。
```shell script
node005 slots=1
```
nccl文件配置，进入env.yaml文件，按照服务器选择配置文件。
```shell script
cd dcu_slime/examples
# wz
NCCL_IB_HCA: "mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
NCCL_TOPO_FILE: "/public/home/linxi1/slime/scripts/topo-input.xml"

# 508 mlxn
#  NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
#  NCCL_PXN_DISABLE: "1"
#  NCCL_NCHANNELS_PER_PEER: "32"
#  NCCL_MIN_P2P_NCHANNELS: "32"
#  NCCL_MAX_P2P_NCHANNELS: "32"
#  RCCL_P2P_XHCL_CHANNEL_NUM: "30"
#  NCCL_TOPO_FILE: "eg: /public/home/linxi1/dcu_slime/examples/built-in-508-topo-input-tj-default.xml"
```
执行运行脚本。
```shell script
cd dcu_slime/examples
# 启动conda环境执行脚本
bash run_conda.sh 
# 启动docker环境执行脚本
bash run_docker.sh 
```
#重要说明
- 本文以GPRO算法训练qwen3-4B为例，进行环境配置以及训练，更多算法以及训练参数搭配需参考[quick_start.md](https://github.com/THUDM/slime/docs/zh/get_started/quick_start.md)和[usage.md](https://github.com/THUDM/slime/docs/zh/get_started/usage.md)。