ART 扩散模型微调
================

本页介绍 DiffusionART 在 SD3.5 Medium 图像文字生成和 Wan2.2 TI2V 5B 视频文字生成中的微调，内容涵盖算法与公式、OCR 数据、训练配置、ART/NFT 奖励与耗时实测结果，以及训练指标定义，并给出 Pi0 在 LIBERO 多步动作轨迹上的实验性 ART 配方。

概览
----

单张图像的文字生成使用图像配方，跨帧文字生成使用视频配方。OCR 指光学字符识别：读取生成画面中的文字，与 prompt 要求的文字比较。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      使用标量奖励的 DiffusionART

   .. grid-item-card:: 模型
      :text-align: center

      SD3.5 Medium / Wan2.2 TI2V 5B / Pi0

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      图像 OCR / 视频 OCR / LIBERO

   .. grid-item-card:: 训练
      :text-align: center

      LoRA + FSDP2；Wan 对比使用 4 张 NVIDIA A800

任务
~~~~

``examples/diffusion/config/`` 中的配置将各任务与对应的 Diffusers checkpoint 和 OCR 奖励 backend 关联。

.. list-table::
   :header-rows: 1
   :widths: 20 39 41

   * - 任务
     - 配置
     - 输入与奖励
   * - 文生图
     - ``sd3_art_ocr``
     - OCR prompt；``visual.ocr`` 计算图像文字相似度。
   * - 文生视频
     - ``wan22_ti2v_5b_art_video_ocr``
     - OCR prompt；``visual.video_ocr`` 计算视频级平均分。
   * - 机器人操作
     - ``libero_object_0_art_openpi`` （``examples/embodiment/config/``）
     - LIBERO Object task 0；环境 episode 回报。

观测与动作
~~~~~~~~~~

图像、视频任务的每个 episode 生成一张图像或一段视频。ART 为每个候选保留一个标量奖励，再比较同一 prompt 下的候选。:ref:`LIBERO 动作配方 <diffusion-art-libero>` 则对完整机器人轨迹分组，按决策位置对齐动作块。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 含义
   * - Observation
     - 从数据集采样的 ``task_descriptions``。
   * - Action
     - 生成的图像或视频 tensor。
   * - Reward
     - ``[0, 1]`` 范围内的 OCR 文字相似度；视频训练取各帧平均分。
   * - Prompt
     - 包含引号内目标文字的描述，例如要求招牌写着 "OPEN"。

ART 训练流程
------------

对同一 prompt 生成多个候选并评分后，ART 将最终的干净 latent 与奖励汇总成一个共享方向：高于组内平均奖励的候选把方向拉向自己的终点，低于平均奖励的候选则把方向推离自己的终点。模型在生成过程中实际经过的去噪状态上，学习一个冻结的速度目标。

当前配方把每个 prompt 的候选分配到两个互不重叠的更新窗口，两次更新都沿用第一次更新前确定的目标。第二次更新既包含奖励引导，也包含对第一次更新所引入偏移的修正。若某个状态的移动已超出阈值，并且继续更新还会向外移动，mask 会屏蔽该项。该更新结构的公式与配置约束见 :ref:`ART 算法目标与实现 <diffusion-art-algorithm>`。

.. _diffusion-art-algorithm:

ART 算法目标与实现
------------------

ART 的目标函数结合分组奖励引导、冻结速度目标和修正 mask。以下公式将论文的算法结构对应到 SD3、Wan 的速度符号和 RLinf 配置约束。算法出自 `DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models <https://vanzll.github.io/diffusionart-project/>`_。

聚合奖励引导
~~~~~~~~~~~~

图像或视频生成完毕后才能得到奖励，模型却需要在多个中间状态上学习去噪。ART 将最终输出的评分转化成同一 prompt 下各个状态共用的方向。例如，一组招牌中有的正确写出 "OPEN"，有的缺少字母，ART 根据它们的 latent 终点和评分，估计哪一个方向与更准确的文字生成相关。奖励函数可以是黑盒。

DiffusionNFT 同样利用最终输出的奖励。它在生成样本上重新加噪，通过前向过程目标训练模型。ART 论文指出，这类目标中存在前向与反向过程的一致性作用力：单个样本的前向速度与模型反向生成时预测的速度可能不同。将不同状态及模型梯度汇总后，正负分支只会部分抵消这股作用力，剩余部分可能干扰奖励引导。这个分析描述目标函数的性质，并不意味着所有 NFT 配置都会崩溃。

论文还分析了指数移动平均，即 EMA。在 RLinf 的 NFT 实现中，rollout/reference 参数按下式跟随 actor：

.. math::

   \bar\theta_{k+1}=(1-\tau_k)\bar\theta_k+\tau_k\theta_{k+1}.

``nft_tau`` 表示当前参数的混合比例。值越小，保留的旧 reference 越多；取 ``1`` 时使用当前 actor。论文将 EMA 的稳定作用归因于朝滞后 reference 的恢复力，但其强度需要按任务调整。ART 从完整 prompt 组直接构造奖励引导，并通过第二次更新相对于冻结目标产生恢复力。

从终点评分得到方向
~~~~~~~~~~~~~~~~~~

对于给定 prompt :math:`c`，高分样本在当前 rollout policy 的生成分布中获得更大权重，形成奖励倾斜后的分布：

.. math::

   p_\lambda(x_0\mid c)=
   \frac{p_{\mathrm{old}}(x_0\mid c)\exp(\lambda R(x_0,c))}{Z_c(\lambda)},
   \qquad
   \left.\frac{\partial}{\partial\lambda}
   \mathbb E_{p_\lambda}[x_0\mid c]\right|_{\lambda=0}
   =\operatorname{Cov}_{p_{\mathrm{old}}}(R,x_0\mid c).

分母 :math:`Z_c` 保证概率归一化。右侧导数说明：只要稍微偏向高奖励样本，样本均值就会沿奖励与终点的协方差方向移动。这里用 :math:`\lambda` 表示论文的倾斜参数 :math:`\tau`，它与 NFT 的 EMA 系数含义不同。

对同一 prompt 的 :math:`K` 个候选，论文公式（5）估计协方差向量，再用均方根 RMS 归一化：

.. math::

   C_c=\frac1K\sum_{i=1}^K
   (R_i-\bar R_c)(x_{0,i}-\bar x_{0,c}),
   \qquad
   G_c=\frac{C_c}{\operatorname{RMS}(C_c)},
   \qquad
   \operatorname{RMS}(z)=\sqrt{\frac1D\sum_{j=1}^D z_j^2}.

每个 :math:`x_{0,i}` 都是具有 :math:`D` 个坐标的干净 latent 终点，不是解码后的 RGB 图像。高于组内平均分的候选贡献正项，低于平均分的候选贡献负项。归一化用于统一不同 prompt 和奖励范围下的方向尺度。若组内奖励完全相同，就没有相对优劣信号，RLinf 会将该组引导清零。

``_art_scalar_covariance`` 在拆分 micro-batch 前，对完整 prompt 组计算统计量。当前配方的 ``art_covariance_reduction=unbiased`` 使用 :math:`K-1` 作分母，``population`` 使用论文中的 :math:`K`。理想 RMS 归一化会抵消正标量倍数。historical 路径先除以奖励的平均绝对偏差，再用带 epsilon 下限的 RMS 归一化，worker 随后再次归一化。信号高于这些数值下限时，默认单位幅度引导满足 :math:`H_c=G_c`；非常小的非零信号可能被 epsilon 下限衰减。``art_zero_signal_mode=exact`` 会显式清除常量奖励产生的浮点残差。

冻结速度目标
~~~~~~~~~~~~

ART 将同一个组方向应用到 ODE 生成时实际经过的状态。论文采用朝向干净样本的速度约定，目标写作 :math:`v^{\mathrm{ART}}_{\mathrm{tgt}}=v^{(0)}+G_c`。RLinf 使用的 Diffusers 模型速度符号相反，噪声水平递减时的 Euler 更新为：

.. math::

   x_{\sigma'}=x_\sigma+(\sigma'-\sigma)v_\theta(x_\sigma,c),
   \qquad \sigma'<\sigma.

在这个约定下，实现中的目标与 loss 为：

.. math::

   v_{\mathrm{tgt}}=\operatorname{stopgrad}(v_{\mathrm{old}}-H_c),
   \qquad
   \mathcal L_\theta(s)=w(s)\frac1D
   \|v_\theta(s)-v_{\mathrm{tgt}}(s)\|_2^2.

由于时间差为负，把速度减去 :math:`H_c` 会让下一个 latent 朝 :math:`H_c` 移动。这解释了 ``_art_loss`` 相对于论文出现的负号。停止梯度操作保持旧预测与奖励方向固定，梯度只更新当前模型。

``_precompute_old_velocities`` 在两次 optimizer update 之前，对全部选中状态重算并缓存旧速度；两个更新窗口共用该缓存。rollout tensor 保存中间状态、当前及下一步噪声水平，以及最终终点。SD3 和 Wan 通过各自的速度 forward 接口接入同一个 ART actor。

双更新恢复
~~~~~~~~~~

模型参数在第一次更新后发生变化，即使某个状态留给第二个窗口，它的预测也可能随之改变。继续沿用原目标，就能让第二次更新感知这部分移动。在 RLinf 的速度约定下定义 :math:`d(s)=v_\theta(s)-v_{\mathrm{old}}(s)`，求导时固定 mask，得到：

.. math::

   -\nabla_{v_\theta}\mathcal L
   =\frac{2w}{D}(-H_c-d(s)).

第一次更新前 :math:`d=0`，输出空间中的下降方向正比于 :math:`-H_c`。第二次更新多出 :math:`-d`，偏移越大，朝旧速度场的恢复作用越强。这就是论文公式（6）换成代码速度符号后的表达。这一过程类似朝固定目的地迈步：第一步走得越多，第二步需要计入的已移动距离就越大。

``art_updates_per_rollout=2`` 创建两个等大的互斥窗口，每个 prompt 各有一半候选进入一个窗口。每个候选只进入其中一个窗口，两次更新都使用完整组估计的协方差。梯度累积同时覆盖所选时刻和 micro-batch。``art_updates_per_rollout=1`` 对应单次更新的消融设置，同时也改变训练动态。

偏移修正 mask
~~~~~~~~~~~~~

第二次更新仍可能让预测偏移过大。ART 用参考高斯转移衡量移动，其均值分别由旧速度和当前速度决定。论文公式（7）给出按坐标平均的 KL 代理量：

.. math::

   m(s)=\frac1D D_{\mathrm{KL}}(q_\theta(\cdot\mid s)\|q_{\mathrm{old}}(\cdot\mid s))
   =\frac{\|\mu_\theta(s)-\mu_{\mathrm{old}}(s)\|_2^2}
   {2D\sigma_{\mathrm{ref}}^2(s)}.

参考高斯转移为偏移提供度量尺度，生成过程仍沿 ODE 轨迹执行。步长记为 :math:`h=\sigma-\sigma'`，参考噪声系数记为 :math:`\eta`，代码采用的代理量为：

.. math::

   m(s)=\frac{a^2\operatorname{MSE}(v_\theta,v_{\mathrm{old}})}{2b^2},
   \qquad
   (a,b^2)=
   \begin{cases}
      (h,\eta^2),&\sigma\simeq1,\\
      (h(1+\eta^2/2),\eta^2\sigma h/(1-\sigma)),&\text{otherwise}.
   \end{cases}

实现对噪声 schedule 边界加入数值下限。采用 ``art_mask_metric=sde_kl`` 时，将论文公式（8）转换成 RLinf 的速度符号，得到：

.. math::

   w(s)=\sigma\left[1-
      \mathbf1\{m(s)\ge\delta\}
      \mathbf1\{\langle d(s),-H_c-d(s)\rangle>0\}\right].

只有偏移达到阈值，而且继续沿下降方向移动还会增加偏移时，该状态才会被屏蔽。即使已超出阈值，朝旧速度场修正的更新仍可保留。因子 :math:`\sigma` 加大高噪声状态的权重，无引导信号的状态 loss 为零。mask 判断不参与 autograd。

更新配置
~~~~~~~~

``sd3_art_ocr.yaml`` 和 ``wan22_ti2v_5b_art_video_ocr.yaml`` 均设置 ``algorithm.loss_type=embodied_art``、``adv_type=raw`` 和 ``actor.model.<model_type>.rl_mode=art``。ART 直接从标量奖励计算方向，训练信号不依赖 GRPO advantage tensor。

.. list-table:: ``algorithm`` 下的 ART 选项
   :header-rows: 1
   :widths: 35 22 43

   * - 选项
     - 配方取值
     - 作用
   * - ``art_updates_per_rollout``
     - ``2``
     - 互斥 optimizer 窗口数；支持用 ``1`` 做消融。
   * - ``art_covariance_reduction``
     - ``unbiased``
     - 协方差分母；``population`` 使用组大小。
   * - ``art_direction_mode``
     - ``historical``
     - 先除 MAD，再使用带 epsilon 下限的 RMS；另可选 ``normalized_aggregate``。
   * - ``art_scale_mode``
     - ``unit_normalized_covariance``
     - 有效组的引导幅度；另可选 ``reward_mad``、``raw_covariance``、``reward_std_normalized_covariance``。
   * - ``art_guidance_gain``
     - ``1.0``
     - 引导幅度的乘数。
   * - ``art_zero_signal_mode`` / ``art_zero_signal_atol``
     - ``exact`` / ``0.0``
     - 显式清零组内奖励完全相同的信号。
   * - ``art_loss_space``
     - ``velocity_linear_sigma``
     - 按噪声水平加权；``velocity_uniform`` 使用单位权重。
   * - ``art_mask_metric``
     - ``sde_kl``
     - 偏移代理量；``fixed_mse`` 直接使用速度 MSE。
   * - ``art_mask_threshold``
     - ``1e-5``
     - 正的偏移阈值 :math:`\delta`。
   * - ``art_reference_noise_level``
     - ``0.8``
     - 偏移代理量中的参考 SDE 噪声系数 :math:`\eta`。
   * - ``art_corrective_mask_enabled``
     - ``True``
     - 启用向外偏移修正 mask。
   * - ``art_exclude_final_step``
     - ``False``
     - 可从时刻池中排除最后一个转移。
   * - ``art_timestep_fraction``
     - SD3 ``0.5``；Wan ``0.25``
     - 每个候选不放回选择 ``max(1, int(pool_size * fraction))`` 个状态。
   * - ``art_reward_reduction``
     - ``mean``
     - 将每个候选的奖励项归约成标量；另可选 ``sum``。
   * - ``art_eps``
     - ``1e-5``
     - 协方差方向归一化的数值下限。

模型配置内的 ``timestep_fraction`` 是另一个字段，``algorithm.art_timestep_fraction`` 控制 ART 训练状态数量。当前每条轨迹记录 10 个转移，SD3 配方选择 5 个状态，Wan 配方选择 2 个。NFT 的 ``resample`` 路径每次准备训练数据时，为每个候选随机选一个状态。这些数量影响 actor 耗时与每条轨迹消耗的训练计算量。

Batch 与 Placement 约束
~~~~~~~~~~~~~~~~~~~~~~~~~~

每个 actor shard 保留完整候选组。:math:`E` 表示 ``env.train.total_num_envs``，:math:`L` 表示 ``env.train.rollout_epoch``，:math:`U` 表示 ``art_updates_per_rollout``，:math:`G` 表示 ``algorithm.group_size``，配置校验要求：

.. math::

   N=EL,\qquad B_{\mathrm{global}}=N/U,\qquad
   E\bmod\bigl(\operatorname{lcm}(W_{\mathrm{env}},W_{\mathrm{actor}})G\bigr)=0.

例如，96 个环境、12 个 rollout epoch 共生成 1152 个候选，分为两个窗口时，``actor.global_batch_size=576``。每个 rank 的窗口大小还必须能被 ``actor.micro_batch_size`` 整除。

实现要求 FSDP、``rollout.pipeline_stage_num=1``、``algorithm.update_epoch=1`` 和 ``runner.weight_sync_interval=1``。更新次数 :math:`U\in\{1,2\}` 必须能整除 :math:`G`，``env.train.group_size`` 必须等于 ``algorithm.group_size``。每个 rollout epoch 必须生成一个完整候选，即 ``env.train.max_steps_per_rollout_epoch=actor.model.num_action_chunks``。SD3 还要求 ``train_sampler=ode`` 和 ``noise_level=0.0``；Wan ART 路径记录确定性的 ODE 轨迹。

实现与论文的范围
~~~~~~~~~~~~~~~~

当前配方用于标量奖励下的 SD3.5 图像 OCR 和 Wan2.2 文生视频 OCR。论文实验覆盖更多模型与奖励：

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 场景
     - 论文实验
     - 当前 ART 配方
   * - 文生图
     - SD3.5 Medium、FLUX.1-dev；GenEval、OCR、PickScore 及多奖励联合优化。
     - SD3.5 Medium，标量 OCR 奖励与 LoRA。
   * - 文生视频
     - HunyuanVideo-13B；文本对齐、画面质量、运动质量。
     - Wan2.2 TI2V 5B 的 ``t2v`` 模式，标量视频 OCR 奖励与 LoRA。
   * - 图生视频
     - 蒸馏 Wan2.1-I2V-14B，画面质量与运动质量奖励。
     - 需为该设置补充 ART 任务、数据配方与验证。
   * - Pairwise 反馈
     - 用胜负或平局标签代替标量评分。
     - 需接入 pairwise 奖励生产端与对应配方。

worker 遵循标量聚合、冻结目标、双窗口更新和修正 mask 的算法结构；RLinf 承担分布式 placement、轨迹收集、梯度累积、日志与 checkpoint 管理。实现入口为 ``rlinf/workers/actor/fsdp_art_policy_worker.py`` 中的 ``EmbodiedARTFSDPPolicy``，模型包装位于 ``rlinf/models/diffusion/``，配置约束位于 ``rlinf/config.py``。训练诊断见 :ref:`扩散模型训练指标 <diffusion-training-metrics>`。

安装
----

扩散模型环境运行于 Linux NVIDIA GPU 主机上的 RLinf 仓库，以下安装命令默认采用 CPU OCR 评分：

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

该命令创建 Python 3.10 环境，安装 Diffusers、PEFT 和 PaddleOCR。两个 ART 配置均设置 ``env.*.reward.use_gpu=False``。GPU OCR 评分对应带 ``--paddle-gpu`` 的安装，以及 train、eval 奖励字段取 ``True`` 的配置。共用依赖和下载要求见 :doc:`扩散模型 <video_models>`。

模型权重
--------

各任务使用对应的 Diffusers checkpoint。SD3.5 权重下载以在 Hugging Face 接受模型访问条款为前提。

.. code-block:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

这些命令将 Diffusers 格式的 checkpoint 保存到本地目录。``actor.model.model_path`` 对应启动时使用的本地路径。

OCR 数据集
----------

配方使用 `DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`_ 的 OCR prompt。每行一个 prompt，``train.txt`` 和 ``test.txt`` 分别对应训练集与测试集。

.. code-block:: bash

   mkdir -p /path/to/dataset/ocr
   curl -fL https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -fL https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

这些命令建立 ``visual.ocr_dataset`` 所需的目录结构。``env.train.dataset.path`` 和 ``env.eval.dataset.path`` 对应该目录路径。

训练命令
--------

以下命令在仓库根目录运行，沿用配置内的 batch 设置；模型和数据集参数对应本地资源目录。

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_art_ocr \
      actor.model.model_path=/path/to/stable-diffusion-3.5-medium \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr \
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

脚本启动同步 runner，将日志写入 ``logs/<timestamp>-<config>/``。actor 使用 FSDP2 训练 LoRA 参数，rollout worker 生成候选，环境计算奖励。其他选项在两个 ART YAML 文件中配置。``runner.max_steps=5`` 对应启动流程的短运行检查；短运行容易受编译和初始化影响，不适合估算稳定阶段的吞吐量。

.. warning::

   调整 batch 或 placement 时，每个 actor shard 必须持有完整的 prompt 组，且 ``actor.global_batch_size`` 必须等于每轮候选总数除以 ``algorithm.art_updates_per_rollout``。精确约束见 :ref:`ART 算法目标与实现 <diffusion-art-algorithm>`。

对比配置
~~~~~~~~

对比实验使用 4 张 NVIDIA A800 运行 Wan2.2。以下三条命令对应在同一组 GPU 上分别执行的三次运行。

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   COMMON_ARGS=(
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers
      env.train.dataset.path=/path/to/dataset/ocr
      env.eval.dataset.path=/path/to/dataset/ocr
      env.train.total_num_envs=96
      env.eval.total_num_envs=32
      actor.micro_batch_size=2
      actor.model.wan22_ti2v_5b.max_generation_batch_size=2
      actor.model.wan22_ti2v_5b.offload_auxiliary_modules=true
      actor.model.wan22_ti2v_5b.compile_transformer_forward=true
      env.train.reward.use_gpu=false
      env.eval.reward.use_gpu=false
      rollout.pipeline_stage_num=1
      runner.max_steps=200
      runner.max_epochs=200
      runner.save_interval=10
   )

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr \
      "${COMMON_ARGS[@]}" env.train.rollout_epoch=12 \
      actor.global_batch_size=576 actor.optim.lr=1e-4 \
      runner.logger.experiment_name=wan22-art-n1152-lr1e-4

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr \
      "${COMMON_ARGS[@]}" env.train.rollout_epoch=12 \
      actor.global_batch_size=576 actor.optim.lr=1e-4 \
      runner.logger.experiment_name=wan22-nft-n1152-lr1e-4

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr \
      "${COMMON_ARGS[@]}" env.train.rollout_epoch=6 \
      actor.global_batch_size=288 actor.optim.lr=4e-4 \
      runner.logger.experiment_name=wan22-nft-n576-lr4e-4

前两组每轮生成 ``96 * 12 = 1152`` 个候选，第三组生成 ``96 * 6 = 576`` 个；按照这里的 batch 设置，三组每轮均执行两次 optimizer update。两组 NFT 使用同一实现，区别在学习率和采样量。这些命令复现配置选择，具体曲线还取决于代码版本、依赖、prompt 和随机种子。

.. _diffusion-art-libero:

LIBERO 动作微调
---------------

``libero_object_0_art_openpi`` 使用与 ``examples/embodiment/config/libero_object_0_nft_openpi.yaml`` 相同的 SFT checkpoint，在 LIBERO Object task 0 上微调 Pi0。配方沿用环境交互与 rollout 流程，使用完整 episode 回报、rollout 缓存速度，以及每轮两次 ART 更新。

.. warning::

   这是一种实验性的轨迹级适配。同组轨迹共享任务与模拟器初始状态，执行不同动作之后，观测会逐渐分化。因此，后续决策的聚合条件与论文的同条件图像、视频设置存在差异。成功率、指导强度和漂移阈值需要通过 LIBERO 实验验证。``art_first_chunk_only=True`` 仅训练初始决策，可作为同初始观测对照。

轨迹回报与动作指导
~~~~~~~~~~~~~~~~~~

一个机器人 episode 包含多次决策，每次决策分别通过去噪生成一个动作块。配方先累加终止之前及终止动作上的奖励，再把完整 episode 回报分配给有效决策，让终局成功反馈参与早期动作训练。

对于组 :math:`g`、第 :math:`h` 次决策和实际执行的动作坐标 :math:`j`，有效轨迹集合记为 :math:`\mathcal V_{g,h,j}`。协方差仅在该集合内计算：

.. math::

   C_{g,h,j}=\frac{1}{n_{g,h,j}-1}
   \sum_{i\in\mathcal V_{g,h,j}}
   (R_i-\bar R_{g,h,j})(a_{i,h,j}-\bar a_{g,h,j}).

其中 :math:`R_i` 是完整 episode 回报，:math:`a_{i,h,j}` 是模型内部的归一化动作坐标，:math:`n_{g,h,j}=|\mathcal V_{g,h,j}|`。有效轨迹少于两条的坐标，其指导信号为零。协方差在有效动作坐标上归一化，再分配给各轨迹；回报完全相同的组产生零指导信号。

Transformer 仍接收完整动作生成长度和模型维度；协方差、目标损失和漂移代理只使用实际执行的动作与真实环境维度。同一个动作块中，终止之后的动作也会被排除。损失归一化统计所有 actor rank 上的有效动作块。

两次更新始终使用 rollout 时缓存的速度。每个初始状态组分别向两个窗口分配一半轨迹，同一条轨迹的全部动作块归入同一窗口。组指导在划分窗口前计算。Pi0 与 SD3/Wan 实现使用相同的速度符号，目标为 :math:`v_{\mathrm{old}}-H_{g,h}`，现有 corrective mask 在有效动作坐标上计算。

配置与启动
~~~~~~~~~~

OpenPi 环境和 checkpoint 设置沿用 :doc:`Pi0 <embodied/pi0>`。LIBERO 安装选项准备该配方使用的模拟器素材：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero \
      --venv .venv-openpi-libero
   source .venv-openpi-libero/bin/activate

这会创建独立的 OpenPi 环境。Pi0 SFT checkpoint 已位于本地时，以下命令从仓库根目录启动实验：

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1

   python examples/embodiment/train_embodied_agent.py \
      --config-path "$EMBODIED_PATH/config" \
      --config-name libero_object_0_art_openpi \
      actor.model.model_path=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
      rollout.model.model_path=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

默认每轮采集 64 条 episode，每条最多 240 个环境步，每次决策执行 5 个动作。终止掩码生效前，每轮共有 :math:`64\times(240/5)=3072` 个动作块，分成两次不重叠更新，因此 ``actor.global_batch_size=1536``。通用关系为 ``total_num_envs * rollout_epoch * decisions / art_updates_per_rollout``。设置 ``algorithm.art_first_chunk_only=True`` 时，决策数为一，该配方对应的 batch 覆盖值为 ``actor.global_batch_size=32``。

模型使用 ``openpi.rl_mode=art``、``is_nft=False``，以及零转移噪声的 ``noise_method=flow_ode``。训练要求 ``auto_reset=False``、``ignore_terminations=False`` 和固定的分组初始状态。四步采样器结合 ``art_timestep_fraction=0.25``，为每个动作块选择一个实际 rollout 去噪状态。完整配置位于 ``examples/embodiment/config/libero_object_0_art_openpi.yaml``。

固定位置分叉采样
~~~~~~~~~~~~~~~~

共享动作前缀将同条件比较扩展到任务中途。``algorithm.art_branch_chunk_index`` 为所有组指定一个从零开始的决策索引。到达该位置之前，同组环境执行首个成员采样的动作块；从分叉处开始，各成员使用冻结的 rollout policy 独立采样并继续完成任务。Actor 只训练分叉动作块，同时读取该位置的观测、去噪状态和缓存旧速度。回报累加范围为分叉处至终止：

.. math::

   R_i^{(b)}=\sum_{t=t_b}^{T_i}r_{i,t}.

分叉模式在现有启动命令上附加以下覆盖值：

.. code-block:: bash

   algorithm.art_first_chunk_only=False \
   algorithm.art_branch_chunk_index=8 \
   actor.global_batch_size=32

每个动作块执行 5 个动作时，索引 8 对应先执行 40 个环境步再分叉；索引 0 用于初始决策对照。每条 episode 提供一个训练动作块，因此 64 条 episode、两次互斥更新对应全局 batch 32。``art_first_chunk_only`` 与非空分叉索引互斥。分叉模式要求同步批次，且每个 environment、rollout 和 actor rank 均包含完整组。评估保持标准的完整任务 rollout。

Rollout 在前缀和分叉处检查同组任务、机器人观测、可获取的模拟器状态及两路相机图像。状态最大差异默认容差为 ``art_branch_state_atol=1e-5``，各相机平均绝对像素差默认容差为 ``art_branch_image_atol=1.0``。任一前缀决策不满足一致性要求，或任一组员在分叉前终止，该组整体退出训练，批次形状保持不变。检查范围为可获取的状态和观测，不包含控制器快照恢复。

验证与指标
~~~~~~~~~~

张量与配置测试覆盖终止掩码、分组对齐、冻结目标和缓存状态回放。模型测试使用小型 Gemma/SigLIP 与构造的观测，可以在本地直接运行：

.. code-block:: bash

   python -m pytest tests/unit_tests/test_diffusion_art.py

启动实验附加 ``runner.max_steps=2``、``runner.max_epochs=2``、``runner.val_check_interval=1`` 和 ``runner.save_interval=1``，会先执行两轮训练、评估与 checkpoint 保存。GPU LIBERO 训练和 checkpoint 恢复需要真实模型与模拟器素材。

``actor/art_reward_mean`` 和 ``actor/art_reward_std`` 描述 episode 回报；``actor/valid_chunk_frac`` 和 ``actor/valid_action_frac`` 描述终止掩码与可选奖励过滤后保留的比例。现有 ``active_group_frac``、``mask_frac`` 和 ``movement_score`` 反映指导信号与漂移。ART/NFT 对照使用相同 SFT 初始化、初始状态和评估状态，同时记录环境交互量、optimizer 更新次数与耗时；两种损失的数值尺度对应不同目标。

分叉实验额外记录 ``actor/branch_chunk_index``、``actor/branch_valid_group_frac``、``actor/branch_state_error_max`` 和 ``actor/branch_image_error_max``。误差汇总前缀检查结果，有效组比例还排除分叉前终止的组。先运行索引 0 和索引 8 的短实验，分别检查初始决策行为与中途分叉，再增加训练预算。这些实验需要 GPU LIBERO 验证；单元测试覆盖采样、缓存对齐、回报截取和无效组屏蔽。

.. _diffusion-art-results:

训练结果
--------

``env/video_ocr`` 记录任务奖励，生成样本补充反映文字渲染质量。ART 引导、mask、NFT 裁剪和计时字段定义见 :ref:`扩散模型训练指标 <diffusion-training-metrics>`。OCR 奖励衡量文字相似度，画面质量与时间一致性属于独立评价维度。

表格汇总每组配置各一次的运行记录：ART 148 轮、NFT-1152 111 轮、NFT-576 70 轮。三组统一使用 ``video_ocr`` 奖励字段；平滑奖励取最近 5760 条轨迹的平均值，对 ART、NFT-1152 分别对应 5 轮，对 NFT-576 对应 10 轮。

.. list-table:: 运行记录
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - 运行
     - 每轮轨迹数
     - 学习率
     - 已记录轮数
     - 已记录累计时间
   * - ART
     - 1152
     - 1e-4
     - 148
     - 41:26:57
   * - NFT-1152
     - 1152
     - 1e-4
     - 111
     - 29:29:42
   * - NFT-576
     - 576
     - 4e-4
     - 70
     - 09:35:58

奖励提升
~~~~~~~~

达标成本对应平滑奖励首次达到阈值的日志点。每项分别列出累计小时数与轨迹数，统计使用完整的 5760 条轨迹平均窗口。

.. list-table:: 达到目标奖励所需的时间与轨迹数
   :header-rows: 1
   :widths: 16 28 28 28

   * - 目标
     - ART，1152，1e-4
     - NFT，1152，1e-4
     - NFT，576，4e-4
   * - 0.2
     - 2.30 h / 9216
     - 10.18 h / 43776
     - 2.44 h / 10368
   * - 0.5
     - 3.18 h / 12672
     - 13.67 h / 58752
     - 3.58 h / 14976
   * - 0.8
     - 5.43 h / 21888
     - 29.50 h 内未达到
     - 7.83 h / 32832

在这些记录中，ART 达到 0.8 的累计时间比 NFT-576 少 30.6%，轨迹数少 33.3%。两组 NFT 同时改变学习率和采样量，本次对比无法拆分各自的贡献。记录长度也不同，不能把最后一个奖励值当作相同预算下的终点比较。这些结果来自单次运行的训练奖励，不能解释成独立测试集成功率，也不代表复现了论文的视频 benchmark。

耗时与诊断
~~~~~~~~~~

执行成本统一统计累计轨迹区间 ``5760 < cumulative trajectories <= 40320``，每组覆盖 34560 条轨迹。对应 ART、NFT-1152 的第 6 至 35 轮，以及 NFT-576 的第 11 至 70 轮。

.. list-table:: 共同区间内的平均耗时
   :header-rows: 1
   :widths: 34 22 22 22

   * - 指标
     - ART
     - NFT，1152，1e-4
     - NFT，576，4e-4
   * - 每轮耗时（s）
     - 1026.24
     - 971.35
     - 495.00
   * - 每轮 actor 训练（s）
     - 105.23
     - 61.56
     - 63.37
   * - 每千条轨迹耗时（s）
     - 890.83
     - 843.18
     - 859.37
   * - 每分钟轨迹数
     - 67.35
     - 71.16
     - 69.82

在每轮 1152 条轨迹的两组中，ART 在该区间内每轮 actor 训练多用约 44 s，整轮多用约 55 s。Wan ART 配方为每条轨迹选择两个去噪状态，预计算对应旧速度，并在两个状态上反向传播；NFT 配方每条轨迹随机选择一个训练状态。额外的 forward 与 backward 是 actor 耗时差异的可能来源，现有 timer 尚未单独量出这部分成本。三组记录中，生成仍占最大的阶段耗时。

``algorithm.art_timestep_fraction=0.1`` 对应训练时刻成本的消融配置。当前生成步数为 10，该值会把每条轨迹的训练状态从两个减为一个。其余设置一致、计时排除初始化的对照可反映这一调整的影响。奖励随累计时间的变化与吞吐量反映不同侧面，因为减少训练状态也会改变学习预算。

本页的指标定义说明更新诊断与奖励、耗时的关系。ART 与 NFT 的 loss 定义不同，仅凭日志中的梯度大小不能判断哪种算法更稳定。论文的图像、视频与 pairwise 实验，以及当前实现的边界见 :ref:`ART 算法目标与实现 <diffusion-art-algorithm>`。

.. _diffusion-training-metrics:

扩散模型训练指标
----------------

这些指标描述 ART 和 NFT 的图像、视频生成训练。任务奖励反映文字生成效果，更新诊断记录模型变化，计时记录执行成本。终端表格会缩短 namespace：Training/Actor 下的 ``actor/art_loss`` 对应日志中的 ``train/actor/art_loss``，Environment 下的 ``video_ocr`` 对应 ``env/video_ocr``。

任务奖励与 rollout 信号
~~~~~~~~~~~~~~~~~~~~~~~

例如 prompt 要求生成 "OPEN"，OCR 会读取生成文字，再与目标比较。评分器去除空格、忽略大小写，按归一化编辑距离评分：

.. math::

   R_{\mathrm{OCR}}=1-\frac{\min(d(\hat y,y),|y|)}{|y|}.

其中 :math:`y` 是目标文字，:math:`\hat y` 是识别文字，:math:`d` 统计插入、删除和替换次数。子串匹配也会得到满分。视频 OCR 对每帧评分，``reward_type=video_level`` 使用所有帧的平均分，``frame_level`` 保留逐帧分数供后续分组。奖励 0.8 表示文字相似度，不代表 80% 的视频文字完全正确。

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 终端指标
     - 含义
   * - ``avg`` / ``video_ocr`` / ``ocr``
     - 奖励 backend 的评分，``reward.key`` 选择训练奖励。统一的 backend 字段支持跨运行对比；帧和 chunk 归约不同，日志聚合结果也可能不同。
   * - ``return``
     - episode 奖励之和。在单步、标量奖励的 ART 配方中，它等于该 episode 的评分。
   * - ``episode_len``
     - 每个 episode 的环境步数。一段视频可以只对应一个 episode，内部仍包含多个去噪步骤。
   * - ``num_trajectories``
     - 当前采集轮次完成的候选数，不是累计计数器。
   * - ``rewards``
     - rollout 侧对训练奖励 tensor 的聚合。
   * - ``advantages_mean`` / ``advantages_min`` / ``advantages_max``
     - NFT 分组比较的相对 advantage 统计。归一化后均值接近零是预期行为。ART 直接从奖励计算引导。
   * - ``loss_mask_fraction``
     - rollout 中标记为可参与训练的项所占比例，与 ART 修正 mask、NFT 裁剪判断含义不同。

ART 引导与更新诊断
~~~~~~~~~~~~~~~~~~

ART 先从完整 prompt 组估计方向，再在选中的去噪状态上拟合冻结速度目标。公式与速度符号约定见 :ref:`ART 算法目标与实现 <diffusion-art-algorithm>`。其 loss 为 :math:`\mathcal L=w\operatorname{MSE}(v_\theta,v_{\mathrm{old}}-H_c)`。

.. list-table:: Training/Actor 下的终端字段
   :header-rows: 1
   :widths: 36 64

   * - 指标
     - 含义
   * - ``actor/art_reward_mean``
     - 经过 ``art_reward_reduction`` 后，各候选标量奖励的平均值。
   * - ``actor/art_reward_std``
     - 每个 actor shard 内候选奖励的总体标准差，再跨 shard 平均。它不等于同一 prompt 内的奖励差异。
   * - ``actor/active_group_frac``
     - 引导非零的 prompt 组占比。组内奖励相同或汇总方向为零时，该组无有效引导。
   * - ``actor/covariance_rms``
     - 有效组的原始奖励与终点协方差 RMS 的平均值，尺度受奖励与 latent 坐标影响。
   * - ``actor/guidance_rms``
     - 每个候选的方向乘幅度后计算 RMS，再对候选平均，包含无效候选。单位归一化时通常接近有效组比例。
   * - ``actor/skipped_zero_guidance``
     - 所有 actor shard 均无引导、跳过优化时为一；至少一个 shard 有信号时为零。
   * - ``actor/art_loss``
     - 加权并经过修正 mask 后的训练 loss，平均时包含 loss 为零的项。
   * - ``actor/unmasked_loss``
     - 修正 mask 生效前的加权 loss，仅对有效项平均，分母与 ``actor/art_loss`` 不同。
   * - ``actor/old_deviate``
     - 训练项上速度 MSE 的平均值，即 :math:`D^{-1}\|v_\theta-v_{\mathrm{old}}\|_2^2`。
   * - ``actor/movement_score``
     - 偏移代理量的均值。``sde_kl`` 使用参考 SDE 噪声与步长；``fixed_mse`` 使用速度 MSE。
   * - ``actor/outward_frac``
     - 有效项中，下降方向与相对于旧速度的偏移内积为正的比例。
   * - ``actor/mask_frac``
     - 有效项中，同时因偏移过大和向外移动条件而被屏蔽的比例。
   * - ``actor/total_loss``
     - ART worker 中等于 ``actor/art_loss``。

这些比例与 loss 汇总各 micro-batch、各训练时刻的统计，再跨 worker 平均。``outward_frac`` 包含任意偏移大小的向外方向，``mask_frac`` 还要求越过阈值。屏蔽更多训练项也会让 masked loss 变小，奖励、有效组比例和屏蔽比例共同构成解释该值的依据。

NFT 能量与裁剪
~~~~~~~~~~~~~~

NFT 根据当前速度变化 :math:`\Delta v=v_\theta-v_{\mathrm{old}}` 构造两个分支：

.. math::

   v_\pm=v_{\mathrm{old}}\pm\beta\Delta v,\qquad
   E_\pm=\operatorname{reduce}\left[
      w_\pm\bigl(T(v_\pm)-x_{\mathrm{target}}\bigr)^2\right].

变换 :math:`T` 将速度转换到配置指定的预测空间，例如 ``nft_target_space=x0``。能量衡量预测误差，``nft_weight_mode`` 和 ``nft_energy_reduction`` 决定它的尺度。正负指两个分支，能量均值通常包含所有候选。采用 ``nft_loss_form=mse`` 时，转换后的 advantage 是 :math:`r\in[0,1]` 的权重，裁剪前的逐项 loss 为：

.. math::

   \ell_{\mathrm{NFT}}=\frac{A_{\max}}{\beta}
   \left[rE_+ +(1-r)E_-\right],
   \qquad A_{\max}=\text{advantage clipping bound}.

这里的裁剪上限为 ``adv_clip_max``。启用裁剪后，实现取裁剪前后逐项 loss 的较大值。它与 ART 目标 MSE 的定义和尺度不同。

.. list-table:: Training/Actor 下的终端字段
   :header-rows: 1
   :widths: 36 64

   * - 指标
     - 含义
   * - ``actor/E_pos_mean`` / ``actor/E_neg_mean``
     - 所有候选的正分支与负分支平均能量。
   * - ``actor/E_pos_mean_pos_only``
     - 转换后 advantage 满足 :math:`r>0.5` 的有效项上的正分支能量。
   * - ``actor/E_neg_mean_neg_only``
     - 满足 :math:`r<0.5` 的有效项上的负分支能量。
   * - ``actor/delta_E_mean``
     - :math:`E_+-E_-` 的均值，相反方向的贡献可能在聚合时相互抵消。
   * - ``actor/delta_v_norm``
     - 按配置的 gather、reduction 轴计算速度变化范数，与 ART 的平均平方偏移不同。
   * - ``actor/clip_frac``
     - 归约后的速度项中，裁剪系数小于一的比例。计数单位由 ``nft_gather_type`` 决定。
   * - ``actor/clip_loss_frac``
     - 有效 loss 项中，裁剪后的 loss 大于原 loss、最终被选中的比例。
   * - ``actor/nft_tau``
     - schedule 插值后的 rollout/reference EMA 当前 actor 混合系数，越小越保留历史参数。
   * - ``actor/nft_loss``
     - 梯度累积缩放前的 NFT 目标值，包含所配置的裁剪行为。
   * - ``actor/total_loss``
     - NFT worker 中除以梯度累积因子后的 loss。使用 SFT co-training 时，还包含 SFT 项。
   * - ``actor/grad_norm`` / ``actor/lr``
     - optimizer 报告的梯度范数与当前学习率，ART 也记录这两个指标。

能量均值、loss 和梯度可用于诊断单次训练，但它们的尺度受权重、目标空间与梯度累积影响。相同时间或采样预算下的任务奖励构成算法表现对比的依据。

耗时与进度
~~~~~~~~~~

runner 的整轮计时与互不重叠的阶段计时反映端到端性能，内部 timer 反映各阶段内的成本。重叠计时相加会重复统计部分执行时间。

.. list-table:: Time 下的终端字段
   :header-rows: 1
   :widths: 43 57

   * - 指标
     - 含义
   * - ``step``
     - runner 当前轮次的耗时。
   * - ``generate_rollouts``
     - 采集阶段，包含生成、奖励反馈和轨迹交付。
   * - ``actor_training`` / ``actor/run_training``
     - runner 和 worker 视角的 actor 训练计时，包含 ART/NFT 输入准备。
   * - ``sync_weights``
     - runner 权重同步阶段。
   * - ``actor/actor/recv_traj``
     - actor 等待、接收轨迹的时间，与 rollout 采集重叠。
   * - ``actor/actor/compute_adv`` / ``cal_adv_and_returns``
     - worker 和 runner 视角的 advantage、return 处理时间；ART 在这个阶段报告 rollout 统计。
   * - ``rollout/predict``
     - 模型生成调用，包含 wrapper 内的生成与解码。
   * - ``env/env_interact_step`` / ``env/env/bootstrap_step``
     - 包含奖励评分的环境交互，以及 bootstrap 工作。
   * - ``env/interact`` / ``env/run_interact_once``
     - 环境侧交互区间，包含通信和等待生成结果。
   * - ``rollout/generate_one_epoch`` / ``rollout/rollout/generate``
     - ``generate_rollouts`` 内部的嵌套 rollout 循环计时。
   * - ``actor/actor/sync_model_to_rollout`` / ``rollout/sync_model_from_actor``
     - actor 与 rollout 两侧的权重传输计时，可能重叠。

表头 Global Step 统计采集与训练轮次，一轮 ART 可以包含两次 optimizer update。Elapsed 是累计墙钟时间，ETA 是预估剩余时间，表头 Step Time 是累计墙钟时间除以已完成轮数得到的平均每轮耗时。Time 区的 ``step`` 记录单轮成本，每千条轨迹耗时与每分钟轨迹数反映不同采样量下的执行成本。相同累计采样区间的对比见 :ref:`训练结果 <diffusion-art-results>`。
