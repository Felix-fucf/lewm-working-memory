"""
module_oot.py
===============================================================================

OOT（Occluded Object Transport）实验专用模型模块。

设计目标
--------
1. 保留 LeWorldModel 原始结构中的：
      图像编码器 encoder
      动作编码器 action_encoder
      Transformer predictor
      projector / pred_proj

2. 在 encoder 与 predictor 之间插入 Leaky-RNN 工作记忆层：

      pixels
        ↓
      encoder + projector
        ↓
      当前时刻视觉 latent embedding z_t
        ↓
      Leaky-RNN memory
        ↓
      带历史信息的 memory embedding m_t
        ↓
      action-conditioned Transformer predictor
        ↓
      预测下一时刻视觉 embedding z_{t+1}

3. 不修改原仓库的 module.py 与 jepa.py，便于保留原始 LeWM 作为基线。

为什么不是直接把 Transformer 换成 RNN
------------------------------------
我们当前想检验的是“显式工作记忆层是否帮助模型跨越遮挡保存对象状态”。
因此最干净的对照是：

    原模型：encoder -> Transformer predictor
    实验组：encoder -> Leaky-RNN -> Transformer predictor

这样 Transformer predictor 的主体尽量保持不变，主要自变量就是是否增加
Leaky-RNN memory。

张量记号
--------
B: batch size
T: 时间长度
D: latent embedding 维度
A: action embedding 维度
H: Leaky-RNN hidden dimension
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn


def _validate_probability(value: float, name: str) -> None:
    """检查一个数是否严格位于 (0, 1) 内。"""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} 必须位于 (0, 1) 内，但收到 {value}。")


def _probability_to_logit(value: float) -> float:
    """
    把概率 p 转换为 logit。

    当 leak_rate 设为可学习参数时，我们不直接优化 alpha，
    而是优化一个无约束实数 raw_alpha，再通过 sigmoid 得到 alpha。
    这样训练过程中 alpha 会始终保持在 (0, 1) 内。
    """
    _validate_probability(value, "leak_rate")
    value_tensor = torch.tensor(value, dtype=torch.float32)
    return torch.log(value_tensor / (1.0 - value_tensor)).item()


class LeakyRNNMemory(nn.Module):
    r"""
    一个显式、可解释的 Leaky-RNN 工作记忆层。

    单步更新公式
    ------------
    candidate_t = tanh(W_x x_t + W_h h_{t-1} + b)

    h_t = (1 - alpha) * h_{t-1} + alpha * candidate_t

    其中：
    - h_{t-1}：上一时刻记忆；
    - candidate_t：当前输入建议写入的新内容；
    - alpha：泄漏率 / 更新率。

    alpha 较小：
        新输入只缓慢改变记忆，长期保存能力更强，但更新可能较迟钝。

    alpha 较大：
        记忆快速跟随当前输入，但更容易忘记遮挡前信息。

    对 OOT 的直觉
    -------------
    当对象进入遮挡区域后，当前图像中不再直接包含对象位置。
    Leaky-RNN 不能只依赖当前图像，而需要让此前的隐藏状态继续影响 h_t。
    同时，动作信息仍可推动隐藏状态更新，从而表示“物体虽不可见但仍在移动”。

    Parameters
    ----------
    input_dim:
        每个时刻输入 x_t 的维度。
    hidden_dim:
        隐藏状态 h_t 的维度。
    leak_rate:
        初始 alpha。
    learnable_leak:
        是否允许 alpha 在训练中学习。
        初版建议先设 False，便于把 alpha 当作清晰的实验变量。
    dropout:
        仅施加在输出序列上的 dropout，不直接破坏递归状态本身。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        leak_rate: float = 0.2,
        learnable_leak: bool = False,
        dropout: float = 0.0,
        initial_state_std: float = 0.01,
    ) -> None:
        super().__init__()

        _validate_probability(leak_rate, "leak_rate")

        initial_state_std = float(initial_state_std)
        if initial_state_std < 0.0:
            raise ValueError(
                "initial_state_std 必须大于等于 0，"
                f"实际收到 {initial_state_std}。"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learnable_leak = bool(learnable_leak)

        # 每个 episode 的初始 hidden state 使用小幅独立高斯噪声。
        #
        # std=0 时严格退化为原先的零初始化；
        # std>0 时，每个 batch 样本获得独立随机初态。
        self.initial_state_std = initial_state_std

        # 当前时刻输入 x_t 对候选记忆的影响。
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=True)

        # 上一时刻隐藏状态 h_{t-1} 对候选记忆的影响。
        # 这里不再添加 bias，避免与 input_proj 的 bias 重复。
        self.hidden_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # 可学习 alpha 使用 logit 参数化；固定 alpha 则注册为 buffer。
        # buffer 会随模型移动到 GPU，也会被保存在 state_dict 中，
        # 但不会被优化器更新。
        if self.learnable_leak:
            initial_logit = _probability_to_logit(leak_rate)
            self.raw_leak = nn.Parameter(torch.tensor(initial_logit))
        else:
            self.register_buffer(
                "fixed_leak",
                torch.tensor(float(leak_rate), dtype=torch.float32),
            )

        self.output_dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化递归权重。

        hidden_proj 使用正交初始化：
        对简单 RNN 来说，正交递归矩阵通常比任意高方差矩阵更不容易在
        长时间展开时立刻产生爆炸或过快衰减。
        """
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.hidden_proj.weight)

    def _sample_initial_state(
        self,
        reference: Tensor,
        batch_size: int,
    ) -> Tensor:
        """生成每个 episode 独立的小幅随机初始 hidden state。"""
        shape = (int(batch_size), self.hidden_dim)

        if self.initial_state_std == 0.0:
            return reference.new_zeros(shape)

        # 使用 float32 采样后再转换，避免 bf16/fp16 随机采样精度过低。
        noise = torch.randn(
            shape,
            device=reference.device,
            dtype=torch.float32,
        )

        return (
            noise
            * self.initial_state_std
        ).to(dtype=reference.dtype)

    @property
    def leak_rate(self) -> Tensor:
        """返回当前 alpha；返回值是标量 Tensor，位于 (0, 1)。"""
        if self.learnable_leak:
            return torch.sigmoid(self.raw_leak)
        return self.fixed_leak

    def forward(
        self,
        x: Tensor,
        initial_state: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        沿时间维展开 Leaky-RNN。

        Parameters
        ----------
        x:
            shape = (B, T, input_dim)

        initial_state:
            shape = (B, hidden_dim)
            若为 None，则按 initial_state_std 采样小幅随机状态。

        valid_mask:
            shape = (B, T) 或 (B, T, 1)
            1 表示该时刻有效；0 表示 padding。
            对无效时刻，隐藏状态保持不变。

        reset_mask:
            shape = (B, T) 或 (B, T, 1)
            1 表示“在处理当前时刻前先清空记忆”。
            常规 OOT 样本通常不会跨 episode，因此一般无需提供；
            但保留此接口可以防止未来数据切片跨 episode 时发生记忆泄漏。

        Returns
        -------
        hidden_sequence:
            shape = (B, T, hidden_dim)

        final_state:
            shape = (B, hidden_dim)
        """
        if x.ndim != 3:
            raise ValueError(
                f"LeakyRNNMemory 期望 x 为三维 (B,T,D)，实际 shape={tuple(x.shape)}"
            )

        batch_size, time_steps, input_dim = x.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"LeakyRNNMemory input_dim={self.input_dim}，"
                f"但实际输入最后一维为 {input_dim}。"
            )

        sampled_initial_state = initial_state is None

        if sampled_initial_state:
            hidden = self._sample_initial_state(
                x,
                batch_size,
            )
        else:
            expected = (batch_size, self.hidden_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(
                    f"initial_state 应为 {expected}，"
                    f"实际为 {tuple(initial_state.shape)}。"
                )
            hidden = initial_state

        def prepare_mask(mask: Optional[Tensor], default: float) -> Tensor:
            """把 mask 统一为 (B,T,1)，并移动到与 x 相同的 dtype/device。"""
            if mask is None:
                return x.new_full((batch_size, time_steps, 1), default)

            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)

            if tuple(mask.shape) != (batch_size, time_steps, 1):
                raise ValueError(
                    "mask 应为 (B,T) 或 (B,T,1)，"
                    f"实际 shape={tuple(mask.shape)}。"
                )

            return mask.to(device=x.device, dtype=x.dtype)

        valid_mask = prepare_mask(valid_mask, default=1.0)
        reset_mask = prepare_mask(reset_mask, default=0.0)

        # alpha 是标量，但广播后可同时作用于整个 batch 和 hidden dimension。
        alpha = self.leak_rate.to(device=x.device, dtype=x.dtype)

        hidden_steps = []

        # 这里显式写出时间循环，而不是使用 nn.RNN。
        # 原因是 Leaky-RNN 的更新公式和 reset/valid mask 都需要我们自己控制。
        for t in range(time_steps):
            current_input = x[:, t]
            current_valid = valid_mask[:, t]
            current_reset = reset_mask[:, t]

            # reset_mask=1 时，在读取 x_t 前为新 episode
            # 重新采样小幅随机初态，而不是清成严格全零。
            # initial_state=None 时，序列开始前已经采样过一次 h0。
            # 因此 t=0 的 episode_start 只表示“这是新 episode”，
            # 不应再次采样并丢弃刚生成的初态。
            should_resample_reset = (
                (t > 0 or not sampled_initial_state)
                and bool(torch.any(current_reset > 0.5))
            )

            if should_resample_reset:
                reset_hidden = self._sample_initial_state(
                    x,
                    batch_size,
                )

                hidden = (
                    (1.0 - current_reset) * hidden
                    + current_reset * reset_hidden
                )

            candidate = torch.tanh(
                self.input_proj(current_input) + self.hidden_proj(hidden)
            )

            updated_hidden = (1.0 - alpha) * hidden + alpha * candidate

            # padding 时刻不应该改变隐藏状态。
            hidden = (
                current_valid * updated_hidden
                + (1.0 - current_valid) * hidden
            )

            hidden_steps.append(hidden)

        hidden_sequence = torch.stack(hidden_steps, dim=1)
        hidden_sequence = self.output_dropout(hidden_sequence)

        return hidden_sequence, hidden


class PointwiseActionPredictor(nn.Module):
    """Predict the next latent independently at every timestep.

    The predictor has no temporal operators: no attention, recurrence,
    positional embedding, or causal mask.  All history must therefore pass
    through the recurrent state supplied by :class:`StrictRecurrentOOTJEPA`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        if state.ndim != 3 or action.ndim != 3:
            raise ValueError("state and action must both have shape (B,T,D)")
        if state.shape[:2] != action.shape[:2]:
            raise ValueError(
                "state/action batch and time dimensions differ: "
                f"state={tuple(state.shape)}, action={tuple(action.shape)}"
            )
        if state.size(-1) != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, got {state.size(-1)}"
            )
        if action.size(-1) != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {action.size(-1)}"
            )
        return self.net(torch.cat((state, action), dim=-1))


class StrictRecurrentOOTJEPA(nn.Module):
    """OOT world model with a strict recurrent information bottleneck.

    Data flow:
        s_t = encoder(o_t)
        h_t = LeakyRNN(h_(t-1), concat(s_t, E(a_(t-1))))
        s_hat_(t+1) = predictor(concat(h_t, E(a_t)))

    ``remember`` deliberately returns the raw hidden sequence as the predictor
    state.  There is no visual residual or hidden-to-embedding bypass.
    """

    strict_recurrent_bottleneck = True

    def __init__(
        self,
        encoder: nn.Module,
        predictor: PointwiseActionPredictor,
        action_encoder: nn.Module,
        memory: LeakyRNNMemory,
        embed_dim: int,
        projector: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.memory = memory
        self.embed_dim = int(embed_dim)
        self.projector = projector or nn.Identity()

        if self.memory.input_dim != 2 * self.embed_dim:
            raise ValueError(
                "strict bottleneck memory must receive concat(s_t, E(a_t-1)); "
                f"expected input_dim={2 * self.embed_dim}, "
                f"got {self.memory.input_dim}"
            )
        if self.memory.hidden_dim != self.embed_dim:
            raise ValueError(
                "strict bottleneck currently requires hidden_dim == embed_dim; "
                f"got {self.memory.hidden_dim} and {self.embed_dim}"
            )

    def encode(self, info: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "pixels" not in info:
            raise KeyError("StrictRecurrentOOTJEPA.encode requires 'pixels'")
        output = dict(info)
        pixels = info["pixels"].float()
        if pixels.ndim < 3:
            raise ValueError(
                "pixels must include batch and time dimensions; "
                f"got {tuple(pixels.shape)}"
            )
        batch_size = pixels.size(0)
        encoder_output = self.encoder(
            rearrange(pixels, "b t ... -> (b t) ..."),
            interpolate_pos_encoding=True,
        )
        cls = encoder_output.last_hidden_state[:, 0]
        projected = self.projector(cls)
        output["emb"] = rearrange(projected, "(b t) d -> b t d", b=batch_size)
        if "action" in info:
            output["act_emb"] = self.action_encoder(info["action"])
        return output

    def remember(
        self,
        emb: Tensor,
        act_emb: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if emb.ndim != 3 or emb.size(-1) != self.embed_dim:
            raise ValueError(
                f"emb must have shape (B,T,{self.embed_dim}), got {tuple(emb.shape)}"
            )
        if act_emb is None:
            raise ValueError("strict bottleneck remember() requires previous action")
        if act_emb.shape != emb.shape:
            raise ValueError(
                "previous action embedding must match visual embedding shape; "
                f"visual={tuple(emb.shape)}, action={tuple(act_emb.shape)}"
            )
        hidden, final_state = self.memory(
            torch.cat((emb, act_emb), dim=-1),
            initial_state=initial_state,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
        )
        return hidden, hidden, final_state

    def predict(self, hidden: Tensor, act_emb: Tensor) -> Tensor:
        return self.predictor(hidden, act_emb)

    @property
    def leak_rate(self) -> Tensor:
        return self.memory.leak_rate


class OOTJEPA(nn.Module):
    """
    在原始 JEPA 接口上加入 Leaky-RNN memory 的 OOT 世界模型。

    该类有三个核心公开接口：

    encode(batch)
        pixels -> visual embeddings
        action -> action embeddings

    remember(emb, act_emb)
        visual embeddings + action information -> recurrent memory embeddings

    predict(memory_emb, act_emb)
        memory embeddings -> next visual embedding prediction

    这样 train_oot.py 可以清晰地把三个阶段分别记录下来，便于后续分析：
        原始视觉 latent
        Leaky-RNN hidden state
        predictor 输出 latent
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        memory: LeakyRNNMemory,
        embed_dim: int,
        projector: Optional[nn.Module] = None,
        pred_proj: Optional[nn.Module] = None,
        action_memory_proj: Optional[nn.Module] = None,
        fuse_action_into_memory: bool = True,
        memory_residual: bool = True,
        memory_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.memory = memory

        self.embed_dim = int(embed_dim)
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

        # 动作不仅条件化 predictor，也可以进入 Leaky-RNN。
        # 这是 OOT 的关键：对象在遮挡期间虽然不可见，但“运输动作”仍提供
        # 状态如何变化的信息。
        self.fuse_action_into_memory = bool(fuse_action_into_memory)
        self.action_memory_proj = action_memory_proj or nn.Identity()

        self.memory_residual = bool(memory_residual)
        self.memory_residual_scale = float(memory_residual_scale)

        # Leaky-RNN hidden_dim 未必等于视觉 latent 的 embed_dim。
        # 因此用线性层把 hidden state 映射回 predictor 所需的维度。
        self.memory_to_embedding = nn.Linear(
            self.memory.hidden_dim,
            self.embed_dim,
            bias=True,
        )

        # 使用较小增益初始化，使模型初始时仍接近原始视觉 embedding，
        # 避免新加的随机递归层立刻破坏 predictor 输入。
        nn.init.xavier_uniform_(self.memory_to_embedding.weight, gain=0.1)
        nn.init.zeros_(self.memory_to_embedding.bias)

        self.memory_output_norm = nn.LayerNorm(self.embed_dim)

    def encode(self, info: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        编码图像和动作。

        这部分基本沿用原始 jepa.JEPA.encode 的逻辑：
        1. 将 (B,T,...) 展平成 (B*T,...)；
        2. 每帧独立送入视觉 encoder；
        3. 取 CLS token；
        4. 经 projector 得到 latent embedding；
        5. 恢复为 (B,T,D)。

        注意：
        我们创建 output=dict(info)，不直接在原 batch 字典上增加中间结果，
        这样 callback 保存原始 batch 时更清晰。
        """
        if "pixels" not in info:
            raise KeyError("OOTJEPA.encode 需要 batch 中包含 'pixels'。")

        output = dict(info)

        pixels = info["pixels"].float()
        if pixels.ndim < 3:
            raise ValueError(
                f"pixels 至少应包含 batch 与时间维，实际 shape={tuple(pixels.shape)}"
            )

        batch_size = pixels.size(0)

        flattened_pixels = rearrange(pixels, "b t ... -> (b t) ...")

        encoder_output = self.encoder(
            flattened_pixels,
            interpolate_pos_encoding=True,
        )

        # 原 LeWM 取 ViT 的 CLS token 作为整帧表示。
        pixels_embedding = encoder_output.last_hidden_state[:, 0]

        projected_embedding = self.projector(pixels_embedding)

        output["emb"] = rearrange(
            projected_embedding,
            "(b t) d -> b t d",
            b=batch_size,
        )

        if "action" in info:
            output["act_emb"] = self.action_encoder(info["action"])

        return output

    def _build_memory_input(
        self,
        emb: Tensor,
        act_emb: Optional[Tensor],
    ) -> Tensor:
        """
        构造 Leaky-RNN 的输入。

        初版采用“视觉 latent + 动作 latent”的加法融合，而不是拼接：
        - 保持输入维度等于 embed_dim；
        - 参数量较小；
        - 与原 predictor 使用相同时间对齐方式；
        - 便于把 action 解释为对当前 latent 状态更新的驱动力。

        若未来发现 action embedding 与 visual embedding 维度不一致，
        可通过 action_memory_proj 投影到 embed_dim。
        """
        if not self.fuse_action_into_memory:
            return emb

        if act_emb is None:
            raise ValueError(
                "fuse_action_into_memory=True，但 remember() 没有收到 act_emb。"
            )

        projected_action = self.action_memory_proj(act_emb)

        if projected_action.shape != emb.shape:
            raise ValueError(
                "动作加入记忆前必须与视觉 embedding 形状一致；"
                f"visual={tuple(emb.shape)}, action={tuple(projected_action.shape)}。"
            )

        return emb + projected_action

    def remember(
        self,
        emb: Tensor,
        act_emb: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        把逐帧视觉 embedding 转换为带历史信息的 memory embedding。

        Returns
        -------
        memory_embedding:
            shape = (B,T,embed_dim)
            这是送给 Transformer predictor 的输入。

        hidden_sequence:
            shape = (B,T,H)
            这是 Leaky-RNN 原始隐藏状态，后续可用于可视化和 probing。

        final_state:
            shape = (B,H)
        """
        if emb.ndim != 3:
            raise ValueError(f"emb 应为 (B,T,D)，实际为 {tuple(emb.shape)}。")

        memory_input = self._build_memory_input(emb, act_emb)

        hidden_sequence, final_state = self.memory(
            memory_input,
            initial_state=initial_state,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
        )

        memory_delta = self.memory_to_embedding(hidden_sequence)

        if self.memory_residual:
            # 残差形式保留当前视觉信息，同时允许递归状态补充历史信息。
            memory_embedding = (
                emb + self.memory_residual_scale * memory_delta
            )
        else:
            memory_embedding = memory_delta

        memory_embedding = self.memory_output_norm(memory_embedding)

        return memory_embedding, hidden_sequence, final_state

    def predict(self, memory_emb: Tensor, act_emb: Tensor) -> Tensor:
        """
        使用原 LeWM 风格的 action-conditioned predictor 预测下一时刻 latent。

        predictor 的因果注意力会保证第 t 个输出只能使用不晚于 t 的输入。
        """
        predictions = self.predictor(memory_emb, act_emb)

        # pred_proj 使用 LayerNorm MLP；禁止 BatchNorm 跨样本统计。
        batch_size = memory_emb.size(0)
        predictions = self.pred_proj(
            rearrange(predictions, "b t d -> (b t) d")
        )
        predictions = rearrange(
            predictions,
            "(b t) d -> b t d",
            b=batch_size,
        )

        return predictions

    @property
    def leak_rate(self) -> Tensor:
        """方便 train_oot.py 记录当前 Leaky-RNN 的 alpha。"""
        return self.memory.leak_rate
