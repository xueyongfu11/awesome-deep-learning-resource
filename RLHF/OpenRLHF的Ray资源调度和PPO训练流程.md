[TOC]



# 理一下OpenRLHF的Ray资源调度和PPO训练流程

以`examples/train_ppo_ray.py`为例，讲解PPO训练中ray的资源调度机制和PPO训练流程。再具体讲解代码之前，先大概介绍下Ray和DeepSpeed的功能和差异。

## Ray和DeepSpeed的功能和差异

Ray 管“把 GPU 这类资源在集群里怎么分、给谁用、什么时候用”；DeepSpeed 管“拿到这些 GPU 之后，训练这件事怎么更快、更省显存、更稳定”。两者不是互相替代，更多是上下游/互补。

>  Ray 的“GPU 管理”到底在管什么

Ray 把 GPU 当作一种可调度资源（像 CPU、内存一样），核心能力在于：

- 资源声明与隔离：某个任务/Actor 声明 `num_gpus=1/2/8...`，Ray 负责分配，让它只看到被分配的卡。
- 集群级调度：多台机器、多张卡，Ray 决定任务放哪台机器、哪几张卡上。
- 并发与队列：同时跑很多训练/推理/数据处理任务，Ray 控制并发、重试、容错。
- 拓扑与放置约束：比如要“同机 8 卡”或“跨机 2×8 卡”，Ray 用 placement groups/策略保证资源布局。

> DeepSpeed 在管什么

DeepSpeed 的关注点是单个训练作业内部：拿到 N 张 GPU 后，怎么把模型/参数/梯度/优化器状态拆分与高效同步。

DeepSpeed 是训练侧的优化库/运行时，核心能力在于：

- 分布式训练实现与加速：数据并行、模型并行（含 ZeRO）、通信优化等。
- 显存优化：ZeRO-1/2/3、offload、activation checkpointing 等，让大模型能塞进有限显存。
- 性能工程：更高吞吐、更低通信开销、更好的稳定性（在特定配置下）。

> 二者的关系

你可以把 Ray 看成外层调度器/作业编排，DeepSpeed 看成内层训练引擎。Ray 负责起训练作业 + 分配 GPU，DeepSpeed 负责作业内分布式训练

- Ray 启动一个训练任务，可能是一个 driver + 多个 worker。
- 每个 worker 在 Ray 分配到的 GPU 上运行。
- worker 内部用 DeepSpeed（以及 torch.distributed/NCCL）做分布式训练。

- DeepSpeed 不负责整个集群上有 100 张卡怎么分给 10 个作业。它更多假设给一组进程/节点/卡”，然后把训练跑好。

接下来进入到代码具体看Ray和DeepSpeed是如何进行互相配合完成PPO训练的

## ray初始化

```python
# 初始化 Ray 运行时（避免重复初始化）
if not ray.is_initialized():
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

# 构建并打印分布式/并行训练策略
strategy = get_strategy(args)
strategy.print(args)
```

## 为actor/ref模型共置或者全模型共置分配资源

有几个地方容易混淆：

- 在PPO中，主要包括actor model、critic model、reference model、reward model
- 在ray中，actor是有状态的编程模型或者抽象，多次调用之间状态会保留，常驻GPU，而task是无状态的函数调用，直接完返回结果。一个ray worker要么执行普通的task，要么承载一个actor。

```python
'''
# 初始化 vLLM / actor / critic / reference / reward 模型
# 如果需要 colocate，先为 actor/ref 创建 placement group 来打包资源
如何理解共置colocate：主要是为了节省GPU资源，actor、ref、rm、value等每个模型负载波动大，共置能让空闲时段被其他模型吃掉。
'''
pg = None
if args.colocate_actor_ref or args.colocate_all_models:
    if args.init_kl_coef > 0:
        assert (
            args.actor_num_nodes == args.ref_num_nodes
            and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

    # 按照actor总的GPU数量生成一组资源包，每个资源包bundle由1个GPU和1个CPU组成，使用PACK策略来创建placement_group
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
    pg = placement_group(bundles, strategy="PACK")
    # 阻塞这个placement group真正分配完成，保证后续创建actor、ref可以绑定到这组资源
    ray.get(pg.ready())
```

## 分配vllm engines资源并创建

```python
'''
初始化 vLLM 引擎（用于生成 rollout 文本）

如果colocate_all_model并且是非异步训练，必须设置actor和vllm engines需要的资源相等。
异步训练（async_train为True）表示actor训练和采样并行执行，并通过队列通信，async_queue_size控制队列的大小。

如果colocate_all_model并且是异步训练，则没有这样的要求，即actor、ref、rm、value共置，vllm engines单独分配资源。
'''
vllm_engines = None
if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
    max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
    if args.colocate_all_models and not args.async_train:
        assert (
            args.actor_num_nodes * args.actor_num_gpus_per_node
            == args.vllm_num_engines * args.vllm_tensor_parallel_size
        ), (
            f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
            f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
            f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
        )

    # placement group 只在 colocate_all_models 且同步训练时传入
    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.pretrain,
        args.seed,
        args.full_determinism,
        args.enable_prefix_caching,
        args.enforce_eager,
        max_len,
        pg if args.colocate_all_models and not args.async_train else None,
        args.vllm_gpu_memory_utilization,
        args.vllm_enable_sleep,
        "processed_logprobs" if args.enable_vllm_is_correction else None,
        agent_func_path=args.agent_func_path,
        remote_rm_url=args.remote_rm_url,
    )
```

## 创建vllm engines

```python
def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    logprobs_mode=None,
    agent_func_path: Optional[str] = None,
    remote_rm_url: Optional[str] = None,
):
    """Spin up a set of vLLM Ray actors with consistent placement."""
    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    # use_hybrid_engine 为True时表示使用colocate_all_model模式并且是非异步训练方式。
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # 混合模式下允许一张卡上放多个 engine，靠 fractional GPU 进行资源切分
        num_gpus = 0.2

    if not use_hybrid_engine:
        # 非混合模式时，创建一个大的 placement group，把所有 engine 打包在一起，
        # 这样 Ray 会尽量把这些 actor 放在尽可能少的节点上，减少跨节点通信开销
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            '''
            tensor_parallel_size > 1 时，一个 vLLM engine 需要占用多块 GPU（多个 bundle）。这时必须显式算出
            一组连续的 bundle_indices，并通过 VLLM_RAY_BUNDLE_INDICES 把这些 bundle 绑定到同一台机器上，
            避免 Ray 把同一引擎的多 GPU 跨节点分散（get_bundle_indices 就是在做“同节点分组”，规避 Ray 
            的 placement bug/行为）。

            而当 tensor_parallel_size == 1 时，每个引擎只需要 1 个 bundle，直接用 i 作为 placement group
            的 bundle index 就够了，不需要做“多 bundle 同节点”的选择。
            '''
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        # `scheduling_strategy`用来设置每个actor vllm绑定placement group的调度策略，`LLMRayActor.options`
        # 用来给actor设置资源和调度策略。
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        actor_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "worker_extension_cls": "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed + i,
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "full_determinism": full_determinism,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices,
            "num_gpus": 0.2 if use_hybrid_engine else 1,
            "enable_sleep_mode": vllm_enable_sleep,
        }

        actor_kwargs.update(
            {
                "agent_func_path": agent_func_path,
                "remote_rm_url": remote_rm_url,
            }
        )

        if logprobs_mode:
            actor_kwargs["logprobs_mode"] = logprobs_mode
            actor_kwargs["max_logprobs"] = 1
            # 只有新版本 vLLM 才支持 logprobs_mode，避免老版本 silent failure
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            # options 用来声明 actor 的资源需求和调度策略
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**actor_kwargs)
        )

    if vllm_enable_sleep:
        # 统一进入 sleep，减少初始化后的显存占用；后续需要时再唤醒
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines
```



```python
def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """Call the same method on a list of engines and gather results."""
    import torch

    if torch.distributed.is_initialized():
        # 多进程分布式时，通常只在 rank 0 触发远程调用，避免重复执行与资源浪费
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        # Ray actor 的方法调用是异步的，remote 返回 ObjectRef 用于后续聚合
        refs.append(method.remote(*args, **kwargs))

    # 等待所有远程调用完成并收集结果
    return ray.get(refs)
```

## 创建policy模型ray actor组

```python
actor_model = RayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    PolicyModelActor,
    pg=pg,
    num_gpus_per_actor=0.2 if pg else 1,
    duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
)
```



```python
class RayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BaseModelActor]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        # duplicate actors is ring_attn_size * tensor_parallel_size
        self.duplicate_actors = duplicate_actors

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        # self._resources 是 Ray 的自定义资源标签，用来控制 actor 调度到哪类节点或隔离特定硬件资源。
        # 比如你在 Ray 集群里给某些节点打了 {"accelerator": 1} 这样的资源标签，这里传入同名 key 就能让这
        # 些 actor 只被调度到对应资源的节点。它在创建 placement group 或 actor 时传给 resources=...，
        # 因此会参与资源匹配和隔离，避免模型被调度到不合适的机器上。
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # 使用 placement group 把同一类模型需要的资源先锁住，避免调度时被其它任务抢占
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        if pg:
            # rank 0 作为 master，负责建立分布式进程组并对外提供地址/端口
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker_actor
        if world_size > 1:
            # 先让 master 生成监听地址/端口，其他 rank 通过它加入同一分布式 world
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    # 每个 worker 固定绑到 placement group 的对应 bundle，确保资源一致且不迁移
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank,
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)
```

## 创建ref模型ray actor组

```python
# Reference 模型仅在启用 KL 惩罚时需要创建RayActorGroup
if args.init_kl_coef > 0:
    ref_model = RayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )
else:
    ref_model = None
```

## 为critic model分配资源并创建ray actor组

```python
# 未 colocate 全部模型时，后续 critic/reward 使用独立 placement group
if not args.colocate_all_models:
    pg = None

# colocate critic/reward 时创建新的 placement group
if args.critic_pretrain and args.colocate_critic_reward:
    assert (
        args.critic_num_nodes == args.reward_num_nodes
        and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
    ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

if args.critic_pretrain:
    # Critic（价值模型）Ray Actor 组
    critic_model = RayActorGroup(
        args.critic_num_nodes,
        args.critic_num_gpus_per_node,
        CriticModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )
else:
    critic_model = None
```

## 为rm model分配资源并创建ray actor组

```python
# Reward 模型（支持多模型），远程 RM 时不在本地起 Actor
if not args.remote_rm_url:
    reward_model = RayActorGroup(
        args.reward_num_nodes,
        args.reward_num_gpus_per_node,
        RewardModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )
else:
    reward_model = None
```

## 初始化PPO Trainer

```python
'''
按同步/异步模式选择 Trainer
PPOTrainer（同步）：一个 Ray actor 内部按“生成 rollout -> 训练 -> 同步 vLLM”的顺序串行执行，采样和训练不重叠
PPOTrainerAsync（异步）：拆成两个 Ray actor：GenerateSamplesActor 只负责生成 rollout，TrainingActor 只
负责训练；两者通过 Queue 传递 batch
'''
if args.async_train:
    from openrlhf.trainer.ppo_trainer_async import PPOTrainerAsync as PPOTrainer
else:
    from openrlhf.trainer.ppo_trainer import PPOTrainer

# 初始化 PPO Trainer（单控制器 Ray Actor）
ppo_trainer = PPOTrainer.remote(
    args.pretrain,
    strategy,
    actor_model,
    critic_model,
    reward_model,
    ref_model,
    vllm_engines,
    # generate kwargs
    do_sample=True,
    prompt_max_len=args.prompt_max_len,
    max_new_tokens=args.generate_max_len,
    max_length=args.max_len,
    temperature=args.temperature,
    top_p=args.top_p,
)

# 训练总步数用于调度/学习率等（由 trainer 内部计算）
max_steps = ray.get(ppo_trainer.get_max_steps.remote())
```

## 初始化 actor/critic/ref/reward model

```python
# 加载 actor/ref/reward 权重（异步触发，最后统一等待）
refs = []
# async_init_model_from_pretrained初始化所有的policy actors
refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps, vllm_engines))
if ref_model is not None:
    # async_init_model_from_pretrained初始化所有的ref_model actors
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
if reward_model is not None and args.reward_pretrain:
    refs.extend(reward_model.async_init_model_from_pretrained(strategy, args.reward_pretrain))
ray.get(refs)

if critic_model is not None and args.critic_pretrain:
    # critic 的调度依赖 max_steps，需在 actor 初始化完成后再启动
    # TODO: 使用第一个 reward model 作为 critic
    refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
    ray.get(refs)
```

Policy Model的模型初始化：

- 初始actor model对象
- 创建训练actor的trainer
- 创建tokenizer、optimizer、scheduler等
- 创建分布式训练环境，单独介绍

```python
@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        '''
        setup_distributed 在每个 worker进程 内部执行，完成 DeepSpeed/Torch 的分布式初始化，并加入同一个进程组
        因此setup_distributed函数内部获取world size，其实是进程组的进程总数。
        跟ray的关系：Ray 负责启动分布式的多个 worker 进程/Actor，也就是ray是进程编排层，setup_distributed是
        进程内部的分布式初始化
        '''
        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            param_dtype=strategy.args.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.attn_implementation,
                param_dtype=strategy.args.param_dtype,  # default: bf16
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        # 这段代码把普通的 actor/optimizer/scheduler 交给 strategy.prepare 做 DeepSpeed 包装，返回可分布式训练的对象
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states = states

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )
```

reference model、critic model、rm model的初始化不再一一介绍

## actor model的分布式训练环境

在初始化actor model过程会创建分布式训练环境



```python
def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
    if self.full_determinism:
        transformers.enable_full_determinism(self.seed)
        # Use deterministic backward in flash attention as, by default, flash attention uses atomic adds
        # https://github.com/Dao-AILab/flash-attention/commit/732654583c2e640adc012ecb60e460bf19dcd9e3
        transformers.modeling_flash_attention_utils.deterministic_g = True
    else:
        transformers.set_seed(self.seed)

    # Take the local rank from args as first priority
    if self.args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(self.args.local_rank)

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank != -1:
        # 让当前进程绑定到对应 GPU，避免后续初始化用错设备
        torch.cuda.set_device(local_rank)

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    deepspeed.init_distributed(timeout=timeout)

    # 这里用 Device Mesh 表达多维并行：数据并行(dp) + ring-attn 并行(sp) + 张量并行(tp)
    self.world_size = dist.get_world_size()  # 总进程数
    # world_size 必须能被 ring_attn_size 与 tp_size 整除，否则网格划分不合理
    # 总的GPU数根据dp、sp、tp进行切分
    dp_size = self.world_size // self.ring_attn_size // self.ds_tensor_parallel_size
    self.ds_device_mesh = init_device_mesh(
        "cuda", (dp_size, self.ring_attn_size, self.ds_tensor_parallel_size), mesh_dim_names=("dp", "sp", "tp")
    )
    # 根据 mesh 初始化 ring-attn 相关通信组
    self.setup_ring_attn(self.ds_device_mesh)

    # 计算实际的梯度累计步数：
    # - 目标总 batch = train_batch_size * ring_attn * tp
    # - 每步用 micro_batch_size
    # - 再按 world_size 均摊到各 rank
    self.accumulated_gradient = (
        self.train_batch_size
        * self.ring_attn_size
        * self.ds_tensor_parallel_size
        // self.micro_train_batch_size
        // self.world_size
    )
```



```python
def setup_ring_attn(self, ds_device_mesh):
    if self.ring_attn_size == 1:
        self.ring_attn_rank = 0
        return

    # 取出 ring-attn(sp 维度)的通信组；同组内 rank 参与环形注意力并行
    group = ds_device_mesh["sp"].get_group()
    # 在该通信组内的 rank 编号，用于后续分片/通信的定位
    self.ring_attn_rank = dist.get_rank(group=group)
    # 保存到全局上下文，供其他模块（例如自定义 flash-attn 替换逻辑）读取
    set_ring_attn_group(group)

    from ring_flash_attn import substitute_hf_flash_attn

    # ring_head_stride 控制环形注意力里 head 的分配步长；
    # 通过替换 HF 的 flash-attn 内核，让注意力计算在 ring group 上协同执行
    self.ring_head_stride = getattr(self.args, "ring_head_stride", 1)
    substitute_hf_flash_attn(self.ring_attn_group, self.ring_head_stride)
```

## actor model的trainer的初始化

在初始化 actor model时会初始化trainer

```python
class ActorPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        # 记录生成相关的可选参数，后续由生成阶段按需使用
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.max_epochs

        # PPO policy loss 的配置集中在 args 里，这里统一构造损失函数
        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            dual_clip=self.args.dual_clip,
            policy_loss_type=self.args.policy_loss_type,
            enable_vllm_is_correction=self.args.enable_vllm_is_correction,
            vllm_is_truncated_threshold=(
                self.args.vllm_is_truncated_threshold if self.args.enable_vllm_is_correction else None
            ),
            use_icepop=self.args.use_icepop,
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # 训练样本回放缓冲区：支持 CPU offload、sample packing、动态 batch
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        # Init torch group for weights sync
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        # colocate_all_models + 非异步训练时可用 CUDA IPC 更高效地传权重
        if backend == "nccl" and self.args.colocate_all_models and not self.args.async_train:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and each of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            # 选择一个空闲端口作为分布式通信入口
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            # vLLM 引擎的全局进程数 + DeepSpeed rank0
            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            # 让每个 vLLM engine 初始化自己的进程组
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                # 用 Ray collective 管理同步组；rank0 先初始化
                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                # 走 torch.distributed 的 stateless 初始化，避免全局默认组冲突
                self._model_update_group = stateless_init_process_group(
                    master_address, master_port, 0, world_size, torch.cuda.current_device()
                )

            # 等待所有 vLLM 端初始化完成
            ray.get(refs)
        # 同步分布式训练和 CUDA 操作
        torch_dist_barrier_and_cuda_sync()
```

初始化部分讲完了，接下来是模型训练

## 异步的PPOTrainer的训练流程

同步的相对比较简单，这里主要介绍异步的PPO训练流程

```python
def fit(self) -> None:
    # 从训练端读取断点，保证生成端/训练端使用同一套进度信息。
    checkpoint_states = ray.get(self.trainer_actor.init_checkpoint_states.remote())

    # 恢复训练进度：episode、global_step，以及已消费的prompt数量。
    start_episode = checkpoint_states["episode"]
    global_step = checkpoint_states["global_step"]
    total_consumed_prompts = checkpoint_states.get("total_consumed_prompts", 0)
    # 断点续训时需要同步：
    # 1) 生成端的数据加载器状态（避免重复或遗漏样本）
    # 2) vLLM 的权重（避免生成用旧权重）
    if global_step > 0:
        ray.get(
            [
                self.generator_actor.load_state_dict.remote(checkpoint_states["data_loader_state_dict"]),
                self.trainer_actor.broadcast_to_vllm.remote(),
            ]
        )

    # 并发启动：生成端不断产出样本，训练端消费样本并更新模型。
    # ray.get 等待二者都结束（生成端会在耗尽数据后发送 "done" 结束训练端）。
    ray.get(
        [
            self.generator_actor.fit.remote(episode=start_episode, total_consumed_prompts=total_consumed_prompts),
            self.trainer_actor.fit.remote(global_step=global_step),
        ]
    )
```

## generator的样本生成

```python
def fit(self, episode: int, total_consumed_prompts: int):
    for episode in range(episode, self.args.num_episodes):
        dataset_length = len(self.prompts_dataloader)
        pbar = tqdm(
            range(dataset_length),
            desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
            initial=total_consumed_prompts % max(dataset_length, 1),
        )
        while True:
            # 背压控制：只有拿到一个“令牌”（队列容量）才允许生成，
            # 否则生成会塞满队列导致训练端无法及时消费。
            self.rollout_slots.get(block=True)

            # vLLM 临界区：生成时不能与权重广播同时发生，
            # 避免读到不一致的权重或状态（推理/训练权重错位）。
            ray.get(self.vllm_lock.acquire.remote())
            try:
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(**self.generate_kwargs)
                )
                total_consumed_prompts += prompts_consumed
            finally:
                ray.get(self.vllm_lock.release.remote())

            produced = bool(rollout_samples)
            if produced:
                # 把生成结果和训练端需要的进度信息一起放入队列，
                # 便于训练端保存断点并与生成端保持进度一致。
                client_states = {
                    "episode": episode,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.rollout_queue.put((rollout_samples, client_states, filter_pass_rate), block=True)
                if prompts_consumed:
                    pbar.update(prompts_consumed)
            else:
                # 未产生样本就不会入队；训练端也不会“消费”这个令牌，
                # 因此这里要手动归还令牌，避免令牌泄漏导致死锁。
                self.rollout_slots.put(None, block=True)

            # 当前 episode 的数据耗尽时退出内层循环。
            if is_exhausted:
                break

        pbar.close()

    # 发送终止信号，通知训练端安全退出。
    self.rollout_queue.put("done", block=True)
```

## ppo 训练

```python
def fit(self, global_step: int):
    while True:
        # 阻塞等待生成端提供新样本；收到 "done" 表示数据结束。
        payload = self.rollout_queue.get(block=True)
        if payload == "done":
            break

        rollout_samples, client_states, filter_pass_rate = payload

        # 训练端消费一个 batch 后，归还一个令牌，
        # 允许生成端继续产出下一批数据。
        self.rollout_slots.put(None, block=True)

        status, global_step = self.train_step(rollout_samples, global_step)

        if self.args.dynamic_filtering:
            # 记录动态过滤通过率，便于监控采样质量。
            status["dynamic_filtering_pass_rate"] = filter_pass_rate

        # 日志中不打印生成样本本体，避免日志过大。
        log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
        logger.info(f"✨ Global step {global_step}: {log_status}")

        # 把训练进度与生成端状态一起保存，保证断点一致可恢复。
        client_states.update({"global_step": global_step})
        self.save_logs_and_checkpoints(global_step, status, client_states)

    # 训练结束后关闭日志器，避免资源泄漏。
    if self.wandb_logger:
        self.wandb_logger.close()
    if self.tensorboard_logger:
        self.tensorboard_logger.close()
```



```python
def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
    # 将采样得到的原始rollout转换为PPO所需的trajectory，并计算奖励/优势等信号。
    # 这里把“环境交互”结果变成“可训练样本”，是PPO更新的核心输入。
    experiences = self.experience_maker.make_experience_batch(rollout_samples)

    # 取第一个样本做快速sanity check：生成文本与reward是否看起来合理。
    # 这有助于尽早发现reward model或采样链路的异常。
    sample0 = [
        self.tokenizer.batch_decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
        experiences[0].info["reward"][0].item(),
    ]
    print(sample0)

    # 若启用动态batch，在不同DP rank之间重新平衡experience长度/数量，
    # 以缓解长序列导致的负载不均与吞吐抖动。
    if self.args.use_dynamic_batch:
        experiences = balance_experiences(experiences, self.args)

    # 将experience推送到actor/critic的分片上，作为后续PPO优化的训练数据。
    # 这里是“采样端 → 训练端”的数据交接点。
    refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
    if self.critic_model_group is not None:
        refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
    ray.get(refs)

    # 执行一次PPO优化（actor/critic），并收集训练指标。
    status = self.ppo_train(global_step)

    # 将最新actor权重同步到vLLM推理引擎，保证后续采样使用新策略。
    if self.vllm_engines is not None:
        self.broadcast_to_vllm()

    # 根据最新的KL统计更新KL控制器，用于调节后续PPO的KL惩罚强度。
    if "kl" in status:
        # TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here.
        self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

    status["generated_samples"] = sample0
    return status, global_step + 1

def ppo_train(self, global_steps: int) -> Dict:
    """Run one PPO train step for critic + actor and return merged status dict."""
    status: dict = {}

    # 决定本轮是否训练critic/actor（actor可以在前若干步冻结）。
    # 这样可以先稳定价值估计，再放开策略更新，降低初期不稳定性。
    run_critic = self.critic_model_group is not None
    run_actor = global_steps > self.args.freezing_actor_steps and self.actor_model_group is not None

    def _run_sleep(group, **kwargs):
        # Sleep模式下，按“加载权重 -> 训练 -> 卸载权重”的顺序执行，
        # 以减少常驻显存占用，适合多模型并行但显存紧张的场景。
        ray.get(group.async_run_method(method_name="reload_states"))
        ref = group.async_run_method(method_name="fit", **kwargs)
        status.update(ray.get(ref)[0])
        ray.get(group.async_run_method(method_name="offload_states"))

    if self.args.deepspeed_enable_sleep:
        # 休眠/同机模式：先训练critic，再训练actor，避免相互抢占显存。
        if run_critic:
            _run_sleep(self.critic_model_group)
        if run_actor:
            _run_sleep(self.actor_model_group, kl_ctl=self.kl_ctl.value)
    else:
        # 异步模式：先并行启动，再统一等待结果并合并指标。
        # 这样可以提高吞吐，但需要注意不同任务完成时间不一致。
        refs = []
        if run_critic:
            refs += self.critic_model_group.async_run_method(method_name="fit")
        if run_actor:
            refs += self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)

        for result in ray.get(refs):
            status.update(result)

    return status
```



```python
def async_run_method_batch(self, method_name, **kwargs):
    """Run method on all actors with batched input data asynchronously using round-robin scheduling.
    Each actor processes one chunk of data at a time. Actors in the same ring / tensor parallel group process the same chunk.

    Args:
        method_name (str): Name of the method to run
        **kwargs: Keyword arguments for the method. Each value should be a list/tensor of the same length.

    Returns:
        List[ray.ObjectRef]: List of remote object references to the results
    """
    # 先做输入检查，保证每个参数都可切片/取长度
    for key, value in kwargs.items():
        if not hasattr(value, "__len__"):
            raise ValueError(f"Parameter {key} must be iterable")

    # 以第一个参数的长度作为 batch 总长度的基准
    first_param = next(iter(kwargs.values()))
    total_length = len(first_param)

    # 各参数必须对齐，否则不同字段无法按样本一一对应切分
    for key, value in kwargs.items():
        if len(value) != total_length:
            raise ValueError(
                f"All parameters must have the same length. {key} has length {len(value)}, expected {total_length}"
            )

    # 计算“有效 actor 数”：同一 ring/tensor 并行组的 duplicate actors 处理同一个 chunk
    num_actors = len(self._actor_handlers)
    effective_actors = num_actors // self.duplicate_actors
    if total_length == 0 or total_length < effective_actors:
        # 每个有效 actor 至少要有一条数据，否则会出现空切片或空闲 actor
        raise ValueError(
            f"Insufficient batch size for async_run_method_batch: total_length={total_length}, "
            f"effective_actors={effective_actors}"
        )
    # 均分切片，不能整除时向上取整，确保覆盖全部样本
    chunk_size = total_length // effective_actors
    if total_length % effective_actors != 0:
        chunk_size += 1

    # 将全量数据一次性放入 Ray object store，避免在每个 actor 间重复拷贝
    all_data_ref = ray.put(kwargs)

    refs = []
    for chunk_idx in range(effective_actors):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_length)

        for j in range(self.duplicate_actors):
            actor_idx = chunk_idx * self.duplicate_actors + j
            actor = self._actor_handlers[actor_idx]

            # 同一 chunk 会被 duplicate actors 并行处理（用于 ring/tensor 并行）
            refs.append(actor.execute_batch.remote(method_name, all_data_ref, start_idx, end_idx))

    return refs
```



## actor model同步到vllm engines

```python
def broadcast_to_vllm(self):
    use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
    cache_reset_refs = []
    if use_prefix_cache and torch.distributed.get_rank() == 0:
        # clear prefix cache
        for engine in self.vllm_engines:
            cache_reset_refs.append(engine.reset_prefix_cache.remote())

    torch.cuda.empty_cache()
    model = self.actor.model.module
    count, num_params = 0, len(list(model.named_parameters()))

    def _broadcast_param(param, count, num_params):
        use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
        # Fire all vllm engines for broadcast
        if torch.distributed.get_rank() == 0:
            # ZeRO-3 情况下参数可能是分片的，ds_shape 记录了完整形状
            shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
            # 先让 vLLM 侧预分配/准备接收（异步），避免同步时占用训练线程
            refs = [
                engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                for engine in self.vllm_engines
            ]

            if use_ray:
                import ray.util.collective as collective

                # 走 Ray collective 的 group（不依赖 torch 默认进程组）
                collective.broadcast(param.data, 0, group_name=self._model_update_group)
            else:
                # 走 torch.distributed 的自定义进程组
                self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
            ray.get(refs)

    def _handle_cuda_ipc(param, count, num_params):
        from torch.multiprocessing.reductions import reduce_tensor

        # CUDA IPC 需要先把权重复制到独立 tensor，避免引用到可能被释放的参数存储
        weight = param.data.clone()
        ipc_handle = reduce_tensor(weight)

        ipc_handle = {get_physical_gpu_id(): ipc_handle}
        ipc_handle_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

        if torch.distributed.get_rank() == 0:
            # 汇总每张 GPU 的 IPC 句柄，传给 vLLM 侧用于直接映射读取
            ipc_handles = {}
            for d in ipc_handle_list:
                ipc_handles.update(d)

            shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
            refs = [
                engine.update_weight_cuda_ipc.remote(
                    name,
                    dtype=param.dtype,
                    shape=shape,
                    ipc_handles=ipc_handles,
                    empty_cache=count == num_params,
                )
                for engine in self.vllm_engines
            ]
            ray.get(refs)
        # 等待 IPC 传输路径准备就绪，避免后续参数更新交错
        torch_dist_barrier_and_cuda_sync()

    for name, param in model.named_parameters():
        count += 1  # empty_cache at last param

        # broadcast
        if not self.use_cuda_ipc:
            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            if self.strategy.args.ds_tensor_parallel_size > 1:
                # TP + ZeRO 时需要先聚合被替换的分片参数
                # 把para的ZeRO+TP分片先allgather成完整参数
                with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                    _broadcast_param(param, count, num_params)
            else:
                # ZeRO-3 先 allgather 到 rank0，再广播给 vLLM
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    _broadcast_param(param, count, num_params)
        # CUDA IPC
        else:
            if self.strategy.args.ds_tensor_parallel_size > 1:
                # TP + CUDA IPC 仍需先把参数聚合成完整权重
                with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                    _handle_cuda_ipc(param, count, num_params)
            else:
                # ZeRO-3 下先 allgather，再用 IPC 交给 vLLM 侧映射
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    _handle_cuda_ipc(param, count, num_params)

    if cache_reset_refs:
        ray.get(cache_reset_refs)
    torch.cuda.empty_cache()
    torch_dist_barrier_and_cuda_sync()
```







