from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import torch
from torch.utils import tensorboard
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.policy import A2CPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

from sit_liveclip.core import PreprocessNet
from sit_liveclip.utils import console


if TYPE_CHECKING:
    from typing import Literal

    import gym

    from sit_liveclip.config import Config


def train(env: gym.Env, log_dir: Path | None, config: Config) -> None:
    # pylint: disable=too-many-locals
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
    policy = _build_policy(env=env, config=config, device=device)

    # logger
    writer = tensorboard.SummaryWriter(log_dir)  # type: ignore
    if log_dir is None:
        log_dir = Path(writer.log_dir)
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    console.print_divider("Config")
    console.print("log_dir:", log_dir)
    console.print(config)

    console.print_divider("Model")
    console.print(policy)

    # training
    console.print_divider("Training")
    train_collector = Collector(policy, env, ReplayBuffer(20000))
    result = onpolicy_trainer(
        policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=config.max_epoch,
        step_per_epoch=config.step_per_epoch,
        repeat_per_collect=config.repeat_per_collect,
        episode_per_test=config.episode_per_test,
        batch_size=config.batch_size,
        step_per_collect=config.step_per_epoch,
        episode_per_collect=config.episode_per_collect,
        stop_fn=lambda mean_rewards: mean_rewards >= config.reward_threshold,
        logger=logger,
    )

    # save model
    console.print_divider("Training Result")
    out_model_file = log_dir / "policy.pth"
    torch.save(policy.state_dict(), out_model_file)
    console.print("Saved trained model to", out_model_file)
    console.print_dict(result)


def evaluate(env: gym.Env, model_path: Path, config: Config) -> None:
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
    policy = _build_policy(env=env, config=config, device=device)

    console.print_divider("Config")
    console.print(config)

    console.print_divider("Model")
    console.print(policy)

    # evaluate
    console.print_divider("Evaluation")
    test_collector = Collector(policy, env)
    policy.load_state_dict(torch.load(model_path))  # type: ignore
    policy.eval()
    result = test_collector.collect(n_episode=1, render=False)

    # show result
    console.print_divider("Evaluation Result")
    console.print_dict(result)


def _build_policy(
    env: gym.Env,
    config: Config,
    device: Literal["cpu", "cuda"],
) -> A2CPolicy:
    preprocess_net = PreprocessNet(
        history_size=config.history_size,
        sliding_window_size=config.sliding_window_size,
        conv1d_out=config.conv1d_out,
        conv1d_kernel_size=config.conv1d_kernel_size,
        device=device,
    )
    actor = Actor(
        preprocess_net,
        env.action_space.n,
        hidden_sizes=config.actor_hidden_size,
        device=device,
    ).to(device)
    critic = Critic(
        preprocess_net,
        hidden_sizes=config.critic_hidden_size,
        device=device,
    ).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate)

    # A2C Policy
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor,
        critic,
        optim,
        dist,
        action_space=env.action_space,
        deterministic_eval=True,
    )

    return policy
