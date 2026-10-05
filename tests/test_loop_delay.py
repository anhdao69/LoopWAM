import math
import random

import numpy as np
import pytest

from fastwam.loop.evaluation import delay_ticks, preserve_rng_state, run_delayed_episode


class Environment:
    def __init__(self, success_at=None, period=.05):
        self.env = self
        self.control_timestep = period
        self.success_at = success_at
        self.actions = []

    def reset(self):
        self.actions = []

    def set_init_state(self, state):
        return dict(step=0)

    def step(self, action):
        self.actions.append(list(action))
        n = len(self.actions)
        return dict(step=n), 0, n == self.success_at, {}


@pytest.mark.parametrize('seconds,ticks', [(0,0),(.001,1),(.05,1),(.050001,2),(.1,2)])
def test_delay_uses_next_control_boundary(seconds, ticks):
    assert delay_ticks(seconds, .05) == ticks


@pytest.mark.parametrize('seconds,period', [(-1,.05),(math.nan,.05),(1,0),(1,math.inf)])
def test_invalid_delay_is_not_a_zero_delay(seconds, period):
    with pytest.raises(ValueError):
        delay_ticks(seconds, period)


def test_exact_processed_command_is_held_and_delay_counts_toward_cap():
    env = Environment()
    requests = []
    command = [.1,.2,.3,.4,.5,.6,1]
    def predict(obs):
        requests.append(obs['step'])
        return [command]*32, .06
    result = run_delayed_episode(env, None, predict, max_steps=25)
    assert requests == [30,42,54]
    assert env.actions[30:32] == [[0,0,0,0,0,0,-1]]*2
    assert env.actions[32:] == [command]*23
    assert result['control_steps'] == 25
    assert result['delay_steps'] == 5
    assert result['policy_steps'] == 20
    assert result['replans'][-1]['delay_steps_executed'] == 1
    assert result['replans'][0]['availability_step'] == 2
    assert result['replans'][0]['quantization_overhead_ms'] == pytest.approx(40.)


def test_success_during_delay_stops_before_using_new_chunk():
    env = Environment(success_at=33)
    result = run_delayed_episode(env, None, lambda obs: ([[1]*7]*32, .3))
    assert result['success']
    assert result['control_steps'] == result['delay_steps'] == 3
    assert result['policy_steps'] == 0
    assert env.actions == [[0,0,0,0,0,0,-1]]*33


def test_zero_delay_matches_original_ten_action_replan_schedule():
    env = Environment()
    requests = []
    chunk = [[i]*6+[-1] for i in range(32)]
    def predict(obs):
        requests.append(obs['step'])
        return chunk, 0.
    result = run_delayed_episode(env, None, predict, max_steps=700)
    assert requests == list(range(30,730,10))
    assert env.actions[30:] == chunk[:10]*70
    assert result['control_steps'] == result['policy_steps'] == 700
    assert result['delay_steps'] == 0


def test_warmup_preserves_python_numpy_and_torch_rng_even_on_error():
    import torch
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    def draw():
        return random.random(),float(np.random.random()),float(torch.rand(()))
    expected = draw()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    with pytest.raises(RuntimeError):
        with preserve_rng_state():
            for _ in range(50):
                draw()
            raise RuntimeError('warmup failure')
    assert draw() == expected


def test_delay_summary_retains_timing_and_rejects_unaccounted_control_steps():
    from fastwam.loop.evaluation import DELAY_PROTOCOL, summarize_tasks
    tasks=[]
    for task in range(10):
        rows=[]
        for episode in range(50):
            result=run_delayed_episode(Environment(success_at=32),None,lambda obs: ([[1]*7]*32,.06))
            result.pop('frames');result['episode_id']=episode
            rows.append(result)
        tasks.append(dict(task_id=task,seed=42,status='complete',total_episodes=50,
            success_episodes=list(range(50)),failure_episodes=[],evaluation_kind='delay',
            delay_protocol=DELAY_PROTOCOL,episode_results=rows))
    summary=summarize_tasks(tasks,42,delay=True)
    assert summary['successes']==500 and summary['delay']['delay_steps']==1000
    assert summary['delay']['decision_wall']['p50_ms']==pytest.approx(60.)
    assert summary['outcomes'][0]['replans'][0]['availability_step']==2
    tasks[0]['episode_results'][0]['control_steps']=701
    with pytest.raises(ValueError,match='control'):
        summarize_tasks(tasks,42,delay=True)
