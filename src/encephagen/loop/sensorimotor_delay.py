"""Sensorimotor Delay: realistic timing between brain and body.

Real sensorimotor loops have 20-100ms conduction delay:
  Sensory receptor → spinal cord → brainstem → cortex: ~30ms
  Motor cortex → spinal cord → muscle: ~20ms
  Total loop: ~50ms

Without delay, the brain gets instant feedback — removing the timing
structure that CPGs and cerebellar coordination evolved to handle.
Adding delay forces the cerebellum and cortex to PREDICT, which is
what enables M1-cerebellum coherence to emerge (Gulati 2024).
"""

from collections import deque
import numpy as np


class SensorimotorDelay:
    """Circular buffer delay between brain and body."""

    def __init__(self, sensory_delay_ms=30, motor_delay_ms=20, dt_body_ms=20):
        """
        Args:
            sensory_delay_ms: delay from body observation to brain (afferent)
            motor_delay_ms: delay from brain command to body action (efferent)
            dt_body_ms: body simulation timestep in ms
        """
        self.sensory_steps = max(1, int(sensory_delay_ms / dt_body_ms))
        self.motor_steps = max(1, int(motor_delay_ms / dt_body_ms))

        self.sensory_buffer = deque(maxlen=self.sensory_steps)
        self.motor_buffer = deque(maxlen=self.motor_steps)

        self._initialized = False

    def init(self, obs_shape, action_shape):
        """Initialize buffers with zeros."""
        for _ in range(self.sensory_steps):
            self.sensory_buffer.append(np.zeros(obs_shape))
        for _ in range(self.motor_steps):
            self.motor_buffer.append(np.zeros(action_shape))
        self._initialized = True

    def delay_sensory(self, obs):
        """Push new observation, return delayed observation."""
        if not self._initialized:
            self.init(obs.shape, (6,))
        self.sensory_buffer.append(obs.copy())
        return self.sensory_buffer[0]  # oldest = most delayed

    def delay_motor(self, action):
        """Push new action, return delayed action."""
        self.motor_buffer.append(action.copy())
        return self.motor_buffer[0]  # oldest = most delayed

    def get_delay_info(self):
        return {
            'sensory_delay_steps': self.sensory_steps,
            'motor_delay_steps': self.motor_steps,
            'total_loop_delay_steps': self.sensory_steps + self.motor_steps,
        }
