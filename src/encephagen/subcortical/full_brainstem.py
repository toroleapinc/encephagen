"""Full brainstem: All neonatal reflexes for a Humanoid body.

Every reflex a real human newborn has on day one, mapped to
the Humanoid-v5 body (torso, 2 arms, 2 legs, 17 joints).

No learning. Pure 先天. Structure → behavior.

Humanoid-v5 action mapping:
  0: abdomen_z (torso rotation)
  1: abdomen_y (torso forward/back)
  2: abdomen_x (torso side bend)
  3: right_hip_x, 4: right_hip_z, 5: right_hip_y, 6: right_knee
  7: left_hip_x, 8: left_hip_z, 9: left_hip_y, 10: left_knee
  11: right_shoulder1, 12: right_shoulder2, 13: right_elbow
  14: left_shoulder1, 15: left_shoulder2, 16: left_elbow
"""

import numpy as np


class FullBrainstem:
    """All neonatal reflexes mapped to Humanoid body."""

    def __init__(self):
        self.moro_timer = 0
        self.moro_phase = 0  # 0=none, 1=extension, 2=flexion
        self.startle_level = 0.0
        self.prev_height = 1.3
        self.prev_tilt = 0.0
        self.prev_obs = None

        # Breathing CPG (simple oscillator ~40 breaths/min = 0.67 Hz)
        self.breath_phase = 0.0
        self.breath_freq = 0.67  # Hz

        # General movements (slow random fidgeting)
        self.fidget_phase = np.random.rand(17) * 2 * np.pi
        self.fidget_freq = np.random.rand(17) * 0.3 + 0.1  # 0.1-0.4 Hz per joint

        # Grasp state
        self.grasp_left = 0.0
        self.grasp_right = 0.0
        self.stepping_drive = 0.0

    def process(self, obs, dt=0.02):
        """Process all reflexes. Returns 17-dim action for Humanoid.

        obs: Humanoid-v5 observation (348-dim)
          obs[0]: z height
          obs[1]: x tilt (forward/back)
          obs[2]: y tilt (left/right)
          obs[22:39]: joint positions
          obs[39:56]: joint velocities
        """
        action = np.zeros(17, dtype=np.float32)
        height = obs[0]
        tilt_fb = obs[1]
        tilt_lr = obs[2]

        # Detect sudden changes for startle/Moro
        height_drop = self.prev_height - height
        tilt_change = abs(tilt_fb - self.prev_tilt)

        # ========================================
        # 1. RIGHTING REFLEX (tonic labyrinthine)
        # Tilt → corrective torso torque
        # Most important — always active
        # ========================================
        action[1] += np.clip(-tilt_fb * 3.0, -0.4, 0.4)  # abdomen forward/back
        action[2] += np.clip(-tilt_lr * 2.0, -0.4, 0.4)  # abdomen side

        # ========================================
        # 2. MORO REFLEX
        # Sudden drop or loud stimulus →
        #   Phase 1: arms extend outward (abduct), fingers spread
        #   Phase 2: arms flex inward (adduct), fists clench
        # ========================================
        if (height_drop > 0.03 or tilt_change > 0.2) and self.moro_timer <= 0:
            self.moro_timer = 30
            self.moro_phase = 1

        if self.moro_timer > 0:
            if self.moro_timer > 15:
                # Phase 1: EXTENSION — arms spread out
                action[11] += 0.3   # right shoulder out
                action[12] += 0.2
                action[13] -= 0.2   # right elbow extend
                action[14] += 0.3   # left shoulder out
                action[15] += 0.2
                action[16] -= 0.2   # left elbow extend
            else:
                # Phase 2: FLEXION — arms come in, fists clench
                action[11] -= 0.3   # right shoulder in
                action[13] += 0.3   # right elbow flex
                action[14] -= 0.3   # left shoulder in
                action[16] += 0.3   # left elbow flex
            self.moro_timer -= 1

        # ========================================
        # 3. STARTLE REFLEX
        # Sudden sensory change → whole body brief jerk
        # ========================================
        if self.prev_obs is not None:
            delta = np.abs(obs[:20] - self.prev_obs[:20]).sum()
            if delta > 3.0:
                self.startle_level = min(delta / 5.0, 1.0)
        self.startle_level *= 0.85

        if self.startle_level > 0.1:
            # Brief jerk in all joints
            for a in range(17):
                action[a] += self.startle_level * 0.08 * np.sin(a * 1.5)

        # ========================================
        # 4. PALMAR GRASP
        # Constant slight elbow flexion (grasp posture)
        # Real newborn grip can support body weight
        # ========================================
        action[13] += 0.15  # right elbow slightly flexed
        action[16] += 0.15  # left elbow slightly flexed

        # ========================================
        # 5. ATNR (Asymmetric Tonic Neck Reflex / Fencing)
        # Head turns right → right arm extends, left arm flexes
        # We use torso rotation as proxy for head turn
        # ========================================
        torso_rotation = obs[0] if len(obs) > 3 else 0  # approximate
        atnr_strength = 0.1
        if abs(tilt_lr) > 0.05:
            # Turn right (positive tilt_lr) → right arm extends
            action[11] += tilt_lr * atnr_strength  # right shoulder
            action[13] -= tilt_lr * atnr_strength  # right elbow extends
            action[14] -= tilt_lr * atnr_strength  # left shoulder
            action[16] += tilt_lr * atnr_strength  # left elbow flexes

        # ========================================
        # 6. GALANT REFLEX
        # Stroke along spine → trunk curves toward stimulus
        # Approximated as slight lateral trunk oscillation
        # ========================================
        # (Embedded in general movements below)

        # ========================================
        # 7. BREATHING CPG
        # Rhythmic abdomen movement (~40/min)
        # ========================================
        self.breath_phase += self.breath_freq * dt * 2 * np.pi
        breath_signal = np.sin(self.breath_phase) * 0.05
        action[1] += breath_signal  # abdomen forward/back (breathing)

        # ========================================
        # 8. GENERAL MOVEMENTS (Prechtl's)
        # Whole-body movements involving arms, legs, trunk
        # Variable, complex, wax and wane
        # NOT rhythmic like CPG — multiple frequencies per joint
        # ========================================
        for j in range(17):
            self.fidget_phase[j] += self.fidget_freq[j] * dt * 2 * np.pi
            fidget = np.sin(self.fidget_phase[j]) * 0.04
            # Modulate amplitude with slow wave (wax and wane)
            slow_mod = 0.5 + 0.5 * np.sin(self.fidget_phase[j] * 0.1)
            action[j] += fidget * slow_mod

        # ========================================
        # 9. STEPPING DRIVE
        # When upright, activate leg CPG
        # (CPG is handled separately in the newborn)
        # ========================================
        self.stepping_drive = 0.3 if height > 0.8 else 0.0

        # ========================================
        # 10. KNEE STABILIZATION
        # Slight constant flexion prevents hyperextension
        # ========================================
        action[6] += 0.05   # right knee
        action[10] += 0.05  # left knee

        # Update state
        self.prev_height = height
        self.prev_tilt = tilt_fb
        self.prev_obs = obs[:20].copy()

        return np.clip(action, -0.4, 0.4)

    def get_active_reflexes(self):
        """Return dict of active reflex intensities for display."""
        return {
            'Righting': min(abs(self.prev_tilt) * 3.0, 1.0),
            'Moro': self.moro_timer / 30.0,
            'Startle': self.startle_level,
            'Grasp': 0.15,  # always active
            'ATNR': 0.1,    # proportional to tilt
            'Breathing': abs(np.sin(self.breath_phase)) * 0.5,
            'General mvt': 0.3,  # always active
            'Stepping': self.stepping_drive,
        }
