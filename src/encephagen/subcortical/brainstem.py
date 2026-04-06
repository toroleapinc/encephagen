"""Brainstem: Hardwired reflex arcs for a newborn human.

The brainstem is the most mature brain structure at birth — fully myelinated.
It produces all the innate reflexes that keep a newborn alive.

Each reflex is a simple stimulus→response arc:
  Sensory input → brainstem nuclei → motor output

Reflexes are prioritized: Moro > righting > stepping > rooting.
Only one reflex can dominate at a time (brainstem mutual inhibition).

References:
    Primitive Reflexes — StatPearls/NCBI
    Prechtl (2005) — General movements assessment
"""

from __future__ import annotations

import numpy as np


class BrainstemReflexes:
    """Brainstem reflex arcs for a newborn.

    Produces motor commands based on sensory input.
    No learning. Purely hardwired. This is what keeps newborns alive.
    """

    def __init__(self):
        # Reflex states
        self.moro_timer = 0       # Moro reflex cooldown
        self.rooting_side = 0.0   # Which direction to root
        self.startle_level = 0.0  # Current startle intensity

        # Previous sensory state (for change detection)
        self.prev_tilt = 0.0
        self.prev_height = 1.3

    def process(self, sensory: dict) -> dict:
        """Process sensory input through brainstem reflex arcs.

        Args:
            sensory: dict with keys:
                'tilt_fb': forward/back tilt (radians)
                'tilt_lr': left/right tilt (radians)
                'height': body height
                'angular_vel': angular velocity
                'touch_left': touch on left side (0-1)
                'touch_right': touch on right side (0-1)
                'loud_sound': sudden loud stimulus (0-1)
                'face_touch': touch near face/mouth (0-1)

        Returns:
            dict with motor commands:
                'righting': corrective torque for balance
                'moro': Moro reflex intensity (0-1)
                'rooting': head turn direction (-1 to 1)
                'startle': startle motor burst (0-1)
                'stepping_drive': CPG drive modulation
                'withdrawal_left': left limb withdrawal (0-1)
                'withdrawal_right': right limb withdrawal (0-1)
        """
        tilt_fb = sensory.get('tilt_fb', 0.0)
        tilt_lr = sensory.get('tilt_lr', 0.0)
        height = sensory.get('height', 1.3)
        angular_vel = sensory.get('angular_vel', 0.0)
        touch_left = sensory.get('touch_left', 0.0)
        touch_right = sensory.get('touch_right', 0.0)
        loud_sound = sensory.get('loud_sound', 0.0)
        face_touch = sensory.get('face_touch', 0.0)

        output = {
            'righting': 0.0,
            'moro': 0.0,
            'rooting': 0.0,
            'startle': 0.0,
            'stepping_drive': 0.0,
            'withdrawal_left': 0.0,
            'withdrawal_right': 0.0,
        }

        # ============================================
        # RIGHTING REFLEX (tonic labyrinthine + vestibular)
        # Highest priority for survival — always active
        # PD control on body tilt
        # ============================================
        output['righting'] = -tilt_fb * 3.0 - angular_vel * 1.0

        # ============================================
        # MORO REFLEX
        # Trigger: sudden drop in height OR sudden head extension
        # Response: arms extend (abduct) → then flex (adduct) + cry
        # Brainstem vestibular nuclei
        # ============================================
        height_drop = self.prev_height - height
        tilt_change = abs(tilt_fb - self.prev_tilt)

        if (height_drop > 0.05 or tilt_change > 0.3 or loud_sound > 0.5) and self.moro_timer <= 0:
            self.moro_timer = 20  # 20 cycles of Moro response
            output['moro'] = 1.0

        if self.moro_timer > 0:
            # Phase 1 (first 10 cycles): extension (arms out)
            # Phase 2 (next 10 cycles): flexion (arms in)
            if self.moro_timer > 10:
                output['moro'] = 0.8  # extension phase
            else:
                output['moro'] = -0.5  # flexion phase
            self.moro_timer -= 1

        # ============================================
        # STARTLE REFLEX
        # Trigger: sudden sensory change
        # Response: brief whole-body motor burst
        # ============================================
        if loud_sound > 0.3:
            self.startle_level = min(loud_sound, 1.0)
        self.startle_level *= 0.85  # decay
        output['startle'] = self.startle_level

        # ============================================
        # ROOTING REFLEX
        # Trigger: touch on cheek → head turns toward touch
        # Brainstem trigeminal nucleus
        # ============================================
        if face_touch > 0.1:
            output['rooting'] = 0.5  # turn toward

        # ============================================
        # STEPPING REFLEX
        # Trigger: feet touch surface while body upright
        # Response: alternating leg movements (spinal CPG activation)
        # ============================================
        if height > 0.8:  # roughly upright
            output['stepping_drive'] = 0.3  # activate CPG
        else:
            output['stepping_drive'] = 0.0  # too far gone, don't step

        # ============================================
        # WITHDRAWAL REFLEX
        # Trigger: noxious stimulus on limb
        # Response: flex (pull away) the stimulated limb
        # Spinal cord level
        # ============================================
        if touch_left > 0.3:
            output['withdrawal_left'] = touch_left
        if touch_right > 0.3:
            output['withdrawal_right'] = touch_right

        # Update state
        self.prev_tilt = tilt_fb
        self.prev_height = height

        return output


class BasalGangliaGating:
    """Basal ganglia: prioritizes which reflex/behavior wins.

    At birth, basal ganglia is relatively mature. It modulates which
    brainstem reflex gets executed when multiple are triggered.

    Priority (hardwired):
      1. Moro/startle (survival — highest)
      2. Righting (balance)
      3. Withdrawal (pain avoidance)
      4. Stepping (locomotion)
      5. Rooting (feeding)
    """

    def gate(self, reflex_outputs: dict) -> dict:
        """Apply priority gating to reflex outputs.

        If a higher-priority reflex is active, lower-priority ones are suppressed.
        """
        gated = dict(reflex_outputs)

        # Moro/startle suppresses everything else
        moro_active = abs(reflex_outputs.get('moro', 0)) > 0.3
        startle_active = reflex_outputs.get('startle', 0) > 0.3

        if moro_active or startle_active:
            gated['stepping_drive'] *= 0.1
            gated['rooting'] *= 0.0
            # Righting still partially active (survival)
            gated['righting'] *= 0.5

        # Withdrawal suppresses stepping on that side
        if reflex_outputs.get('withdrawal_left', 0) > 0.3:
            gated['stepping_drive'] *= 0.5
        if reflex_outputs.get('withdrawal_right', 0) > 0.3:
            gated['stepping_drive'] *= 0.5

        return gated
