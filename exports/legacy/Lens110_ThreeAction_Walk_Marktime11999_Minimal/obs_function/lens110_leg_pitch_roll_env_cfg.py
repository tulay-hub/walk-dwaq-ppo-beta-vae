from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
import legged_lab.tasks.locomotion.amp.mdp.lens110_leg_pitch_roll as leg_pr_mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import ObservationsCfg as AmpObservationsCfg
from legged_lab.tasks.locomotion.amp.config.lens110.lens110_amp_env_cfg import (
    ANIMATION_TERM_NAME,
    Lens110AmpEnvCfg,
    Lens110AmpEnvCfg_PLAY,
)


WALK_MOTION_NAMES = [
    "02_02_stageii_walk",
    "16_34_stageii_walk2stand",
    "B10_-__Walk_turn_left_45_stageii",
    "B13_-__Walk_turn_right_90_stageii",
    "B14_-__Walk_turn_right_45_t2_stageii",
    "B15_-__Walk_turn_around_stageii",
    "B9_-__Walk_turn_left_90_stageii",
    "move_back",
    "move_l",
    "move_r",
    "turn_l",
    "turn_r",
]


@configclass
class LegPitchRollDiscriminatorCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    joint_pos = ObsTerm(
        func=leg_pr_mdp.leg_pitch_roll_joint_pos,
        params={"asset_cfg": leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG},
    )
    joint_vel = ObsTerm(
        func=leg_pr_mdp.leg_pitch_roll_joint_vel,
        params={"asset_cfg": leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1
        self.history_length = 3
        self.flatten_history_dim = False


@configclass
class LegPitchRollDiscriminatorDemoCfg(ObsGroup):
    ref_root_lin_vel_b = ObsTerm(
        func=mdp.ref_root_lin_vel_b,
        params={
            "animation": ANIMATION_TERM_NAME,
            "flatten_steps_dim": False,
        },
    )
    ref_root_ang_vel_b = ObsTerm(
        func=mdp.ref_root_ang_vel_b,
        params={
            "animation": ANIMATION_TERM_NAME,
            "flatten_steps_dim": False,
        },
    )
    ref_joint_pos = ObsTerm(
        func=leg_pr_mdp.ref_leg_pitch_roll_joint_pos,
        params={
            "animation": ANIMATION_TERM_NAME,
            "flatten_steps_dim": False,
        },
    )
    ref_joint_vel = ObsTerm(
        func=leg_pr_mdp.ref_leg_pitch_roll_joint_vel,
        params={
            "animation": ANIMATION_TERM_NAME,
            "flatten_steps_dim": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1


@configclass
class LegPitchRollObservationsCfg(AmpObservationsCfg):
    disc: LegPitchRollDiscriminatorCfg = LegPitchRollDiscriminatorCfg()
    disc_demo: LegPitchRollDiscriminatorDemoCfg = LegPitchRollDiscriminatorDemoCfg()


@configclass
class Lens110LegPitchRollAmpEnvCfg(Lens110AmpEnvCfg):
    """Lens110 AMP walking config whose policy ankle IO is ankle_pitch/ankle_roll."""

    observations: LegPitchRollObservationsCfg = LegPitchRollObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._use_leg_pitch_roll_io()
        if self.__class__.__name__ == "Lens110LegPitchRollAmpEnvCfg":
            self.disable_zero_weight_rewards()

    def _use_leg_pitch_roll_io(self):
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=leg_pr_mdp.LEG_PITCH_ROLL_JOINT_NAMES,
            preserve_order=True,
            scale=0.25,
            use_default_offset=True,
        )
        self.actions.scripted_arm_swing = leg_pr_mdp.LegDrivenArmSwingActionCfg(
            asset_name="robot",
            joint_names=leg_pr_mdp.SCRIPTED_ARM_SWING_JOINT_NAMES,
            preserve_order=True,
            shoulder_pitch_gain=-1.6,
            shoulder_pitch_limit=0.55,
            elbow_gain=1.0,
            elbow_limit=0.25,
            smoothing=0.20,
        )
        self.scene.robot.actuators["feet"].stiffness = {
            ".*ankle_pitch_joint": 10.0,
            ".*ankle_roll_joint": 10.0,
        }
        self.scene.robot.actuators["feet"].damping = {
            ".*ankle_pitch_joint": 10.0,
            ".*ankle_roll_joint": 10.0,
        }

        self.observations.policy.joint_pos = ObsTerm(
            func=leg_pr_mdp.leg_pitch_roll_joint_pos_rel,
            params={"asset_cfg": leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 降低噪声: -0.03 -> -0.01 (减少左右腿感知差异)
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=leg_pr_mdp.leg_pitch_roll_joint_vel_rel,
            params={"asset_cfg": leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG},
            noise=Unoise(n_min=-0.8, n_max=0.8),  # 降低噪声: -1.75 -> -0.8 (减少左右腿感知差异)
        )

        self.observations.critic.joint_pos = ObsTerm(
            func=leg_pr_mdp.leg_pitch_roll_joint_pos,
            params={"asset_cfg": leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG},
        )
        self.observations.critic.joint_vel = ObsTerm(
            func=leg_pr_mdp.leg_pitch_roll_joint_vel,
            params={"asset_cfg": leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG},
        )
        self.observations.disc = LegPitchRollDiscriminatorCfg()
        self.observations.disc_demo = LegPitchRollDiscriminatorDemoCfg()
        self.observations.disc.key_body_pos_b = None
        self.observations.disc_demo.ref_key_body_pos_b = None

        self.motion_data.motion_dataset.motion_data_weights = {
            "02_02_stageii_walk": (2.5, "auto"),
            "16_34_stageii_walk2stand": (1.8, "auto"),
            "B10_-__Walk_turn_left_45_stageii": (0.6, "auto"),
            "B13_-__Walk_turn_right_90_stageii": (0.6, "auto"),
            "B14_-__Walk_turn_right_45_t2_stageii": (0.6, "auto"),
            "B15_-__Walk_turn_around_stageii": (0.5, "auto"),
            "B9_-__Walk_turn_left_90_stageii": (0.6, "auto"),
            "move_back": (0.3, "auto"),
            "move_l": (0.3, "auto"),
            "move_r": (0.3, "auto"),
            "turn_l": (0.4, "auto"),
            "turn_r": (0.4, "auto"),
        }

        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.02, 0.02)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)
        self.curriculum.lin_vel_cmd_levels.params["lin_vel_x_limit"] = [0.15, 0.8]
        self.curriculum.lin_vel_cmd_levels.params["lin_vel_y_limit"] = [-0.02, 0.02]

        self.rewards.joint_vel_l2.params["asset_cfg"] = leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG
        self.rewards.joint_acc_l2.params["asset_cfg"] = leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG
        self.rewards.joint_pos_limits.params["asset_cfg"] = leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG
        self.rewards.joint_torques_l2.params["asset_cfg"] = leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG
        self.rewards.joint_regularization.params["asset_cfg"] = leg_pr_mdp.LEG_PITCH_ROLL_ASSET_CFG
        self.rewards.ang_vel_xy_l2.weight = -0.18
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.joint_acc_l2.weight = -4.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.joint_torques_l2.weight = -1.5e-5
        self.rewards.feet_air_time.weight = 0.8
        self.rewards.feet_air_time.params["threshold"] = 0.32
        self.rewards.feet_slide.weight = -0.22
        self.rewards.feet_distance.weight = 0.18
        self.rewards.knee_distance.weight = 0.12
        self.rewards.joint_deviation_arms.weight = 0.0
        self.rewards.joint_deviation_torso.weight = 0.0
        self.rewards.arm_symmetry.weight = 0.0
        self.rewards.arm_vel_symmetry.weight = 0.0


@configclass
class Lens110LegPitchRollAmpEnvCfg_PLAY(Lens110AmpEnvCfg_PLAY, Lens110LegPitchRollAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
