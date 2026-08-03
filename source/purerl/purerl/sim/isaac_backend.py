"""Isaac Sim 5.1 backend implemented directly on its batched APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from purerl.config.env import EnvCfg
from purerl.sensors import HeightFieldSampler
from purerl.terrain import assign_terrain_tiles, combine_terrain_meshes, generate_terrain_tiles

from .articulation import ArticulationIndexMap


@dataclass
class IsaacArticulationState:
    """Policy-ordered articulation state refreshed after every control step."""

    root_position: Any
    root_quaternion: Any
    root_linear_velocity: Any
    root_angular_velocity: Any
    joint_positions: Any
    joint_velocities: Any
    joint_accelerations: Any
    joint_torques: Any
    body_positions: Any
    body_linear_velocities: Any
    body_angular_velocities: Any
    net_contact_forces: Any


class IsaacSimBackend:
    """Direct Isaac Sim backend for cloned flat-ground TienKung environments.

    ``SimulationApp`` must already be running before :meth:`initialize` is
    called. All Isaac Sim imports are deliberately kept inside methods so the
    rest of PureRL remains importable without the simulator installed.
    """

    def __init__(self, *, asset_cache_dir: str | Path = "/tmp/purerl-assets"):
        self.asset_cache_dir = Path(asset_cache_dir)
        self.num_envs = 0
        self.device = ""
        self.state: IsaacArticulationState | None = None
        self.all_env_ids: Any = None
        self.env_origins: Any = None
        self.index_map: ArticulationIndexMap | None = None
        self.joint_limits: Any = None
        self.joint_names: tuple[str, ...] = ()
        self.body_names: tuple[str, ...] = ()
        self.terrain_levels: Any = None
        self.terrain_columns: Any = None
        self.terrain_tile_indices: Any = None
        self._cfg: EnvCfg | None = None
        self._torch: Any = None
        self._world: Any = None
        self._articulation: Any = None
        self._body_view: Any = None
        self._contact_view: Any = None
        self._joint_indices: Any = None
        self._body_view_indices: Any = None
        self._num_bodies = 0
        self._default_joint_positions: Any = None
        self._previous_joint_velocities: Any = None
        self._terrain_sampler: HeightFieldSampler | None = None
        self._terrain_tile_origins: Any = None
        self._terrain_num_rows = 0
        self._terrain_num_cols = 0
        self._root_body_view_index = 0
        self._default_root_masses: Any = None
        self._default_root_com_positions: Any = None
        self._closed = False

    def initialize(self, cfg: EnvCfg) -> None:
        if self._world is not None:
            raise RuntimeError("IsaacSimBackend is already initialized")
        try:
            import numpy as np
            import omni.kit.commands
            import torch
            from isaacsim.core.api import World
            from isaacsim.core.api.sensors import RigidContactView
            from isaacsim.core.cloner import GridCloner
            from isaacsim.core.prims import Articulation, RigidPrim
            from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
            from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim must be installed and SimulationApp must be launched before creating the backend"
            ) from exc

        cfg.validate()
        self._cfg = cfg
        self._torch = torch
        self.num_envs = cfg.scene.num_envs
        self.device = cfg.sim.device
        self.all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        terrain_tiles = ()
        terrain_assignment = None
        if cfg.task_kind == "rough":
            terrain_tiles = generate_terrain_tiles(cfg.terrain, seed=cfg.seed)
            terrain_assignment = assign_terrain_tiles(
                terrain_tiles,
                cfg.terrain,
                num_envs=self.num_envs,
                seed=cfg.seed,
            )

        World.clear_instance()
        self._world = World(
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt * cfg.sim.render_interval,
            stage_units_in_meters=1.0,
            backend="torch",
            device=self.device,
        )
        if cfg.task_kind == "flat":
            self._world.scene.add_ground_plane(
                size=max(100.0, cfg.scene.env_spacing * self.num_envs**0.5 * 2.0),
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        robot_usd = self._import_robot_usd(omni.kit.commands, cfg)
        stage = get_current_stage()
        if terrain_tiles:
            self._create_rough_terrain(
                stage,
                terrain_tiles,
                np,
                UsdGeom,
                UsdPhysics,
                UsdShade,
                Vt,
            )
        base_env_path = "/World/envs/env_0"
        cloner = GridCloner(spacing=0.0 if terrain_assignment is not None else cfg.scene.env_spacing)
        cloner.define_base_env(base_env_path)
        UsdGeom.Xform.Define(stage, base_env_path)
        add_reference_to_stage(str(robot_usd), f"{base_env_path}/Robot")
        body_names, body_patterns = self._prepare_body_views(
            stage,
            f"{base_env_path}/Robot",
            PhysxSchema,
            Usd,
            UsdPhysics,
        )

        env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        origins = cloner.clone(
            source_prim_path=base_env_path,
            prim_paths=env_paths,
            position_offsets=None if terrain_assignment is None else terrain_assignment.origins,
            replicate_physics=True,
            enable_env_ids=terrain_assignment is not None,
        )
        origin_array = np.asarray(origins, dtype=np.float32)
        self.env_origins = torch.as_tensor(origin_array, dtype=torch.float32, device=self.device)
        if terrain_assignment is None:
            self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.terrain_columns = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.terrain_tile_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self.terrain_levels = torch.as_tensor(
                terrain_assignment.levels, dtype=torch.long, device=self.device
            )
            self.terrain_columns = torch.as_tensor(
                terrain_assignment.columns, dtype=torch.long, device=self.device
            )
            self.terrain_tile_indices = torch.as_tensor(
                terrain_assignment.tile_indices, dtype=torch.long, device=self.device
            )
            self._terrain_tile_origins = torch.as_tensor(
                np.stack([tile.origin for tile in terrain_tiles]),
                dtype=torch.float32,
                device=self.device,
            )
            self._terrain_num_rows = cfg.terrain.num_rows
            self._terrain_num_cols = cfg.terrain.num_cols
            self._terrain_sampler = HeightFieldSampler(
                np.stack([tile.height_field for tile in terrain_tiles]),
                np.stack([tile.origin for tile in terrain_tiles]),
                terrain_assignment.tile_indices,
                size=cfg.terrain.size,
                horizontal_scale=cfg.terrain.horizontal_scale,
                device=self.device,
            )

        self._body_view = self._world.scene.add(
            RigidPrim(
                prim_paths_expr=body_patterns,
                name="tienkung_body_view",
                reset_xform_properties=False,
            )
        )
        # Isaac Sim 5.1's Scene.add() expects a public ``name`` property that
        # RigidContactView does not provide, so this tensor view is initialized
        # explicitly after the world starts playing.
        self._contact_view = RigidContactView(
            prim_paths_expr=body_patterns,
            filter_paths_expr=[[] for _ in body_patterns],
            name="tienkung_contact_view",
            prepare_contact_sensors=False,
        )
        self._articulation = self._world.scene.add(
            Articulation(
                prim_paths_expr="/World/envs/env_.*/Robot",
                name="tienkung_view",
                reset_xform_properties=False,
            )
        )
        self._world.reset()
        self._contact_view.initialize()

        if self._articulation.count != self.num_envs:
            raise RuntimeError(
                f"Articulation view found {self._articulation.count} robots, expected {self.num_envs}"
            )
        self.joint_names = tuple(self._articulation.dof_names)
        self.body_names = tuple(self._articulation.body_names)
        self.index_map = ArticulationIndexMap.resolve(cfg.robot, self.joint_names, self.body_names)
        if self._body_view.count != self.num_envs * len(body_names):
            raise RuntimeError(
                f"Rigid body view found {self._body_view.count} bodies, "
                f"expected {self.num_envs * len(body_names)}"
            )
        missing_view_bodies = [name for name in self.body_names if name not in body_names]
        if missing_view_bodies:
            raise RuntimeError(
                "Rigid body view is missing articulation links: " + ", ".join(missing_view_bodies)
            )
        self._num_bodies = len(body_names)
        self._body_view_indices = torch.as_tensor(
            [body_names.index(name) for name in self.body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._root_body_view_index = body_names.index(cfg.robot.root_body_name)
        self._joint_indices = torch.as_tensor(
            self.index_map.joint_indices, dtype=torch.long, device=self.device
        )
        simulator_joint_limits = self._articulation.get_dof_limits()
        limit_indices = self._joint_indices.to(device=simulator_joint_limits.device)
        self.joint_limits = simulator_joint_limits[:, limit_indices].clone().to(self.device)
        self._default_joint_positions = torch.as_tensor(
            cfg.robot.default_joint_positions, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._previous_joint_velocities = torch.zeros_like(self._default_joint_positions)

        self._configure_articulation()
        root_body_indices = [self.index_map.root_body_index]
        self._default_root_masses = self._articulation.get_body_masses(
            body_indices=root_body_indices, clone=True
        )
        self._default_root_com_positions, _ = self._articulation.get_body_coms(
            body_indices=root_body_indices, clone=True
        )
        self.reset(self.all_env_ids)
        self.refresh()

    def zeros(self, shape: tuple[int, ...], *, dtype: str) -> Any:
        if self._torch is None:
            raise RuntimeError("Backend is not initialized")
        dtype_map = {
            "bool": self._torch.bool,
            "float": self._torch.float32,
            "long": self._torch.long,
        }
        try:
            torch_dtype = dtype_map[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported tensor dtype: {dtype}") from exc
        return self._torch.zeros(shape, dtype=torch_dtype, device=self.device)

    def set_joint_position_targets(self, targets: Any) -> None:
        self._require_initialized()
        expected_shape = (self.num_envs, len(self._cfg.robot.joint_names))
        if tuple(targets.shape) != expected_shape:
            raise ValueError(f"Joint targets have shape {tuple(targets.shape)}, expected {expected_shape}")
        self._articulation.set_joint_position_targets(targets, joint_indices=self._joint_indices)

    def simulate(self, *, render: bool) -> None:
        self._require_initialized()
        self._world.step(render=render)

    def refresh(self) -> None:
        self._require_initialized()
        root_position, root_quaternion = self._articulation.get_world_poses(clone=True)
        root_velocity = self._articulation.get_velocities(clone=True)
        joint_positions = self._articulation.get_joint_positions(
            joint_indices=self._joint_indices, clone=True
        )
        joint_velocities = self._articulation.get_joint_velocities(
            joint_indices=self._joint_indices, clone=True
        )
        joint_accelerations = (
            joint_velocities - self._previous_joint_velocities
        ) / self._cfg.sim.step_dt
        self._previous_joint_velocities.copy_(joint_velocities)
        joint_torques = self._articulation.get_applied_joint_efforts(
            joint_indices=self._joint_indices, clone=True
        )
        body_positions, _ = self._body_view.get_world_poses(clone=True)
        body_velocities = self._body_view.get_velocities(clone=True)
        net_contact_forces = self.get_net_contact_forces()
        body_positions = self._reshape_body_data(body_positions)
        body_velocities = self._reshape_body_data(body_velocities)
        self.state = IsaacArticulationState(
            root_position=root_position,
            root_quaternion=root_quaternion,
            root_linear_velocity=root_velocity[:, :3],
            root_angular_velocity=root_velocity[:, 3:],
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_accelerations=joint_accelerations,
            joint_torques=joint_torques,
            body_positions=body_positions,
            body_linear_velocities=body_velocities[..., :3],
            body_angular_velocities=body_velocities[..., 3:],
            net_contact_forces=net_contact_forces,
        )

    def get_net_contact_forces(self) -> Any:
        self._require_initialized()
        forces = self._contact_view.get_net_contact_forces(clone=True, dt=self._cfg.sim.dt)
        return self._reshape_body_data(forces)

    def sample_terrain_heights(self, world_points_xy: Any) -> Any:
        self._require_initialized()
        if self._terrain_sampler is None:
            return self._torch.zeros(
                world_points_xy.shape[:-1], dtype=self._torch.float32, device=self.device
            )
        return self._terrain_sampler.sample(world_points_xy)

    def set_terrain_levels(self, env_ids: Any, levels: Any) -> None:
        self._require_initialized()
        if self._terrain_sampler is None:
            raise RuntimeError("Terrain levels are only available for generated terrain")
        env_ids = self._normalize_env_ids(env_ids)
        levels = self._torch.as_tensor(
            levels, dtype=self._torch.long, device=self.device
        ).flatten()
        if env_ids.shape != levels.shape:
            raise ValueError("env_ids and levels must have the same shape")
        if levels.numel() and (levels.min() < 0 or levels.max() >= self._terrain_num_rows):
            raise ValueError("Terrain level is outside the configured grid")

        tile_indices = levels * self._terrain_num_cols + self.terrain_columns[env_ids]
        self.terrain_levels[env_ids] = levels
        self.terrain_tile_indices[env_ids] = tile_indices
        self.env_origins[env_ids] = self._terrain_tile_origins[tile_indices]
        self._terrain_sampler.update_env_tiles(env_ids, tile_indices)

    def randomize_body_properties(
        self, env_ids: Any, mass_delta: Any, com_offset: Any
    ) -> None:
        self._require_initialized()
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        if tuple(mass_delta.shape) != (env_ids.numel(),) or tuple(com_offset.shape) != (
            env_ids.numel(),
            3,
        ):
            raise ValueError("Body randomization values do not match env_ids")

        cpu_ids = env_ids.detach().cpu()
        root_body_indices = [self.index_map.root_body_index]
        mass_ids = env_ids.to(self._default_root_masses.device)
        masses = self._default_root_masses[mass_ids] + mass_delta.to(
            self._default_root_masses.device
        )[:, None]
        com_ids = env_ids.to(self._default_root_com_positions.device)
        positions = self._default_root_com_positions[com_ids] + com_offset.to(
            self._default_root_com_positions.device
        )[:, None, :]
        self._articulation.set_body_masses(
            masses,
            indices=cpu_ids,
            body_indices=root_body_indices,
        )
        self._articulation.set_body_coms(
            positions=positions,
            indices=cpu_ids,
            body_indices=root_body_indices,
        )

    def set_actuator_gains(
        self, env_ids: Any, stiffness_scale: Any, damping_scale: Any
    ) -> None:
        self._require_initialized()
        env_ids = self._normalize_env_ids(env_ids)
        expected = (env_ids.numel(), len(self._cfg.robot.joint_names))
        if tuple(stiffness_scale.shape) != expected or tuple(damping_scale.shape) != expected:
            raise ValueError("Actuator gain scales do not match env_ids and joint count")
        stiffness = self._torch.as_tensor(self._cfg.robot.stiffness).repeat(env_ids.numel(), 1)
        damping = self._torch.as_tensor(self._cfg.robot.damping).repeat(env_ids.numel(), 1)
        self._articulation.set_gains(
            kps=stiffness * stiffness_scale.detach().cpu(),
            kds=damping * damping_scale.detach().cpu(),
            indices=env_ids.detach().cpu(),
            joint_indices=self._joint_indices.detach().cpu(),
        )

    def randomize_reset_state(
        self,
        env_ids: Any,
        root_xy: Any,
        root_yaw: Any,
        joint_position_scale: Any,
    ) -> None:
        self._require_initialized()
        env_ids = self._normalize_env_ids(env_ids)
        count = env_ids.numel()
        expected_joints = (count, len(self._cfg.robot.joint_names))
        if tuple(root_xy.shape) != (count, 2) or tuple(root_yaw.shape) != (count,):
            raise ValueError("Root reset randomization values do not match env_ids")
        if tuple(joint_position_scale.shape) != expected_joints:
            raise ValueError("Joint reset scales do not match env_ids and joint count")

        root_positions = self.env_origins[env_ids].clone()
        root_positions[:, :2] += root_xy
        root_positions[:, 2] += self._cfg.robot.default_root_height
        root_quaternions = self._torch.zeros((count, 4), dtype=self._torch.float32, device=self.device)
        root_quaternions[:, 0] = self._torch.cos(root_yaw * 0.5)
        root_quaternions[:, 3] = self._torch.sin(root_yaw * 0.5)
        root_velocities = self._torch.zeros((count, 6), dtype=self._torch.float32, device=self.device)
        joint_positions = self._default_joint_positions[env_ids] * joint_position_scale
        joint_velocities = self._torch.zeros_like(joint_positions)

        self._articulation.set_world_poses(root_positions, root_quaternions, indices=env_ids)
        self._articulation.set_velocities(root_velocities, indices=env_ids)
        self._articulation.set_joint_positions(
            joint_positions, indices=env_ids, joint_indices=self._joint_indices
        )
        self._articulation.set_joint_velocities(
            joint_velocities, indices=env_ids, joint_indices=self._joint_indices
        )
        self._articulation.set_joint_position_targets(
            joint_positions, indices=env_ids, joint_indices=self._joint_indices
        )
        self._previous_joint_velocities[env_ids] = 0.0

    def apply_root_wrench(self, env_ids: Any, forces: Any, torques: Any) -> None:
        self._require_initialized()
        env_ids = self._normalize_env_ids(env_ids)
        expected = (env_ids.numel(), 3)
        if tuple(forces.shape) != expected or tuple(torques.shape) != expected:
            raise ValueError("Root wrench values do not match env_ids")
        body_indices = self._root_body_view_index * self.num_envs + env_ids
        self._body_view.apply_forces_and_torques_at_pos(
            forces=forces,
            torques=torques,
            indices=body_indices,
            is_global=False,
        )

    def add_root_velocity(self, env_ids: Any, linear_velocity_xy: Any) -> None:
        self._require_initialized()
        env_ids = self._normalize_env_ids(env_ids)
        if tuple(linear_velocity_xy.shape) != (env_ids.numel(), 2):
            raise ValueError("Root velocity values do not match env_ids")
        velocities = self._articulation.get_velocities(indices=env_ids, clone=True)
        velocities[:, :2] += linear_velocity_xy
        self._articulation.set_velocities(velocities, indices=env_ids)
        if self.state is not None:
            self.state.root_linear_velocity[env_ids, :2] = velocities[:, :2]

    def reset(self, env_ids: Any) -> None:
        self._require_initialized()
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        root_positions = self.env_origins[env_ids].clone()
        root_positions[:, 2] += self._cfg.robot.default_root_height
        root_quaternions = self._torch.zeros((count, 4), dtype=self._torch.float32, device=self.device)
        root_quaternions[:, 0] = 1.0
        root_velocities = self._torch.zeros((count, 6), dtype=self._torch.float32, device=self.device)
        joint_positions = self._default_joint_positions[env_ids]
        joint_velocities = self._torch.zeros_like(joint_positions)

        self._articulation.set_world_poses(root_positions, root_quaternions, indices=env_ids)
        self._articulation.set_velocities(root_velocities, indices=env_ids)
        self._articulation.set_joint_positions(
            joint_positions, indices=env_ids, joint_indices=self._joint_indices
        )
        self._articulation.set_joint_velocities(
            joint_velocities, indices=env_ids, joint_indices=self._joint_indices
        )
        self._articulation.set_joint_position_targets(
            joint_positions, indices=env_ids, joint_indices=self._joint_indices
        )
        self._previous_joint_velocities[env_ids] = 0.0

    def nonzero(self, mask: Any) -> Any:
        return self._torch.nonzero(mask, as_tuple=False).flatten()

    def clone(self, value: Any) -> Any:
        return value.clone()

    def close(self) -> None:
        if self._closed:
            return
        if self._world is not None:
            world_type = type(self._world)
            self._world.stop()
            self._world.clear()
            world_type.clear_instance()
        self._world = None
        self._articulation = None
        self._body_view = None
        self._contact_view = None
        self._closed = True

    def _import_robot_usd(self, kit_commands: Any, cfg: EnvCfg) -> Path:
        self.asset_cache_dir.mkdir(parents=True, exist_ok=True)
        source = cfg.robot.urdf_path.resolve()
        destination = self.asset_cache_dir / f"{source.stem}-isaacsim-5.1.usd"
        if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
            return destination

        status, import_config = kit_commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Isaac Sim failed to create a URDF import configuration")
        import_config.merge_fixed_joints = cfg.robot.merge_fixed_joints
        import_config.fix_base = False
        import_config.make_default_prim = True
        import_config.create_physics_scene = False
        import_config.import_inertia_tensor = True
        import_config.distance_scale = 1.0
        import_config.set_self_collision(cfg.robot.self_collisions)
        import_config.set_collision_from_visuals(False)
        status, imported_path = kit_commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(source),
            import_config=import_config,
            dest_path=str(destination),
            get_articulation_root=True,
        )
        if not status or not destination.is_file():
            raise RuntimeError(
                f"Isaac Sim failed to import {source} to {destination}; result={imported_path!r}"
            )
        return destination

    def _configure_articulation(self) -> None:
        num_envs = self.num_envs
        robot = self._cfg.robot

        def policy_vector(values: tuple[float, ...]) -> Any:
            return self._torch.as_tensor(values, dtype=self._torch.float32, device=self.device).repeat(
                num_envs, 1
            )

        self._articulation.set_gains(
            kps=policy_vector(robot.stiffness),
            kds=policy_vector(robot.damping),
            joint_indices=self._joint_indices,
        )
        self._articulation.set_max_efforts(
            policy_vector(robot.effort_limits), joint_indices=self._joint_indices
        )
        self._articulation.set_max_joint_velocities(
            policy_vector(robot.velocity_limits), joint_indices=self._joint_indices
        )
        self._articulation.set_armatures(
            policy_vector(robot.armature), joint_indices=self._joint_indices
        )
        self._articulation.set_solver_position_iteration_counts(
            self._torch.full(
                (num_envs,), robot.solver_position_iterations, dtype=self._torch.int32, device=self.device
            )
        )
        self._articulation.set_solver_velocity_iteration_counts(
            self._torch.full(
                (num_envs,), robot.solver_velocity_iterations, dtype=self._torch.int32, device=self.device
            )
        )
        self._articulation.set_enabled_self_collisions(
            self._torch.full(
                (num_envs,), robot.self_collisions, dtype=self._torch.bool, device=self.device
            )
        )

    def _create_rough_terrain(
        self,
        stage: Any,
        terrain_tiles: Any,
        np: Any,
        usd_geom: Any,
        usd_physics: Any,
        usd_shade: Any,
        vt: Any,
    ) -> None:
        vertices, faces = combine_terrain_meshes(terrain_tiles)
        mesh = usd_geom.Mesh.Define(stage, "/World/terrain/mesh")
        mesh.CreatePointsAttr().Set(vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexIndicesAttr().Set(vt.IntArray.FromNumpy(faces.reshape(-1)))
        face_counts = np.full(len(faces), 3, dtype=np.int32)
        mesh.CreateFaceVertexCountsAttr().Set(vt.IntArray.FromNumpy(face_counts))
        mesh.CreateSubdivisionSchemeAttr().Set(usd_geom.Tokens.none)

        terrain_prim = mesh.GetPrim()
        usd_physics.CollisionAPI.Apply(terrain_prim)
        mesh_collision = usd_physics.MeshCollisionAPI.Apply(terrain_prim)
        mesh_collision.CreateApproximationAttr().Set("none")

        material = usd_shade.Material.Define(stage, "/World/terrain/physicsMaterial")
        material_api = usd_physics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(1.0)
        material_api.CreateDynamicFrictionAttr().Set(1.0)
        material_api.CreateRestitutionAttr().Set(0.0)
        binding = usd_shade.MaterialBindingAPI.Apply(terrain_prim)
        binding.Bind(
            material,
            bindingStrength=usd_shade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

    def _prepare_body_views(
        self,
        stage: Any,
        robot_path: str,
        physx_schema: Any,
        usd: Any,
        usd_physics: Any,
    ) -> tuple[tuple[str, ...], list[str]]:
        robot_prim = stage.GetPrimAtPath(robot_path)
        body_prims = [
            prim
            for prim in usd.PrimRange(robot_prim, usd.TraverseInstanceProxies())
            if prim.HasAPI(usd_physics.RigidBodyAPI)
        ]
        if not body_prims:
            raise RuntimeError(f"Imported robot at {robot_path} does not contain rigid bodies")

        body_names = tuple(prim.GetName() for prim in body_prims)
        if len(body_names) != len(set(body_names)):
            raise RuntimeError("Imported robot contains duplicate rigid body names")
        patterns = []
        for prim in body_prims:
            contact_api = (
                physx_schema.PhysxContactReportAPI(prim)
                if prim.HasAPI(physx_schema.PhysxContactReportAPI)
                else physx_schema.PhysxContactReportAPI.Apply(prim)
            )
            contact_api.CreateThresholdAttr().Set(0.0)
            patterns.append(str(prim.GetPath()).replace("/env_0/", "/env_.*/", 1))
        return body_names, patterns

    def _reshape_body_data(self, values: Any) -> Any:
        body_major = values.reshape(self._num_bodies, self.num_envs, values.shape[-1])
        env_major = body_major.transpose(0, 1)
        return env_major[:, self._body_view_indices]

    def _normalize_env_ids(self, env_ids: Any) -> Any:
        if isinstance(env_ids, self._torch.Tensor):
            return env_ids.to(device=self.device, dtype=self._torch.long).flatten()
        return self._torch.as_tensor(env_ids, dtype=self._torch.long, device=self.device).flatten()

    def _require_initialized(self) -> None:
        if (
            self._world is None
            or self._articulation is None
            or self._body_view is None
            or self._contact_view is None
            or self._cfg is None
        ):
            raise RuntimeError("Backend is not initialized")
