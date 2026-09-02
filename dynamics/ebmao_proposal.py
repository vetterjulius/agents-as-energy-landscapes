import random
import torch


class EBMAOAssignmentProposal:
    def __init__(
        self,
        energy_registry,
        lambda_align=0.5,
        lambda_memory=None,
        num_tasks=4,
        block_size=3,
        agent_sample_size=None,
        mode="guided",
    ):
        self.energy_registry = energy_registry

        self.lambda_align = float(
            lambda_align
        )

        self.lambda_memory = (
            float(lambda_memory)
            if lambda_memory is not None
            else self.lambda_align
        )

        self.num_tasks = max(
            1,
            int(num_tasks),
        )

        self.block_size = max(
            1,
            int(block_size),
        )

        if agent_sample_size is None:
            self.agent_sample_size = None
        else:
            self.agent_sample_size = max(
                1,
                int(agent_sample_size),
            )

        self.mode = mode

        if self.mode not in (
            "guided",
            "random",
        ):
            raise ValueError(
                f"Unknown proposal mode: {self.mode}. "
                f"Expected 'guided' or 'random'."
            )

    def propose(self, state):
        if self.mode == "random":
            X_prop = self._random_swap(state)

        else:
            r = random.random()

            # Guided single-task proposal.
            if r < 0.45:
                X_prop = self._guided_single_swap(
                    state
                )

            # Guided multi-task proposal.
            elif r < 0.75:
                X_prop = self._guided_block_move(
                    state
                )

            # Random exploration.
            elif r < 0.95:
                X_prop = self._random_swap(
                    state
                )

            # Occasional exhaustive local search.
            else:
                X_prop = self._full_single_reassignment(
                    state
                )

        # --------------------------------------------------------------
        # Avoid no-op proposals whenever possible.
        # --------------------------------------------------------------

        if (
            state.N > 1
            and torch.equal(
                X_prop,
                state.X,
            )
        ):
            X_prop = self._random_swap(
                state
            )

        self._validate_assignment(
            state,
            X_prop,
        )

        return X_prop

    def _guided_single_swap(self, state):
        """
        Select high-priority tasks according to the EBMAO assignment
        criterion and find the best single-task reassignment.
        """

        tasks = self._select_tasks(
            state,
            self.num_tasks,
        )

        return self._best_swap_among_tasks(
            state,
            tasks,
        )

    def _guided_block_move(self, state):

        tasks = self._select_tasks(
            state,
            self.block_size,
        )

        return self._best_block_move(
            state,
            tasks,
        )

    def _random_swap(self, state):

        X_new = state.X.clone()

        if state.M <= 0:
            return X_new

        if state.N <= 1:
            return X_new

        # Select one task uniformly.
        task_idx = random.randint(
            0,
            state.M - 1,
        )

        # Current agent.
        old_agent = torch.argmax(
            state.X[:, task_idx]
        ).item()

        # Alternative agents.
        candidates = [
            agent_idx
            for agent_idx in range(state.N)
            if agent_idx != old_agent
        ]

        if not candidates:
            return X_new

        new_agent = random.choice(
            candidates
        )

        # Reassign exactly one task.
        X_new[
            old_agent,
            task_idx,
        ] = 0.0

        X_new[
            new_agent,
            task_idx,
        ] = 1.0

        return X_new

    def _select_tasks(self, state, k):
        """
        Select tasks whose current assignments are comparatively weak
        according to the EBMAO assignment structure.

        For task i assigned to agent a:

            importance_i =
                ||s_a - c_i||²
                - lambda_align * s_a^T c_i
                - lambda_memory * s_a^T kappa_a

        Higher importance means that the current assignment is less
        attractive and therefore more promising for reassignment.
        """

        if state.M <= 0:
            return []

        if k >= state.M:
            return list(range(state.M))

        k = max(
            1,
            int(k),
        )

        # Current agent assigned to each task.
        assigned_agents = torch.argmax(
            state.X,
            dim=0,
        )

        # Capability vectors of assigned agents.
        assigned_capabilities = state.s[
            assigned_agents
        ]

        # Memory vectors of assigned agents.
        assigned_memory = state.kappa[
            assigned_agents
        ]

        # --------------------------------------------------------------
        # Assignment distance
        # --------------------------------------------------------------

        distance = torch.sum(
            (
                assigned_capabilities
                - state.c
            ) ** 2,
            dim=1,
        )

        # --------------------------------------------------------------
        # Capability/task alignment
        # --------------------------------------------------------------

        alignment = torch.sum(
            assigned_capabilities
            * state.c,
            dim=1,
        )

        # --------------------------------------------------------------
        # Memory alignment
        # --------------------------------------------------------------

        memory_alignment = torch.sum(
            assigned_capabilities
            * assigned_memory,
            dim=1,
        )

        # --------------------------------------------------------------
        # EBMAO task importance
        # --------------------------------------------------------------

        importance = (
            distance
            - self.lambda_align * alignment
            - self.lambda_memory * memory_alignment
        )

        _, top_indices = torch.topk(
            importance,
            k,
            largest=True,
        )

        return top_indices.tolist()

    # ==================================================================
    # Candidate-agent selection
    # ==================================================================

    def _candidate_agents(self, state, old_agent):
        """
        Return candidate replacement agents.

        If no agent sampling is configured, all alternative agents
        are evaluated.

        Otherwise a random subset is evaluated.
        """

        candidates = [
            agent_idx
            for agent_idx in range(state.N)
            if agent_idx != old_agent
        ]

        if not candidates:
            return []

        if (
            self.agent_sample_size is None
            or self.agent_sample_size >= len(candidates)
        ):
            return candidates

        return random.sample(
            candidates,
            self.agent_sample_size,
        )

    # ==================================================================
    # Guided single-task reassignment
    # ==================================================================

    def _best_swap_among_tasks(self, state, tasks):
        """
        Search for the best single-task reassignment among the
        selected tasks and candidate agents.

        The original state is always restored before returning.
        """

        X_orig = state.X.clone()

        best_X = X_orig.clone()

        best_E, _ = self.energy_registry.compute(
            state
        )
        best_E = best_E.item()

        for task_idx in tasks:

            old_agent = torch.argmax(
                X_orig[:, task_idx]
            ).item()

            candidates = self._candidate_agents(
                state,
                old_agent,
            )

            for new_agent in candidates:

                X_prop = X_orig.clone()

                X_prop[
                    old_agent,
                    task_idx,
                ] = 0.0

                X_prop[
                    new_agent,
                    task_idx,
                ] = 1.0

                state.X = X_prop

                E, _ = self.energy_registry.compute(
                    state
                )
                E_val = E.item()

                if E_val < best_E:
                    best_E = E_val
                    best_X = X_prop.clone()

        # Restore state.
        state.X = X_orig

        return best_X

    # ==================================================================
    # Guided block reassignment
    # ==================================================================

    def _best_block_move(self, state, tasks):
        """
        Construct a sequentially improved block proposal.

        Each selected task is considered using the current intermediate
        block assignment.

        The complete block proposal is returned to the sampler.
        """

        X_orig = state.X.clone()

        X_prop = X_orig.clone()

        best_E, _ = self.energy_registry.compute(
            state
        )
        best_E = best_E.item()

        for task_idx in tasks:

            old_agent = torch.argmax(
                X_prop[:, task_idx]
            ).item()

            candidates = self._candidate_agents(
                state,
                old_agent,
            )

            best_local_X = X_prop.clone()
            best_local_E = best_E

            for new_agent in candidates:

                X_trial = X_prop.clone()

                X_trial[
                    old_agent,
                    task_idx,
                ] = 0.0

                X_trial[
                    new_agent,
                    task_idx,
                ] = 1.0

                state.X = X_trial

                E, _ = self.energy_registry.compute(
                    state
                )
                E_val = E.item()

                if E_val < best_local_E:
                    best_local_E = E_val
                    best_local_X = X_trial.clone()

            X_prop = best_local_X

            if best_local_E < best_E:
                best_E = best_local_E

        # Restore original state.
        state.X = X_orig

        return X_prop

    # ==================================================================
    # Exhaustive single-task reassignment
    # ==================================================================

    def _full_single_reassignment(self, state):
        """
        Exhaustively evaluate every possible single-task reassignment.

        This is intentionally used only occasionally by guided mode
        because it is more computationally expensive.
        """

        X_orig = state.X.clone()

        best_X = X_orig.clone()

        best_E, _ = self.energy_registry.compute(
            state
        )
        best_E = best_E.item()

        for task_idx in range(state.M):

            old_agent = torch.argmax(
                X_orig[:, task_idx]
            ).item()

            for new_agent in range(state.N):

                if new_agent == old_agent:
                    continue

                X_prop = X_orig.clone()

                X_prop[
                    old_agent,
                    task_idx,
                ] = 0.0

                X_prop[
                    new_agent,
                    task_idx,
                ] = 1.0

                state.X = X_prop

                E, _ = self.energy_registry.compute(
                    state
                )
                E_val = E.item()

                if E_val < best_E:
                    best_E = E_val
                    best_X = X_prop.clone()

        # Restore original state.
        state.X = X_orig

        return best_X

    # ==================================================================
    # Assignment invariant
    # ==================================================================

    @staticmethod
    def _validate_assignment(state, X):
        """
        Verify that every task is assigned to exactly one agent.
        """

        expected = torch.ones(
            state.M,
            dtype=X.dtype,
            device=X.device,
        )

        column_sums = X.sum(
            dim=0
        )

        if not torch.allclose(
            column_sums,
            expected,
        ):
            raise RuntimeError(
                "EBMAOAssignmentProposal produced an invalid assignment: "
                "every task must be assigned exactly once."
            )