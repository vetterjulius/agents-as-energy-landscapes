import random
import torch


class AssignmentProposal:
    """
    Proposal mechanism for the classical Energy-based orchestrator.

    The proposal mechanism is deliberately separated from the sampler:

        Proposal
            -> generates a candidate X'

        SimulatedAnnealingSampler
            -> evaluates E(X')
            -> computes dE
            -> applies Metropolis acceptance

    Therefore:

        mode="random"
            means random proposal generation.

        mode="guided"
            means energy-guided proposal generation.

    The sampler remains responsible for deciding whether a proposal
    is actually accepted.
    """

    def __init__(
        self,
        energy_registry,
        lambda_align=0.5,
        num_tasks=4,
        block_size=3,
        agent_sample_size=None,
        mode="guided",
    ):
        self.energy_registry = energy_registry
        self.lambda_align = float(lambda_align)

        self.num_tasks = max(1, int(num_tasks))
        self.block_size = max(1, int(block_size))

        if agent_sample_size is None:
            self.agent_sample_size = None
        else:
            self.agent_sample_size = max(
                1,
                int(agent_sample_size),
            )

        self.mode = mode

        if self.mode not in ("guided", "random"):
            raise ValueError(
                f"Unknown proposal mode: {self.mode}. "
                f"Expected 'guided' or 'random'."
            )

    # ==================================================================
    # Public API
    # ==================================================================

    def propose(self, state):
        """
        Generate one valid assignment proposal.

        Random mode
        -----------
        Generates exactly one random single-task reassignment.

        Guided mode
        -----------
        Uses a mixture of:
            - guided single-task reassignment
            - guided block reassignment
            - random single-task reassignment
            - exhaustive single-task reassignment

        The proposal itself does NOT perform Metropolis acceptance.
        Acceptance is handled exclusively by the sampler.

        Returns
        -------
        torch.Tensor
            Proposed assignment matrix X' with shape [N, M].

        Invariant
        ---------
        Every task remains assigned to exactly one agent:

            X'.sum(dim=0) == 1
        """

        if self.mode == "random":
            X_prop = self._random_swap(state)

        else:
            r = random.random()

            if r < 0.45:
                X_prop = self._guided_single_swap(state)

            elif r < 0.75:
                X_prop = self._guided_block_move(state)

            elif r < 0.95:
                X_prop = self._random_swap(state)

            else:
                X_prop = self._full_single_reassignment(state)

        # --------------------------------------------------------------
        # Avoid no-op proposals whenever possible.
        # For N=1 (single agent), all proposals result in identical X.
        # For N>1, retry with random swap if proposal didn't change anything.
        # --------------------------------------------------------------

        if torch.equal(X_prop, state.X) and state.N > 1:
            # Fallback: force a change with random swap
            X_prop = self._random_swap(state)
            # Note: Even after retry, X_prop might still equal state.X in edge cases.
            # This is acceptable - the sampler will simply reject with 100% probability.

        self._validate_assignment(
            state,
            X_prop,
        )

        return X_prop

    # ==================================================================
    # Guided proposal selection
    # ==================================================================

    def _guided_single_swap(self, state):
        """
        Select promising tasks and find the best single reassignment
        among the sampled candidate agents.
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
        """
        Sequentially improve a small block of selected tasks.

        This is still a proposal mechanism, not a separate optimization
        algorithm. The resulting candidate is later evaluated and
        accepted/rejected by the sampler.
        """

        tasks = self._select_tasks(
            state,
            self.block_size,
        )

        return self._best_block_move(
            state,
            tasks,
        )

    # ==================================================================
    # Random proposal
    # ==================================================================

    def _random_swap(self, state):
        """
        Randomly reassign exactly one task to another agent.

        This is the proposal used by pure SA.

        Important:
            "random" refers only to how X' is generated.
            The sampler still performs the Metropolis acceptance test.

        Assignment invariant:

            X.sum(dim=0) == 1
        """

        X_new = state.X.clone()

        if state.M <= 0:
            return X_new

        if state.N <= 1:
            return X_new

        # Select one task uniformly at random.
        task_idx = random.randint(
            0,
            state.M - 1,
        )

        # Current agent assigned to this task.
        old_agent = torch.argmax(
            state.X[:, task_idx]
        ).item()

        # All alternative agents.
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
        X_new[old_agent, task_idx] = 0.0
        X_new[new_agent, task_idx] = 1.0

        return X_new

    # ==================================================================
    # Task selection for guided proposals
    # ==================================================================

    def _select_tasks(self, state, k):
        """
        Select tasks that currently have high assignment mismatch.

        The score is based on the currently assigned agent and the
        task embedding.

        Higher score means that the task is considered a stronger
        candidate for reassignment.
        """

        if state.M <= 0:
            return []

        if k >= state.M:
            return list(range(state.M))

        k = max(
            1,
            int(k),
        )

        # Current assignment for every task.
        assigned_agents = torch.argmax(
            state.X,
            dim=0,
        )

        # Capability vector of the currently assigned agent.
        assigned_capabilities = state.s[
            assigned_agents
        ]

        # Squared Euclidean mismatch.
        distance = torch.sum(
            (
                assigned_capabilities
                - state.c
            ) ** 2,
            dim=1,
        )

        # Alignment contribution.
        alignment = torch.sum(
            assigned_capabilities
            * state.c,
            dim=1,
        )

        # Higher value = stronger candidate for reassignment.
        importance = (
            distance
            - self.lambda_align * alignment
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

        If agent_sample_size is None or larger than the available
        candidate set, all alternative agents are evaluated.

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
    # Guided single-task move
    # ==================================================================

    def _best_swap_among_tasks(self, state, tasks):
        """
        Find the best single-task reassignment among the selected tasks.

        The current state is restored before returning.
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

            candidate_agents = self._candidate_agents(
                state,
                old_agent,
            )

            for new_agent in candidate_agents:

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

        # Always restore the original state.
        state.X = X_orig

        return best_X

    # ==================================================================
    # Guided block move
    # ==================================================================

    def _best_block_move(self, state, tasks):
        """
        Construct a guided multi-task proposal.

        Tasks are processed sequentially. For each selected task,
        the locally best alternative agent is chosen according to
        the current energy.

        The final block candidate is returned to the sampler, which
        decides whether the complete proposal is accepted.
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

            candidate_agents = self._candidate_agents(
                state,
                old_agent,
            )

            best_local_X = X_prop.clone()
            best_local_E = best_E

            for new_agent in candidate_agents:

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

        # Restore original state before returning proposal.
        state.X = X_orig

        return X_prop

    # ==================================================================
    # Exhaustive reassignment
    # ==================================================================

    def _full_single_reassignment(self, state):
        """
        Exhaustively search all single-task / alternative-agent moves.

        This is the strongest guided proposal and is intentionally
        used only occasionally in guided mode.
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
        Validate the fundamental assignment invariant.

        Every task must be assigned to exactly one agent.
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
                "AssignmentProposal produced an invalid assignment: "
                "every task must be assigned exactly once."
            )