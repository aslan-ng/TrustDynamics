class FailurePolicy:

    def __init__(
        self,
        transitions: dict[str, float],
    ):
        """
        Parameters
        ----------
        transitions
            Mapping from task names to probabilities indicating where
            the workflow should resume after a task fails review.

            The probabilities must sum to 1.

            Example:
                {
                    "Design": 0.8,
                    "Requirements": 0.2,
                }

            means that after failure, there is an 80% chance of
            redoing "Design" and a 20% chance of returning to
            "Requirements".
        """
        if not transitions:
            raise ValueError("transitions cannot be empty.")

        if any(probability < 0 for probability in transitions.values()):
            raise ValueError("Probabilities must be non-negative.")

        total_probability = sum(transitions.values())

        if abs(total_probability - 1.0) > 1e-9:
            raise ValueError(
                "Transition probabilities must sum to 1."
            )

        self.transitions = transitions