def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    # Write code here
    return base_lr*(
        (current_step/warmup_steps)*int(current_step<warmup_steps)+
        (
            (total_steps-current_step)/(total_steps-warmup_steps)*(current_step>=warmup_steps)
        )
    )