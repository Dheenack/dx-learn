def multi_objective_score(
    accuracy: float,
    explainability: float,
    train_time: float,
    w_acc=0.6,
    w_exp=0.3,
    w_time=0.1,
):
    return (
        w_acc * accuracy +
        w_exp * explainability -
        w_time * train_time
    )
