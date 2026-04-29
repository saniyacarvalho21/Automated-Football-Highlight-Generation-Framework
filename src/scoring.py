def semantic(
    motion,
    audio,
    goal,
    crowd,
    whistle,
    replay,
    zoom,
    sentiment
):

    weights = {
        "motion":0.25,
        "audio":0.25,
        "goal":0.15,
        "crowd":0.10,
        "whistle":0.05,
        "replay":0.10,
        "zoom":0.05,
        "sentiment":0.05
    }

    score = (
        weights["motion"]*motion +
        weights["audio"]*audio +
        weights["goal"]*goal +
        weights["crowd"]*crowd +
        weights["whistle"]*whistle +
        weights["replay"]*replay +
        weights["zoom"]*zoom +
        weights["sentiment"]*sentiment
    )

    return score