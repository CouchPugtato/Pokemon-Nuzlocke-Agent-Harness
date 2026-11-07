def loop_warning(hashes):
    if len(hashes) >= 4 and len(set(hashes[:4])) == 1:
        return "Four identical screenshots. Reassess: input may be ineffective; inspect menus or stop."
    if len(hashes) >= 6 and len(set(hashes[:6])) <= 2:
        return "Repeated visual states suggest an action loop. Try a different bounded action or stop."
    return None


def input_signature(actions):
    return tuple((action["kind"], tuple(sorted(action["buttons"]))) for action in actions)


def cycle_signals(frames, repetitions=3, max_period=8, similarity=3):
    """heuristic cycles over attempted inputs and resulting screenshots"""
    signals = []
    for kind in ("exact_visual", "similar_visual", "input_cycle"):
        for period in range(1, max_period + 1):
            count = period * repetitions
            if len(frames) < count:
                continue
            tail = frames[-count:]

            def matches(left, right):
                if kind == "exact_visual":
                    return left["image_hash"] is not None and left["image_hash"] == right["image_hash"]
                if kind == "input_cycle":
                    return left["signature"] == right["signature"]
                a, b = left.get("features"), right.get("features")
                return bool(a and b and all(
                    sum(abs(x - y) for x, y in zip(a[region], b[region])) / len(a[region]) <= similarity
                    for region in ("scene", "dialogue")))

            if all(matches(tail[index], tail[index % period]) for index in range(period, count)):
                signals.append({"kind": kind, "period": period, "repetitions": repetitions,
                                "evidence_observation_ids": list(dict.fromkeys(
                                    frame["observation_id"] for frame in tail if frame["observation_id"] is not None))})
                break
    return signals


def image_features(png):
    """coarse scene and dialogue similarity features without semantic claims"""
    import io
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        image = image.convert("L")
        split = max(1, image.height * 2 // 3)
        return {"scene": list(image.crop((0, 0, image.width, split)).resize((8, 8)).tobytes()),
                "dialogue": list(image.crop((0, min(split, image.height - 1), image.width, image.height)).resize((8, 8)).tobytes())}
