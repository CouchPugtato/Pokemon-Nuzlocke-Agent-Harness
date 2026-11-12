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
                # requiring text detail before asserting visual similarity
                if not a or not b or any(key not in value for value in (a, b)
                                         for key in ("dialogue_edges", "interface_edges")):
                    return False
                for left_tile, right_tile in zip(a["interface_edges"], b["interface_edges"]):
                    x, y = set(left_tile), set(right_tile)
                    if len(x ^ y) > max(8, len(x | y) * 0.1):
                        return False
                edges_a, edges_b = set(a["dialogue_edges"]), set(b["dialogue_edges"])
                union = edges_a | edges_b
                if union and len(edges_a ^ edges_b) / len(union) > 0.2:
                    return False
                return bool(all(
                    sum(abs(x - y) for x, y in zip(a[region], b[region])) / len(a[region]) <= similarity
                    for region in ("scene", "dialogue")))

            if all(matches(tail[index], tail[index % period]) for index in range(period, count)):
                signals.append({"kind": kind, "period": period, "repetitions": repetitions,
                                "evidence_observation_ids": list(dict.fromkeys(
                                    frame["observation_id"] for frame in tail if frame["observation_id"] is not None))})
                break
    return signals


def image_features(png):
    """scene brightness and dialogue edges without semantic claims"""
    import io
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        image = image.convert("L")
        split = max(1, image.height * 2 // 3)
        # retaining text edges that coarse brightness averages hide
        left, top = image.width * 5 // 100, image.height * 72 // 100
        text = image.crop((left, top, max(left + 1, image.width * 95 // 100),
                           max(top + 1, image.height * 94 // 100))).resize((240, 64))
        pixels = list(text.getdata())
        edges = [y * 240 + x for y in range(63) for x in range(239)
                 if max(abs(pixels[y * 240 + x] - pixels[y * 240 + x + 1]),
                        abs(pixels[y * 240 + x] - pixels[(y + 1) * 240 + x])) >= 32]
        # retaining local changes throughout menus and upper text fields
        pixels = list(image.resize((240, 160)).tobytes())
        tiles = [[] for _ in range(24)]
        for y in range(159):
            for x in range(239):
                value = pixels[y * 240 + x]
                if max(abs(value - pixels[y * 240 + x + 1]), abs(value - pixels[(y + 1) * 240 + x])) >= 32:
                    tiles[(y // 40) * 6 + x // 40].append((y % 40) * 40 + x % 40)
        return {"scene": list(image.crop((0, 0, image.width, split)).resize((8, 8)).tobytes()),
                "dialogue": list(image.crop((0, min(split, image.height - 1), image.width, image.height)).resize((8, 8)).tobytes()),
                "dialogue_edges": edges, "interface_edges": tiles}
