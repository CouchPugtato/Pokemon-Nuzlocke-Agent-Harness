def loop_warning(hashes):
    if len(hashes) >= 4 and len(set(hashes[:4])) == 1:
        return "Four identical screenshots. Reassess: input may be ineffective; inspect menus or stop."
    if len(hashes) >= 6 and len(set(hashes[:6])) <= 2:
        return "Repeated visual states suggest an action loop. Try a different bounded action or stop."
    return None
