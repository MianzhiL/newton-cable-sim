EVALUATION_EPISODES = 285


def validate_admission(manifest, admission):
    train = manifest['splits']['train']
    test = manifest['splits']['test']
    quarantined = admission['quarantined_train_ids']
    active = admission['active_train_ids']
    if (admission.get('mode') != 'per_state_admission'
            or admission.get('bank_sha256') != manifest['sha256']
            or not admission.get('infrastructure_passed')
            or not admission.get('canonical_grip_unchanged')
            or not admission.get('exact_retained_state_copy')
            or admission.get('mapped_parent_scan_count') != len(train) + len(test)
            or admission.get('peer_reset_checks', 0) < 3
            or not admission.get('train_test_separation')
            or not set(quarantined).issubset(set(train))
            or len(quarantined) != len(set(quarantined))
            or active != [initial for initial in train if initial not in set(quarantined)]
            or len(active) < 2048 or len(active) < .9 * len(train)
            or admission.get('evaluation_ids') != test or len(test) != EVALUATION_EPISODES
            or len(set(train + test)) != len(train) + len(test)
            or set(train + test) != set(range(len(train) + len(test)))
            or admission.get('maximum_reset_attempts') != 3):
        raise RuntimeError('Invalid per-state admission contract')
    return active
