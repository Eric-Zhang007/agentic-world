"""Check persisted evidence from the live Pi replay, without another API call."""
from world_tools import ROOT, WorldDB, read, save, scene_from_db

out = ROOT/'outputs/pi_migration_20261102'
parent = WorldDB(ROOT/'outputs/video_reconcile_guard_v2_20261009/world.sqlite')
branch = WorldDB(out/'world.sqlite')
try:
    history = branch.get_meta('whitebox_history')
    inherited = parent.get_meta('whitebox_history')
    scene = scene_from_db(branch)
    tiger = next(e for e in scene['entities'] if e['id']=='tiger_guard')
    checks = {
        'inherited_history_unchanged': history[:-1] == inherited,
        'one_commit_only': len(history) == len(inherited)+1,
        'partial_action_retained': branch.get_meta('whitebox_video_action')['action_complete'] is False,
        'actual_crouch_committed': tiger['posture_weights']['crouched'] == 1.,
        'next_plan_starts_at_actual_scene': read(out/'next_plan/scene.json') == scene,
        'next_plan_uses_current_revision': read(out/'next_plan/status.json')['revision'] == branch.get_meta('whitebox_revision'),
        'source_revision_unchanged': parent.get_meta('whitebox_revision') == 1,
    }
    save(out/'migration_verification.json', {'checks': checks, 'passed': all(checks.values()),
         'scope': 'Real Pi tool calls and historical video replay; no fresh H3 generation'})
    print(checks)
    assert all(checks.values())
finally:
    parent.connection.close()
    branch.connection.close()
