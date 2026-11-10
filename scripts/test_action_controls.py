import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from compiler.world_controls import render_controls,compile_timeline
from compiler.projection import project_object
from state.action_session import validate_plan,initialize_world,start_action,propose_timeline,apply_assessment
from state.db import WorldDB
from agent.action_planner import assess_phase


def plan():
    return {"objects":[{"object_id":"actor","category":"person","description":"blue shirt",
        "bbox_normalized":[.4,.2,.6,.8],"approx_height":1.75,"confidence":.8}],"render_only":["cloth"],
        "phases":[{"name":"step","render_prompt":"Take a step","duration_sec":73/24,
        "moves":[{"object_id":"actor","delta_world":[.3,0,0]}],"camera_delta":[0,0,0],
        "camera_target_id":None,"completion_criterion":"Foot settles"} for _ in range(2)]}


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.world=WorldDB(Path(self.tmp.name)/"world.sqlite")
        self.plan=plan();validate_plan(self.plan,73/24,3)
        initialize_world(self.world,self.plan,"original.png",320,192)
        start_action(self.world,"step twice",self.plan)
        self.timeline=propose_timeline(self.world,self.plan["phases"][0],73)

    def tearDown(self):
        self.world.connection.close();self.tmp.cleanup()

    def decision(self,**kw):
        return {"action":"ACCEPT","phase_complete":True,"action_complete":False,
                "confidence":.9,"completion_evidence":["foot settles"],**kw}

    def test_next_action_blocked(self):
        with self.assertRaises(ValueError):start_action(self.world,"another",self.plan)

    def test_proposal_does_not_mutate_world(self):
        self.assertNotEqual(self.timeline[-1]["objects"][0]["position"],self.world.get_object("actor").position)

    def test_reject_preserves_world(self):
        before=self.world.snapshot()
        result=apply_assessment(self.world,self.decision(action="REGENERATE"),self.timeline,"a")
        self.assertEqual(before,self.world.snapshot());self.assertFalse(result["user_input_ready"])

    def test_partial_does_not_teleport(self):
        before=self.world.snapshot()
        apply_assessment(self.world,self.decision(phase_complete=False),self.timeline,"a")
        self.assertEqual(before,self.world.snapshot())

    def test_no_premature_action_completion(self):
        result=apply_assessment(self.world,self.decision(action_complete=True),self.timeline,"a")
        self.assertEqual(result["phase_index"],1);self.assertEqual(result["status"],"RUNNING")
        self.assertEqual(self.world.get_object("actor").provenance,"AGENT_CREATED")

    def test_two_phases_complete(self):
        apply_assessment(self.world,self.decision(),self.timeline,"a")
        timeline=propose_timeline(self.world,self.plan["phases"][1],73)
        result=apply_assessment(self.world,self.decision(action_complete=True),timeline,"b")
        self.assertTrue(result["user_input_ready"]);self.assertEqual(result["status"],"COMPLETE")

    def test_unfinished_final_goal_does_not_repeat_endpoint_commit(self):
        apply_assessment(self.world,self.decision(),self.timeline,"a")
        before=self.world.snapshot()
        timeline=propose_timeline(self.world,self.plan["phases"][1],73)
        result=apply_assessment(self.world,self.decision(),timeline,"b")
        self.assertEqual(before,self.world.snapshot());self.assertFalse(result["user_input_ready"])

    def test_uncertain_does_not_complete(self):
        result=apply_assessment(self.world,self.decision(confidence=.3),self.timeline,"a")
        self.assertEqual(result["phase_index"],0)

    def test_unknown_move_rejected(self):
        p=copy.deepcopy(self.plan);p["phases"][0]["moves"][0]["object_id"]="invented"
        with self.assertRaises(ValueError):validate_plan(p,73/24,3)

    def test_agent_can_choose_longer_phase(self):
        p=copy.deepcopy(self.plan);p["phases"][0]["duration_sec"]=124/24
        validate_plan(p,73/24,3)
        timeline=propose_timeline(self.world,p["phases"][0],124)
        self.assertEqual(len(timeline),124)

    def test_controls_and_depth(self):
        maps,projection,depth,known=render_controls(self.world.query_world(),self.world.get_meta("camera"),["actor"])
        self.assertTrue(known.any());self.assertTrue(np.isfinite(depth[known]).all())
        self.assertEqual(maps["layout"].shape,(192,320,3))
        self.assertEqual(maps["layout"][0,0].tolist(),[255,255,255])
        self.assertEqual(maps["layout"][96,160].tolist(),[255,255,255])
        p=project_object(self.world.get_object("actor"),self.world.get_meta("camera"))
        np.testing.assert_allclose(p["center"],[160,96],atol=.01)

    def test_identity_only_not_forced_into_layout(self):
        obj=self.world.get_object("actor");obj.attributes["control_level"]="identity_only"
        maps,projections,depth,known=render_controls([obj],self.world.get_meta("camera"),["actor"])
        self.assertFalse(known.any());self.assertFalse(projections)
        self.assertTrue((maps["layout"]==255).all())

    def test_no_control_timeline_padding(self):
        with self.assertRaises(ValueError):compile_timeline(self.timeline,Path(self.tmp.name)/"controls",124)

    def test_camera_continuity(self):
        np.testing.assert_allclose(self.timeline[0]["camera"]["world_to_camera"],self.world.get_meta("camera")["world_to_camera"])

    def test_motion_leaves_settling_interval(self):
        self.assertEqual(self.timeline[55]["objects"][0]["position"],self.timeline[-1]["objects"][0]["position"])

    def test_transition_rollback(self):
        timeline=copy.deepcopy(self.timeline)
        extra=copy.deepcopy(timeline[-1]["objects"][0]);extra["object_id"]="generated_tree"
        timeline[-1]["objects"].append(extra)
        before=self.world.snapshot();action=self.world.get_meta("active_action")
        with self.assertRaises(ValueError):apply_assessment(self.world,self.decision(),timeline,"bad")
        self.assertEqual(before,self.world.snapshot());self.assertEqual(action,self.world.get_meta("active_action"))

    def test_review_keeps_original_and_committed_references_distinct(self):
        with patch("agent.action_planner.json_call",return_value={}) as call:
            assess_phase({},"goal",{}, {},[{"path":"rollout.png","timestamp":0}],
                         "original.png",True,[{"phase_index":0}],"committed.png")
        self.assertEqual(call.call_args.args[3],["original.png","committed.png","rollout.png"])
        self.assertEqual(call.call_args.args[2]["completed_phases"],[{"phase_index":0}])

    def test_held_prop_follows_holder_without_fine_pose_control(self):
        from state.models import WorldObject
        self.world.save(WorldObject("staff","staff",position=[0,4,1],attributes={"control_level":"identity_only"}),"test")
        self.world.set_meta("relations",[{"subject":"staff","predicate":"held_by","object":"actor"}])
        timeline=propose_timeline(self.world,self.plan["phases"][0],73)
        child=next(o for o in timeline[-1]["objects"] if o["object_id"]=="staff")
        np.testing.assert_allclose(child["position"],[.3,4,1])

    def test_cyclic_attachment_rejected(self):
        p=copy.deepcopy(self.plan)
        other=copy.deepcopy(p["objects"][0]);other["object_id"]="other";p["objects"].append(other)
        p["relations"]=[{"subject":"actor","predicate":"held_by","object":"other"},
                        {"subject":"other","predicate":"held_by","object":"actor"}]
        with self.assertRaises(ValueError):validate_plan(p,73/24,3)


if __name__=="__main__":unittest.main()
