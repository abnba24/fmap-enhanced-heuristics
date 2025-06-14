(define (problem depotprob7654)
(:domain depot)
(:objects
 depot0 depot1 depot2 - depot
 distributor0 distributor1 distributor2 - distributor
 truck0 truck1 - truck
 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 - crate
 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 pallet8 pallet9 - pallet
 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 - hoist
)
(:shared-data
  (clear ?x - (either surface hoist))
  ((at ?t - truck) - place)
  ((pos ?c - crate) - (either place truck))
  ((on ?c - crate) - (either surface hoist truck)) - 
(either depot0 depot2 distributor0 distributor1 distributor2 truck0 truck1)
)
(:init
 (myAgent depot1)
 (= (pos crate0) distributor1)
 (clear crate0)
 (= (on crate0) pallet8)
 (= (pos crate1) depot0)
 (not (clear crate1))
 (= (on crate1) pallet9)
 (= (pos crate2) distributor0)
 (not (clear crate2))
 (= (on crate2) pallet3)
 (= (pos crate3) distributor2)
 (clear crate3)
 (= (on crate3) pallet6)
 (= (pos crate4) depot0)
 (clear crate4)
 (= (on crate4) pallet0)
 (= (pos crate5) depot0)
 (clear crate5)
 (= (on crate5) crate1)
 (= (pos crate6) distributor1)
 (not (clear crate6))
 (= (on crate6) pallet4)
 (= (pos crate7) distributor1)
 (clear crate7)
 (= (on crate7) crate6)
 (= (pos crate8) depot1)
 (clear crate8)
 (= (on crate8) pallet1)
 (= (pos crate9) distributor0)
 (clear crate9)
 (= (on crate9) crate2)
 (= (at truck0) depot1)
 (= (at truck1) depot2)
 (= (located hoist0) depot0)
 (clear hoist0)
 (= (located hoist1) depot1)
 (clear hoist1)
 (= (located hoist2) depot2)
 (clear hoist2)
 (= (located hoist3) distributor0)
 (clear hoist3)
 (= (located hoist4) distributor1)
 (clear hoist4)
 (= (located hoist5) distributor2)
 (clear hoist5)
 (= (placed pallet0) depot0)
 (not (clear pallet0))
 (= (placed pallet1) depot1)
 (not (clear pallet1))
 (= (placed pallet2) depot2)
 (clear pallet2)
 (= (placed pallet3) distributor0)
 (not (clear pallet3))
 (= (placed pallet4) distributor1)
 (not (clear pallet4))
 (= (placed pallet5) distributor2)
 (clear pallet5)
 (= (placed pallet6) distributor2)
 (not (clear pallet6))
 (= (placed pallet7) depot1)
 (clear pallet7)
 (= (placed pallet8) distributor1)
 (not (clear pallet8))
 (= (placed pallet9) depot0)
 (not (clear pallet9))
)
(:global-goal (and
 (= (on crate1) pallet8)
 (= (on crate2) pallet3)
 (= (on crate4) pallet0)
 (= (on crate5) pallet5)
 (= (on crate6) pallet1)
 (= (on crate7) crate6)
 (= (on crate9) crate7)
))
)
