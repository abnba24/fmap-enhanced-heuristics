(define (problem depotprob7654)
(:domain depot)
(:objects
 depot0 depot1 depot2 - depot
 distributor0 distributor1 distributor2 - distributor
 truck0 truck1 - truck
 crate0 crate1 crate2 crate3 crate4 crate5 - crate
 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 - pallet
 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 - hoist
)
(:shared-data
  (clear ?x - (either surface hoist))
  ((at ?t - truck) - place)
  ((pos ?c - crate) - (either place truck))
  ((on ?c - crate) - (either surface hoist truck)) - 
(either depot1 depot2 distributor0 distributor1 distributor2 truck0 truck1)
)
(:init
 (myAgent depot0)
 (= (pos crate0) depot1)
 (clear crate0)
 (= (on crate0) pallet1)
 (= (pos crate1) depot0)
 (clear crate1)
 (= (on crate1) pallet0)
 (= (pos crate2) distributor2)
 (not (clear crate2))
 (= (on crate2) pallet5)
 (= (pos crate3) distributor2)
 (clear crate3)
 (= (on crate3) crate2)
 (= (pos crate4) depot2)
 (clear crate4)
 (= (on crate4) pallet2)
 (= (pos crate5) distributor0)
 (clear crate5)
 (= (on crate5) pallet3)
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
 (not (clear pallet2))
 (= (placed pallet3) distributor0)
 (not (clear pallet3))
 (= (placed pallet4) distributor1)
 (clear pallet4)
 (= (placed pallet5) distributor2)
 (not (clear pallet5))
)
(:global-goal (and
 (= (on crate0) crate4)
 (= (on crate2) pallet3)
 (= (on crate3) pallet0)
 (= (on crate4) pallet5)
))
)
