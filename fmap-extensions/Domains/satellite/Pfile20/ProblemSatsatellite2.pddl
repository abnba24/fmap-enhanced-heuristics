(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
 satellite0 satellite1 satellite2 satellite3 satellite4 - satellite
 instrument0 instrument1 instrument2 instrument3 instrument4 instrument5 instrument6 instrument7 instrument8 instrument9 instrument10 instrument11 instrument12 instrument13 instrument14 instrument15 instrument16 instrument17 instrument18 instrument19 instrument20 instrument21 instrument22 instrument23 instrument24 instrument25 instrument26 instrument27 instrument28 - instrument
 spectrograph6 infrared1 thermograph8 infrared5 image3 infrared9 image2 thermograph7 image4 spectrograph0 - mode
 star0 star2 star4 groundstation3 groundstation1 phenomenon5 star6 planet7 phenomenon8 planet9 star10 star11 phenomenon12 planet13 phenomenon14 planet15 star16 planet17 planet18 phenomenon19 star20 phenomenon21 planet22 star23 star24 - direction
)
(:shared-data
  ((pointing ?s - satellite) - direction)
  (have_image ?d - direction ?m - mode) - (either satellite0 satellite1 satellite3 satellite4)
)
(:init (mySatellite satellite2)
 (power_avail satellite2)
 (not (power_on instrument0))
 (not (calibrated instrument0))
 (= (calibration_target instrument0) star2)
 (not (power_on instrument1))
 (not (calibrated instrument1))
 (= (calibration_target instrument1) star4)
 (not (power_on instrument2))
 (not (calibrated instrument2))
 (= (calibration_target instrument2) star4)
 (not (power_on instrument3))
 (not (calibrated instrument3))
 (= (calibration_target instrument3) star0)
 (not (power_on instrument4))
 (not (calibrated instrument4))
 (= (calibration_target instrument4) groundstation3)
 (not (power_on instrument5))
 (not (calibrated instrument5))
 (= (calibration_target instrument5) groundstation3)
 (not (power_on instrument6))
 (not (calibrated instrument6))
 (= (calibration_target instrument6) groundstation3)
 (not (power_on instrument7))
 (not (calibrated instrument7))
 (= (calibration_target instrument7) groundstation1)
 (not (power_on instrument8))
 (not (calibrated instrument8))
 (= (calibration_target instrument8) star2)
 (not (power_on instrument9))
 (not (calibrated instrument9))
 (= (calibration_target instrument9) star4)
 (not (power_on instrument10))
 (not (calibrated instrument10))
 (= (calibration_target instrument10) star4)
 (not (power_on instrument11))
 (not (calibrated instrument11))
 (= (calibration_target instrument11) groundstation3)
 (not (power_on instrument12))
 (not (calibrated instrument12))
 (= (calibration_target instrument12) groundstation3)
 (not (power_on instrument13))
 (not (calibrated instrument13))
 (= (calibration_target instrument13) star4)
 (not (power_on instrument14))
 (not (calibrated instrument14))
 (= (calibration_target instrument14) groundstation3)
 (not (power_on instrument15))
 (not (calibrated instrument15))
 (= (calibration_target instrument15) star2)
 (not (power_on instrument16))
 (not (calibrated instrument16))
 (= (calibration_target instrument16) star4)
 (not (power_on instrument17))
 (not (calibrated instrument17))
 (= (calibration_target instrument17) star0)
 (not (power_on instrument18))
 (not (calibrated instrument18))
 (= (calibration_target instrument18) groundstation1)
 (not (power_on instrument19))
 (not (calibrated instrument19))
 (= (calibration_target instrument19) groundstation3)
 (not (power_on instrument20))
 (not (calibrated instrument20))
 (= (calibration_target instrument20) groundstation1)
 (not (power_on instrument21))
 (not (calibrated instrument21))
 (= (calibration_target instrument21) star2)
 (not (power_on instrument22))
 (not (calibrated instrument22))
 (= (calibration_target instrument22) star4)
 (not (power_on instrument23))
 (not (calibrated instrument23))
 (= (calibration_target instrument23) star2)
 (not (power_on instrument24))
 (not (calibrated instrument24))
 (= (calibration_target instrument24) star2)
 (not (power_on instrument25))
 (not (calibrated instrument25))
 (= (calibration_target instrument25) star2)
 (not (power_on instrument26))
 (not (calibrated instrument26))
 (= (calibration_target instrument26) star4)
 (not (power_on instrument27))
 (not (calibrated instrument27))
 (= (calibration_target instrument27) groundstation3)
 (not (power_on instrument28))
 (not (calibrated instrument28))
 (= (calibration_target instrument28) groundstation1)
 (not (have_image phenomenon5 thermograph8))
 (not (have_image phenomenon5 spectrograph0))
 (not (have_image phenomenon5 image3))
 (not (have_image star6 spectrograph0))
 (not (have_image star6 spectrograph6))
 (not (have_image star6 image3))
 (not (have_image planet7 spectrograph6))
 (not (have_image planet7 infrared5))
 (not (have_image planet7 image2))
 (not (have_image phenomenon8 spectrograph6))
 (not (have_image phenomenon8 infrared5))
 (not (have_image phenomenon8 thermograph7))
 (not (have_image planet9 spectrograph6))
 (not (have_image star10 spectrograph6))
 (not (have_image star11 thermograph7))
 (not (have_image star11 image4))
 (not (have_image star11 image3))
 (not (have_image phenomenon12 image4))
 (not (have_image planet13 infrared5))
 (not (have_image planet13 spectrograph6))
 (not (have_image planet13 image2))
 (not (have_image phenomenon14 thermograph7))
 (not (have_image planet15 image3))
 (not (have_image star16 image3))
 (not (have_image star16 image4))
 (not (have_image planet18 infrared9))
 (not (have_image planet18 infrared5))
 (not (have_image planet18 thermograph7))
 (not (have_image phenomenon19 image2))
 (not (have_image phenomenon19 image4))
 (not (have_image star20 spectrograph0))
 (not (have_image phenomenon21 image4))
 (not (have_image phenomenon21 image2))
 (not (have_image phenomenon21 thermograph7))
 (not (have_image planet22 image2))
 (not (have_image planet22 spectrograph6))
 (not (have_image star23 image2))
 (not (have_image star23 infrared9))
 (not (have_image star24 spectrograph6))
 (not (have_image star24 infrared5))
 (= (pointing satellite2) phenomenon14)
 (= (on_board satellite2) {instrument16 instrument17 instrument18 instrument19})
 (not (= (on_board satellite2) {instrument0 instrument1 instrument2 instrument3 instrument4 instrument5 instrument6 instrument7 instrument8 instrument9 instrument10 instrument11 instrument12 instrument13 instrument14 instrument15 instrument20 instrument21 instrument22 instrument23 instrument24 instrument25 instrument26 instrument27 instrument28}))
 (= (supports instrument0) {image3})
 (not (= (supports instrument0) {spectrograph6 infrared1 thermograph8 infrared5 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument1) {infrared9})
 (not (= (supports instrument1) {spectrograph6 infrared1 thermograph8 infrared5 image3 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument2) {thermograph8 image2 image4})
 (not (= (supports instrument2) {spectrograph6 infrared1 infrared5 image3 infrared9 thermograph7 spectrograph0}))
 (= (supports instrument3) {infrared9})
 (not (= (supports instrument3) {spectrograph6 infrared1 thermograph8 infrared5 image3 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument4) {thermograph8 image3})
 (not (= (supports instrument4) {spectrograph6 infrared1 infrared5 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument5) {infrared9 image4})
 (not (= (supports instrument5) {spectrograph6 infrared1 thermograph8 infrared5 image3 image2 thermograph7 spectrograph0}))
 (= (supports instrument6) {infrared1})
 (not (= (supports instrument6) {spectrograph6 thermograph8 infrared5 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument7) {spectrograph6 thermograph8})
 (not (= (supports instrument7) {infrared1 infrared5 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument8) {infrared9 thermograph7 spectrograph0})
 (not (= (supports instrument8) {spectrograph6 infrared1 thermograph8 infrared5 image3 image2 image4}))
 (= (supports instrument9) {thermograph7})
 (not (= (supports instrument9) {spectrograph6 infrared1 thermograph8 infrared5 image3 infrared9 image2 image4 spectrograph0}))
 (= (supports instrument10) {spectrograph6 infrared1 thermograph8})
 (not (= (supports instrument10) {infrared5 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument11) {infrared1 thermograph7 image4})
 (not (= (supports instrument11) {spectrograph6 thermograph8 infrared5 image3 infrared9 image2 spectrograph0}))
 (= (supports instrument12) {thermograph8 infrared5 infrared9})
 (not (= (supports instrument12) {spectrograph6 infrared1 image3 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument13) {infrared1 image2})
 (not (= (supports instrument13) {spectrograph6 thermograph8 infrared5 image3 infrared9 thermograph7 image4 spectrograph0}))
 (= (supports instrument14) {image3})
 (not (= (supports instrument14) {spectrograph6 infrared1 thermograph8 infrared5 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument15) {thermograph7})
 (not (= (supports instrument15) {spectrograph6 infrared1 thermograph8 infrared5 image3 infrared9 image2 image4 spectrograph0}))
 (= (supports instrument16) {infrared9 image2})
 (not (= (supports instrument16) {spectrograph6 infrared1 thermograph8 infrared5 image3 thermograph7 image4 spectrograph0}))
 (= (supports instrument17) {infrared5})
 (not (= (supports instrument17) {spectrograph6 infrared1 thermograph8 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument18) {infrared9})
 (not (= (supports instrument18) {spectrograph6 infrared1 thermograph8 infrared5 image3 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument19) {infrared5 image2})
 (not (= (supports instrument19) {spectrograph6 infrared1 thermograph8 image3 infrared9 thermograph7 image4 spectrograph0}))
 (= (supports instrument20) {image3 image2 image4})
 (not (= (supports instrument20) {spectrograph6 infrared1 thermograph8 infrared5 infrared9 thermograph7 spectrograph0}))
 (= (supports instrument21) {thermograph8 infrared5 image3})
 (not (= (supports instrument21) {spectrograph6 infrared1 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument22) {thermograph8 infrared5})
 (not (= (supports instrument22) {spectrograph6 infrared1 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument23) {thermograph8 image3})
 (not (= (supports instrument23) {spectrograph6 infrared1 infrared5 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument24) {thermograph8})
 (not (= (supports instrument24) {spectrograph6 infrared1 infrared5 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument25) {infrared5})
 (not (= (supports instrument25) {spectrograph6 infrared1 thermograph8 image3 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument26) {image3})
 (not (= (supports instrument26) {spectrograph6 infrared1 thermograph8 infrared5 infrared9 image2 thermograph7 image4 spectrograph0}))
 (= (supports instrument27) {infrared9 image2})
 (not (= (supports instrument27) {spectrograph6 infrared1 thermograph8 infrared5 image3 thermograph7 image4 spectrograph0}))
 (= (supports instrument28) {thermograph7 image4 spectrograph0})
 (not (= (supports instrument28) {spectrograph6 infrared1 thermograph8 infrared5 image3 infrared9 image2}))
)
(:global-goal (and
 (= (pointing satellite1) phenomenon19)
 (have_image phenomenon5 thermograph8)
 (have_image phenomenon5 spectrograph0)
 (have_image phenomenon5 image3)
 (have_image star6 spectrograph0)
 (have_image star6 spectrograph6)
 (have_image star6 image3)
 (have_image planet7 spectrograph6)
 (have_image planet7 infrared5)
 (have_image planet7 image2)
 (have_image phenomenon8 spectrograph6)
 (have_image phenomenon8 infrared5)
 (have_image phenomenon8 thermograph7)
 (have_image planet9 spectrograph6)
 (have_image star10 spectrograph6)
 (have_image star11 thermograph7)
 (have_image star11 image4)
 (have_image star11 image3)
 (have_image phenomenon12 image4)
 (have_image planet13 infrared5)
 (have_image planet13 spectrograph6)
 (have_image planet13 image2)
 (have_image phenomenon14 thermograph7)
 (have_image planet15 image3)
 (have_image star16 image3)
 (have_image star16 image4)
 (have_image planet18 infrared9)
 (have_image planet18 infrared5)
 (have_image planet18 thermograph7)
 (have_image phenomenon19 image2)
 (have_image phenomenon19 image4)
 (have_image star20 spectrograph0)
 (have_image phenomenon21 image4)
 (have_image phenomenon21 image2)
 (have_image phenomenon21 thermograph7)
 (have_image planet22 image2)
 (have_image planet22 spectrograph6)
 (have_image star23 image2)
 (have_image star23 infrared9)
 (have_image star24 spectrograph6)
 (have_image star24 infrared5)
))
)
