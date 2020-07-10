(ns nn-clojure.train-test
  (:require [clojure.test :refer :all]
            [nn-clojure.train :refer :all]
            [nn-clojure.datatypes :as dt]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def seed 42) ;; arbitrary value
(def ctx
  (binding [dt/*r* (java.util.Random. seed)] ;; create the random number generator
     (let [nn            (dt/nn [2 4 1])
           activation-fn ::dt/sigmoid

           learning-rate      1.5
           max-epochs      3000
           target-err-rate    0.01

           eval-fn   (fn [ctx] (< (max-err ctx) 0.1))

           size       2
           ratio      0.5
           pattern-00 {:train/inputs     [0 0]
                       :train/range-down [0.0 0.0]
                       :train/range-up   [0.0 0.0]
                       :train/goal       [0]}
           pattern-01 {:train/inputs [0 1]
                       :train/range-down [0.0 0.0]
                       :train/range-up [0.0 0]
                       :train/goal [1]}
           pattern-10 {:train/inputs [1 0]
                       :train/range-down [0.0 0.0]
                       :train/range-up [0.0 0.0]
                       :train/goal [1]}
           pattern-11 {:train/inputs [1 1]
                       :train/range-down [0.0 0.0]
                       :train/range-up [0.0 0]
                       :train/goal [1]}
           batch-size 4

           ctx {::dt/nn             nn
                ::dt/learning-rate  learning-rate
                :activation/fn-name activation-fn
                ::dt/rnd            dt/*r*
                :train/eval-fn      eval-fn
                :train/target       target-err-rate
                :train/batch-size   4
                :train/min-epochs   1
                :train/max-epochs   max-epochs}
           ]

       (-> ctx
           (+training-data size
                           ratio
                           pattern-00
                           pattern-01
                           pattern-10
                           pattern-11)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(deftest evaluate-test
  (testing "OR network training"
    (let [epochs   1
          ctx      (-> ctx train evaluate)]
      (is (and (= ((:train/eval-fn ctx) ctx) true)
               (= (:train/epochs ctx) 109))))))

(deftest evaluate-result-test
  (testing "Evaluation report string"
    (let [epochs   1
          ctx      (-> ctx train)
          bad-ctx  (merge ctx {:train/epochs epochs
                               :train/eval-fn (fn [_] false)})
          good-ctx (merge ctx {:train/epochs epochs
                               :train/eval-fn (fn [_] true)})]
      (is (= (evaluate-result bad-ctx)  "Training FAILED after 1 epochs."))
      (is (= (evaluate-result good-ctx) "Training SUCCEEDED after 1 epochs.")))))
